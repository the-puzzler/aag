"""Persistent global Gaussian assignment (report Section 1).

The whole point of the report is that a *persistent* per-example particle,
transported by cheap 1D rank operations, beats minibatch Gaussianization.
Nothing here trains a network; we just move points z_i in R^d, keeping the
identity x_i <-> z_i fixed throughout (Section 1.1).

Primitives:
  * greedy global rank transport            (1.2)
  * conditional offset-slab cleanup         (1.3)
  * periodic radial chi_d calibration       (1.4)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import stats


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _gaussian_quantiles(n: int, device, dtype) -> torch.Tensor:
    """q_i = Phi^{-1}((rank_i + 1/2)/n), sorted ascending (report 1.2)."""
    ranks = torch.arange(n, device=device, dtype=dtype)
    u = (ranks + 0.5) / n
    return torch.special.ndtri(u)


def whiten(h: torch.Tensor, rotate: bool = True):
    """Center and PCA-whiten the latent cloud so it starts isotropic.

    Returns (z0, mean, W) with z0 = (h - mean) @ W.  The inverse map is
    h = z @ W_inv + mean, needed if we ever want to decode back to AE latents.
    """
    mean = h.mean(0, keepdim=True)
    hc = h - mean
    if not rotate:
        # Scale-only: no eigenvector rotation, so coordinate j of z still MEANS
        # coordinate j of h. For a spatial AE latent that preserves the grid
        # topology, which a full PCA rotation destroys by mixing every grid
        # position into every coordinate. Leaves correlations for the transport
        # to remove -- it searches arbitrary directions anyway.
        sd = hc.std(0, keepdim=True).clamp_min(1e-8)
        W = torch.diag(1.0 / sd.squeeze(0))
        W_inv = torch.diag(sd.squeeze(0))
        return hc @ W, mean, W, W_inv
    cov = (hc.T @ hc) / (h.shape[0] - 1)
    cov = cov + 1e-5 * torch.eye(cov.shape[0], device=h.device, dtype=h.dtype)
    evals, evecs = torch.linalg.eigh(cov)
    # order by DESCENDING variance so whitened coordinate 0 is the top principal
    # direction; this makes a rank-k prior (zeroing coords k..d) keep the k most
    # informative directions, matching the report's "rank-k embedded prior".
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    W = evecs @ torch.diag(evals.clamp_min(1e-8).rsqrt())
    W_inv = torch.diag(evals.clamp_min(1e-8).sqrt()) @ evecs.T
    return hc @ W, mean, W, W_inv


def w2_to_standard_normal(vals: torch.Tensor) -> torch.Tensor:
    """W_2^2 between the empirical 1D distribution of `vals` and N(0,1).

    With equal-mass samples this is the mean squared gap between sorted samples
    and matched Gaussian quantiles — the objective maximized in 1.2 and 1.3.
    """
    s, _ = torch.sort(vals)
    q = _gaussian_quantiles(s.numel(), vals.device, vals.dtype)
    return ((s - q) ** 2).mean()


def _rand_unit(k: int, d: int, device, dtype) -> torch.Tensor:
    a = torch.randn(k, d, device=device, dtype=dtype)
    return a / a.norm(dim=1, keepdim=True)


# --------------------------------------------------------------------------- #
# 1.2  greedy global rank transport
# --------------------------------------------------------------------------- #
def greedy_rank_transport_step(z, *, search_subset, n_dirs, alpha, gen):
    """One global step: find the most non-Gaussian projection, rank-transport it.

    Direction search uses only `search_subset` points (report 1.2: "the
    direction search is performed on a fixed-size subset"); the winning update
    is then applied to the *full* dataset.
    """
    N, d = z.shape
    m = min(search_subset, N)
    idx = torch.randperm(N, device=z.device, generator=gen)[:m]
    zs = z[idx]

    dirs = _rand_unit(n_dirs, d, z.device, z.dtype)
    proj = zs @ dirs.T                                    # (m, n_dirs)
    # W2^2 of each candidate on the subset
    s, _ = torch.sort(proj, dim=0)
    q = _gaussian_quantiles(m, z.device, z.dtype).unsqueeze(1)
    scores = ((s - q) ** 2).mean(0)
    best = int(torch.argmax(scores))
    a = dirs[best]                                        # a*

    # full-dataset rank transport along a*
    proj_full = z @ a                                     # (N,)
    order = torch.argsort(proj_full)
    q_full = _gaussian_quantiles(N, z.device, z.dtype)
    target = torch.empty_like(proj_full)
    target[order] = q_full                                # target[i] = q at rank of i
    z.add_(alpha * (target - proj_full).unsqueeze(1) * a.unsqueeze(0))
    return float(scores[best])


# --------------------------------------------------------------------------- #
# 1.2b  attribute-conditional rank transport
# --------------------------------------------------------------------------- #
def conditional_rank_transport_step(z, cond, *, k, n_dirs, alpha, gen):
    """Same primitive as greedy_rank_transport_step, but the subset is a
    k-nearest-neighborhood in a binary attribute space (Hamming distance)
    around one randomly sampled real particle's condition vector, and the
    rank transport is applied to *just that k-subset* (not propagated to the
    full population). Interleaved with global steps, this additionally
    Gaussianizes each attribute-similar neighborhood on its own terms, so a
    downstream conditional generator sees a well-behaved z | condition.
    """
    N, d = z.shape
    qi = int(torch.randint(N, (1,), device=z.device, generator=gen))
    dist = (cond != cond[qi:qi + 1]).sum(1)              # Hamming distance, (N,)
    k = min(k, N)
    idx = torch.topk(dist, k, largest=False).indices
    zs = z[idx]

    dirs = _rand_unit(n_dirs, d, z.device, z.dtype)
    proj = zs @ dirs.T
    s, _ = torch.sort(proj, dim=0)
    q = _gaussian_quantiles(k, z.device, z.dtype).unsqueeze(1)
    scores = ((s - q) ** 2).mean(0)
    best = int(torch.argmax(scores))
    a = dirs[best]

    proj_sub = zs @ a
    order = torch.argsort(proj_sub)
    q_sub = _gaussian_quantiles(k, z.device, z.dtype)
    target = torch.empty_like(proj_sub)
    target[order] = q_sub
    zs = zs + alpha * (target - proj_sub).unsqueeze(1) * a.unsqueeze(0)
    z[idx] = zs
    return float(scores[best])


# --------------------------------------------------------------------------- #
# 1.2e  purely continuous conditional rank transport
# --------------------------------------------------------------------------- #
def cond_distance(cond, qi, metric="cosine"):
    """Distance from particle qi's condition to every other, for k-NN selection.

    'cosine' compares direction only -- for AE latents the magnitude often tracks
    brightness/contrast rather than content, so two frames of the same scene at
    different exposure stay near each other. 'l2' keeps magnitude as signal.
    """
    q = cond[qi:qi + 1]
    if metric == "cosine":
        return 1.0 - torch.nn.functional.cosine_similarity(q, cond, dim=1)
    if metric == "l2":
        # not torch.cdist: it materialises cond.pow(2), a second full-size copy.
        # At 1.66M x 6144 that is another 38 GiB on top of the 38 GiB already
        # resident and OOMs a 95 GiB card -- which is exactly how this failed.
        return _l2_to_all(q, cond).squeeze(0).clamp_min_(0).sqrt_()
    raise ValueError(f"unknown metric {metric!r}; expected 'cosine' or 'l2'")


def continuous_knn_transport_step(z, cond, *, k, n_dirs, alpha, gen, metric="cosine"):
    """Conditional transport when the condition is a plain continuous vector.

    Same primitive as everywhere else -- define a distance, take k neighbours,
    rank-transport that subset. Used for first-frame-conditioned video, where c is
    the per-frame AE embedding of frame 0 and there is no discrete tag to filter
    on: conditional_rank_transport_step measures Hamming distance over bits and
    the group/action steps need a categorical tag.
    """
    N, d = z.shape
    qi = int(torch.randint(N, (1,), device=z.device, generator=gen))
    dist = cond_distance(cond, qi, metric)
    k = min(k, N)
    idx = torch.topk(dist, k, largest=False).indices
    zs = z[idx]

    dirs = _rand_unit(n_dirs, d, z.device, z.dtype)
    s, _ = torch.sort(zs @ dirs.T, dim=0)
    q = _gaussian_quantiles(k, z.device, z.dtype).unsqueeze(1)
    scores = ((s - q) ** 2).mean(0)
    a = dirs[int(torch.argmax(scores))]

    proj = zs @ a
    target = torch.empty_like(proj)
    target[torch.argsort(proj)] = _gaussian_quantiles(k, z.device, z.dtype)
    z[idx] = zs + alpha * (target - proj).unsqueeze(1) * a.unsqueeze(0)
    return float(scores.max())



_SQNORM_CACHE = {}


def _sq_norms(x, chunk=65536):
    """Cached row-wise squared norms, computed in chunks.

    torch.cdist internally does x.pow(2).sum(-1), and x.pow(2) materialises a
    full-size temporary -- a second 38 GiB copy of the conditioning tensor at
    1.66M particles, which OOMs a 95 GiB card that already holds the original.
    Chunking avoids the temporary, and caching is free because cond never
    changes during an assignment: only z does.
    """
    key = (x.data_ptr(), x.shape, x.dtype)
    hit = _SQNORM_CACHE.get(key)
    if hit is not None:
        return hit
    out = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    for i in range(0, x.shape[0], chunk):
        out[i:i + chunk] = x[i:i + chunk].pow(2).sum(1)
    _SQNORM_CACHE.clear()          # only ever one cond per run; do not leak
    _SQNORM_CACHE[key] = out
    return out


_UNIT_CACHE = {}
_UNIT_CACHE_MAX = 8


def _unit(x, chunk=65536):
    """Cached row-normalised copy of x, built in chunks.

    cosine needs x / ||x|| for every row. Recomputing it per firing allocates a
    full-size copy each time -- 12.6 GB at 512k particles, 40.9 GB at 1.66M.
    cond never changes during an assignment, so build it once.

    Holds several entries rather than one. A multi-scale run cycles between
    context lengths within a single step, and a single-entry cache would
    renormalise every tensor on every step -- billions of elements per step,
    which dwarfs the transport itself. Eviction is oldest-first; at 512k
    particles the five scales 1/3/5/12/24 cost 23.6 GB together, against 83 GB
    free on this card.
    """
    key = (x.data_ptr(), tuple(x.shape), x.dtype)
    hit = _UNIT_CACHE.get(key)
    if hit is not None:
        return hit
    out = torch.empty_like(x)
    for i in range(0, x.shape[0], chunk):
        blk = x[i:i + chunk]
        out[i:i + chunk] = blk / blk.norm(dim=1, keepdim=True).clamp_min(1e-12)
    while len(_UNIT_CACHE) >= _UNIT_CACHE_MAX:
        _UNIT_CACHE.pop(next(iter(_UNIT_CACHE)))
    _UNIT_CACHE[key] = out
    return out


def _l2_to_all(q, x):
    """||q - x||^2 for every pair, via the matmul form so no (N, D) temporary
    is ever allocated. Returns squared distances, which rank identically."""
    qs = q.pow(2).sum(1, keepdim=True)                 # (B,1) -- q is tiny
    xs = _sq_norms(x).unsqueeze(0)                     # (1,N) -- cached
    return (qs + xs - 2.0 * (q @ x.T)).clamp_min_(0)


def continuous_knn_transport_batch(z, cond, *, k, n_dirs, alpha, gen, n_fire,
                                   metric="cosine", qis=None):
    """n_fire firings of continuous_knn_transport_step, one pass over `cond`.

    Identical in result to calling that function n_fire times: same query
    distribution, same neighbourhoods, same sequential updates to z. The only
    change is WHEN the distances are computed.

    It is worth doing because cond never changes -- only z does -- so the
    per-firing distance pass was re-reading a static 12.6 GB tensor. At 112
    firings per step that is 1.4 TB of memory traffic per step, which measured
    at 6.5 s/step and would have been ~90 hours at 1.66M particles. Computing
    all n_fire distance rows in one pass makes it one read instead of n_fire.

    Neighbour selection is hoisted (it depends only on cond); the transports
    still run one at a time against the live z, so this is not a Jacobi
    relaxation of the original -- it is the original.

    Note the queries are drawn as one batch rather than one per firing, so for a
    given seed the DRAWS differ from the loop even though the distribution and
    the algorithm do not. Pass `qis` to pin them (used by the equivalence test).
    """
    N, d = z.shape
    if qis is None:                       # qis is injectable so the equivalence
        qis = torch.randint(N, (n_fire,), device=z.device, generator=gen)
    k = min(k, N)
    q = cond[qis]
    if metric == "cosine":
        qn = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
        dist = 1.0 - qn @ _unit(cond).T          # cached; cond is static
    elif metric == "l2":
        dist = _l2_to_all(q, cond)          # squared; ranking is unchanged
    else:
        raise ValueError(f"unknown metric {metric!r}")
    idxs = torch.topk(dist, k, largest=False, dim=1).indices      # (n_fire, k)
    del dist

    qg = _gaussian_quantiles(k, z.device, z.dtype)
    total = 0.0
    for b in range(n_fire):
        idx = idxs[b]
        zs = z[idx]
        dirs = _rand_unit(n_dirs, d, z.device, z.dtype)
        s, _ = torch.sort(zs @ dirs.T, dim=0)
        scores = ((s - qg.unsqueeze(1)) ** 2).mean(0)
        a = dirs[int(torch.argmax(scores))]
        proj = zs @ a
        target = torch.empty_like(proj)
        target[torch.argsort(proj)] = qg
        z[idx] = zs + alpha * (target - proj).unsqueeze(1) * a.unsqueeze(0)
        total += float(scores.max())
    return total / max(n_fire, 1)


# --------------------------------------------------------------------------- #
# 1.2d  hybrid: exact discrete group + continuous k-NN inside it
# --------------------------------------------------------------------------- #
def action_knn_transport_step(z, cond, action_ids, *, k, n_dirs, alpha, gen):
    """Conditional transport when the condition is (continuous vector, discrete tag).

    The world-model condition is c = (h_{t-3..t-1}, action): part continuous,
    part an exact 18-way categorical. Neither existing step fits --
    conditional_rank_transport_step assumes Hamming distance over bits, and
    group_rank_transport_step assumes the whole condition is categorical. So:
    filter EXACTLY to the sampled particle's action (no approximation needed,
    the groups are real), then take the k nearest neighbours in continuous
    context space within that group, and rank-transport that subset.
    """
    N, d = z.shape
    qi = int(torch.randint(N, (1,), device=z.device, generator=gen))
    same = (action_ids == action_ids[qi]).nonzero(as_tuple=True)[0]
    if same.numel() < 64:
        return 0.0
    dist = torch.cdist(cond[qi:qi + 1], cond[same]).squeeze(0)   # continuous L2
    k = min(k, same.numel())
    idx = same[torch.topk(dist, k, largest=False).indices]
    zs = z[idx]

    dirs = _rand_unit(n_dirs, d, z.device, z.dtype)
    proj = zs @ dirs.T
    s, _ = torch.sort(proj, dim=0)
    q = _gaussian_quantiles(k, z.device, z.dtype).unsqueeze(1)
    scores = ((s - q) ** 2).mean(0)
    best = int(torch.argmax(scores))
    a = dirs[best]

    proj_sub = zs @ a
    order = torch.argsort(proj_sub)
    target = torch.empty_like(proj_sub)
    target[order] = _gaussian_quantiles(k, z.device, z.dtype)
    z[idx] = zs + alpha * (target - proj_sub).unsqueeze(1) * a.unsqueeze(0)
    return float(scores[best])


def action_dist_knn_transport_step(z, cond, act_vec, *, k, k_act, n_dirs, alpha, gen):
    """Conditional transport with a DISTANCE on the action too, not an exact group.

    action_knn_transport_step filters to the sampled particle's exact action id.
    That works when actions are a small clean categorical (Doom's 18). The VPT
    action is movement x turn x pitch where turn/pitch are sign-with-deadzone, so
    an exact match throws away mouse MAGNITUDE -- and magnitude explains more of
    the frame-to-frame change than every key combined (16.4% vs 2.1% of variance).

    So: take the k_act nearest neighbours in continuous ACTION space, then the k
    nearest in continuous CONTEXT space within them, then rank-transport that
    subset. Two nested nearest-k rather than one blended metric, because AE latent
    units and mouse pixels have no common scale -- a blend needs a weight that
    cannot be derived, whereas nested k's only need group sizes.

    act_vec must be pre-transformed (signed log1p + standardised) or the raw mouse
    range dominates: measured, dx alone is 81% of raw L2 variance and the key bits
    are ~0%.
    """
    N, d = z.shape
    qi = int(torch.randint(N, (1,), device=z.device, generator=gen))
    k_act = min(k_act, N)
    adist = torch.cdist(act_vec[qi:qi + 1], act_vec).squeeze(0)
    near_a = torch.topk(adist, k_act, largest=False).indices
    if near_a.numel() < 64:
        return 0.0

    cdist = torch.cdist(cond[qi:qi + 1], cond[near_a]).squeeze(0)
    kk = min(k, near_a.numel())
    idx = near_a[torch.topk(cdist, kk, largest=False).indices]
    zs = z[idx]

    dirs = _rand_unit(n_dirs, d, z.device, z.dtype)
    s, _ = torch.sort(zs @ dirs.T, dim=0)
    q = _gaussian_quantiles(kk, z.device, z.dtype).unsqueeze(1)
    scores = ((s - q) ** 2).mean(0)
    best = int(torch.argmax(scores))
    a = dirs[best]

    proj = zs @ a
    target = torch.empty_like(proj)
    target[torch.argsort(proj)] = _gaussian_quantiles(kk, z.device, z.dtype)
    z[idx] = zs + alpha * (target - proj).unsqueeze(1) * a.unsqueeze(0)
    return float(scores[best])


# --------------------------------------------------------------------------- #
# 1.2c  discrete-group conditional rank transport
# --------------------------------------------------------------------------- #
_GROUP_CACHE = {}


def _group_weights(group_ids, max_group):
    """Group ids, and sampling weights that equalise touches PER PARTICLE.

    Sampling a group uniformly does not spread transport evenly over particles.
    One firing touches min(n, max_group) members of a class of size n, so the
    per-member touch rate is p(n) * min(n, max_group) / n. Holding that constant
    across classes gives p(n) proportional to n / min(n, max_group), i.e.
    max(1, n / max_group).

    Measured on VPT's 81 action classes with max_group=8192: uniform sampling
    gave a9 (n=74,291) 0.022 touches per member per step against a46 (n=371) at
    0.198, a 9x disparity biased against the commonest actions -- which are the
    ones used at inference. It showed up as corr(log n, per-action ratio) = +0.75
    with the big classes worst decorrelated.
    """
    key = (group_ids.data_ptr(), tuple(group_ids.shape), max_group)
    hit = _GROUP_CACHE.get(key)
    if hit is not None:
        return hit
    groups, counts = torch.unique(group_ids, return_counts=True)
    cap = counts.clamp(max=max_group) if max_group else counts
    wts = counts.to(torch.float32) / cap.to(torch.float32)
    # keep several: an interleaved run cycles between groupings inside one step,
    # and a single-entry cache would re-run torch.unique over 512k ids every
    # firing. These are tiny (one entry per distinct group id).
    while len(_GROUP_CACHE) >= 16:
        _GROUP_CACHE.pop(next(iter(_GROUP_CACHE)))
    _GROUP_CACHE[key] = (groups, wts)
    return groups, wts


def group_rank_transport_step(z, group_ids, *, n_dirs, alpha, gen, max_group=None,
                              size_weighted=True):
    """Conditional transport for DISCRETE, disjoint condition groups.

    conditional_rank_transport_step approximates "same condition" by a k-NN
    neighbourhood in a binary attribute space, because CelebA's 40-bit attribute
    vector has no exact groups. When the condition is categorical (e.g. a CIFAR-10
    class label) the groups ARE exact, so we transport the true conditional
    population p(z | class) instead of a k-NN proxy -- a cleaner test of the
    independence condition, with no neighbourhood-size hyperparameter.

    Samples one group, finds its worst random 1D projection, and rank-transports
    that group to Gaussian quantiles along it.
    """
    N, d = z.shape
    if size_weighted:
        groups, wts = _group_weights(group_ids, max_group)
        gi = groups[int(torch.multinomial(wts, 1, generator=gen))]
    else:
        groups = torch.unique(group_ids)
        gi = groups[int(torch.randint(len(groups), (1,), device=z.device,
                                      generator=gen))]
    idx = (group_ids == gi).nonzero(as_tuple=True)[0]
    if max_group is not None and idx.numel() > max_group:
        idx = idx[torch.randperm(idx.numel(), device=z.device, generator=gen)[:max_group]]
    m = idx.numel()
    if m < 64:
        return 0.0
    zs = z[idx]

    dirs = _rand_unit(n_dirs, d, z.device, z.dtype)
    proj = zs @ dirs.T
    s, _ = torch.sort(proj, dim=0)
    q = _gaussian_quantiles(m, z.device, z.dtype).unsqueeze(1)
    scores = ((s - q) ** 2).mean(0)
    best = int(torch.argmax(scores))
    a = dirs[best]

    proj_sub = zs @ a
    order = torch.argsort(proj_sub)
    target = torch.empty_like(proj_sub)
    target[order] = _gaussian_quantiles(m, z.device, z.dtype)
    z[idx] = zs + alpha * (target - proj_sub).unsqueeze(1) * a.unsqueeze(0)
    return float(scores[best])


# --------------------------------------------------------------------------- #
# 1.3  conditional offset-slab cleanup
# --------------------------------------------------------------------------- #
def offset_slab_cleanup_step(z, *, search_subset, n_slabs, eps, alpha, gen):
    """Localize a slab |n^T z - b| < eps, Gaussianize an orthogonal tangent there.

    For a true standard Gaussian, an orthogonal coordinate is still N(0,1)
    inside any such slab, so the in-slab tangent is rank-transported to N(0,1).
    This catches localized tail spikes that global slices barely register (1.3).
    """
    N, d = z.shape
    m = min(search_subset, N)
    idx = torch.randperm(N, device=z.device, generator=gen)[:m]
    zs = z[idx]

    best = None  # (score, n, b, t)
    for _ in range(n_slabs):
        nrm = _rand_unit(1, d, z.device, z.dtype)[0]
        t = _rand_unit(1, d, z.device, z.dtype)[0]
        t = t - (t @ nrm) * nrm
        t = t / t.norm()
        pn = zs @ nrm
        b = pn[torch.randint(m, (1,), device=z.device, generator=gen)].item()
        mask = (pn - b).abs() < eps
        if int(mask.sum()) < 64:
            continue
        score = float(w2_to_standard_normal((zs[mask] @ t)))
        if best is None or score > best[0]:
            best = (score, nrm, b, t)
    if best is None:
        return 0.0

    _, nrm, b, t = best
    pn_full = z @ nrm
    mask = (pn_full - b).abs() < eps
    if int(mask.sum()) < 8:
        return best[0]
    sub = z[mask]
    coord = sub @ t
    order = torch.argsort(coord)
    q = _gaussian_quantiles(int(mask.sum()), z.device, z.dtype)
    target = torch.empty_like(coord)
    target[order] = q
    sub.add_(alpha * (target - coord).unsqueeze(1) * t.unsqueeze(0))
    z[mask] = sub
    return best[0]


# --------------------------------------------------------------------------- #
# 1.4  radial chi_d calibration
# --------------------------------------------------------------------------- #
def radial_chi_calibration(z, *, d, alpha_r):
    """Rank-correct radii toward the exact Gaussian shell r ~ chi_d (report 1.4).

    Directions are preserved; only the radial magnitude is nudged.  This removes
    the high-d shell error that survives every 1D projection test.
    """
    N = z.shape[0]
    r = z.norm(dim=1)
    order = torch.argsort(r)
    u = (torch.arange(N, device=z.device, dtype=torch.float64) + 0.5) / N
    r_target = torch.as_tensor(
        stats.chi.ppf(u.cpu().numpy(), df=d), device=z.device, dtype=z.dtype
    )
    target = torch.empty_like(r)
    target[order] = r_target
    scale = (1 - alpha_r) + alpha_r * target / r.clamp_min(1e-12)
    z.mul_(scale.unsqueeze(1))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
@dataclass
class AssignConfig:
    steps: int = 400
    search_subset: int = 2048
    n_dirs: int = 64
    alpha: float = 1.0
    n_slabs: int = 32
    slab_eps: float = 0.5
    slab_alpha: float = 1.0
    cleanup_every: int = 2        # interleave a slab cleanup every k global steps
    chi_every: int = 20           # radial calibration cadence (report 1.4)
    alpha_r: float = 1.0
    seed: int = 0
    log_every: int = 50


def build_assignment(h: torch.Tensor, cfg: AssignConfig, log=print):
    """Construct the persistent Gaussian assignment from raw latents h.

    Returns dict with z (assigned particles), whitening params, and history.
    """
    device = h.device
    z, mean, W, W_inv = whiten(h)
    z = z.contiguous()
    d = z.shape[1]
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    hist = []
    for step in range(cfg.steps):
        s = greedy_rank_transport_step(
            z, search_subset=cfg.search_subset, n_dirs=cfg.n_dirs,
            alpha=cfg.alpha, gen=gen,
        )
        if cfg.cleanup_every and step % cfg.cleanup_every == 0:
            offset_slab_cleanup_step(
                z, search_subset=cfg.search_subset, n_slabs=cfg.n_slabs,
                eps=cfg.slab_eps, alpha=cfg.slab_alpha, gen=gen,
            )
        if cfg.chi_every and (step + 1) % cfg.chi_every == 0:
            radial_chi_calibration(z, d=d, alpha_r=cfg.alpha_r)
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            log(f"  [assign] step {step:4d}/{cfg.steps}  max_proj_W2={s:.4f}")
            hist.append((step, s))
    return {"z": z, "mean": mean, "W": W, "W_inv": W_inv, "hist": hist}
