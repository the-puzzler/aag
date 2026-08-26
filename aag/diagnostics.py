"""Assignment-quality diagnostics and intrinsic-dimension estimation.

Table 1 of the report reports projection / conditional / shell statistics as
*ratios to a same-size Gaussian finite-sample baseline* — a value near 1 means
"indistinguishable from a real Gaussian draw of the same N and d".  We replicate
that: every metric is computed on the assignment and on a fresh N(0, I_d) sample
of identical size, and we return the ratio.
"""
from __future__ import annotations

import numpy as np
import torch

from .gaussianize import w2_to_standard_normal


@torch.no_grad()
def _mean_projection_w2(z, n_dirs, gen):
    d = z.shape[1]
    dirs = torch.randn(n_dirs, d, device=z.device, generator=gen)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    proj = z @ dirs.T
    return torch.stack([w2_to_standard_normal(proj[:, k]) for k in range(n_dirs)]).mean()


@torch.no_grad()
def _mean_conditional_w2(z, n_slabs, eps, gen):
    N, d = z.shape
    vals = []
    for _ in range(n_slabs):
        nrm = torch.randn(d, device=z.device, generator=gen); nrm /= nrm.norm()
        t = torch.randn(d, device=z.device, generator=gen)
        t = t - (t @ nrm) * nrm; t /= t.norm()
        pn = z @ nrm
        b = pn[torch.randint(N, (1,), device=z.device, generator=gen)].item()
        mask = (pn - b).abs() < eps
        if int(mask.sum()) < 64:
            continue
        vals.append(w2_to_standard_normal(z[mask] @ t))
    return torch.stack(vals).mean() if vals else torch.tensor(float("nan"))


@torch.no_grad()
def _radius_stats(z, d):
    from scipy import stats
    N = z.shape[0]
    r = z.norm(dim=1).sort().values
    u = (np.arange(N) + 0.5) / N
    q = torch.as_tensor(stats.chi.ppf(u, df=d), device=z.device, dtype=z.dtype)
    qq_rmse = ((r - q) ** 2).mean().sqrt()
    p99 = torch.quantile(r, 0.99)
    return p99, qq_rmse


@torch.no_grad()
def assignment_diagnostics(z, *, d, n_dirs=64, n_slabs=64, eps=0.5, seed=0):
    """Return dict of ratios-to-Gaussian, matching Table 1 columns."""
    gen = torch.Generator(device=z.device).manual_seed(seed)
    genb = torch.Generator(device=z.device).manual_seed(seed + 1)
    ref = torch.randn_like(z)

    proj = _mean_projection_w2(z, n_dirs, gen)
    proj_ref = _mean_projection_w2(ref, n_dirs, gen)
    cond = _mean_conditional_w2(z, n_slabs, eps, genb)
    cond_ref = _mean_conditional_w2(ref, n_slabs, eps, genb)
    p99, qq = _radius_stats(z, d)
    p99_ref, _ = _radius_stats(ref, d)

    return {
        "proj_over_gauss": float(proj / proj_ref),
        "cond_over_gauss": float(cond / cond_ref),
        "p99_radius_over_gauss": float(p99 / p99_ref),
        "radius_qq_rmse": float(qq),
    }


@torch.no_grad()
def intrinsic_dimension_twonn(h: torch.Tensor, frac: float = 0.9, max_pts: int = 4000):
    """TwoNN intrinsic-dimension estimator (Facco et al., 2017).

    Report 8.2: "Measure effective/intrinsic dimension of real AE latents before
    assuming the 64D synthetic failure transfers directly to images."  Uses the
    ratio mu = r2/r1 of the two nearest-neighbour distances; d = fit of the
    linear region of the empirical -log(1-F(mu)) vs log(mu) relation.
    """
    h = h.float()
    if h.shape[0] > max_pts:
        idx = torch.randperm(h.shape[0])[:max_pts]
        h = h[idx]
    d2 = torch.cdist(h, h)
    d2.fill_diagonal_(float("inf"))
    r1 = d2.min(dim=1).values
    d2.scatter_(1, d2.argmin(dim=1, keepdim=True), float("inf"))
    r2 = d2.min(dim=1).values
    mu = (r2 / r1.clamp_min(1e-12))
    mu = mu[torch.isfinite(mu) & (mu > 1)]
    mu, _ = torch.sort(mu)
    n = mu.numel()
    keep = int(frac * n)
    mu = mu[:keep]
    F = (torch.arange(1, keep + 1, device=mu.device).float()) / n
    x = torch.log(mu)
    y = -torch.log(1 - F.clamp_max(1 - 1e-9))
    # least squares through the origin: d = sum(xy)/sum(x^2)
    return float((x * y).sum() / (x * x).sum())


# --------------------------------------------------------------------------- #
# conditional (attribute-aware) diagnostics
# --------------------------------------------------------------------------- #
def conditional_group_w2(z, cond, *, k, n_eval=20, gen=None, n_dirs=16):
    """Mean max-projected-W2 over n_eval k-NN-by-Hamming attribute neighbourhoods.

    This is the same statistic the conditional transport step optimises, so on
    its own it only shows the optimiser working -- always pair it with
    random_subset_w2() below and report the RATIO.
    """
    import torch
    from .gaussianize import _gaussian_quantiles
    scores = []
    for _ in range(n_eval):
        qi = int(torch.randint(z.shape[0], (1,), device=z.device, generator=gen))
        dist = (cond != cond[qi:qi + 1]).sum(1)
        idx = torch.topk(dist, min(k, z.shape[0]), largest=False).indices
        zs = z[idx]
        dirs = torch.randn(n_dirs, z.shape[1], device=z.device, generator=gen)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        s, _ = torch.sort(zs @ dirs.T, dim=0)
        q = _gaussian_quantiles(zs.shape[0], z.device, z.dtype).unsqueeze(1)
        scores.append(float(((s - q) ** 2).mean(0).max()))
    return sum(scores) / len(scores)


def action_knn_w2(z, cond, action_ids, *, k, n_eval=20, gen=None, n_dirs=16):
    """conditional_group_w2's analogue for a (continuous vector, discrete tag)
    condition: exact action filter, then continuous k-NN inside that group.

    Pair with random_subset_w2 at the same k and report the RATIO -- alone this
    only shows the transport step optimising its own objective.
    """
    import torch
    from .gaussianize import _gaussian_quantiles
    scores = []
    for _ in range(n_eval):
        qi = int(torch.randint(z.shape[0], (1,), device=z.device, generator=gen))
        same = (action_ids == action_ids[qi]).nonzero(as_tuple=True)[0]
        if same.numel() < 64:
            continue
        dist = torch.cdist(cond[qi:qi + 1], cond[same]).squeeze(0)
        idx = same[torch.topk(dist, min(k, same.numel()), largest=False).indices]
        zs = z[idx]
        dirs = torch.randn(n_dirs, z.shape[1], device=z.device, generator=gen)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        s, _ = torch.sort(zs @ dirs.T, dim=0)
        q = _gaussian_quantiles(zs.shape[0], z.device, z.dtype).unsqueeze(1)
        scores.append(float(((s - q) ** 2).mean(0).max()))
    return sum(scores) / max(len(scores), 1)


def continuous_knn_w2(z, cond, *, k, n_eval=20, gen=None, n_dirs=16, metric="cosine"):
    """Diagnostic matching continuous_knn_transport_step -- pass the SAME metric.
    Pair with random_subset_w2 at the same k and report the RATIO."""
    import torch
    from .gaussianize import _gaussian_quantiles, cond_distance
    scores = []
    for _ in range(n_eval):
        qi = int(torch.randint(z.shape[0], (1,), device=z.device, generator=gen))
        dist = cond_distance(cond, qi, metric)
        idx = torch.topk(dist, min(k, z.shape[0]), largest=False).indices
        zs = z[idx]
        dirs = torch.randn(n_dirs, z.shape[1], device=z.device, generator=gen)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        s, _ = torch.sort(zs @ dirs.T, dim=0)
        q = _gaussian_quantiles(zs.shape[0], z.device, z.dtype).unsqueeze(1)
        scores.append(float(((s - q) ** 2).mean(0).max()))
    return sum(scores) / len(scores)


def knn_preservation(z0, z, *, k=10, n_sample=4000, gen=None):
    """Fraction of each particle's k nearest neighbours in z0 still among its k in z.

    Measured in a 2D study to predict amortised-generator error far better than
    displacement does: between 50 and 5000 transport steps displacement moves 4%
    while this falls 0.50 -> 0.20 and generator MSE triples. Displacement is a
    coarse summary that misses *rearrangement* -- transport keeps changing who is
    next to whom long after it stops moving anyone further.
    """
    import torch
    N = z0.shape[0]
    idx = torch.arange(N, device=z0.device) if N <= n_sample else \
        torch.randperm(N, device=z0.device, generator=gen)[:n_sample]
    a = torch.cdist(z0[idx], z0).topk(k + 1, largest=False).indices[:, 1:]
    b = torch.cdist(z[idx], z).topk(k + 1, largest=False).indices[:, 1:]
    hit = (a.unsqueeze(2) == b.unsqueeze(1)).any(2).float().sum(1)
    return float((hit / k).mean())


def r_dispersion(z, h, *, n_probes=1000, k=32, gen=None, chunk=256):
    """Prior-local decodability: sample probes u ~ N(0,I), take the k nearest
    ASSIGNED z, and measure how dispersed their h targets are (normalised by h's
    global scale). Lower = nearby prior coordinates decode to similar targets.

    Probes come from the PRIOR, not the data, so this inspects the regions fresh
    z will actually land in -- including gaps between assigned points. That is
    what makes it a generation diagnostic rather than a training-set one.
    """
    import torch
    d = z.shape[1]
    u = torch.randn(n_probes, d, device=z.device, generator=gen)
    hs = h.std(0).norm().clamp_min(1e-8)
    out = []
    for i in range(0, n_probes, chunk):
        idx = torch.cdist(u[i:i + chunk], z).topk(k, largest=False).indices
        hn = h[idx]
        out.append((hn - hn.mean(1, keepdim=True)).norm(dim=-1).mean(1))
    return float((torch.cat(out) / hs).median())


def r_affine(z, h, *, n_probes=250, k=48, ridge=1e-3, gen=None):
    """Finer variant of r_dispersion: fit h ~ Az+b on k-1 local neighbours of each
    prior probe and report held-out error. Lower = locally affine-decodable."""
    import torch
    d = z.shape[1]
    u = torch.randn(n_probes, d, device=z.device, generator=gen)
    hs = h.std(0).norm().clamp_min(1e-8)
    eye = torch.eye(d + 1, device=z.device)
    errs = []
    for i in range(n_probes):
        idx = torch.cdist(u[i:i + 1], z).topk(k, largest=False).indices[0]
        zz, hh = z[idx], h[idx]
        zt = torch.cat([zz, torch.ones(k, 1, device=z.device)], 1)
        A = torch.linalg.solve(zt[:-1].T @ zt[:-1] + ridge * eye, zt[:-1].T @ hh[:-1])
        errs.append((zt[-1:] @ A - hh[-1:]).norm())
    return float((torch.stack(errs) / hs).median())


def f_z(h, cond, *, n_probes=400, m_cond=2048, gen=None):
    """Fraction of representation variance left for z after conditioning:

        f_z = E_c[ tr Cov(h | c) ] / tr Cov(h)

    Estimated over local condition neighbourhoods, the same geometry r_rel uses
    with `cond`. Near 0 means the condition already determines h and z has almost
    nothing to encode; near 1 means the condition explains little.

    This is what makes R_rel interpretable under conditioning: R_rel measures
    failure RELATIVE to the local variance, so when f_z is small a poor R_rel
    costs little in absolute terms.
    """
    import torch
    N = h.shape[0]
    tot = float(h.var(0, unbiased=False).sum())
    cn = cond / cond.norm(dim=1, keepdim=True).clamp_min(1e-8)
    pick = torch.randperm(N, device=h.device, generator=gen)[:n_probes]
    acc = 0.0
    for i in range(n_probes):
        near = (cn[pick[i]:pick[i] + 1] @ cn.T).topk(min(m_cond, N), dim=1).indices[0]
        acc += float(h[near].var(0, unbiased=False).sum())
    return (acc / n_probes) / max(tot, 1e-12)


def r_rel(z, h, *, n_probes=400, k=64, hold=8, ridge=1e-3, gen=None, cond=None,
          m_cond=4096):
    """Relative local roughness: local affine prediction MSE / local target variance.

        R_rel = ||h_test - (A z_test + b)||^2  /  ||h_test - mean(h_train)||^2

    i.e. 1 - R^2 of a local affine fit around a prior probe. Scale-free and
    comparable across datasets, dimensions and representations, which the
    globally-normalised variants are not -- they conflate "this region genuinely
    has high variance" with "the fit is bad".

        0   perfectly locally decodable from z
        1   z tells you nothing beyond the local mean
        >1  worse than predicting the local mean

    Pass `cond` to measure in the conditional (decoder-input) geometry.
    """
    import torch
    N, d = z.shape
    u = torch.randn(n_probes, d, device=z.device, generator=gen)
    eye = torch.eye(d + 1, device=z.device)
    if cond is not None:
        cn = cond / cond.norm(dim=1, keepdim=True).clamp_min(1e-8)
        pick = torch.randperm(N, device=z.device, generator=gen)[:n_probes]
    # Denominator. Unconditional: local target variance. Conditional: the RESIDUAL
    # variance Var(h - E[h|c]) -- the variance z is actually supposed to explain.
    # Using the local (c and z) neighbourhood spread instead would shrink as
    # z-localisation improves, moving the yardstick with the treatment. With this
    # choice f_z * R_rel telescopes to (local error / Var h): the fraction of TOTAL
    # representation variance the decoder mis-models.
    resid = None
    if cond is not None:
        acc = 0.0
        for i in range(n_probes):
            near = (cn[pick[i]:pick[i] + 1] @ cn.T).topk(min(m_cond, N), dim=1).indices[0]
            acc += float(((h[near] - h[near].mean(0)) ** 2).sum(1).mean())
        resid = acc / n_probes                       # E[ ||h - E[h|c]||^2 ]
    num = den = 0.0
    n_te = 0
    for i in range(n_probes):
        if cond is None:
            idx = torch.cdist(u[i:i + 1], z).topk(k, largest=False).indices[0]
        else:
            near = (cn[pick[i]:pick[i] + 1] @ cn.T).topk(min(m_cond, N), dim=1).indices[0]
            idx = near[torch.cdist(u[i:i + 1], z[near]).topk(k, largest=False).indices[0]]
        zz, hh = z[idx], h[idx]
        zt = torch.cat([zz, torch.ones(k, 1, device=z.device)], 1)
        tr, te = slice(0, k - hold), slice(k - hold, k)
        A = torch.linalg.solve(zt[tr].T @ zt[tr] + ridge * eye, zt[tr].T @ hh[tr])
        num += float(((zt[te] @ A - hh[te]) ** 2).sum())
        den += float(((hh[te] - hh[tr].mean(0)) ** 2).sum())
        n_te += (k - (k - hold))
    if resid is not None:
        return num / max(n_te * resid, 1e-12)
    return num / max(den, 1e-12)


def r_cond(z, h, cond, *, n_probes=800, k=32, m_cond=4096, gen=None, chunk=128):
    """r_dispersion measured in the DECODER INPUT geometry: restrict to a local
    condition neighbourhood first, then find nearby assigned z. Without the
    restriction, 'neighbours' mix incompatible conditions and the dispersion
    reflects the condition rather than the assignment."""
    import torch
    N, d = z.shape
    idx_c = torch.randperm(N, device=z.device, generator=gen)[:n_probes]
    u = torch.randn(n_probes, d, device=z.device, generator=gen)
    hs = h.std(0).norm().clamp_min(1e-8)
    cn = cond / cond.norm(dim=1, keepdim=True).clamp_min(1e-8)
    out = []
    for i in range(0, n_probes, chunk):
        ci = idx_c[i:i + chunk]
        near_c = (cn[ci] @ cn.T).topk(min(m_cond, N), dim=1).indices
        sub = z[near_c]
        sel = (sub - u[i:i + chunk, None]).norm(dim=-1).topk(k, largest=False).indices
        rows = torch.arange(len(ci), device=z.device)[:, None]
        hn = h[near_c[rows, sel]]
        out.append((hn - hn.mean(1, keepdim=True)).norm(dim=-1).mean(1))
    return float((torch.cat(out) / hs).median())


def action_dist_knn_w2(z, cond, act_vec, *, k, k_act, n_eval=20, gen=None, n_dirs=16):
    """Diagnostic matching action_dist_knn_transport_step: action-nearest-k_act,
    then context-nearest-k inside it. Pair with random_subset_w2 at the same k
    and report the RATIO -> 1.0.
    """
    import torch
    from .gaussianize import _gaussian_quantiles
    scores = []
    for _ in range(n_eval):
        qi = int(torch.randint(z.shape[0], (1,), device=z.device, generator=gen))
        ad = torch.cdist(act_vec[qi:qi + 1], act_vec).squeeze(0)
        ka = min(k_act, z.shape[0])
        near = torch.topk(ad, ka, largest=False).indices
        cd = torch.cdist(cond[qi:qi + 1], cond[near]).squeeze(0)
        kk = min(k, near.numel())
        idx = near[torch.topk(cd, kk, largest=False).indices]
        zs = z[idx]
        dirs = torch.randn(n_dirs, z.shape[1], device=z.device, generator=gen)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        srt, _ = torch.sort(zs @ dirs.T, dim=0)
        q = _gaussian_quantiles(kk, z.device, z.dtype).unsqueeze(1)
        scores.append(((srt - q) ** 2).mean(0).max().item())
    return sum(scores) / len(scores)


def per_action_w2(z, action_ids, *, n_eval_per=4, gen=None, n_dirs=16, min_size=256,
                  max_group=4096, floor_reps=8):
    """Max-projected W2 per DISCRETE action class, against a SIZE-MATCHED floor.

    Returns {action_id: (ratio, size, w2, floor)}.

    The size match is the whole point and easy to get wrong: dividing every class
    by a floor computed at one fixed k makes small classes look dependent purely
    from finite-sample W2. Measured here, that error inflated the mean ratio from
    2.46 to 4.98 and the max from 6.50 to 13.81, and it reordered which actions
    looked worst -- the apparent worst two (n~300) were fine once matched, while
    the real worst were mid-sized classes.
    """
    import torch
    from .gaussianize import _gaussian_quantiles
    out = {}
    for a in torch.unique(action_ids).tolist():
        member = (action_ids == a).nonzero(as_tuple=True)[0]
        if member.numel() < min_size:
            continue
        vals = []
        for _ in range(n_eval_per):
            sel = member
            if sel.numel() > max_group:
                perm = torch.randperm(sel.numel(), device=z.device, generator=gen)
                sel = sel[perm[:max_group]]
            zs = z[sel]
            dirs = torch.randn(n_dirs, z.shape[1], device=z.device, generator=gen)
            dirs = dirs / dirs.norm(dim=1, keepdim=True)
            srt, _ = torch.sort(zs @ dirs.T, dim=0)
            q = _gaussian_quantiles(len(zs), z.device, z.dtype).unsqueeze(1)
            vals.append(((srt - q) ** 2).mean(0).max().item())
        cls = sum(vals) / len(vals)
        n = min(int(member.numel()), max_group)
        fl = []
        for _ in range(floor_reps):
            sel = torch.randperm(z.shape[0], device=z.device, generator=gen)[:n]
            zs = z[sel]
            dirs = torch.randn(n_dirs, z.shape[1], device=z.device, generator=gen)
            dirs = dirs / dirs.norm(dim=1, keepdim=True)
            srt, _ = torch.sort(zs @ dirs.T, dim=0)
            q = _gaussian_quantiles(n, z.device, z.dtype).unsqueeze(1)
            fl.append(((srt - q) ** 2).mean(0).max().item())
        floor = sum(fl) / len(fl)
        out[a] = (cls / max(floor, 1e-12), int(member.numel()), cls, floor)
    return out


def random_subset_w2(z, *, k, n_eval=20, gen=None, n_dirs=16):
    """Same statistic on RANDOM subsets of matched size = the independence floor.

    If z is independent of the condition, an attribute-selected neighbourhood is
    statistically indistinguishable from a random subset, so
    conditional_group_w2 / random_subset_w2 -> 1.0 is the convergence target.
    """
    import torch
    from .gaussianize import _gaussian_quantiles
    scores = []
    for _ in range(n_eval):
        idx = torch.randperm(z.shape[0], device=z.device, generator=gen)[:k]
        zs = z[idx]
        dirs = torch.randn(n_dirs, z.shape[1], device=z.device, generator=gen)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        s, _ = torch.sort(zs @ dirs.T, dim=0)
        q = _gaussian_quantiles(zs.shape[0], z.device, z.dtype).unsqueeze(1)
        scores.append(float(((s - q) ** 2).mean(0).max()))
    return sum(scores) / len(scores)


def independence_ratio(z, cond, *, k, n_eval=20, gen=None):
    """conditional_group_w2 / random_subset_w2. 1.0 == z independent of cond."""
    return (conditional_group_w2(z, cond, k=k, n_eval=n_eval, gen=gen)
            / random_subset_w2(z, k=k, n_eval=n_eval, gen=gen))


def transport_objective_floor(n, d, *, search_subset=2048, n_dirs=64, reps=30,
                              device="cuda", gen=None):
    """Noise floor of the greedy transport objective: its value on a TRUE N(0,I)
    cloud of matched size. The assignment is done improving once the running
    objective enters this band -- past that it cannot see progress while
    transport displacement keeps accumulating."""
    import numpy as np, torch
    from .gaussianize import _gaussian_quantiles, _rand_unit
    z = torch.randn(n, d, device=device, generator=gen)
    out = []
    for _ in range(reps):
        idx = torch.randperm(n, device=device, generator=gen)[:search_subset]
        zs = z[idx]
        dirs = _rand_unit(n_dirs, d, device, z.dtype)
        s, _ = torch.sort(zs @ dirs.T, dim=0)
        q = _gaussian_quantiles(search_subset, device, z.dtype).unsqueeze(1)
        out.append(float(((s - q) ** 2).mean(0).max()))
    return float(np.mean(out)), float(np.std(out))


def group_w2(z, group_ids, *, n_eval=20, gen=None, n_dirs=16, max_group=4096):
    """Mean max-projected-W2 over DISCRETE condition groups (exact, not k-NN).
    Pair with random_subset_w2 at matched size and report the RATIO -> 1.0."""
    import torch
    from .gaussianize import _gaussian_quantiles
    groups = torch.unique(group_ids)
    scores, sizes = [], []
    for _ in range(n_eval):
        gi = groups[int(torch.randint(len(groups), (1,), device=z.device, generator=gen))]
        idx = (group_ids == gi).nonzero(as_tuple=True)[0]
        if idx.numel() > max_group:
            idx = idx[torch.randperm(idx.numel(), device=z.device, generator=gen)[:max_group]]
        zs = z[idx]
        dirs = torch.randn(n_dirs, z.shape[1], device=z.device, generator=gen)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        s, _ = torch.sort(zs @ dirs.T, dim=0)
        q = _gaussian_quantiles(zs.shape[0], z.device, z.dtype).unsqueeze(1)
        scores.append(float(((s - q) ** 2).mean(0).max())); sizes.append(idx.numel())
    return sum(scores) / len(scores), int(sum(sizes) / len(sizes))


def group_independence_ratio(z, group_ids, *, n_eval=20, gen=None, max_group=4096):
    """group_w2 / random_subset_w2 at matched size. 1.0 == z independent of group."""
    gw, k = group_w2(z, group_ids, n_eval=n_eval, gen=gen, max_group=max_group)
    return gw / random_subset_w2(z, k=k, n_eval=n_eval, gen=gen)
