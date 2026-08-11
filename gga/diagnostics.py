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
