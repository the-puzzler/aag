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
