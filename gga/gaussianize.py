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


def whiten(h: torch.Tensor):
    """Center and PCA-whiten the latent cloud so it starts isotropic.

    Returns (z0, mean, W) with z0 = (h - mean) @ W.  The inverse map is
    h = z @ W_inv + mean, needed if we ever want to decode back to AE latents.
    """
    mean = h.mean(0, keepdim=True)
    hc = h - mean
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
