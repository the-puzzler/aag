"""Direct decoder stage (report 1.5) and fresh-prior generation evaluation.

After the assignment is frozen we train a plain MLP D: z_i* -> h_i on the fixed
pairs.  Inference is one forward pass from fresh Gaussian noise (Section 1.5):
    z ~ N(0, I)  ->  D(z)  ->  AE.dec  ->  image.

CIFAR has no exact source density (unlike the synthetic study), so "valid mass"
is estimated against a kNN density model fit to the real AE latents h_i — a
faithful analogue of the report's "Fresh > real 1% density" metric.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ResidualDecoder(nn.Module):
    """z (whitened Gaussian coords) -> h (AE latent).  Plain residual MLP."""

    def __init__(self, dim: int, width: int = 512, blocks: int = 4):
        super().__init__()
        self.inp = nn.Linear(dim, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(blocks)
        )
        self.out = nn.Linear(width, dim)

    def forward(self, z):
        x = self.inp(z)
        for b in self.blocks:
            x = x + b(x)
        return self.out(x)


def train_decoder(z, h, *, dim, epochs, lr, batch, device, val_frac=0.2, log=print):
    """Train D(z)~=h on `1-val_frac` of pairs; return (model, held_out_mse).

    Mirrors report Section 4: hold out pairs, record held-out MSE, then fine-tune
    on all pairs before fresh-prior generation is evaluated.
    """
    N = z.shape[0]
    perm = torch.randperm(N)
    n_val = int(val_frac * N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    z, h = z.to(device), h.to(device)

    model = ResidualDecoder(dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def run(idx, train):
        model.train(train)
        total, n = 0.0, 0
        order = idx[torch.randperm(idx.numel())] if train else idx
        for i in range(0, order.numel(), batch):
            b = order[i:i + batch]
            with torch.set_grad_enabled(train):
                pred = model(z[b])
                loss = nn.functional.mse_loss(pred, h[b])
                if train:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += loss.item() * b.numel(); n += b.numel()
        return total / n

    for ep in range(epochs):
        tr = run(tr_idx, True)
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            with torch.no_grad():
                vl = run(val_idx, False)
            log(f"  [dec] epoch {ep+1}/{epochs}  train_mse={tr:.5f}  val_mse={vl:.5f}")
    with torch.no_grad():
        held_out_mse = run(val_idx, False)

    # fine-tune on all pairs (report Section 4)
    for _ in range(max(1, epochs // 4)):
        run(perm, True)
    return model, held_out_mse


@torch.no_grad()
def _knn_logdensity(query, reference, k=10, chunk=1024):
    """Unnormalized log-density estimate: logp ~ -d * log(r_k) at each query.

    r_k is the distance to the k-th nearest reference point.  Constant factors
    cancel because we only compare against percentiles of the reference set.
    """
    d = reference.shape[1]
    out = torch.empty(query.shape[0], device=query.device)
    for i in range(0, query.shape[0], chunk):
        qq = query[i:i + chunk]
        dist = torch.cdist(qq, reference)
        rk = dist.kthvalue(min(k + 1, reference.shape[0]), dim=1).values
        out[i:i + chunk] = -d * torch.log(rk.clamp_min(1e-12))
    return out


@torch.no_grad()
def fresh_prior_metrics(model, h_real, *, dim, n_samples=6000, rank=None,
                        device="cpu", k=10):
    """Evaluate one-pass generation from a fresh Gaussian prior.

    rank=None  -> full N(0, I_dim) prior.
    rank=k     -> rank-k prior embedded in the dim-wide space (report 5.1):
                  only the first `rank` whitened coordinates are sampled.

    Returns valid@1% / valid@5% (fraction of fresh decodes above the 1st/5th
    percentile of the real-latent density) and a latent Frechet distance.
    """
    h_real = h_real.to(device)
    z = torch.randn(n_samples, dim, device=device)
    if rank is not None and rank < dim:
        z[:, rank:] = 0.0
    h_gen = model(z)

    # density thresholds from the real latents themselves (leave-one-out-ish)
    logp_real = _knn_logdensity(h_real, h_real, k=k)
    logp_gen = _knn_logdensity(h_gen, h_real, k=k)
    thr1 = torch.quantile(logp_real, 0.01)
    thr5 = torch.quantile(logp_real, 0.05)

    fd = _frechet(h_real, h_gen)
    return {
        "valid@1%": float((logp_gen > thr1).float().mean()),
        "valid@5%": float((logp_gen > thr5).float().mean()),
        "logp_gap": float(logp_gen.mean() - logp_real.mean()),
        "latent_frechet": fd,
    }


@torch.no_grad()
def _frechet(a, b):
    """Frechet distance between two Gaussians fit to a and b (FID formula)."""
    a, b = a.double().cpu().numpy(), b.double().cpu().numpy()
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca = np.cov(a, rowvar=False)
    cb = np.cov(b, rowvar=False)
    from scipy import linalg
    covmean = linalg.sqrtm(ca @ cb)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    return float(diff @ diff + np.trace(ca + cb - 2 * covmean))


@torch.no_grad()
def decode_samples(model, ae, *, dim, n, mean, W_inv, rank=None, device="cpu"):
    """Full one-pass generation to images: z ~ N(0,I) -> D -> AE.dec -> image."""
    z = torch.randn(n, dim, device=device)
    if rank is not None and rank < dim:
        z[:, rank:] = 0.0
    h = model(z)  # decoder already targets AE latent space directly
    return ae.dec(h)
