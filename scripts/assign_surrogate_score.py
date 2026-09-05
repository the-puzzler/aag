#!/usr/bin/env python
"""Score real assignment checkpoints WITHOUT training the big generator (toy_step_predictor.py's fd_sur on real data).

For each assignment .pt: hold out 5% of pairs; train a tiny MLP z -> latent (2x512, 1500 steps, ~3 s)
on the rest; report
  fd_sur    Frechet distance in (standardised) latent space between MLP(fresh z) and the held-out latents
  fd_ref    the same for MLP(held-out assigned z)     (ceiling: how well the pairs alone can be fit)
  hold_sur  held-out MSE of the tiny MLP
  obj       max-sliced W2^2 over 256 random dirs (m=8192) and the Gaussian floor for that estimator
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from aag.gaussianize import _rand_unit, _gaussian_quantiles

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+"); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--out", default=None)
a = ap.parse_args(); dev = "cuda"

def frechet(A, B):
    muA, muB = A.mean(0), B.mean(0); cA, cB = torch.cov(A.T), torch.cov(B.T)
    ev = torch.linalg.eigvals(cA @ cB).real.clamp_min(0)
    return float(((muA - muB) ** 2).sum() + torch.trace(cA) + torch.trace(cB) - 2 * ev.sqrt().sum())

@torch.no_grad()
def msw2(z, gen, m=8192, ndirs=256):
    N, d = z.shape; sub = z[torch.randperm(N, device=dev, generator=gen)[:m]]
    dirs = _rand_unit(ndirs, d, dev, z.dtype); s, _ = torch.sort(sub @ dirs.T, dim=0); q = _gaussian_quantiles(m, dev, z.dtype).unsqueeze(1)
    return float(((s - q) ** 2).mean(0).max())

def score(path):
    d = torch.load(path, map_location="cpu", mmap=True)
    z = d["z"].float().to(dev); x = d["h"].float().to(dev); N, dz = z.shape
    x = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
    g = torch.Generator(device=dev).manual_seed(7); perm = torch.randperm(N, device=dev, generator=g); ho, tr = perm[: N // 20], perm[N // 20:]
    torch.manual_seed(0)
    G = nn.Sequential(nn.Linear(dz, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, x.shape[1])).to(dev)
    opt = torch.optim.AdamW(G.parameters(), lr=2e-3, weight_decay=0.01); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    for _ in range(a.steps):
        b = tr[torch.randint(len(tr), (512,), device=dev)]
        loss = F.mse_loss(G(z[b]), x[b]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    G.eval()
    with torch.no_grad():
        zf = torch.randn(len(ho) * 4, dz, device=dev)
        r = {"fd_sur": frechet(G(zf), x[ho]), "fd_ref": frechet(G(z[ho]), x[ho]), "hold_sur": F.mse_loss(G(z[ho]), x[ho]).item(),
             "obj": msw2(z, g), "floor": msw2(torch.randn_like(z), g), "steps": d.get("steps"), "alpha": d.get("alpha"), "refine": d.get("refine_steps")}
    return r
rows = {}
print(f"{'checkpoint':44s} {'fd_sur':>8s} {'fd_ref':>8s} {'hold_sur':>8s} {'obj/floor':>9s}", flush=True)
for p in a.ckpts:
    r = score(p); rows[Path(p).name] = r
    print(f"{Path(p).name[:44]:44s} {r['fd_sur']:8.2f} {r['fd_ref']:8.2f} {r['hold_sur']:8.4f} {r['obj']/r['floor']:9.2f}", flush=True)
if a.out: Path(a.out).write_text(json.dumps(rows, indent=1))
