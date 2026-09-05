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
ap.add_argument("ckpts", nargs="+"); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--out", default=None); ap.add_argument("--seeds", type=int, default=1, help="average fd_sur over this many surrogate seeds")
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
    rs = [score_seed(d, sd) for sd in range(a.seeds)]
    r = {k: (sum(x[k] for x in rs) / len(rs) if isinstance(rs[0][k], float) else rs[0][k]) for k in rs[0]}
    r["fd_sur_seeds"] = [x["fd_sur"] for x in rs]; return r

def score_seed(d, seed):
    z = d["z"].float().to(dev); x = d["h"].float().to(dev); N, dz = z.shape
    if N > 200000: a.steps = max(a.steps, 6000)        # ImageNet-scale: more surrogate steps for 1000 classes
    x = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
    g = torch.Generator(device=dev).manual_seed(7); perm = torch.randperm(N, device=dev, generator=g); ho, tr = perm[: N // 20], perm[N // 20:]
    torch.manual_seed(seed)
    lab = d["label"].to(dev); n_cls = int(lab.max()) + 1 if lab.numel() and int(lab.max()) > 0 else 0
    class CondMLP(nn.Module):                      # class-conditional when the assignment carries labels
        def __init__(s):
            super().__init__(); s.emb = nn.Embedding(n_cls, 128) if n_cls else None
            s.net = nn.Sequential(nn.Linear(dz + (128 if n_cls else 0), 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, x.shape[1]))
        def forward(s, z, y=None): return s.net(torch.cat([z, s.emb(y)], 1) if s.emb is not None else z)
    G = CondMLP().to(dev)
    opt = torch.optim.AdamW(G.parameters(), lr=2e-3, weight_decay=0.01); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    for _ in range(a.steps):
        b = tr[torch.randint(len(tr), (512,), device=dev)]
        loss = F.mse_loss(G(z[b], lab[b]), x[b]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    G.eval()
    with torch.no_grad():
        nf = min(len(ho) * 4, 50000); zf = torch.randn(nf, dz, device=dev); yf = lab[torch.randint(N, (nf,), device=dev)]
        r = {"fd_sur": frechet(G(zf, yf), x[ho]), "fd_ref": frechet(G(z[ho], lab[ho]), x[ho]), "hold_sur": F.mse_loss(G(z[ho], lab[ho]), x[ho]).item(),
             "obj": msw2(z, g), "floor": msw2(torch.randn_like(z), g), "steps": d.get("steps"), "alpha": d.get("alpha"), "refine": d.get("refine_steps")}
    return r
rows = {}
print(f"{'checkpoint':44s} {'fd_sur':>8s} {'fd_ref':>8s} {'hold_sur':>8s} {'obj/floor':>9s}", flush=True)
for p in a.ckpts:
    r = score(p); rows[Path(p).name] = r
    print(f"{Path(p).name[:44]:44s} {r['fd_sur']:8.2f} {r['fd_ref']:8.2f} {r['hold_sur']:8.4f} {r['obj']/r['floor']:9.2f}   seeds " + " ".join(f"{v:.1f}" for v in r["fd_sur_seeds"]), flush=True)
if a.out: Path(a.out).write_text(json.dumps(rows, indent=1))
