#!/usr/bin/env python
"""Score saved toy assignment snapshots (from toy_step_predictor.py) with several generator recipes.

Question: does the FD-optimal number of transport steps depend on how hard the generator fits the
pairs?  For every snapshot <method>_<step>.pt we train each recipe (width x depth x steps x weight decay)
and report FD at fresh z, so the optimum can be read per recipe.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--snaps", type=Path, required=True, help="…_snaps directory")
ap.add_argument("--D", type=int, default=128); ap.add_argument("--k", type=int, default=16); ap.add_argument("--N", type=int, default=28000)
ap.add_argument("--noise", type=float, default=0.02); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--methods", default="random")
ap.add_argument("--recipes", default="w1024d3s6000,w1024d3s1500,w512d2s6000,w512d2s1500,w1024d3s6000wd0.3,w256d2s1500")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); dev = "cuda"

# regenerate the exact same data as toy_step_predictor.py (same seed path)
torch.manual_seed(a.seed)
def make_f(k, D, gen):
    W1 = torch.randn(k, 256, generator=gen) / k ** 0.5; W2 = torch.randn(256, 256, generator=gen) / 16; W3 = torch.randn(256, D, generator=gen) / 16
    def f(u): return torch.tanh(torch.tanh(u @ W1.to(u.device)) @ W2.to(u.device)) @ W3.to(u.device)
    return f
g = torch.Generator().manual_seed(a.seed); f = make_f(a.k, a.D, g)
u = torch.randn(a.N + 5000, a.k, generator=g); x_all = f(u) + a.noise * torch.randn(a.N + 5000, a.D, generator=g)
x_all = (x_all - x_all.mean(0)) / x_all.std(0).clamp_min(1e-6)
x, x_test = x_all[: a.N].to(dev), x_all[a.N:].to(dev)
N = a.N; hold = torch.zeros(N, dtype=torch.bool, device=dev); hold[torch.randperm(N, device=dev, generator=torch.Generator(device=dev).manual_seed(7))[: N // 50]] = True
tr_idx = (~hold).nonzero(as_tuple=True)[0]; ho_idx = hold.nonzero(as_tuple=True)[0]

def frechet(A, B):
    muA, muB = A.mean(0), B.mean(0); cA, cB = torch.cov(A.T), torch.cov(B.T)
    ev = torch.linalg.eigvals(cA @ cB).real.clamp_min(0)
    return float(((muA - muB) ** 2).sum() + torch.trace(cA) + torch.trace(cB) - 2 * ev.sqrt().sum())

def train(z, width, depth, steps, wd, seed=0):
    torch.manual_seed(seed); d = z.shape[1]; D = x.shape[1]
    layers = [nn.Linear(d, width), nn.SiLU()]
    for _ in range(depth - 1): layers += [nn.Linear(width, width), nn.SiLU()]
    G = nn.Sequential(*layers, nn.Linear(width, D)).to(dev)
    opt = torch.optim.AdamW(G.parameters(), lr=1e-3 if width >= 1024 else 2e-3, weight_decay=wd); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        b = tr_idx[torch.randint(len(tr_idx), (512,), device=dev)]
        loss = F.mse_loss(G(z[b]), x[b]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    G.eval()
    with torch.no_grad():
        return {"FD_fresh": frechet(G(torch.randn(5000, d, device=dev)), x_test), "heldout_mse": F.mse_loss(G(z[ho_idx]), x[ho_idx]).item(),
                "train_mse": F.mse_loss(G(z[tr_idx[:5000]]), x[tr_idx[:5000]]).item()}

def parse(r):
    m = re.match(r"w(\d+)d(\d+)s(\d+)(?:wd([0-9.]+))?$", r); return int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4) or 0.01)
recipes = a.recipes.split(",")
res = {"config": vars(a), "results": {}}
for method in a.methods.split(","):
    files = sorted(a.snaps.glob(f"{method}_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    print(f"\n== {method}   FD_fresh per recipe (held-out mse in brackets)\n{'step':>7s} | " + " ".join(f"{r:>22s}" for r in recipes), flush=True)
    for p in files:
        step = int(p.stem.split("_")[-1]); z = torch.load(p).float().to(dev); row = {}
        for r in recipes: row[r] = train(z, *parse(r))
        res["results"][f"{method}_{step}"] = row
        print(f"{step:7d} | " + " ".join(f"{row[r]['FD_fresh']:12.3f} ({row[r]['heldout_mse']:.4f})" for r in recipes), flush=True)
    for r in recipes:
        steps = [int(k.split("_")[-1]) for k in res["results"] if k.startswith(method + "_")]
        fds = [res["results"][f"{method}_{s}"][r]["FD_fresh"] for s in steps]; i = min(range(len(fds)), key=lambda j: fds[j])
        print(f"   {r:22s} best FD {fds[i]:8.3f} at step {steps[i]}", flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1, default=str)); print("saved", a.out)
