#!/usr/bin/env python
"""Head-to-head of direction-search strategies for the AAG rank transport, on real latents.

Same start (whitened particles), same 1-D rank transport, same step budget; only how each step
picks its direction differs:
  random      worst of 64 random unit vectors on a 2048 subset (the published recipe)
  refine      that, then 30 Adam steps on the unit vector maximising the projected W2 (max-sliced)
  population  a population of directions optimised by SGD on skew^2+kurt^2 of standardised
              projections of fresh minibatches, then verified by 1-D W2 and the argmax used
              (user's projection-pursuit proposal, 2026-09-05)
Readouts every --eval-every steps: learned-direction W2 (the residual a 300-step optimised probe
finds), assigned-vs-fresh MLP classifier accuracy, A->A kNN fraction, mean displacement from the
start, and neighbourhood preservation (fraction of each point's 10 NN at the start still among
its 10 NN now) as the locality proxy. Radial chi calibration every 20 steps for all; no slab step,
to isolate the direction search.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from aag.gaussianize import (whiten, greedy_rank_transport_step, radial_chi_calibration, population_direction,
                             rank_transport_along, _gaussian_quantiles)

ap = argparse.ArgumentParser()
ap.add_argument("--particles", required=True); ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--eval-every", type=int, default=2000); ap.add_argument("--methods", default="random,refine,population")
ap.add_argument("--pop", type=int, default=32); ap.add_argument("--probe-steps", type=int, default=20)
ap.add_argument("--probe-batch", type=int, default=4096); ap.add_argument("--eval-subset", type=int, default=8192)
ap.add_argument("--refine-steps", type=int, default=30); ap.add_argument("--alpha", type=float, default=1.0, help="transport step size along the chosen direction (1 = full rank match)")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); dev = "cuda"
P = torch.load(a.particles, map_location="cpu", weights_only=False)
h = P["h"].float().to(dev); z0, *_ = whiten(h, rotate=False); z0 = z0.contiguous(); N, d = z0.shape
sub = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:8192].to(dev)
D0 = torch.cdist(z0[sub], z0[sub]); D0.fill_diagonal_(float("inf")); nn0 = D0.topk(10, largest=False).indices

def w2_1d(v):
    s, _ = torch.sort(v); q = _gaussian_quantiles(len(v), v.device, v.dtype); return ((s - q) ** 2).mean()

def readouts(z):
    r = {}
    u = nn.Parameter(torch.randn(d, device=dev)); opt = torch.optim.Adam([u], lr=0.05)
    for _ in range(300):
        loss = -w2_1d(z @ (u / u.norm())); opt.zero_grad(); loss.backward(); opt.step()
    r["learned_dir_w2"] = float(w2_1d(z @ (u / u.norm()).detach()))
    zf = torch.randn(N, d, device=dev); X = torch.cat([z, zf]); y = torch.cat([torch.ones(N), torch.zeros(N)]).to(dev)
    perm = torch.randperm(2 * N, device=dev); tr, te = perm[: int(1.6 * N)], perm[int(1.6 * N):]
    clf = nn.Sequential(nn.Linear(d, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 1)).to(dev); o = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for ep in range(20):
        for i in range(0, len(tr), 1024):
            b = tr[i:i + 1024]; l = F.binary_cross_entropy_with_logits(clf(X[b]).squeeze(1), y[b]); o.zero_grad(); l.backward(); o.step()
    with torch.no_grad(): r["classifier_acc"] = ((clf(X[te]).squeeze(1) > 0).float() == y[te]).float().mean().item()
    za, zg = z[sub], torch.randn(len(sub), d, device=dev); Xk = torch.cat([za, zg]); lab = torch.cat([torch.zeros(len(sub)), torch.ones(len(sub))]).to(dev)
    Dk = torch.cdist(Xk, Xk); Dk.fill_diagonal_(float("inf")); nnk = Dk.topk(10, largest=False).indices
    r["A_to_A"] = (lab[nnk[: len(sub)]] == 0).float().mean().item()
    D1 = torch.cdist(z[sub], z[sub]); D1.fill_diagonal_(float("inf")); nn1 = D1.topk(10, largest=False).indices
    r["nn_preserved"] = float(torch.tensor([len(set(nn0[i].tolist()) & set(nn1[i].tolist())) for i in range(len(sub))]).float().mean() / 10)
    r["displacement"] = float((z - z0).norm(dim=1).mean()); r["displacement_frac"] = r["displacement"] / d ** 0.5
    return r

results = {"config": vars(a) | {"N": N, "d": d}, "start": readouts(z0), "methods": {}}
print(f"N={N} d={d}  start: {json.dumps({k: round(v, 4) for k, v in results['start'].items()})}", flush=True)
for method in a.methods.split(","):
    z = z0.clone(); gen = torch.Generator(device=dev).manual_seed(1); curve = []; t0 = time.time(); tsearch = 0.0
    for step in range(1, a.steps + 1):
        if method == "random":
            greedy_rank_transport_step(z, search_subset=2048, n_dirs=64, alpha=a.alpha, gen=gen, return_score=False)
        elif method == "refine":
            greedy_rank_transport_step(z, search_subset=a.eval_subset, n_dirs=64, alpha=a.alpha, gen=gen, return_score=False, refine_steps=a.refine_steps)
        elif method == "population":
            v, _ = population_direction(z, n_pop=a.pop, probe_steps=a.probe_steps, batch=a.probe_batch, eval_subset=a.eval_subset, lr=0.05, gen=gen)
            rank_transport_along(z, v, alpha=a.alpha)
        if step % 20 == 0: radial_chi_calibration(z, d=d, alpha_r=1.0)
        if step % a.eval_every == 0 or step == a.steps:
            torch.cuda.synchronize(); el = time.time() - t0
            r = readouts(z) | {"step": step, "seconds": round(el)}; curve.append(r)
            print(f"{method:10s} step {step:6d} ({el:5.0f}s)  learned-dir W2 {r['learned_dir_w2']:.5f}  clf acc {r['classifier_acc']:.3f}  A->A {r['A_to_A']:.3f}  "
                  f"NN kept {r['nn_preserved']:.3f}  disp {r['displacement_frac']:.2f}", flush=True)
            t0 = time.time() - el   # keep elapsed excluding readouts roughly comparable
    results["methods"][method] = curve
    torch.save(z.cpu(), a.out.with_name(a.out.stem + f"_{method}_z.pt"))
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(results, indent=1, default=str)); print("saved", a.out)
