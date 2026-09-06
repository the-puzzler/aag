#!/usr/bin/env python
"""Which assignment makes the best GENERATOR? A synthetic end-to-end benchmark (user, 2026-09-05).

Data: x = f(u) + eps in R^D with u ~ N(0, I_k) on a smooth random nonlinear manifold (2-layer tanh MLP
with a random linear embedding), k << D, N points for the assignment/generator and a held-out set.
For every assignment method the SAME MLP generator G: z -> x is trained on the (z_i, x_i) pairs
and judged on what matters in AAG -- samples from fresh z ~ N(0, I):
  FD_fresh   Frechet distance between G(fresh z) and held-out real x, in data space
  C2ST_fresh accuracy of a kNN two-sample test generated-vs-real (0.5 = indistinguishable)
  FD_assigned  the same with the assigned z (the ceiling set by the generator + pairs)
  heldout_mse  MSE of G on never-trained assigned pairs (the AAG held-out criterion)
Assignment methods share the rank transport; only the direction choice (and alpha) differs.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from aag.gaussianize import (whiten, greedy_rank_transport_step, radial_chi_calibration, offset_slab_cleanup_step,
                             population_direction, rank_transport_along, aag2_block_step, aag2_defect, aag2_floor, _rand_unit)

ap = argparse.ArgumentParser()
ap.add_argument("--D", type=int, default=128); ap.add_argument("--k", type=int, default=16); ap.add_argument("--N", type=int, default=28000)
ap.add_argument("--noise", type=float, default=0.02); ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--methods", default="random,random_x10,population,refine,refine_a0.3,refine_a0.1")
ap.add_argument("--gen-steps", type=int, default=6000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); dev = "cuda"; torch.manual_seed(a.seed)

# ---- synthetic manifold data
def make_f(k, D, gen):
    W1 = torch.randn(k, 256, generator=gen) / k ** 0.5; W2 = torch.randn(256, 256, generator=gen) / 16; W3 = torch.randn(256, D, generator=gen) / 16
    def f(u): return torch.tanh(torch.tanh(u @ W1.to(u.device)) @ W2.to(u.device)) @ W3.to(u.device)
    return f
g = torch.Generator().manual_seed(a.seed); f = make_f(a.k, a.D, g)
u = torch.randn(a.N + 5000, a.k, generator=g); x_all = f(u) + a.noise * torch.randn(a.N + 5000, a.D, generator=g)
x_all = (x_all - x_all.mean(0)) / x_all.std(0).clamp_min(1e-6)     # like a whitened AE latent
x, x_test = x_all[: a.N].to(dev), x_all[a.N:].to(dev)

def frechet(A, B):
    muA, muB = A.mean(0), B.mean(0); cA, cB = torch.cov(A.T), torch.cov(B.T)
    ev = torch.linalg.eigvals(cA @ cB).real.clamp_min(0)
    return float(((muA - muB) ** 2).sum() + torch.trace(cA) + torch.trace(cB) - 2 * ev.sqrt().sum())
def c2st(A, B, k=10):
    X = torch.cat([A, B]); lab = torch.cat([torch.zeros(len(A)), torch.ones(len(B))]).to(A.device)
    Dm = torch.cdist(X, X); Dm.fill_diagonal_(float("inf")); nn_ = Dm.topk(k, largest=False).indices
    pred = (lab[nn_].mean(1) > 0.5).float(); return float((pred == lab).float().mean())

def train_generator(z, x, seed=0):
    torch.manual_seed(seed); N, d = z.shape; D = x.shape[1]
    hold = torch.zeros(N, dtype=torch.bool, device=dev); hold[torch.randperm(N, device=dev)[: N // 50]] = True
    G = nn.Sequential(nn.Linear(d, 1024), nn.SiLU(), nn.Linear(1024, 1024), nn.SiLU(), nn.Linear(1024, 1024), nn.SiLU(), nn.Linear(1024, D)).to(dev)
    opt = torch.optim.AdamW(G.parameters(), lr=1e-3, weight_decay=0.01); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.gen_steps)
    tr = (~hold).nonzero(as_tuple=True)[0]
    for step in range(a.gen_steps):
        b = tr[torch.randint(len(tr), (512,), device=dev)]
        loss = F.mse_loss(G(z[b]), x[b]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    G.eval()
    with torch.no_grad():
        hm = F.mse_loss(G(z[hold]), x[hold]).item()
        zf = torch.randn(5000, d, device=dev); xg = G(zf); xa = G(z[torch.randperm(N, device=dev)[:5000]])
        return {"heldout_mse": hm, "FD_fresh": frechet(xg, x_test), "C2ST_fresh": c2st(xg, x_test), "FD_assigned": frechet(xa, x_test), "C2ST_assigned": c2st(xa, x_test)}

z0, *_ = whiten(x, rotate=False); z0 = z0.contiguous(); d = z0.shape[1]
def run(method):
    z = z0.clone(); gen = torch.Generator(device=dev).manual_seed(1); t0 = time.time()
    import re
    m_ = re.match(r"random_(\d+)k$", method); steps = int(m_.group(1)) * 1000 if m_ else (a.steps * 10 if method == "random_x10" else a.steps)
    m2 = re.match(r"(refine_a[0-9.]+)_(\d+)k$", method)
    if m2: method, steps = m2.group(1), int(m2.group(2)) * 1000
    alpha = {"refine_a0.3": 0.3, "refine_a0.1": 0.1}.get(method, 1.0)
    m3 = re.match(r"aag2(?:s(\d+))?(?:_r([0-9.]+))?_(\d+|floor)$", method)   # aag2[s<search>][_r<ridge>]_<blocks|floor>
    if m3:
        search = int(m3.group(1) or 1); ridge = float(m3.group(2) or 0.02); dirs = _rand_unit(256, d, dev, z.dtype); egen = torch.Generator(device=dev).manual_seed(7)
        floor, _ = aag2_floor(z.shape[0], d, dirs, gen=egen, device=dev); crossed = None
        nb = 400 if m3.group(3) == "floor" else int(m3.group(3))
        for b in range(1, nb + 1):
            aag2_block_step(z, ridge=ridge, gen=gen, search=search)
            r_ = aag2_defect(z, dirs) / floor
            if m3.group(3) == "floor" and r_ <= 1.0: crossed = b; break
        torch.cuda.synchronize(); ta = time.time() - t0
        zg = torch.randn(8192, d, device=dev); aa = c2st(z[torch.randperm(z.shape[0], device=dev)[:8192]], zg)
        r = train_generator(z, x) | {"assign_seconds": round(ta), "z_C2ST_vs_gaussian": aa, "disp": float((z - z0).norm(dim=1).mean() / d ** 0.5), "blocks": crossed or nb, "G_ratio": r_}
        print(f"    [{method}: blocks={crossed or nb} final G/floor={r_:.3f}]", flush=True); return r
    for step in range(1, steps + 1):
        if method.startswith("random"):
            greedy_rank_transport_step(z, search_subset=2048, n_dirs=64, alpha=1.0, gen=gen, return_score=False)
        elif method.startswith("refine"):
            greedy_rank_transport_step(z, search_subset=8192, n_dirs=64, alpha=alpha, gen=gen, return_score=False, refine_steps=30)
        elif method == "population":
            v, _ = population_direction(z, n_pop=32, probe_steps=20, batch=4096, eval_subset=8192, lr=0.05, gen=gen); rank_transport_along(z, v)
        if step % 2 == 0: offset_slab_cleanup_step(z, search_subset=2048, n_slabs=32, eps=0.5, alpha=1.0, gen=gen, return_score=False)
        if step % 20 == 0: radial_chi_calibration(z, d=d, alpha_r=1.0)
    torch.cuda.synchronize(); ta = time.time() - t0
    zg = torch.randn(8192, d, device=dev); aa = c2st(z[torch.randperm(z.shape[0], device=dev)[:8192]], zg)
    r = train_generator(z, x) | {"assign_seconds": round(ta), "z_C2ST_vs_gaussian": aa, "disp": float((z - z0).norm(dim=1).mean() / d ** 0.5)}
    return r
res = {"config": vars(a), "data_ref": {"FD_test_vs_train": frechet(x[:5000], x_test), "C2ST_test_vs_train": c2st(x[:5000], x_test)}, "methods": {}}
res["methods"]["no_transport (whitened x as z)"] = run.__wrapped__(z0) if False else train_generator(z0, x) | {"assign_seconds": 0, "z_C2ST_vs_gaussian": c2st(z0[:8192], torch.randn(8192, d, device=dev)), "disp": 0.0}
print(f"D={a.D} k={a.k} N={a.N}  reference: FD(test,train)={res['data_ref']['FD_test_vs_train']:.3f}  C2ST={res['data_ref']['C2ST_test_vs_train']:.3f}", flush=True)
hdr = f"{'method':28s} {'z C2ST':>7s} {'disp':>5s} {'assign s':>8s} | {'FD fresh':>9s} {'C2ST fresh':>10s} | {'FD assigned':>11s} {'heldout mse':>11s}"
print(hdr, flush=True)
def show(name, r): print(f"{name:28s} {r['z_C2ST_vs_gaussian']:7.3f} {r['disp']:5.2f} {r['assign_seconds']:8d} | {r['FD_fresh']:9.3f} {r['C2ST_fresh']:10.3f} | {r['FD_assigned']:11.3f} {r['heldout_mse']:11.4f}", flush=True)
show("no_transport", res["methods"]["no_transport (whitened x as z)"])
for m in a.methods.split(","):
    res["methods"][m] = run(m); show(m, res["methods"][m])
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1, default=str)); print("saved", a.out)
