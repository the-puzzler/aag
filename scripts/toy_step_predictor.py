#!/usr/bin/env python
"""Can we pick the assignment step budget BEFORE training a generator? (user, 2026-09-05)

Same synthetic manifold as toy_assignment_bench.py.  One long assignment per method; z is
snapshotted at log-spaced step counts.  At every snapshot we compute
  (a) assignment-only metrics -- things computable on a real (z, x) assignment with no
      generator and no ground truth, so they transfer to CelebA/ImageNet checkpoints:
        msw2_rand    max over 256 random directions of the projected W2^2 to N(0,1) (the transport objective)
        msw2_ref     the same after Adam-refining the best direction (max-sliced W2)
        c2st_z       kNN two-sample accuracy assigned z vs fresh N(0,I) (0.5 = Gaussian)
        cover        fresh-z coverage: median NN distance from fresh z to the assigned set / median
                     NN distance among assigned z  (>1: fresh samples land in gaps -- "nowhere safe")
        knn_hold     kNN-regression (k=8, inverse-distance) MSE predicting x from z on 2% held-out pairs
                     -- a non-parametric generator, the held-out criterion without training
        knn_fresh    kNN-regression at fresh z: weighted variance of the neighbours' x (map ambiguity
                     where the generator will actually be sampled)
        fd_knn       the kNN regressor used AS the generator: Frechet distance between its predictions at
                     fresh z and held-out real x (on real data: latent-space FD vs held-out latents);
                     c2st_knn the matching kNN two-sample accuracy
        fd_sur       a ~2 s surrogate generator (2-layer MLP, 1500 steps): FD at fresh z; hold_sur its held-out MSE
z snapshots are saved (fp16) next to --out so new metrics can be scored offline without redoing anything.
        lip          median ||x_i - x_j|| / ||z_i - z_j|| over z-nearest-neighbour pairs (locality; user distrusts)
  (b) the ground truth: an MLP generator trained on the pairs, FD_fresh / C2ST_fresh / heldout_mse.
Finally the Spearman correlation of every metric with FD_fresh across snapshots, and the step each
metric would have picked vs the step FD_fresh picks.
"""
from __future__ import annotations
import argparse, json, math, time, re
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from aag.gaussianize import (whiten, greedy_rank_transport_step, radial_chi_calibration, offset_slab_cleanup_step,
                             refine_direction, _rand_unit, _gaussian_quantiles)

ap = argparse.ArgumentParser()
ap.add_argument("--D", type=int, default=128); ap.add_argument("--k", type=int, default=16); ap.add_argument("--N", type=int, default=28000)
ap.add_argument("--noise", type=float, default=0.02)
ap.add_argument("--methods", default="random,refine_a0.3")
ap.add_argument("--snaps", default="0,250,500,1000,2000,3000,5000,7000,10000,15000,20000,30000,50000,100000")
ap.add_argument("--max-refine-steps", type=int, default=30000, help="refine variants are ~20x slower; cap their sweep")
ap.add_argument("--gen-steps", type=int, default=6000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); dev = "cuda"; torch.manual_seed(a.seed)

def make_f(k, D, gen):
    W1 = torch.randn(k, 256, generator=gen) / k ** 0.5; W2 = torch.randn(256, 256, generator=gen) / 16; W3 = torch.randn(256, D, generator=gen) / 16
    def f(u): return torch.tanh(torch.tanh(u @ W1.to(u.device)) @ W2.to(u.device)) @ W3.to(u.device)
    return f
g = torch.Generator().manual_seed(a.seed); f = make_f(a.k, a.D, g)
u = torch.randn(a.N + 5000, a.k, generator=g); x_all = f(u) + a.noise * torch.randn(a.N + 5000, a.D, generator=g)
x_all = (x_all - x_all.mean(0)) / x_all.std(0).clamp_min(1e-6)
x, x_test = x_all[: a.N].to(dev), x_all[a.N:].to(dev)

def frechet(A, B):
    muA, muB = A.mean(0), B.mean(0); cA, cB = torch.cov(A.T), torch.cov(B.T)
    ev = torch.linalg.eigvals(cA @ cB).real.clamp_min(0)
    return float(((muA - muB) ** 2).sum() + torch.trace(cA) + torch.trace(cB) - 2 * ev.sqrt().sum())
def c2st(A, B, k=10):
    X = torch.cat([A, B]); lab = torch.cat([torch.zeros(len(A)), torch.ones(len(B))]).to(A.device)
    Dm = torch.cdist(X, X); Dm.fill_diagonal_(float("inf")); nn_ = Dm.topk(k, largest=False).indices
    pred = (lab[nn_].mean(1) > 0.5).float(); return float((pred == lab).float().mean())

# fixed held-out split shared by the kNN proxy and the MLP generator
N = a.N; hold = torch.zeros(N, dtype=torch.bool, device=dev); hold[torch.randperm(N, device=dev, generator=torch.Generator(device=dev).manual_seed(7))[: N // 50]] = True
tr_idx = (~hold).nonzero(as_tuple=True)[0]; ho_idx = hold.nonzero(as_tuple=True)[0]

def train_generator(z, seed=0):
    torch.manual_seed(seed); d = z.shape[1]; D = x.shape[1]
    G = nn.Sequential(nn.Linear(d, 1024), nn.SiLU(), nn.Linear(1024, 1024), nn.SiLU(), nn.Linear(1024, 1024), nn.SiLU(), nn.Linear(1024, D)).to(dev)
    opt = torch.optim.AdamW(G.parameters(), lr=1e-3, weight_decay=0.01); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.gen_steps)
    for step in range(a.gen_steps):
        b = tr_idx[torch.randint(len(tr_idx), (512,), device=dev)]
        loss = F.mse_loss(G(z[b]), x[b]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    G.eval()
    with torch.no_grad():
        hm = F.mse_loss(G(z[ho_idx]), x[ho_idx]).item()
        zf = torch.randn(5000, d, device=dev); xg = G(zf)
        return {"heldout_mse": hm, "FD_fresh": frechet(xg, x_test), "C2ST_fresh": c2st(xg, x_test)}

def surrogate_fd(z, steps=1500, seed=0):
    """A ~2 s stand-in for the generator: tiny MLP z->x on the training pairs, FD at fresh z."""
    torch.manual_seed(seed); d = z.shape[1]; D = x.shape[1]
    G = nn.Sequential(nn.Linear(d, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, D)).to(dev)
    opt = torch.optim.AdamW(G.parameters(), lr=2e-3, weight_decay=0.01); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        b = tr_idx[torch.randint(len(tr_idx), (512,), device=dev)]
        loss = F.mse_loss(G(z[b]), x[b]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    G.eval()
    with torch.no_grad():
        return frechet(G(torch.randn(5000, d, device=dev)), x_test), F.mse_loss(G(z[ho_idx]), x[ho_idx]).item()

@torch.no_grad()
def knn_regress(zq, zref, xref, k=8):
    """inverse-distance weighted kNN prediction and the weighted variance of the neighbours' x."""
    Dm = torch.cdist(zq, zref); dist, idx = Dm.topk(k, largest=False)
    w = 1.0 / dist.clamp_min(1e-6); w = w / w.sum(1, keepdim=True)
    xn = xref[idx]                                        # (q, k, D)
    pred = (w.unsqueeze(-1) * xn).sum(1)
    var = (w.unsqueeze(-1) * (xn - pred.unsqueeze(1)) ** 2).sum(1).mean(1)   # per-query mean over D
    return pred, var

@torch.no_grad()
def assignment_metrics(z, gen):
    d = z.shape[1]; m = min(8192, N)
    sub = z[torch.randperm(N, device=dev, generator=gen)[:m]]
    # transport objective, measured identically for every method
    dirs = _rand_unit(256, d, dev, z.dtype); s, _ = torch.sort(sub @ dirs.T, dim=0); q = _gaussian_quantiles(m, dev, z.dtype).unsqueeze(1)
    scores = ((s - q) ** 2).mean(0); best = scores.argmax(); msw2_rand = float(scores[best])
    with torch.enable_grad():
        ar = refine_direction(sub, dirs[best], steps=30)
    sr, _ = torch.sort(sub @ ar); msw2_ref = float(((sr - q.squeeze(1)) ** 2).mean())
    zf = torch.randn(m, d, device=dev, generator=gen)
    c2 = c2st(sub, zf)
    # coverage: fresh->assigned NN distance vs assigned->assigned NN distance
    d_fa = torch.cdist(zf[:4096], z).min(1).values
    Daa = torch.cdist(sub[:4096], z); Daa[Daa < 1e-9] = float("inf")     # drop self-matches
    d_aa = Daa.min(1).values
    cover = float(d_fa.median() / d_aa.median())
    # kNN regression proxies
    pred, _ = knn_regress(z[ho_idx], z[tr_idx], x[tr_idx]); knn_hold = float(F.mse_loss(pred, x[ho_idx]))
    pred_f, var_f = knn_regress(zf[:5000], z, x); knn_fresh = float(var_f.mean())
    # the kNN regressor used AS the generator at fresh z: FD / C2ST against held-out real x
    fd_knn = frechet(pred_f, x_test); c2st_knn = c2st(pred_f, x_test)
    with torch.enable_grad():
        fd_sur, hold_sur = surrogate_fd(z)
    return {"msw2_rand": msw2_rand, "msw2_ref": msw2_ref, "c2st_z": c2, "cover": cover, "knn_hold": knn_hold, "knn_fresh": knn_fresh,
            "fd_knn": fd_knn, "c2st_knn": c2st_knn, "fd_sur": fd_sur, "hold_sur": hold_sur}

z0, *_ = whiten(x, rotate=False); z0 = z0.contiguous(); d = z0.shape[1]
snaps = sorted(int(s) for s in a.snaps.split(","))
cols = ["msw2_rand", "msw2_ref", "c2st_z", "cover", "knn_hold", "knn_fresh", "fd_knn", "c2st_knn", "fd_sur", "hold_sur", "heldout_mse", "FD_fresh", "C2ST_fresh"]
def show(step, r, t): print(f"{step:7d} {t:6.0f}s | " + " ".join(f"{r[c]:9.4f}" for c in cols), flush=True)

def run(method):
    z = z0.clone(); gen = torch.Generator(device=dev).manual_seed(1); mgen = torch.Generator(device=dev).manual_seed(3)
    alpha = float(re.search(r"_a([0-9.]+)", method).group(1)) if "_a" in method else 1.0
    refine = method.startswith("refine"); ss = sorted(s for s in snaps if not refine or s <= a.max_refine_steps)
    # locality needs x-index bookkeeping; compute inline here where we know row identity
    rows = []; t0 = time.time(); step = 0
    print(f"\n== {method}  alpha={alpha}  refine={refine}\n{'step':>7s} {'time':>7s} | " + " ".join(f"{c:>9s}" for c in cols), flush=True)
    for target in ss:
        while step < target:
            step += 1
            if refine: greedy_rank_transport_step(z, search_subset=8192, n_dirs=64, alpha=alpha, gen=gen, return_score=False, refine_steps=30)
            else: greedy_rank_transport_step(z, search_subset=2048, n_dirs=64, alpha=alpha, gen=gen, return_score=False)
            if step % 2 == 0: offset_slab_cleanup_step(z, search_subset=2048, n_slabs=32, eps=0.5, alpha=1.0, gen=gen, return_score=False)
            if step % 20 == 0: radial_chi_calibration(z, d=d, alpha_r=1.0)
        torch.cuda.synchronize(); ta = time.time() - t0
        r = assignment_metrics(z, mgen)
        # locality: z-NN pairs within a subset, ratio of x distance to z distance
        with torch.no_grad():
            sidx = torch.randperm(N, device=dev, generator=mgen)[:4096]; Dz = torch.cdist(z[sidx], z); Dz[torch.arange(4096, device=dev), sidx] = float("inf")
            nnd, nni = Dz.min(1); r["lip"] = float(((x[sidx] - x[nni]).norm(dim=1) / nnd).median())
        snapdir = a.out.parent / (a.out.stem + "_snaps"); snapdir.mkdir(parents=True, exist_ok=True)
        torch.save(z.half().cpu(), snapdir / f"{method}_{step}.pt")
        r |= train_generator(z); r["step"] = step; r["assign_seconds"] = round(ta); rows.append(r); show(step, r, ta)
    return rows

def spearman(u, v):
    ru = torch.tensor(u).argsort().argsort().float(); rv = torch.tensor(v).argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([ru, rv]))[0, 1])

res = {"config": vars(a), "data_ref": {"FD_test_vs_train": frechet(x[:5000], x_test)}, "methods": {}}
print(f"D={a.D} k={a.k} N={a.N} (N/d={a.N/a.D:.0f})  reference FD(test,train)={res['data_ref']['FD_test_vs_train']:.3f}", flush=True)
for m in a.methods.split(","):
    rows = run(m); res["methods"][m] = rows
    fd = [r["FD_fresh"] for r in rows]; best = rows[min(range(len(rows)), key=lambda i: fd[i])]["step"]
    print(f"-- {m}: FD_fresh picks step {best}.  Spearman(metric, FD_fresh) and the step each metric picks (argmin; c2st/cover argmin too):")
    for c in ["msw2_rand", "msw2_ref", "c2st_z", "cover", "knn_hold", "knn_fresh", "fd_knn", "c2st_knn", "fd_sur", "hold_sur", "lip", "heldout_mse"]:
        v = [r[c] for r in rows]; pick = rows[min(range(len(rows)), key=lambda i: v[i])]["step"]
        fd_at = next(r["FD_fresh"] for r in rows if r["step"] == pick)
        print(f"   {c:12s} rho={spearman(v, fd):+.2f}  picks {pick:6d}  FD there {fd_at:8.3f} (best {min(fd):.3f})", flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1, default=str)); print("saved", a.out)
