#!/usr/bin/env python
"""Gaussianise VPT world-model particles, with the action side as a DISTANCE.

Mirrors run_assignment_doom_ctx.py's budget split -- mostly context-only steps
plus some action-conditioned ones -- but the action steps use
action_dist_knn_transport_step (action-nearest-k_act, then context-nearest-k)
instead of an exact categorical filter, because the 81-way index discards mouse
magnitude and magnitude explains more frame-to-frame change than all keys
combined.

Reports, every --eval-every steps:
  G  global gaussian defect
  I_ctx     context-nearest-k W2 / random-subset floor   -> 1.0
  I_act     action-nearest-k  W2 / random-subset floor   -> 1.0
  per-action worst classes, so one bad action is visible rather than averaged
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from aag.gaussianize import (whiten, greedy_rank_transport_step,
                             continuous_knn_transport_step,
                             action_dist_knn_transport_step)
from aag.diagnostics import (random_subset_w2, continuous_knn_w2,
                             action_dist_knn_w2, per_action_w2)

ap = argparse.ArgumentParser()
ap.add_argument("--particles", required=True)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--search-subset", type=int, default=2048)
ap.add_argument("--n-dirs", type=int, default=64)
ap.add_argument("--ctx-per-step", type=int, default=8)
ap.add_argument("--act-per-step", type=int, default=8)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--cond-alpha", type=float, default=0.25)
ap.add_argument("--k", type=int, default=2048)
ap.add_argument("--k-act", type=int, default=8192)
ap.add_argument("--eval-k", type=int, default=2048)
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
P = torch.load(a.particles, map_location="cpu", weights_only=False)
h = P["h_target"].to(dev).float()
cond = P["h_context"].to(dev).float()
act = P["action"].to(dev)
av = P["action_vec"].to(dev).float()
N, d = h.shape
print(f"{N:,} particles  dim={d}  cond_dim={cond.shape[1]}  "
      f"context={P['context']} gamma={P['gamma']}", flush=True)
print(f"budget: {a.steps} global x ({a.ctx_per_step} ctx + {a.act_per_step} act) "
      f"= {a.steps*(a.ctx_per_step+a.act_per_step):,} conditional firings", flush=True)
print("target: BOTH ratios -> 1.0 (below 1.0 = over-transported)", flush=True)

# rotate=False keeps coordinate j of z meaning coordinate j of h, which matters
# for a spatial AE latent whose grid topology a PCA rotation would destroy.
z, mean, W, W_inv = whiten(h, rotate=False)
z = z.contiguous()
gen = torch.Generator(device=dev).manual_seed(a.seed)

curve = {"step": [], "ctx_ratio": [], "act_ratio": [], "floor": [], "G": []}


def gdefect(t):
    dirs = torch.randn(64, t.shape[1], device=t.device, generator=gen)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    s, _ = torch.sort(t @ dirs.T, dim=0)
    q = torch.special.ndtri(
        (torch.arange(len(t), device=t.device, dtype=t.dtype) + .5) / len(t)).unsqueeze(1)
    return ((s - q) ** 2).mean().item()


for step in range(1, a.steps + 1):
    greedy_rank_transport_step(z, search_subset=a.search_subset, n_dirs=a.n_dirs,
                               alpha=a.alpha, gen=gen)
    for _ in range(a.ctx_per_step):
        continuous_knn_transport_step(z, cond, k=a.k, n_dirs=a.n_dirs,
                                      alpha=a.cond_alpha, gen=gen, metric="l2")
    for _ in range(a.act_per_step):
        action_dist_knn_transport_step(z, cond, av, k=a.k, k_act=a.k_act,
                                       n_dirs=a.n_dirs, alpha=a.cond_alpha, gen=gen)

    if step % a.eval_every == 0 or step == 1:
        floor = random_subset_w2(z, k=a.eval_k, n_eval=20, gen=gen)
        ctx = continuous_knn_w2(z, cond, k=a.eval_k, n_eval=20, gen=gen, metric="l2")
        actw = action_dist_knn_w2(z, cond, av, k=a.eval_k, k_act=a.k_act,
                                  n_eval=20, gen=gen)
        G = gdefect(z)
        curve["step"].append(step); curve["floor"].append(floor); curve["G"].append(G)
        curve["ctx_ratio"].append(ctx / max(floor, 1e-12))
        curve["act_ratio"].append(actw / max(floor, 1e-12))
        print(f"step {step:5d}  G={G:.5f}  floor={floor:.5f}  "
              f"I_ctx={ctx/max(floor,1e-12):.3f}  I_act={actw/max(floor,1e-12):.3f}",
              flush=True)

pa = per_action_w2(z, act, gen=gen)
floor = random_subset_w2(z, k=a.eval_k, n_eval=40, gen=gen)
rows = sorted(((r, k, n) for k, (r, n, _, _) in pa.items()), reverse=True)
print(f"\nper-action W2 / SIZE-MATCHED floor  ({len(rows)} classes with >=256 members)")
print(f"  worst 6: " + "  ".join(f"a{k}:{r:.2f}(n={n})" for r, k, n in rows[:6]))
print(f"  best  6: " + "  ".join(f"a{k}:{r:.2f}(n={n})" for r, k, n in rows[-6:]))
rs = np.array([r for r, _, _ in rows]); ns = np.array([n for _, _, n in rows])
print(f"  mean {rs.mean():.3f}  median {np.median(rs):.3f}  max {rs.max():.3f}")
print(f"  corr(log n, ratio) {np.corrcoef(np.log(ns), rs)[0,1]:+.2f} "
      f"(near 0 = the floor match is working)")

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"z": z.cpu(), "h": h.cpu(), "cond": cond.cpu(), "action": act.cpu(),
            "action_vec": av.cpu(), "mean": mean.cpu(), "W": W.cpu(), "W_inv": W_inv.cpu(),
            "curve": curve, "per_action": pa,
            "steps": a.steps, "k": a.k, "k_act": a.k_act,
            "ctx_per_step": a.ctx_per_step, "act_per_step": a.act_per_step,
            "cond_alpha": a.cond_alpha, "context": P["context"], "gamma": P["gamma"],
            "particles": a.particles}, out)
print(f"\nsaved -> {out}")
