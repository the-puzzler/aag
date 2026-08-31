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

from aag.gaussianize import (continuous_knn_transport_batch, group_rank_transport_step, whiten, greedy_rank_transport_step,
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
ap.add_argument("--grp-per-step", type=int, default=0,
                help="Exact-action-class transport firings per step. The other "
                     "action step decorrelates z from the CONTINUOUS act_vec, but "
                     "the generator conditions on a discrete 81-way one-hot -- a "
                     "different partition. Measured on the 16k run: z sits 2.49x "
                     "further off-centre within an action class than chance, "
                     "against 1.03x for context, which is why a fresh z degrades. "
                     "This transports the true p(z | class).")
ap.add_argument("--max-group", type=int, default=8192,
                help="cap per class so one huge class (a0 has 207k members) does "
                     "not dominate the step cost")
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--cond-alpha", type=float, default=0.25)
ap.add_argument("--k", type=int, default=2048)
ap.add_argument("--k-act", type=int, default=8192)
ap.add_argument("--eval-k", type=int, default=2048)
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save-every", type=int, default=0,
                help="If >0, write the assignment every N steps as well as at the "
                     "end. A long run that only saves on completion loses "
                     "everything to a crash, and cannot be stopped early to take "
                     "what it has.")
ap.add_argument("--ctx-metric", choices=["l2", "cosine"], default="l2",
                help="Distance defining the CONTEXT neighbourhood. The doom run "
                     "that reached I 0.860 used cosine; VPT has been using l2. "
                     "cond_distance's rationale: AE latent magnitude tracks "
                     "brightness/contrast rather than content, so under l2 a "
                     "neighbourhood groups frames by overall brightness as much "
                     "as by scene -- and Minecraft spans day, night, caves and "
                     "biomes. Applied to the eval too: measuring l2 independence "
                     "while transporting cosine neighbourhoods would be "
                     "meaningless.")
ap.add_argument("--resume-z", default=None,
                help="Continue transport from a saved assignment's z instead of "
                     "re-whitening from scratch. The 512k run looked plateaued "
                     "at I_ctx 7.5 around step 5000 and still reached 1.035 by "
                     "16000, so an apparent plateau is not evidence of a floor "
                     "-- this makes testing that cheap rather than a restart.")
a = ap.parse_args()

dev = "cuda"
import gc
P = torch.load(a.particles, map_location="cpu", weights_only=False)
h = P["h_target"].to(dev).float()
cond = P["h_context"].to(dev).float()
act = P["action"].to(dev)
av = P["action_vec"].to(dev).float()
N, d = h.shape
# h_context is 40.9 GB at 1.66M particles and is dead once cond is on the GPU.
# Holding it alongside the GPU copy AND the save-time cond.cpu() copy is what
# OOM-killed the first 32k attempt -- earlyoom fired at VmRSS 117 GB.
_keep = {k: P[k] for k in ("chunk", "frame", "episode", "cache", "checkpoint",
                           "context", "gamma") if k in P}
P.clear(); P.update(_keep)
gc.collect()
print(f"{N:,} particles  dim={d}  cond_dim={cond.shape[1]}  "
      f"context={P['context']} gamma={P['gamma']} ctx_metric={a.ctx_metric}",
      flush=True)
print(f"budget: {a.steps} global x ({a.ctx_per_step} ctx + {a.act_per_step} act "
      f"+ {a.grp_per_step} grp) = "
      f"{a.steps*(a.ctx_per_step+a.act_per_step+a.grp_per_step):,} conditional firings",
      flush=True)
print("NOTE: these ratios do NOT select a good assignment. Audited\n       2026-08-31: the doom assignment that actually won logged\n       joint 1.14 / 1.02 / 0.86 / 1.12 (the last two the SAME step),\n       and VPT's runs live at 0.92-1.17 -- the same regime on a\n       +/-0.13 instrument. The 0.860 and 0.778 figures previously\n       quoted as targets were single noisy evals. Read neighbouring\n       evals before quoting any value, and judge an assignment by the\n       fresh-z MSE of a generator trained on it, which is what the\n       doom run selected on.", flush=True)

# rotate=False keeps coordinate j of z meaning coordinate j of h, which matters
# for a spatial AE latent whose grid topology a PCA rotation would destroy.
step0 = 0
if a.resume_z:
    R = torch.load(a.resume_z, map_location="cpu", weights_only=False)
    if R["z"].shape != h.shape:
        raise SystemExit(f"resume z is {tuple(R['z'].shape)} but these particles "
                         f"are {tuple(h.shape)} -- different particle file")
    z = R["z"].to(dev).float().contiguous()
    mean = R["mean"].to(dev); W = R["W"].to(dev); W_inv = R["W_inv"].to(dev)
    step0 = int(R.get("steps", 0))
    print(f"resumed z from {a.resume_z} at step {step0:,} "
          f"(I_ctx was {R['curve']['ctx_ratio'][-1]:.3f}, "
          f"I_act {R['curve']['act_ratio'][-1]:.3f})", flush=True)
    _curve0 = {k: list(v) for k, v in R["curve"].items()}
    R.clear(); R["curve"] = _curve0    # its cond/h were a third 40.9 GB copy
    gc.collect()
else:
    z, mean, W, W_inv = whiten(h, rotate=False)
    z = z.contiguous()
gen = torch.Generator(device=dev).manual_seed(a.seed)

curve = ({k: list(v) for k, v in R["curve"].items()} if a.resume_z
         else {"step": [], "ctx_ratio": [], "act_ratio": [], "floor": [], "G": []})


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
    # one pass over cond for all ctx firings instead of one per firing --
    # bit-identical, and cond is 12.6 GB at 512k particles
    continuous_knn_transport_batch(z, cond, k=a.k, n_dirs=a.n_dirs,
                                   alpha=a.cond_alpha, gen=gen,
                                   n_fire=a.ctx_per_step, metric=a.ctx_metric)
    for _ in range(a.act_per_step):
        action_dist_knn_transport_step(z, cond, av, k=a.k, k_act=a.k_act,
                                       n_dirs=a.n_dirs, alpha=a.cond_alpha, gen=gen)
    for _ in range(a.grp_per_step):
        group_rank_transport_step(z, act, n_dirs=a.n_dirs, alpha=a.cond_alpha,
                                  gen=gen, max_group=a.max_group)

    if step % a.eval_every == 0 or step == 1:
        floor = random_subset_w2(z, k=a.eval_k, n_eval=20, gen=gen)
        ctx = continuous_knn_w2(z, cond, k=a.eval_k, n_eval=20, gen=gen,
                                metric=a.ctx_metric)
        actw = action_dist_knn_w2(z, cond, av, k=a.eval_k, k_act=a.k_act,
                                  n_eval=20, gen=gen)
        G = gdefect(z)
        curve["step"].append(step0 + step)
        curve["floor"].append(floor); curve["G"].append(G)
        curve["ctx_ratio"].append(ctx / max(floor, 1e-12))
        curve["act_ratio"].append(actw / max(floor, 1e-12))
        print(f"step {step0 + step:5d}  G={G:.5f}  floor={floor:.5f}  "
              f"I_ctx={ctx/max(floor,1e-12):.3f}  I_act={actw/max(floor,1e-12):.3f}",
              flush=True)

    if a.save_every and step % a.save_every == 0 and step < a.steps:
        _p = Path(a.out); _p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"z": z.cpu(), "h": h.cpu(), "cond": cond.cpu(), "action": act.cpu(),
                    "chunk": P.get("chunk"), "frame": P.get("frame"),
                    "episode": P.get("episode"), "cache": P.get("cache"),
                    "ae_checkpoint": P.get("checkpoint"),
                    "action_vec": av.cpu(), "mean": mean.cpu(), "W": W.cpu(),
                    "W_inv": W_inv.cpu(), "curve": curve, "per_action": None,
                    "steps": step0 + step, "k": a.k, "k_act": a.k_act,
                    "ctx_per_step": a.ctx_per_step, "act_per_step": a.act_per_step,
                    "cond_alpha": a.cond_alpha, "ctx_metric": a.ctx_metric,
                    "context": P["context"],
                    "gamma": P["gamma"], "particles": a.particles,
                    "partial": True}, str(_p) + ".tmp")
        Path(str(_p) + ".tmp").replace(_p)   # atomic: never a truncated file
        gc.collect()
        print(f"  [checkpoint at step {step0 + step}]", flush=True)

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
# the assignment permutes nothing -- row i of z is still particle i -- so the
# cache coordinates carry through unchanged, and the generator can index frames
torch.save({"z": z.cpu(), "h": h.cpu(), "cond": cond.cpu(), "action": act.cpu(),
            "chunk": P.get("chunk"), "frame": P.get("frame"),
            "episode": P.get("episode"), "cache": P.get("cache"),
            "ae_checkpoint": P.get("checkpoint"),
            "action_vec": av.cpu(), "mean": mean.cpu(), "W": W.cpu(), "W_inv": W_inv.cpu(),
            "curve": curve, "per_action": pa,
            "steps": step0 + a.steps, "k": a.k, "k_act": a.k_act,
            "ctx_per_step": a.ctx_per_step, "act_per_step": a.act_per_step,
            "cond_alpha": a.cond_alpha, "grp_per_step": a.grp_per_step,
            "ctx_metric": a.ctx_metric,
            "context": P["context"], "gamma": P["gamma"],
            "particles": a.particles}, out)
print(f"\nsaved -> {out}")
