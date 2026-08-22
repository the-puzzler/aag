"""Doom world-model assignment that spends most of its conditional budget on
CONTEXT-only neighbourhoods.

run_assignment_doom.py filters each conditional step to one action and then takes
k-NN on the 3-frame context, so every step works inside the intersection. But the
action factor of the independence ratio is already ~1.23 at cond-per-step 32 while
the frames factor is still 33.8 -- so half of each step's effort goes to a
constraint that has converged.

Here the conditional steps are split: --ctx-per-step continuous k-NN steps on the
context alone (ignoring the action), plus --act-per-step action-filtered k-NN steps
to keep the action factor from drifting back.

Output format matches run_assignment_doom.py exactly so the same diagnostics and
generator scripts work unchanged.
"""
import argparse, json, os, sys
from pathlib import Path


import torch

from aag.gaussianize import (whiten, greedy_rank_transport_step,
                             action_knn_transport_step, continuous_knn_transport_step)
from aag.diagnostics import (knn_preservation, assignment_diagnostics, action_knn_w2,
                             random_subset_w2, continuous_knn_w2, group_w2,
                             intrinsic_dimension_twonn)

ap = argparse.ArgumentParser()
ap.add_argument("--particles", required=True)
ap.add_argument("--steps", type=int, default=16000)
ap.add_argument("--search-subset", type=int, default=2048)
ap.add_argument("--n-dirs", type=int, default=64)
ap.add_argument("--ctx-per-step", type=int, default=112)
ap.add_argument("--act-per-step", type=int, default=16)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--cond-alpha", type=float, default=0.25)
ap.add_argument("--k", type=int, default=4096)
ap.add_argument("--eval-k", type=int, default=4096)
ap.add_argument("--eval-every", type=int, default=400)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
P = torch.load(a.particles, map_location=dev, weights_only=False)
h, cond, act = P["h_target"].to(dev), P["h_context"].to(dev), P["action"].to(dev)
N, d = h.shape
print(f"{N:,} particles, dim={d}, cond_dim={cond.shape[1]}", flush=True)
print(f"conditional split per global step: {a.ctx_per_step} context-only + "
      f"{a.act_per_step} action-filtered", flush=True)

z, mean, W, W_inv = whiten(h)
z0 = z.clone()
gen = torch.Generator(device=dev).manual_seed(a.seed)
curve = {"step": [], "ratio": [], "cond_w2": [], "floor_w2": [],
         "displacement": [], "proj_over_gauss": [], "knn": [],
         "frames_ratio": [], "action_ratio": []}

for s in range(1, a.steps + 1):
    greedy_rank_transport_step(z, search_subset=a.search_subset,
                               n_dirs=a.n_dirs, alpha=a.alpha, gen=gen)
    for _ in range(a.ctx_per_step):
        continuous_knn_transport_step(z, cond, k=a.k, n_dirs=a.n_dirs,
                                      alpha=a.cond_alpha, gen=gen, metric="cosine")
    for _ in range(a.act_per_step):
        action_knn_transport_step(z, cond, act, k=a.k, n_dirs=a.n_dirs,
                                  alpha=a.cond_alpha, gen=gen)
    if s % a.eval_every == 0 or s == a.steps:
        cw = action_knn_w2(z, cond, act, k=a.eval_k, gen=gen)
        fw = random_subset_w2(z, k=a.eval_k, gen=gen)
        fr = continuous_knn_w2(z, cond, k=a.eval_k, gen=gen)
        ar, _ = group_w2(z, act, gen=gen, max_group=a.eval_k)
        diag = assignment_diagnostics(z, d=d)
        disp = float((z - z0).norm(dim=1).mean())
        knn = knn_preservation(z0, z, k=10, gen=gen)
        curve["step"].append(s); curve["ratio"].append(cw / max(fw, 1e-12))
        curve["cond_w2"].append(cw); curve["floor_w2"].append(fw)
        curve["displacement"].append(disp)
        curve["proj_over_gauss"].append(diag["proj_over_gauss"])
        curve["knn"].append(knn)
        curve["frames_ratio"].append(fr / max(fw, 1e-12))
        curve["action_ratio"].append(ar / max(fw, 1e-12))
        print(f"  [{s}/{a.steps}] joint={cw/max(fw,1e-12):.2f} "
              f"frames={fr/max(fw,1e-12):.2f} action={ar/max(fw,1e-12):.2f} "
              f"disp={disp:.3f} G={diag['proj_over_gauss']:.2f} knn={knn:.4f}", flush=True)

diag = assignment_diagnostics(z, d=d)
print("assignment diagnostics:", diag, flush=True)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
torch.save({"z": z.cpu(), "h": h.cpu(), "cond": cond.cpu(), "action": act.cpu(),
            "chunk": P["chunk"], "frame": P["frame"], "episode": P["episode"],
            "mean": mean.cpu(), "W": W.cpu(), "W_inv": W_inv.cpu(),
            "diagnostics": diag, "curve": curve, "N": N,
            "alpha": a.alpha, "cond_alpha": a.cond_alpha, "steps": a.steps,
            "k": a.k, "eval_k": a.eval_k,
            "ctx_per_step": a.ctx_per_step, "act_per_step": a.act_per_step},
           out / "assignment.pt")
(out / "curve.json").write_text(json.dumps(curve, indent=2))
print("saved:", out / "assignment.pt", flush=True)
print("CTX_ASSIGN_DONE", flush=True)
