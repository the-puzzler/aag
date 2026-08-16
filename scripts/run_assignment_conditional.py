#!/usr/bin/env python
"""4k-global-step assignment with the conditional budget CONCENTRATED so
independence plateaus inside that budget.

Rationale: transport displacement (which degrades z->image locality and hurts
the generator) is driven by GLOBAL steps -- in the 60k run global contributed
~9.8B point-updates vs the conditional steps' ~46M. So capping global steps at
4k (where the transport objective already reaches its noise floor, 0.00428) while
running many conditional steps per global step should give independence at
flagship-level displacement.

Budget maths: reaching ratio ~1.1 previously took ~45k conditional firings.
Packed into 4000 global steps that is ~12 per step (default).
Logs displacement, the independence ratio against its random-subset floor, the
raw transport objective, and proj_over_gauss at every checkpoint."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from aag.diagnostics import assignment_diagnostics
from aag.datasets import collect_conditions, spec
from aag.gaussianize import (whiten, greedy_rank_transport_step, offset_slab_cleanup_step, group_rank_transport_step,
                             radial_chi_calibration, conditional_rank_transport_step,
                             _gaussian_quantiles)
from aag.diagnostics import conditional_group_w2, random_subset_w2, group_w2

ap = argparse.ArgumentParser()
ap.add_argument("--h-source", default="results_celeba/full_pipeline_lpips_ae/assignment.pt")
ap.add_argument("--data-root", default="data")
ap.add_argument("--N", type=int, default=50000)
ap.add_argument("--dataset", choices=["celeba","cifar10","doom","doom_frames"], default="celeba")
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--cond-per-step", type=int, default=12)
ap.add_argument("--k", type=int, default=1024)
ap.add_argument("--cond-alpha", type=float, default=1.0,
                help="step size of each conditional rank transport. 1.0 moves points\n                      fully to their target quantiles; lower damps the movement, which\n                      cuts accumulated transport displacement (displacement, not\n                      Gaussianity, is what degrades z->image learnability).")
ap.add_argument("--eval-every", type=int, default=200)
ap.add_argument("--out", type=Path, default=Path("results_celeba_conditional/assign_4k_dense"))
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
args.out.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)

ap_h = args.h_source
h = torch.load(ap_h, map_location=dev, weights_only=False)["h"].to(dev)
DISCRETE = spec(args.dataset)["discrete"]
if args.dataset == "celeba":
    cond = torch.load("results_celeba/attrs.pt", map_location=dev, weights_only=False)["attrs"].to(dev)
    groups = None
else:
    _, groups = collect_conditions(args.dataset, args.data_root, 256, args.N, device=dev)
    cond = None
z, mean, W, W_inv = whiten(h)
z = z.contiguous(); z0 = z.clone()
d = z.shape[1]
gen = torch.Generator(device=dev).manual_seed(args.seed)

log(f"assignment: {args.steps} global steps x {args.cond_per_step} conditional/step "
    f"= {args.steps*args.cond_per_step:,} conditional firings, k={args.k}, cond_alpha={args.cond_alpha}")
log(f"(target: independence RATIO -> 1.0; going BELOW 1.0 means over-transport --\n the group becomes more Gaussian than a random subset of equal size)")

curve = {"step": [], "ratio": [], "cond_group_w2": [], "random_floor_w2": [],
         "displacement": [], "proj_over_gauss": [], "conv_step": [], "conv_score": []}
for step in range(args.steps):
    s = greedy_rank_transport_step(z, search_subset=2048, n_dirs=64, alpha=1.0, gen=gen)
    curve["conv_step"].append(step + 1); curve["conv_score"].append(s)
    if step % 2 == 0:
        offset_slab_cleanup_step(z, search_subset=2048, n_slabs=32, eps=0.5, alpha=1.0, gen=gen)
    for _ in range(args.cond_per_step):
        if DISCRETE:
            group_rank_transport_step(z, groups, n_dirs=64, alpha=args.cond_alpha, gen=gen, max_group=args.k)
        else:
            conditional_rank_transport_step(z, cond, k=args.k, n_dirs=64, alpha=args.cond_alpha, gen=gen)
    if (step + 1) % 20 == 0:
        radial_chi_calibration(z, d=d, alpha_r=1.0)
    if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
        if DISCRETE:
            cw, kk = group_w2(z, groups, n_eval=20, gen=gen, max_group=args.k)
        else:
            cw, kk = conditional_group_w2(z, cond, k=args.k, n_eval=20, gen=gen), args.k
        rf = random_subset_w2(z, k=kk, n_eval=20, gen=gen)
        disp = (z - z0).norm(dim=1).mean().item()
        diag = assignment_diagnostics(z, d=d, seed=args.seed)
        curve["step"].append(step + 1); curve["ratio"].append(cw / rf)
        curve["cond_group_w2"].append(cw); curve["random_floor_w2"].append(rf)
        curve["displacement"].append(disp); curve["proj_over_gauss"].append(diag["proj_over_gauss"])
        log(f"  step {step+1:5d}/{args.steps}  RATIO={cw/rf:5.2f}x  disp={disp:.3f}  "
            f"proj/gauss={diag['proj_over_gauss']:.3f}  obj={s:.5f}")
        torch.save({"z": z.cpu(), "h": h.cpu(), "mean": mean.cpu(), "W_inv": W_inv.cpu(),
                    "curve": curve}, args.out / "assignment.pt")
        (args.out / "curve.json").write_text(json.dumps(curve, indent=2))
log("ALL DONE")
