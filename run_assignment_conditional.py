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
from gga.diagnostics import assignment_diagnostics
from gga.gaussianize import (whiten, greedy_rank_transport_step, offset_slab_cleanup_step,
                             radial_chi_calibration, conditional_rank_transport_step,
                             _gaussian_quantiles)
from gga.diagnostics import conditional_group_w2, random_subset_w2

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--cond-per-step", type=int, default=12)
ap.add_argument("--k", type=int, default=1024)
ap.add_argument("--eval-every", type=int, default=200)
ap.add_argument("--out", type=Path, default=Path("results_celeba_conditional/assign_4k_dense"))
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
args.out.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)

h = torch.load("results_celeba/full_pipeline_lpips_ae/assignment.pt",
               map_location=dev, weights_only=False)["h"].to(dev)
cond = torch.load("results_celeba/attrs.pt", map_location=dev, weights_only=False)["attrs"].to(dev)
z, mean, W, W_inv = whiten(h)
z = z.contiguous(); z0 = z.clone()
d = z.shape[1]
gen = torch.Generator(device=dev).manual_seed(args.seed)

log(f"4k-dense assignment: {args.steps} global steps x {args.cond_per_step} conditional/step "
    f"= {args.steps*args.cond_per_step:,} conditional firings, k={args.k}")
log(f"(for reference: 60k run used ~45,000 firings and reached ratio 1.14 at displacement 3.41;")
log(f" flagship 4k used 0 firings at displacement 1.17)")

curve = {"step": [], "ratio": [], "cond_group_w2": [], "random_floor_w2": [],
         "displacement": [], "proj_over_gauss": [], "conv_step": [], "conv_score": []}
for step in range(args.steps):
    s = greedy_rank_transport_step(z, search_subset=2048, n_dirs=64, alpha=1.0, gen=gen)
    curve["conv_step"].append(step + 1); curve["conv_score"].append(s)
    if step % 2 == 0:
        offset_slab_cleanup_step(z, search_subset=2048, n_slabs=32, eps=0.5, alpha=1.0, gen=gen)
    for _ in range(args.cond_per_step):
        conditional_rank_transport_step(z, cond, k=args.k, n_dirs=64, alpha=1.0, gen=gen)
    if (step + 1) % 20 == 0:
        radial_chi_calibration(z, d=d, alpha_r=1.0)
    if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
        cw = conditional_group_w2(z, cond, k=args.k, n_eval=20, gen=gen)
        rf = random_subset_w2(z, k=args.k, n_eval=20, gen=gen)
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
