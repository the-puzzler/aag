#!/usr/bin/env python
"""Global + first-frame-conditional assignment for Doom video generation.

Condition is a plain continuous vector (the per-frame AE embedding of frame 0),
so the conditional step is continuous_knn_transport_step and the diagnostic is
continuous_knn_w2 / random_subset_w2 at matched k.

On choosing k: see docs/METHOD.md §5. Keep --eval-k FIXED when sweeping --k.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch

from aag.gaussianize import (whiten, greedy_rank_transport_step, continuous_knn_transport_step,
                             offset_slab_cleanup_step, radial_chi_calibration)
from aag.diagnostics import (r_dispersion, r_cond, knn_preservation, assignment_diagnostics, continuous_knn_w2,
                             random_subset_w2, intrinsic_dimension_twonn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--search-subset", type=int, default=2048)
    ap.add_argument("--n-dirs", type=int, default=64)
    ap.add_argument("--cond-per-step", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=1.0,
                     help="GLOBAL transport damping. A 2D study shows lower alpha buys "
                          "locality (kNN) at identical Gaussianity, for more steps -- and "
                          "steps are the cheap resource. Was hardcoded to 1.0 before.")
    ap.add_argument("--cond-alpha", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--eval-k", type=int, default=4096)
    ap.add_argument("--no-rotate", action="store_true",
                     help="scale-only normalisation: preserves the AE latent's grid topology "
                          "in z (PCA rotation destroys it by mixing all coordinates)")
    ap.add_argument("--cond-metric", choices=["cosine", "l2"], default="cosine",
                     help="distance used for the k-NN condition neighbourhood")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--cleanup", action="store_true",
                     help="interleave the slab-cleanup (every 2 steps) and radial chi "
                          "calibration (every 20) used by the CelebA/CIFAR pipeline")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    P = torch.load(args.particles, map_location=dev, weights_only=False)
    h, cond = P["h_target"].to(dev), P["cond_first_frame"].to(dev)
    N, d = h.shape
    print(f"{N:,} clips, dim={d}, cond_dim={cond.shape[1]} (first frame), "
          f"{P['episode'].unique().numel():,} episodes", flush=True)
    print(f"intrinsic dim (TwoNN) = {intrinsic_dimension_twonn(h):.2f}", flush=True)

    z, mean, W, W_inv = whiten(h, rotate=not args.no_rotate)
    print(f"normalisation: {'scale-only (grid topology preserved)' if args.no_rotate else 'PCA whitening'}",
          flush=True)
    z0 = z.clone()
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    curve = {"step": [], "ratio": [], "cond_w2": [], "floor_w2": [],
             "displacement": [], "proj_over_gauss": [], "knn": [], "R": [], "Rc": [],
             # per-step objective, so plot_assignment.py can show it against the
             # N(0,I) noise floor -- the panel that says whether we transported enough
             "conv_step": [], "conv_score": []}

    for s in range(1, args.steps + 1):
        sc = greedy_rank_transport_step(z, search_subset=args.search_subset,
                                        n_dirs=args.n_dirs, alpha=args.alpha, gen=gen)
        curve["conv_step"].append(s); curve["conv_score"].append(sc)
        if args.cleanup:
            if s % 2 == 0:
                offset_slab_cleanup_step(z, search_subset=args.search_subset,
                                         n_slabs=32, eps=0.5, alpha=1.0, gen=gen)
            if s % 20 == 0:
                radial_chi_calibration(z, d=d, alpha_r=1.0)
        for _ in range(args.cond_per_step):
            continuous_knn_transport_step(z, cond, k=args.k, n_dirs=args.n_dirs,
                                          alpha=args.cond_alpha, gen=gen,
                                          metric=args.cond_metric)
        if s % args.eval_every == 0 or s == args.steps:
            cw = continuous_knn_w2(z, cond, k=args.eval_k, gen=gen, metric=args.cond_metric)
            fw = random_subset_w2(z, k=args.eval_k, gen=gen)
            diag = assignment_diagnostics(z, d=d)
            disp = float((z - z0).norm(dim=1).mean())
            knn = knn_preservation(z0, z, k=10, gen=gen)
            R = r_dispersion(z, h, gen=gen); Rc = r_cond(z, h, cond, gen=gen)
            curve["step"].append(s); curve["ratio"].append(cw / max(fw, 1e-12))
            curve["cond_w2"].append(cw); curve["floor_w2"].append(fw)
            curve["displacement"].append(disp)
            curve["proj_over_gauss"].append(diag["proj_over_gauss"])
            curve["knn"].append(knn); curve["R"].append(R); curve["Rc"].append(Rc)
            print(f"  [{s}/{args.steps}] ratio={cw/max(fw,1e-12):.2f} cond_w2={cw:.5f} "
                  f"floor={fw:.5f} disp={disp:.3f} proj/gauss={diag['proj_over_gauss']:.3f}",
                  flush=True)

    diag = assignment_diagnostics(z, d=d)
    print("assignment diagnostics:", diag, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"z": z.cpu(), "h": h.cpu(), "cond": cond.cpu(),
                "episode": P["episode"], "action_seqs": P["action_seqs"],
                "mean": mean.cpu(), "W": W.cpu(), "W_inv": W_inv.cpu(),
                "diagnostics": diag, "curve": curve, "N": N,
                "alpha": args.alpha, "cond_alpha": args.cond_alpha, "k": args.k, "eval_k": args.eval_k,
                "cond_metric": args.cond_metric, "rotate": not args.no_rotate,
                "steps": args.steps}, args.out / "assignment.pt")
    (args.out / "curve.json").write_text(json.dumps(curve, indent=2))
    print("saved:", args.out / "assignment.pt", flush=True)


if __name__ == "__main__":
    main()
