#!/usr/bin/env python
"""Global then conditional assignment for the Doom world model.

Condition is c_t = (h_{t-3..t-1}, a_t): continuous context plus an exact 18-way
action. The conditional step therefore filters exactly by action and takes a
continuous k-NN inside that group (action_knn_transport_step), rather than
CelebA's Hamming k-NN or CIFAR's pure categorical groups.

Reported metric is the independence RATIO (action-knn W2 / random-subset W2 at
matched k): -> 1.0 means p(z | c) is indistinguishable from p(z), which is what
makes sampling z independently of c valid at generation time.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch

from aag.gaussianize import (whiten, greedy_rank_transport_step,
                             action_knn_transport_step)
from aag.diagnostics import (knn_preservation, assignment_diagnostics, action_knn_w2,
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
    ap.add_argument("--k", type=int, default=4096, help="transport neighbourhood size")
    ap.add_argument("--eval-k", type=int, default=4096,
                     help="diagnostic neighbourhood size; hold FIXED when sweeping --k, "
                          "otherwise the measuring stick changes with the treatment")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = torch.load(args.particles, map_location=dev, weights_only=False)
    h, cond, act = P["h_target"].to(dev), P["h_context"].to(dev), P["action"].to(dev)
    N, d = h.shape
    print(f"{N:,} particles, dim={d}, cond_dim={cond.shape[1]}, "
          f"{P['episode'].unique().numel():,} episodes", flush=True)
    print(f"intrinsic dim (TwoNN) = {intrinsic_dimension_twonn(h):.2f}", flush=True)

    z, mean, W, W_inv = whiten(h)
    z0 = z.clone()
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    curve = {"step": [], "ratio": [], "cond_w2": [], "floor_w2": [],
             "displacement": [], "proj_over_gauss": [], "knn": []}

    for s in range(1, args.steps + 1):
        greedy_rank_transport_step(z, search_subset=args.search_subset,
                                   n_dirs=args.n_dirs, alpha=args.alpha, gen=gen)
        for _ in range(args.cond_per_step):
            action_knn_transport_step(z, cond, act, k=args.k, n_dirs=args.n_dirs,
                                      alpha=args.cond_alpha, gen=gen)
        if s % args.eval_every == 0 or s == args.steps:
            cw = action_knn_w2(z, cond, act, k=args.eval_k, gen=gen)
            fw = random_subset_w2(z, k=args.eval_k, gen=gen)
            diag = assignment_diagnostics(z, d=d)
            disp = float((z - z0).norm(dim=1).mean())
            knn = knn_preservation(z0, z, k=10, gen=gen)
            curve["step"].append(s); curve["ratio"].append(cw / max(fw, 1e-12))
            curve["cond_w2"].append(cw); curve["floor_w2"].append(fw)
            curve["displacement"].append(disp)
            curve["proj_over_gauss"].append(diag["proj_over_gauss"])
            curve["knn"].append(knn)
            print(f"  [{s}/{args.steps}] ratio={cw/max(fw,1e-12):.2f} "
                  f"cond_w2={cw:.5f} floor={fw:.5f} disp={disp:.3f} "
                  f"proj_over_gauss={diag['proj_over_gauss']:.3f}", flush=True)

    diag = assignment_diagnostics(z, d=d)
    print("assignment diagnostics:", diag, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"z": z.cpu(), "h": h.cpu(), "cond": cond.cpu(), "action": act.cpu(),
                "chunk": P["chunk"], "frame": P["frame"], "episode": P["episode"],
                "mean": mean.cpu(), "W": W.cpu(), "W_inv": W_inv.cpu(),
                "diagnostics": diag, "curve": curve, "N": N,
                "alpha": args.alpha, "cond_alpha": args.cond_alpha, "steps": args.steps,
                "k": args.k, "eval_k": args.eval_k},
               args.out / "assignment.pt")
    (args.out / "curve.json").write_text(json.dumps(curve, indent=2))
    print("saved:", args.out / "assignment.pt", flush=True)


if __name__ == "__main__":
    main()
