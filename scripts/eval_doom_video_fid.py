#!/usr/bin/env python
"""Held-out generation quality for the first-frame-conditioned video generators.

Training MSE measures how well the assigned pairs are FIT, which is not the same
as how well fresh z ~ N(0,I) GENERATES -- a lesson the 2D study taught the hard
way. This evaluates the thing we actually care about:

  condition: first frame of a VAL episode (never seen by AE, assignment or generator)
  latent:    fresh z ~ N(0,I)
  metric:    FID over generated frames vs real val frames

Two generators are compared at a matched checkpoint, differing only in which
assignment they were trained on.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch

from aag.fid import get_activations, activation_stats, calculate_frechet_distance
from aag.ae import AutoEncoder, AdaLNDecoder3d, SpatialCondDecoder3d

DEV = "cuda"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-ae", required=True)
    ap.add_argument("--generators", nargs="+", required=True)
    ap.add_argument("--val-cache", default="/data/doom/cache_val")
    ap.add_argument("--n", type=int, default=1500, help="clips to generate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    segs = np.load(f"{args.val_cache}/segments.npy", mmap_mode="r")
    n_val = len(np.load(f"{args.val_cache}/labels.npy"))
    n = min(args.n, n_val)
    real = torch.from_numpy(np.ascontiguousarray(segs[:n])).to(DEV)      # (n,T,H,W,3)
    real_f = real.permute(0, 1, 4, 2, 3).float().div_(255.)              # (n,T,3,H,W)

    fk = torch.load(args.frame_ae, map_location=DEV, weights_only=False)
    fae = AutoEncoder(fk["latent_dim"], ch=fk["channels"], architecture=fk["architecture"],
                      image_size=fk["image_size"]).to(DEV).eval()
    fae.load_state_dict(fk["model_state_dict"])
    cond = fae.enc(real_f[:, 0] * 2 - 1)                                 # encode FIRST frame
    print(f"{n} held-out val clips, condition dim {cond.shape[1]}", flush=True)

    # reference stats from real val frames (subsample frames to keep it comparable)
    fr_idx = torch.arange(0, real_f.shape[1], 4)
    ref = real_f[:, fr_idx].reshape(-1, 3, 64, 64)
    mu_r, sig_r = activation_stats(get_activations(ref.cpu(), DEV))
    print(f"reference: {ref.shape[0]} real frames", flush=True)

    out = {}
    for gp in args.generators:
        g = torch.load(gp, map_location=DEV, weights_only=False)
        mode = g.get("cond_mode", "adaln")
        if mode == "none":
            from aag.ae import ResidualDecoder3d
            m = ResidualDecoder3d(g["dim_z"], ch=64, image_size=64, frames=16).to(DEV).eval()
        elif mode == "spatial":
            grid = 4
            m = SpatialCondDecoder3d(g["dim_z"], g["cond_dim"] // (grid*grid), ch=64,
                                     image_size=64, frames=16, cond_grid=grid).to(DEV).eval()
        else:
            m = AdaLNDecoder3d(g["dim_z"], g["cond_dim"], ch=64, image_size=64, frames=16).to(DEV).eval()
        m.load_state_dict(g["model_state_dict"])
        torch.manual_seed(args.seed)
        frames = []
        for i in range(0, n, 64):
            c = cond[i:i + 64]
            z = torch.randn(c.shape[0], g["dim_z"], device=DEV)          # FRESH z
            v = (m(z) if mode == "none" else m(z, c)).clamp(-1, 1).add(1).div(2)
            frames.append(v[:, :, fr_idx].permute(0, 2, 1, 3, 4).reshape(-1, 3, 64, 64).cpu())
        acts = get_activations(torch.cat(frames), DEV)
        mu_g, sig_g = activation_stats(acts)
        fid = calculate_frechet_distance(mu_r, sig_r, mu_g, sig_g)
        out[gp] = fid
        print(f"  {Path(gp).parent.name:<30} {mode:<8} ep{g['epoch']:<3} FID = {fid:.3f}", flush=True)
    json.dump(out, open("results_doom/heldout_fid.json", "w"), indent=2)


if __name__ == "__main__":
    main()
