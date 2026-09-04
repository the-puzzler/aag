#!/usr/bin/env python
"""Is the generator already at the autoencoder's reconstruction floor?

The question matters structurally, not just as bookkeeping. In AAG, z is the ONLY
channel carrying target-specific information, and z is a transported version of
h_target = AE_enc(frame). So detail the AE does not encode has no representation
in z at all: the assignment never had a handle on it, and it was therefore never
gaussianised. At generation time that detail is unspecified, and an MSE + LPIPS
objective resolves unspecified detail to its conditional mean -- which is blur,
concentrated exactly on fine texture.

If that is right, the AE's reconstruction error is a CEILING on the generator
rather than a term added to it, and the generator should already be sitting close
to it. This measures both on the same frames in the same units so the ratio is
meaningful.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F

from aag.ae import AutoEncoder
from aag.datasets import open_segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", type=Path,
                    default=Path("/data/aag_results/results_vpt/assign_12d_lag1/"
                                 "assignment.pt"))
    ap.add_argument("--ae", type=Path,
                    default=Path("/data/aag_results/results_vpt/"
                                 "ae_dcae_ch192_dim256_cont/checkpoints/"
                                 "ae_doom_frames_dcae_lpips_ch192_dim256_ep4.pt"))
    ap.add_argument("--cache", default="/opt/dlami/nvme/vpt_full")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--gen-mse", type=float, default=0.00809,
                    help="generator single-step MSE to compare against")
    ap.add_argument("--gen-lpips", type=float, default=0.09888)
    args = ap.parse_args()

    dev = "cuda"
    ac = torch.load(args.ae, map_location=dev, weights_only=False)
    ae = AutoEncoder(ac["latent_dim"], ch=ac["channels"],
                     architecture=ac["architecture"], image_size=ac["image_size"],
                     grid=ac.get("grid", 4)).to(dev).eval()
    sd = ac["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    ae.load_state_dict(sd)

    A = torch.load(args.assignment, map_location="cpu", weights_only=False)
    ci, fi = A["chunk"].numpy(), A["frame"].numpy()
    segs = open_segments(args.cache)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ci), args.n, replace=False)

    per = lpips.LPIPS(net="vgg").to(dev).eval()
    m_tot = l_tot = 0.0
    with torch.no_grad():
        for i in range(0, len(sel), 256):
            b = sel[i:i + 256]
            x = np.stack([np.asarray(segs[int(ci[p])][int(fi[p])]) for p in b])
            x = (torch.from_numpy(x).permute(0, 3, 1, 2).float()
                 .div_(127.5).sub_(1.0).to(dev))
            r = ae.dec(ae.enc(x)).clamp(-1, 1)
            m_tot += float(F.mse_loss(r, x)) * len(b)
            l_tot += float(per(r, x).mean()) * len(b)
    mse, lpv = m_tot / len(sel), l_tot / len(sel)

    print(f"AE reconstruction on the particles' own target frames (n={len(sel):,}):")
    print(f"   AE recon MSE    {mse:.5f}")
    print(f"   AE recon LPIPS  {lpv:.5f}\n")
    print(f"generator single-step, same units:")
    print(f"   MSE    {args.gen_mse:.5f}   -> {args.gen_mse/mse:.2f}x the AE floor")
    print(f"   LPIPS  {args.gen_lpips:.5f}   -> {args.gen_lpips/lpv:.2f}x the AE floor")
    print()
    print("A ratio near 1 means the generator has extracted essentially all the")
    print("information the AE latent carries, and no amount of generator capacity,")
    print("rollout or adversary can recover detail the AE never encoded. A ratio")
    print("well above 1 means there is headroom left in the generator itself.")


if __name__ == "__main__":
    main()
