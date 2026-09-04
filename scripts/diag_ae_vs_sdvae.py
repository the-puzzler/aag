#!/usr/bin/env python
"""How far is our 256-number AE from a heavily-trained open-source one?

The comparison is only meaningful at EQUAL RATE, and it happens to be exactly
equal: our DC-AE takes 64x64x3 = 12,288 values to 256, a 48x compression. Stable
Diffusion's VAE is f8c4, so on a 64x64 input it produces 8x8x4 = 256 values --
the same 256 numbers per frame. Nothing has to be rescaled to make the budgets
match.

The caveat to keep with the result: SD's VAE was trained at 512x512 and is being
run here at 64x64, well below its comfort zone, so this understates it. It is
still the right comparison for the question actually being asked -- at OUR rate
and OUR resolution, how much reconstruction quality is a well-trained encoder
worth?
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
    ap.add_argument("--sd-vae", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--cache", default="/opt/dlami/nvme/vpt_full")
    ap.add_argument("--n", type=int, default=2048)
    args = ap.parse_args()

    dev = "cuda"
    A = torch.load(args.assignment, map_location="cpu", weights_only=False)
    ci, fi = A["chunk"].numpy(), A["frame"].numpy()
    segs = open_segments(args.cache)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ci), args.n, replace=False)

    ac = torch.load(args.ae, map_location=dev, weights_only=False)
    ours = AutoEncoder(ac["latent_dim"], ch=ac["channels"],
                       architecture=ac["architecture"],
                       image_size=ac["image_size"],
                       grid=ac.get("grid", 4)).to(dev).eval()
    sd = ac["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    ours.load_state_dict(sd)
    print(f"ours: dcae, latent 256, trained {ac['epochs']} epochs, "
          f"gan_weight {ac.get('gan_weight')}", flush=True)

    from diffusers import AutoencoderKL
    sdv = AutoencoderKL.from_pretrained(args.sd_vae).to(dev).eval()
    nz = sum(p.numel() for p in sdv.parameters())
    print(f"sd-vae: {args.sd_vae}, f8c4 -> 8x8x4 = 256 numbers at 64x64, "
          f"{nz/1e6:.1f}M params", flush=True)

    per = lpips.LPIPS(net="vgg").to(dev).eval()
    acc = {k: 0.0 for k in ("ours_mse", "ours_lpips", "sd_mse", "sd_lpips")}
    with torch.no_grad():
        for i in range(0, len(sel), 128):
            b = sel[i:i + 128]
            x = np.stack([np.asarray(segs[int(ci[p])][int(fi[p])]) for p in b])
            x = (torch.from_numpy(x).permute(0, 3, 1, 2).float()
                 .div_(127.5).sub_(1.0).to(dev))
            r1 = ours.dec(ours.enc(x)).clamp(-1, 1)
            lat = sdv.encode(x).latent_dist.mode()
            r2 = sdv.decode(lat).sample.clamp(-1, 1)
            if i == 0:
                print(f"  sd latent tensor {tuple(lat.shape)} = "
                      f"{lat[0].numel()} numbers per frame", flush=True)
            acc["ours_mse"] += float(F.mse_loss(r1, x)) * len(b)
            acc["ours_lpips"] += float(per(r1, x).mean()) * len(b)
            acc["sd_mse"] += float(F.mse_loss(r2, x)) * len(b)
            acc["sd_lpips"] += float(per(r2, x).mean()) * len(b)
    for k in acc:
        acc[k] /= len(sel)

    def psnr(m):
        # pixels are in [-1,1], so peak-to-peak is 2
        return 10.0 * np.log10(4.0 / m)

    print(f"\nreconstruction on {len(sel):,} VPT target frames, 256 numbers each:")
    print(f"  {'model':22s} {'MSE':>9s} {'PSNR dB':>8s} {'LPIPS':>8s}")
    print(f"  {'ours (dcae, 4 ep)':22s} {acc['ours_mse']:9.5f} "
          f"{psnr(acc['ours_mse']):8.2f} {acc['ours_lpips']:8.5f}")
    print(f"  {'SD-VAE f8c4':22s} {acc['sd_mse']:9.5f} "
          f"{psnr(acc['sd_mse']):8.2f} {acc['sd_lpips']:8.5f}")
    print(f"\n  MSE   ours/sd = {acc['ours_mse']/acc['sd_mse']:.2f}x")
    print(f"  LPIPS ours/sd = {acc['ours_lpips']/acc['sd_lpips']:.2f}x")
    print("\nCaveat: SD's VAE was trained at 512x512 and is run here at 64x64,")
    print("below its comfort zone, so this understates it.")


if __name__ == "__main__":
    main()
