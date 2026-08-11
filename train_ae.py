#!/usr/bin/env python
"""Same as probe_celeba_ae_train_curve.py but with LPIPS perceptual loss
added alongside pixel MSE for the AE's own reconstruction objective --
testing whether the LPIPS win we found for the DINO generator also helps
when training our own AE (encoder+decoder jointly) from scratch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F

from gga.ae import AutoEncoder
from gga.celeba_data import celeba_loaders


def test_metrics(ae, loader, device, perceptual):
    ae.eval()
    total_mse, total_lpips, count = 0.0, 0.0, 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            pred = ae(images)
            total_mse += F.mse_loss(pred, images, reduction="sum").item()
            total_lpips += perceptual(pred.clamp(-1, 1), images).sum().item()
            count += images.shape[0]
    ae.train()
    return total_mse / (count * images[0].numel()), total_lpips / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["residual", "spatial", "hybrid"], default="residual")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--lpips-weight", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--data", default="/data/hf_cache")
    ap.add_argument("--out", type=Path, default=Path("results_celeba"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    train_loader, _, test_loader, n_avail = celeba_loaders(
        args.data, args.batch, n_particles=1, image_size=args.image_size,
    )
    print(f"AE trains on {n_avail} CelebA images at {args.image_size}x{args.image_size}, "
          f"arch={args.arch}, lpips_weight={args.lpips_weight}", flush=True)

    ae = AutoEncoder(args.dim, ch=args.ch, architecture=args.arch,
                     image_size=args.image_size).to(device)
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"model params: {n_params:,}", flush=True)

    perceptual = lpips.LPIPS(net="vgg").to(device).eval()
    for p in perceptual.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(ae.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    curve = {"train_epoch": [], "train_mse": [], "train_lpips": [],
             "test_epoch": [], "test_mse": [], "test_lpips": [], "arch": args.arch}
    ae.train()
    for ep in range(args.epochs):
        running_mse, running_lpips, n = 0.0, 0.0, 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            xr = ae(x)
            mse = F.mse_loss(xr, x)
            perc = perceptual(xr.clamp(-1, 1), x).mean()
            loss = mse + args.lpips_weight * perc
            loss.backward()
            opt.step()
            sched.step()
            running_mse += mse.item() * x.size(0)
            running_lpips += perc.item() * x.size(0)
            n += x.size(0)
        curve["train_epoch"].append(ep + 1)
        curve["train_mse"].append(running_mse / n)
        curve["train_lpips"].append(running_lpips / n)

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            tm, tl = test_metrics(ae, test_loader, device, perceptual)
            curve["test_epoch"].append(ep + 1)
            curve["test_mse"].append(tm)
            curve["test_lpips"].append(tl)
            print(f"[{args.arch}] epoch {ep+1}/{args.epochs}  train_mse={running_mse/n:.5f} "
                  f"train_lpips={running_lpips/n:.5f}  test_mse={tm:.5f} test_lpips={tl:.5f}",
                  flush=True)
            ckpt_dir = args.out / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"ae_celeba_{args.arch}_lpips_ch{args.ch}_dim{args.dim}_ep{ep+1}.pt"
            torch.save({
                "model_state_dict": ae.state_dict(), "latent_dim": args.dim,
                "channels": args.ch, "architecture": args.arch, "image_size": args.image_size,
                "epochs": ep + 1, "test_mse": tm, "test_lpips": tl, "seed": args.seed,
            }, ckpt_path)
        else:
            print(f"[{args.arch}] epoch {ep+1}/{args.epochs}  train_mse={running_mse/n:.5f} "
                  f"train_lpips={running_lpips/n:.5f}", flush=True)

    best_i = min(range(len(curve["test_lpips"])), key=lambda i: curve["test_lpips"][i])
    best_epoch, best_lpips = curve["test_epoch"][best_i], curve["test_lpips"][best_i]
    print(f"best test LPIPS {best_lpips:.5f} at epoch {best_epoch}", flush=True)

    out_json = args.out / f"ae_train_curve_{args.arch}_lpips_{args.epochs}ep.json"
    out_json.write_text(json.dumps(curve, indent=2))
    print("saved curve:", out_json)


if __name__ == "__main__":
    main()
