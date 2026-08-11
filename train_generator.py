#!/usr/bin/env python
"""Direct-to-pixel DINO generator, retrained on the new (20000-step)
assignment, with LPIPS perceptual loss added alongside pixel MSE. Motivated
by the earlier finding that held-out pixel MSE alone doesn't track visual
quality well for this generator (epoch 200 looked as good or better than
epoch 10 despite much worse MSE) -- LPIPS should track perceptual quality
more directly, and give us a second, more meaningful axis to pick a
checkpoint from."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from gga.ae import ResidualDecoder as ConvDecoder

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def collect_targets(data_dir, n, image_size, particle_order=False):
    """particle_order=False: natural dataset order (matches extract_dino_features.py /
    extract_lejepa_features.py, which iterate ds[i] directly).
    particle_order=True: gga.celeba_data.celeba_loaders' permuted particle subset
    (matches run_celeba_gaussianize.py, used for the from-scratch AE's assignment.pt)
    -- required whenever the assignment being trained on came from that path, or z
    and image silently mismatch."""
    os.environ.setdefault("HF_HOME", data_dir)
    if particle_order:
        from gga.celeba_data import celeba_loaders
        _, enc_loader, _, _ = celeba_loaders(data_dir, 256, n_particles=n, image_size=image_size)
        imgs = []
        for x, _ in enc_loader:
            imgs.append(x)
        return torch.cat(imgs, 0)
    from datasets import load_dataset
    hf = load_dataset("flwrlabs/celeba", "img_align+identity+attr")
    ds = hf["train"]
    tf = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    targets = torch.empty(n, 3, image_size, image_size)
    for i in range(n):
        targets[i] = tf(ds[i]["image"].convert("RGB"))
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", type=Path,
                    default=Path("results_celeba_dino/full_pipeline_dim384_moresteps/assignment.pt"))
    ap.add_argument("--lpips-weight", type=float, default=0.5)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--data", default="/data/hf_cache")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--particle-order", action="store_true",
                    help="use gga.celeba_data's permuted particle order for targets "
                         "instead of natural dataset order -- needed for assignments "
                         "built via run_celeba_gaussianize.py (our own AE)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def log(*a):
        print(*a, flush=True)

    assign = torch.load(args.assignment, map_location=device, weights_only=False)
    z = assign["z"].to(device)
    dim = z.shape[1]
    N = z.shape[0]
    log(f"loaded assignment: N={N} dim={dim} from {args.assignment}")

    log("collecting pixel targets (regenerated from dataset, same order as embeddings) ...")
    targets = collect_targets(args.data, N, args.image_size, particle_order=args.particle_order).to(device)
    log(f"targets: {tuple(targets.shape)}")

    log("loading LPIPS (VGG) ...")
    perceptual = lpips.LPIPS(net="vgg").to(device).eval()
    for p in perceptual.parameters():
        p.requires_grad_(False)

    perm = torch.randperm(N)
    n_val = int(args.val_frac * N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = ConvDecoder(dim, ch=args.ch, image_size=args.image_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"generator params: {n_params:,}  lpips_weight={args.lpips_weight}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = (tr_idx.numel() + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    def run(idx, train):
        model.train(train)
        order = idx[torch.randperm(idx.numel())] if train else idx
        total_mse, total_lpips, n = 0.0, 0.0, 0
        for i in range(0, order.numel(), args.batch):
            b = order[i:i + args.batch]
            with torch.set_grad_enabled(train):
                pred = model(z[b])
                mse = F.mse_loss(pred, targets[b])
                perc = perceptual(pred.clamp(-1, 1), targets[b]).mean()
                loss = mse + args.lpips_weight * perc
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
                    sched.step()
            total_mse += mse.item() * b.numel()
            total_lpips += perc.item() * b.numel()
            n += b.numel()
        return total_mse / n, total_lpips / n

    curve = {"train_epoch": [], "train_mse": [], "train_lpips": [],
             "val_epoch": [], "val_mse": [], "val_lpips": []}
    for ep in range(args.epochs):
        tr_mse, tr_lpips = run(tr_idx, True)
        curve["train_epoch"].append(ep + 1)
        curve["train_mse"].append(tr_mse)
        curve["train_lpips"].append(tr_lpips)
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            vl_mse, vl_lpips = run(val_idx, False)
            curve["val_epoch"].append(ep + 1)
            curve["val_mse"].append(vl_mse)
            curve["val_lpips"].append(vl_lpips)
            log(f"epoch {ep+1}/{args.epochs}  train_mse={tr_mse:.5f} train_lpips={tr_lpips:.5f}  "
                f"val_mse={vl_mse:.5f} val_lpips={vl_lpips:.5f}")
            torch.save({
                "model_state_dict": model.state_dict(), "dim": dim, "ch": args.ch,
                "image_size": args.image_size, "epoch": ep + 1,
                "val_mse": vl_mse, "val_lpips": vl_lpips,
            }, ckpt_dir / f"generator_ep{ep+1}.pt")
        else:
            log(f"epoch {ep+1}/{args.epochs}  train_mse={tr_mse:.5f} train_lpips={tr_lpips:.5f}")

    best_i = min(range(len(curve["val_lpips"])), key=lambda i: curve["val_lpips"][i])
    best_epoch_lpips = curve["val_epoch"][best_i]
    best_i_mse = min(range(len(curve["val_mse"])), key=lambda i: curve["val_mse"][i])
    best_epoch_mse = curve["val_epoch"][best_i_mse]
    log(f"best val LPIPS at epoch {best_epoch_lpips} ({curve['val_lpips'][best_i]:.5f})")
    log(f"best val MSE at epoch {best_epoch_mse} ({curve['val_mse'][best_i_mse]:.5f})")
    (args.out / "generator_train_curve.json").write_text(json.dumps(curve, indent=2))

    for tag, ep in [("lpips", best_epoch_lpips), ("mse", best_epoch_mse)]:
        ckpt = torch.load(ckpt_dir / f"generator_ep{ep}.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        torch.manual_seed(0)
        with torch.no_grad():
            zs = torch.randn(64, dim, device=device)
            imgs = model(zs)
        save_image((imgs.clamp(-1, 1) + 1) / 2, args.out / f"samples_best_{tag}_ep{ep}.png", nrow=8)
        log(f"saved samples: {args.out / f'samples_best_{tag}_ep{ep}.png'}")


if __name__ == "__main__":
    main()
