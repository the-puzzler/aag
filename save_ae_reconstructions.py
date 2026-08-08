#!/usr/bin/env python
"""Train a CIFAR-10 autoencoder and save original/reconstruction pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from gga.ae import AutoEncoder, reconstruction_loss, train_autoencoder
from gga.data import cifar_loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--arch", choices=["plain", "residual", "spatial"],
                        default="plain")
    parser.add_argument("--topk-percent", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", type=Path, default=Path("results/cifar10_full"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-plain-checkpoint", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    train_loader, _, test_loader, _ = cifar_loaders(
        args.data, args.batch, n_particles=1
    )
    model = AutoEncoder(args.dim, ch=64, architecture=args.arch).to(device)
    train_autoencoder(
        model,
        train_loader,
        epochs=args.epochs,
        lr=2e-3,
        device=device,
        topk_percent=args.topk_percent,
        log=lambda message: print(message, flush=True),
    )

    def test_metrics(ae):
        ae.eval()
        total = 0.0
        hard_total = 0.0
        count = 0
        images_seen = 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                prediction = ae(images)
                total += F.mse_loss(prediction, images, reduction="sum").item()
                hard_total += reconstruction_loss(
                    prediction, images, topk_percent=args.topk_percent
                ).item() * images.size(0)
                count += images.numel()
                images_seen += images.size(0)
        return total / count, hard_total / images_seen

    model_test_mse, model_hard_mse = test_metrics(model)
    print(f"{args.arch} test_mse={model_test_mse:.6f} "
          f"top{args.topk_percent:g}%_mse={model_hard_mse:.6f}")

    originals, _ = next(iter(test_loader))
    originals = originals[:8].to(device)
    with torch.no_grad():
        reconstructions = model(originals)

    comparison_rows = [originals]
    loss_label = "mse" if args.topk_percent == 100 else f"top{args.topk_percent:g}pct"
    image_stem = (
        f"ae_{args.arch}_{loss_label}_{args.epochs}ep_"
        f"dim{args.dim}_reconstructions"
    )
    if args.compare_plain_checkpoint:
        plain = AutoEncoder(args.dim, ch=64, architecture="plain").to(device)
        state = torch.load(args.compare_plain_checkpoint, map_location=device)
        plain.load_state_dict(state["model_state_dict"])
        baseline_epochs = state.get("epochs", "baseline")
        plain_test_mse, plain_hard_mse = test_metrics(plain)
        print(f"plain_baseline test_mse={plain_test_mse:.6f} "
              f"top{args.topk_percent:g}%_mse={plain_hard_mse:.6f}")
        with torch.no_grad():
            plain_reconstructions = plain(originals)
        comparison_rows.append(plain_reconstructions)
        image_stem = (
            f"ae_plain_mse_{baseline_epochs}ep_vs_{args.arch}_{loss_label}_"
            f"{args.epochs}ep_dim{args.dim}_reconstructions"
        )
    comparison_rows.append(reconstructions)

    # First row: originals. Second row: corresponding reconstructions.
    comparison = torch.cat(comparison_rows, dim=0)
    comparison = (comparison.clamp(-1, 1) + 1) / 2
    image_path = args.out / f"{image_stem}.png"
    save_image(comparison.cpu(), image_path, nrow=8, padding=3)

    checkpoint_path = args.out / (
        f"ae_{args.arch}_{loss_label}_{args.epochs}ep_dim{args.dim}.pt"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_dim": args.dim,
            "channels": 64,
            "architecture": args.arch,
            "topk_percent": args.topk_percent,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        checkpoint_path,
    )
    print(f"saved image: {image_path}")
    print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
