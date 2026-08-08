#!/usr/bin/env python
"""Complete the 64/128/256/512 spatial-latent CIFAR compression sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torchvision.utils import save_image

from gga.ae import AutoEncoder
from gga.data import cifar_loaders
from train_spatial_ae_comparison import train_model


def load_model(path, latent_dim, device):
    state = torch.load(path, map_location=device)
    model = AutoEncoder(latent_dim, ch=state.get("channels", 64),
                        architecture="spatial").to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def add_derived_metrics(row):
    mse = row["final_test_mse"]
    row["compression_ratio"] = 3072 / row["latent_dim"]
    row["psnr_db"] = 10 * math.log10(4 / mse)  # data range is [-1, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", type=Path,
                        default=Path("results/cifar10_spatial_compression_sweep"))
    parser.add_argument("--existing-results", type=Path,
                        default=Path("results/cifar10_spatial_compression_sweep/ae_spatial_comparison.json"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    train_loader, _, test_loader, _ = cifar_loaders(
        args.data, args.batch, n_particles=1
    )
    previous = json.loads(args.existing_results.read_text())
    results = {
        "config": {
            "epochs": args.epochs,
            "batch": args.batch,
            "hidden": args.hidden,
            "lr": args.lr,
            "seed": args.seed,
            "optimizer": "Adam",
            "scheduler": None,
            "loss": "MSE",
            "input_values": 3072,
        },
        "models": {},
    }
    models = {}

    for latent_dim in (64, 512):
        row = previous["models"][str(latent_dim)].copy()
        add_derived_metrics(row)
        results["models"][str(latent_dim)] = row
        models[latent_dim] = load_model(Path(row["checkpoint"]), latent_dim, device)

    for latent_dim in (128, 256):
        model, row = train_model(
            latent_dim, train_loader, test_loader, args, device
        )
        checkpoint = args.out / f"ae_spatial_dim{latent_dim}_{args.epochs}ep.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "latent_dim": latent_dim,
                "latent_shape": row["latent_shape"],
                "channels": args.hidden,
                "architecture": "spatial",
                "epochs": args.epochs,
                "seed": args.seed,
            },
            checkpoint,
        )
        row["checkpoint"] = str(checkpoint)
        add_derived_metrics(row)
        results["models"][str(latent_dim)] = row
        models[latent_dim] = model
        (args.out / "ae_spatial_compression_sweep.json").write_text(
            json.dumps(results, indent=2)
        )

    originals, _ = next(iter(test_loader))
    originals = originals[:8].to(device)
    rows = [originals]
    with torch.no_grad():
        for latent_dim in (64, 128, 256, 512):
            rows.append(models[latent_dim](originals))
    grid = (torch.cat(rows).clamp(-1, 1) + 1) / 2
    image_path = args.out / "ae_spatial_compression_sweep.png"
    save_image(grid.cpu(), image_path, nrow=8, padding=3)
    results["comparison_image"] = str(image_path)
    results["comparison_rows"] = ["original", "spatial64", "spatial128",
                                  "spatial256", "spatial512"]
    (args.out / "ae_spatial_compression_sweep.json").write_text(
        json.dumps(results, indent=2)
    )
    print(json.dumps({
        dim: {
            "compression_ratio": results["models"][dim]["compression_ratio"],
            "test_mse": results["models"][dim]["final_test_mse"],
            "psnr_db": results["models"][dim]["psnr_db"],
        }
        for dim in ("64", "128", "256", "512")
    }, indent=2))
    print(f"saved image: {image_path}")


if __name__ == "__main__":
    main()
