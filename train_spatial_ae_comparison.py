#!/usr/bin/env python
"""Train 64- and 512-value spatial CIFAR autoencoders and compare reconstructions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from gga.ae import AutoEncoder
from gga.data import cifar_loaders


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    squared_error = 0.0
    elements = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        squared_error += F.mse_loss(model(images), images, reduction="sum").item()
        elements += images.numel()
    return squared_error / elements


def train_model(latent_dim, loader, test_loader, args, device):
    torch.manual_seed(args.seed)
    model = AutoEncoder(latent_dim, ch=args.hidden, architecture="spatial").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            reconstruction = model(images)
            loss = F.mse_loss(reconstruction, images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
            count += images.size(0)
        train_mse = total / count
        test_mse = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "train_mse": train_mse, "test_mse": test_mse})
        print(
            f"[spatial d={latent_dim}] epoch {epoch}/{args.epochs} "
            f"train_mse={train_mse:.6f} test_mse={test_mse:.6f}",
            flush=True,
        )

    return model, {
        "latent_dim": latent_dim,
        "latent_shape": [latent_dim // 16, 4, 4],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seconds": time.time() - started,
        "history": history,
        "final_test_mse": history[-1]["test_mse"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", type=Path,
                        default=Path("results/cifar10_spatial_compression_sweep"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("results/cifar10_full/ae_dim64.pt"))
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

    models = {}
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
        },
        "models": {},
    }
    for latent_dim in (64, 512):
        model, metrics = train_model(
            latent_dim, train_loader, test_loader, args, device
        )
        models[latent_dim] = model
        results["models"][str(latent_dim)] = metrics
        checkpoint = args.out / f"ae_spatial_dim{latent_dim}_{args.epochs}ep.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "latent_dim": latent_dim,
                "latent_shape": metrics["latent_shape"],
                "channels": args.hidden,
                "architecture": "spatial",
                "epochs": args.epochs,
                "seed": args.seed,
            },
            checkpoint,
        )
        metrics["checkpoint"] = str(checkpoint)
        (args.out / "ae_spatial_comparison.json").write_text(
            json.dumps(results, indent=2)
        )

    baseline = AutoEncoder(64, ch=64, architecture="plain").to(device)
    baseline_state = torch.load(args.baseline, map_location=device)
    baseline.load_state_dict(baseline_state["model_state_dict"])
    baseline_test_mse = evaluate(baseline, test_loader, device)
    results["plain_64_baseline"] = {
        "parameters": sum(parameter.numel() for parameter in baseline.parameters()),
        "epochs": baseline_state.get("epochs"),
        "final_test_mse": baseline_test_mse,
        "checkpoint": str(args.baseline),
    }

    originals, _ = next(iter(test_loader))
    originals = originals[:8].to(device)
    comparison = [originals]
    with torch.no_grad():
        comparison.append(baseline.eval()(originals))
        comparison.append(models[64].eval()(originals))
        comparison.append(models[512].eval()(originals))
    grid = (torch.cat(comparison).clamp(-1, 1) + 1) / 2
    image_path = args.out / "ae_flat64_vs_spatial64_vs_spatial512.png"
    save_image(grid.cpu(), image_path, nrow=8, padding=3)
    results["comparison_image"] = str(image_path)
    (args.out / "ae_spatial_comparison.json").write_text(json.dumps(results, indent=2))

    print(f"plain d=64 test_mse={baseline_test_mse:.6f}")
    print(f"saved image: {image_path}")
    print(f"saved results: {args.out / 'ae_spatial_comparison.json'}")


if __name__ == "__main__":
    main()
