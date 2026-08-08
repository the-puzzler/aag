#!/usr/bin/env python
"""Use an AE only to couple CIFAR images to low-rank Gaussian pixel generators."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import save_image

from gga.ae import AutoEncoder, SpatialResidualUpBlock, encode_all
from gga.data import cifar_loaders
from gga.decoder import _frechet
from gga.diagnostics import assignment_diagnostics
from gga.gaussianize import AssignConfig, build_assignment


class DirectPixelGenerator(nn.Module):
    """Asymmetric kD Gaussian -> CIFAR image generator."""

    def __init__(self, input_dim: int, base_channels: int = 64):
        super().__init__()
        stem_channels = base_channels * 4
        self.stem_channels = stem_channels
        self.fc = nn.Sequential(
            nn.Linear(input_dim, stem_channels * 4 * 4),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            SpatialResidualUpBlock(stem_channels, base_channels * 2),  # 8x8
            SpatialResidualUpBlock(base_channels * 2, base_channels), # 16x16
            SpatialResidualUpBlock(base_channels, base_channels),     # 32x32
            nn.Conv2d(base_channels, 3, 3, 1, 1),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, self.stem_channels, 4, 4)
        return torch.tanh(self.net(x))


class CoupledImages(Dataset):
    def __init__(self, z: torch.Tensor, image_dataset):
        self.z = z.cpu()
        self.image_dataset = image_dataset

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, index):
        image, _ = self.image_dataset[index]
        return self.z[index], image


@torch.no_grad()
def image_pair_mse(model, loader, device):
    model.eval()
    total = 0.0
    elements = 0
    for z, images in loader:
        z = z.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        total += F.mse_loss(model(z), images, reduction="sum").item()
        elements += images.numel()
    return total / elements


def train_generator(rank, z, image_dataset, args, device, log):
    dataset = CoupledImages(z, image_dataset)
    split_gen = torch.Generator().manual_seed(args.seed + rank)
    order = torch.randperm(len(dataset), generator=split_gen)
    n_val = int(args.val_fraction * len(dataset))
    val_indices = order[:n_val].tolist()
    train_indices = order[n_val:].tolist()
    train_loader = DataLoader(
        Subset(dataset, train_indices), args.gen_batch, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), args.gen_batch, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    all_loader = DataLoader(
        dataset, args.gen_batch, shuffle=True, num_workers=2, pin_memory=True,
    )

    torch.manual_seed(args.seed + rank)
    model = DirectPixelGenerator(rank, args.gen_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.gen_lr)
    history = []
    started = time.time()
    for epoch in range(1, args.gen_epochs + 1):
        model.train()
        total = 0.0
        elements = 0
        for batch_z, images in train_loader:
            batch_z = batch_z.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)
            reconstruction = model(batch_z)
            loss = F.mse_loss(reconstruction, images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.numel()
            elements += images.numel()
        train_mse = total / elements
        val_mse = image_pair_mse(model, val_loader, device)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        log(
            f"[rank {rank}] epoch {epoch}/{args.gen_epochs} "
            f"train_mse={train_mse:.6f} val_mse={val_mse:.6f}"
        )

    held_out_mse = history[-1]["val_mse"]
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        total = 0.0
        elements = 0
        for batch_z, images in all_loader:
            batch_z = batch_z.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)
            reconstruction = model(batch_z)
            loss = F.mse_loss(reconstruction, images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.numel()
            elements += images.numel()
        log(
            f"[rank {rank}] all-pairs finetune {epoch}/{args.finetune_epochs} "
            f"mse={total / elements:.6f}"
        )
    all_pair_mse = image_pair_mse(
        model,
        DataLoader(dataset, args.gen_batch, shuffle=False, num_workers=2,
                   pin_memory=True),
        device,
    )
    return model, {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seconds": time.time() - started,
        "held_out_pair_mse": held_out_mse,
        "all_pair_mse_after_finetune": all_pair_mse,
        "history": history,
    }


@torch.no_grad()
def generate_images(model, rank, count, batch, device):
    model.eval()
    output = []
    for start in range(0, count, batch):
        n = min(batch, count - start)
        output.append(model(torch.randn(n, rank, device=device)).cpu())
    return torch.cat(output)


@torch.no_grad()
def encode_tensor_images(ae, images, batch, device):
    output = []
    for start in range(0, images.shape[0], batch):
        output.append(ae.enc(images[start:start + batch].to(device)).cpu())
    return torch.cat(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-checkpoint", type=Path, default=Path(
        "results/cifar10_spatial_compression_sweep/ae_spatial_dim128_10ep.pt"
    ))
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", type=Path,
                        default=Path("results/cifar10_direct_pixel_pca_gaussian"))
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--N", type=int, default=50000)
    parser.add_argument("--n-eval", type=int, default=6000)
    parser.add_argument("--encode-batch", type=int, default=256)
    parser.add_argument("--assign-steps", type=int, default=400)
    parser.add_argument("--search-subset", type=int, default=2048)
    parser.add_argument("--gen-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--gen-batch", type=int, default=256)
    parser.add_argument("--gen-channels", type=int, default=64)
    parser.add_argument("--gen-lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.N != 50000:
        raise ValueError("this all-data experiment expects N=50000")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    def log(message):
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    log(f"device={device} N={args.N} ranks={args.ranks}")
    ae_state = torch.load(args.ae_checkpoint, map_location=device)
    ae = AutoEncoder(128, ch=ae_state.get("channels", 64),
                     architecture="spatial").to(device)
    ae.load_state_dict(ae_state["model_state_dict"])
    ae.eval()

    _, particle_loader, _, _ = cifar_loaders(
        args.data, args.encode_batch, args.N
    )
    image_dataset = particle_loader.dataset
    log("encoding all 50,000 CIFAR-10 training images with the coupling AE")
    h = encode_all(ae, particle_loader, device).float()
    h_mean = h.mean(0, keepdim=True)
    centered = h - h_mean
    covariance = centered.T @ centered / (h.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    max_rank = max(args.ranks)
    pca_components = eigenvectors[:, :max_rank]
    scores = centered @ pca_components
    explained = eigenvalues.cumsum(0) / eigenvalues.sum()
    coupling_path = args.out / "pca_coupling_teacher.pt"
    torch.save(
        {
            "h_mean": h_mean,
            "pca_components": pca_components,
            "eigenvalues": eigenvalues,
            "ae_checkpoint": str(args.ae_checkpoint),
        },
        coupling_path,
    )

    results = {
        "config": {
            **vars(args),
            "ae_checkpoint": str(args.ae_checkpoint),
            "out": str(args.out),
            "device": device,
        },
        "pca_coupling_teacher": str(coupling_path),
        "models": {},
    }

    # Whiten the teacher latents once for scale-independent feature diagnostics.
    evals = eigenvalues.clamp_min(1e-8)
    full_components = eigenvectors
    h_white = centered @ (full_components @ torch.diag(evals.rsqrt()))

    for rank in args.ranks:
        log(f"[rank {rank}] constructing the persistent Gaussian coupling")
        assignment = build_assignment(
            scores[:, :rank].to(device),
            AssignConfig(
                steps=args.assign_steps,
                search_subset=args.search_subset,
                seed=args.seed + rank,
            ),
            log=log,
        )
        z = assignment["z"].cpu()
        assignment_metrics = assignment_diagnostics(
            assignment["z"], d=rank, seed=args.seed + rank
        )
        assignment_path = args.out / f"assignment_rank{rank}.pt"
        torch.save(
            {
                "z": z,
                "mean": assignment["mean"].cpu(),
                "W": assignment["W"].cpu(),
                "W_inv": assignment["W_inv"].cpu(),
                "history": assignment["hist"],
                "rank": rank,
                "N": args.N,
            },
            assignment_path,
        )

        log(f"[rank {rank}] training the direct pixel generator")
        generator, metrics = train_generator(
            rank, z, image_dataset, args, device, log
        )
        generator_path = args.out / f"direct_pixel_generator_rank{rank}.pt"
        torch.save(
            {
                "model_state_dict": generator.state_dict(),
                "rank": rank,
                "base_channels": args.gen_channels,
                "epochs": args.gen_epochs,
                "finetune_epochs": args.finetune_epochs,
                "seed": args.seed,
            },
            generator_path,
        )

        fresh_images = generate_images(
            generator, rank, args.n_eval, args.gen_batch, device
        )
        fresh_path = args.out / f"fresh_samples_rank{rank}.png"
        save_image(
            (fresh_images[:64].clamp(-1, 1) + 1) / 2,
            fresh_path, nrow=8,
        )
        fresh_h = encode_tensor_images(
            ae, fresh_images, args.encode_batch, device
        ).float()
        fresh_centered = fresh_h - h_mean
        fresh_white = fresh_centered @ (
            full_components @ torch.diag(evals.rsqrt())
        )
        feature_frechet = _frechet(h_white, fresh_white)
        variance_ratio = float(
            fresh_white.var(0, unbiased=False).sum()
            / h_white.var(0, unbiased=False).sum()
        )

        # First row: fresh images. Second row: nearest training images in AE space.
        distances = torch.cdist(
            fresh_h[:8].to(device), h.to(device)
        )
        nearest_indices = distances.argmin(1).cpu().tolist()
        nearest_images = torch.stack(
            [image_dataset[index][0] for index in nearest_indices]
        )
        nearest_grid = torch.cat([fresh_images[:8], nearest_images])
        nearest_path = args.out / f"fresh_vs_nearest_train_rank{rank}.png"
        save_image(
            (nearest_grid.clamp(-1, 1) + 1) / 2,
            nearest_path, nrow=8, padding=3,
        )

        metrics.update({
            "rank": rank,
            "N": args.N,
            "N_pow_1_over_rank": float(args.N ** (1 / rank)),
            "pca_explained_variance": float(explained[rank - 1]),
            "assignment": assignment_metrics,
            "assignment_checkpoint": str(assignment_path),
            "generator_checkpoint": str(generator_path),
            "fresh_sample_grid": str(fresh_path),
            "fresh_vs_nearest_grid": str(nearest_path),
            "fresh_ae_feature_frechet": feature_frechet,
            "fresh_ae_feature_variance_ratio": variance_ratio,
            "fresh_ae_feature_mean_norm": float(fresh_white.mean(0).norm()),
            "nearest_train_feature_distance_mean": float(
                distances.min(1).values.mean()
            ),
        })
        results["models"][str(rank)] = metrics
        (args.out / "results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
        log(
            f"[rank {rank}] held_out={metrics['held_out_pair_mse']:.6f} "
            f"feature_frechet={feature_frechet:.3f} "
            f"variance_ratio={variance_ratio:.3f}"
        )

    log(f"ALL DONE: {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
