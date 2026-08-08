#!/usr/bin/env python
"""Compare real AE reconstructions, assigned-pair decodes, and fresh 512D decodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from gga.ae import AutoEncoder
from gga.data import cifar_loaders
from gga.decoder import ResidualDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae", type=Path,
                        default=Path("results/cifar10_spatial_compression_sweep/ae_spatial_dim512_10ep.pt"))
    parser.add_argument("--assignment", type=Path,
                        default=Path("results/cifar10_spatial512_decoder/assignment_spatial512.pt"))
    parser.add_argument("--decoder", type=Path,
                        default=Path("results/cifar10_spatial512_decoder/direct_decoder_spatial512.pt"))
    parser.add_argument("--results", type=Path,
                        default=Path("results/cifar10_spatial512_decoder/results.json"))
    parser.add_argument("--data", default="./data")
    parser.add_argument("--N", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    ae_state = torch.load(args.ae, map_location=device)
    ae = AutoEncoder(512, ch=ae_state.get("channels", 64),
                     architecture="spatial").to(device)
    ae.load_state_dict(ae_state["model_state_dict"])
    ae.eval()

    decoder_state = torch.load(args.decoder, map_location=device)
    decoder = ResidualDecoder(512).to(device)
    decoder.load_state_dict(decoder_state["model_state_dict"])
    decoder.eval()
    assigned_z = torch.load(args.assignment, map_location="cpu")["z"]

    _, particle_loader, _, _ = cifar_loaders(args.data, args.batch, args.N)
    original_error = 0.0
    pair_error = 0.0
    pair_to_ae_error = 0.0
    elements = 0
    offset = 0
    preview = None
    with torch.no_grad():
        for originals, _ in particle_loader:
            originals = originals.to(device)
            batch = originals.size(0)
            h = ae.enc(originals)
            ae_recon = ae.dec(h)
            pair_h = decoder(assigned_z[offset:offset + batch].to(device))
            pair_recon = ae.dec(pair_h)
            original_error += F.mse_loss(
                ae_recon, originals, reduction="sum"
            ).item()
            pair_error += F.mse_loss(
                pair_recon, originals, reduction="sum"
            ).item()
            pair_to_ae_error += F.mse_loss(
                pair_recon, ae_recon, reduction="sum"
            ).item()
            elements += originals.numel()
            if preview is None:
                fresh_h = decoder(torch.randn(8, 512, device=device))
                preview = [
                    originals[:8], ae_recon[:8], pair_recon[:8], ae.dec(fresh_h),
                ]
            offset += batch

    grid = (torch.cat(preview).clamp(-1, 1) + 1) / 2
    image_path = args.results.parent / "decoder_stage_comparison.png"
    save_image(grid.cpu(), image_path, nrow=8, padding=3)

    results = json.loads(args.results.read_text())
    results["image_diagnostics"] = {
        "ae_reconstruction_mse_on_particles": original_error / elements,
        "assigned_pair_decode_mse_to_original": pair_error / elements,
        "assigned_pair_decode_mse_to_ae_reconstruction": pair_to_ae_error / elements,
        "comparison_grid": str(image_path),
        "comparison_rows": [
            "original", "AE reconstruction", "direct decoder at assigned z",
            "direct decoder at fresh Gaussian z",
        ],
    }
    args.results.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["image_diagnostics"], indent=2))


if __name__ == "__main__":
    main()
