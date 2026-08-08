#!/usr/bin/env python
"""Run the full persistent-Gaussian decoder experiment on a saved spatial 512D AE."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from gga.ae import AutoEncoder, encode_all
from gga.data import cifar_loaders
from gga.decoder import _frechet, decode_samples, fresh_prior_metrics, train_decoder
from gga.diagnostics import assignment_diagnostics, intrinsic_dimension_twonn
from gga.gaussianize import AssignConfig, build_assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-checkpoint", type=Path,
                        default=Path("results/cifar10_spatial_compression_sweep/ae_spatial_dim512_10ep.pt"))
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", type=Path,
                        default=Path("results/cifar10_spatial512_decoder"))
    parser.add_argument("--N", type=int, default=6000)
    parser.add_argument("--n-eval", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--assign-steps", type=int, default=400)
    parser.add_argument("--search-subset", type=int, default=2048)
    parser.add_argument("--dec-epochs", type=int, default=200)
    parser.add_argument("--dec-batch", type=int, default=512)
    parser.add_argument("--dec-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    def log(message):
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    log(f"device={device} latent_dim=512 N={args.N}")
    checkpoint = torch.load(args.ae_checkpoint, map_location=device)
    ae = AutoEncoder(512, ch=checkpoint.get("channels", 64),
                     architecture="spatial").to(device)
    ae.load_state_dict(checkpoint["model_state_dict"])
    ae.eval()

    _, particle_loader, _, _ = cifar_loaders(
        args.data, args.batch, args.N
    )
    log("encoding the fixed CIFAR-10 particle subset")
    h = encode_all(ae, particle_loader, device).to(device)

    results = {
        "config": {
            **vars(args),
            "ae_checkpoint": str(args.ae_checkpoint),
            "out": str(args.out),
            "device": device,
        },
        "latent_dim": 512,
        "latent_shape": [32, 4, 4],
        "N": args.N,
        "N_pow_1_over_ambient_dim": float(args.N ** (1 / 512)),
    }

    log("estimating intrinsic dimension")
    intrinsic_dim = intrinsic_dimension_twonn(h.cpu())
    results["intrinsic_dim"] = intrinsic_dim
    results["N_pow_1_over_intrinsic_dim"] = float(
        args.N ** (1 / max(intrinsic_dim, 1e-6))
    )
    log(f"intrinsic_dim={intrinsic_dim:.2f}")

    log("building the 512D persistent Gaussian assignment")
    assignment_started = time.time()
    assignment = build_assignment(
        h,
        AssignConfig(
            steps=args.assign_steps,
            search_subset=args.search_subset,
            seed=args.seed,
        ),
        log=log,
    )
    results["assignment_seconds"] = time.time() - assignment_started
    results["assignment"] = assignment_diagnostics(
        assignment["z"], d=512, seed=args.seed
    )
    log(f"assignment diagnostics={results['assignment']}")

    assignment_path = args.out / "assignment_spatial512.pt"
    torch.save(
        {
            "z": assignment["z"].cpu(),
            "mean": assignment["mean"].cpu(),
            "W": assignment["W"].cpu(),
            "W_inv": assignment["W_inv"].cpu(),
            "history": assignment["hist"],
            "N": args.N,
            "latent_dim": 512,
            "seed": args.seed,
        },
        assignment_path,
    )
    results["assignment_checkpoint"] = str(assignment_path)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=str))

    log("training the direct 512D Gaussian-to-AE-latent decoder")
    decoder_started = time.time()
    decoder, held_out_mse = train_decoder(
        assignment["z"], h, dim=512, epochs=args.dec_epochs,
        lr=args.dec_lr, batch=args.dec_batch, device=device, log=log,
    )
    results["decoder_seconds"] = time.time() - decoder_started
    results["held_out_pair_mse"] = held_out_mse
    decoder_path = args.out / "direct_decoder_spatial512.pt"
    torch.save(
        {
            "model_state_dict": decoder.state_dict(),
            "input_dim": 512,
            "output_dim": 512,
            "epochs": args.dec_epochs,
            "seed": args.seed,
        },
        decoder_path,
    )
    results["decoder_checkpoint"] = str(decoder_path)

    log("evaluating 6,000 fresh full-rank Gaussian samples")
    fresh = fresh_prior_metrics(
        decoder, h, dim=512, n_samples=args.n_eval, device=device
    )
    results["fresh_prior"] = fresh

    with torch.no_grad():
        z_fresh = torch.randn(args.n_eval, 512, device=device)
        h_generated = decoder(z_fresh)
        h_real_white = (h - assignment["mean"]) @ assignment["W"]
        h_generated_white = (h_generated - assignment["mean"]) @ assignment["W"]
        results["fresh_prior_whitened_frechet"] = _frechet(
            h_real_white, h_generated_white
        )
        results["fresh_prior_variance_ratio"] = float(
            h_generated_white.var(0, unbiased=False).sum()
            / h_real_white.var(0, unbiased=False).sum()
        )
        results["fresh_prior_whitened_mean_norm"] = float(
            h_generated_white.mean(0).norm()
        )
        images = decode_samples(
            decoder, ae, dim=512, n=64,
            mean=assignment["mean"], W_inv=assignment["W_inv"],
            device=device,
        )
    sample_path = args.out / "samples_spatial512_full_prior.png"
    save_image((images.clamp(-1, 1) + 1).cpu() / 2, sample_path, nrow=8)
    results["sample_grid"] = str(sample_path)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=str))

    log(f"held_out_pair_mse={held_out_mse:.6f}")
    log(f"fresh_prior={fresh}")
    log(f"whitened_frechet={results['fresh_prior_whitened_frechet']:.3f} "
        f"variance_ratio={results['fresh_prior_variance_ratio']:.3f}")
    log(f"ALL DONE: {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
