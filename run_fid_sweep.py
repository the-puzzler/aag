#!/usr/bin/env python
"""FID sweep across every full generative pipeline we have (encode -> assign
-> generator -> pixels), for both the best-val-metric checkpoint(s) and the
final-epoch checkpoint of each. Some methods (VAE, PCA/no-AE) only ever saved
a single final checkpoint (no periodic checkpointing was done for those), so
best==final there.

Each job generates N_GEN fresh samples from z~N(0,I) (fixed seed=0, matching
the seed already used for every samples_*.png in this project) and computes
FID against the held-out CelebA test-split reference stats.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/data/hf_cache")

import torch

from fid_common import get_activations, fid_from_stats
import numpy as np

N_GEN = 10000
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to01(imgs):
    return (imgs.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def gen_pixel_direct(ckpt_path):
    from gga.ae import ResidualDecoder as ConvDecoder
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = ConvDecoder(ck["dim"], ch=ck["ch"], image_size=ck["image_size"]).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    torch.manual_seed(SEED)
    out = []
    for i in range(0, N_GEN, 500):
        n = min(500, N_GEN - i)
        z = torch.randn(n, ck["dim"], device=DEVICE)
        out.append(to01(model(z)).cpu())
    return torch.cat(out, 0)


_AE_CACHE = {}


@torch.no_grad()
def gen_two_stage_ae(ckpt_path):
    from gga.ae import AutoEncoder
    from gga.decoder import ResidualDecoder as Generator
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    ae_ckpt_path = ck["ae_checkpoint"]
    if ae_ckpt_path not in _AE_CACHE:
        ae_ckpt = torch.load(ae_ckpt_path, map_location=DEVICE, weights_only=False)
        ae = AutoEncoder(ae_ckpt["latent_dim"], ch=ae_ckpt["channels"],
                         architecture=ae_ckpt["architecture"],
                         image_size=ae_ckpt["image_size"]).to(DEVICE)
        ae.load_state_dict(ae_ckpt["model_state_dict"])
        ae.eval()
        _AE_CACHE[ae_ckpt_path] = ae
    ae = _AE_CACHE[ae_ckpt_path]
    model = Generator(ck["dim"]).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    torch.manual_seed(SEED)
    out = []
    for i in range(0, N_GEN, 500):
        n = min(500, N_GEN - i)
        z = torch.randn(n, ck["dim"], device=DEVICE)
        h = model(z)
        out.append(to01(ae.dec(h)).cpu())
    return torch.cat(out, 0)


_VAE_CACHE = {}


@torch.no_grad()
def gen_two_stage_vae(ckpt_path):
    from gga.decoder import ResidualDecoder as Generator
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if "vae" not in _VAE_CACHE:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE).eval()
        _VAE_CACHE["vae"] = vae
        _VAE_CACHE["scale"] = vae.config.scaling_factor
    vae, scale = _VAE_CACHE["vae"], _VAE_CACHE["scale"]
    model = Generator(ck["dim"]).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    lat_shape = ck["lat_shape"]
    torch.manual_seed(SEED)
    out = []
    for i in range(0, N_GEN, 500):
        n = min(500, N_GEN - i)
        z = torch.randn(n, ck["dim"], device=DEVICE)
        h = model(z)
        lat = h.view(-1, *lat_shape)
        imgs = vae.decode(lat / scale).sample
        out.append(to01(imgs).cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def gen_pca(ckpt_path, assignment_path, image_size=64):
    from gga.decoder import ResidualDecoder as Generator
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    assign = torch.load(assignment_path, map_location=DEVICE, weights_only=False)
    V, mean_pix = assign["V"].to(DEVICE), assign["mean_pix"].to(DEVICE)
    model = Generator(ck["dim"]).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    torch.manual_seed(SEED)
    out = []
    for i in range(0, N_GEN, 500):
        n = min(500, N_GEN - i)
        z = torch.randn(n, ck["dim"], device=DEVICE)
        h = model(z)
        pix = (h @ V.T + mean_pix).view(-1, 3, image_size, image_size)
        out.append(to01(pix).cpu())
    return torch.cat(out, 0)


def fid_for(imgs01, real_mu, real_sigma):
    acts = get_activations(imgs01, DEVICE, batch=200)
    return fid_from_stats(real_mu, real_sigma, acts)


def main():
    real = np.load("results_fid/real_stats.npz")
    real_mu, real_sigma = real["mu"], real["sigma"]
    print(f"loaded real stats (n={int(real['n'])})", flush=True)

    R = Path("results_celeba")
    RD = Path("results_celeba_dino")
    RL16 = Path("results_celeba_lejepa")
    RL64 = Path("results_celeba_lejepa_dim64")
    RV = Path("results_celeba_vae")
    RP64 = Path("results_celeba_no_ae")
    RP128 = Path("results_celeba_no_ae_dim128")

    jobs = [
        # (method, checkpoint_role, kind, ckpt_path, extra_args)
        ("ae_two_stage_lpips_dim64", "best_lpips", "two_stage_ae",
         R / "full_pipeline_lpips_ae/generator/checkpoints/generator_ep90.pt", {}),
        ("ae_two_stage_lpips_dim64", "best_mse", "two_stage_ae",
         R / "full_pipeline_lpips_ae/generator/checkpoints/generator_ep60.pt", {}),
        ("ae_two_stage_lpips_dim64", "final_ep200", "two_stage_ae",
         R / "full_pipeline_lpips_ae/generator/checkpoints/generator_ep200.pt", {}),

        ("ae_pixel_direct_dim64", "best_val_ep25", "pixel_direct",
         R / "pixel_generator/checkpoints/pixel_generator_ep25.pt", {}),
        ("ae_pixel_direct_dim64", "final_ep40", "pixel_direct",
         R / "pixel_generator/checkpoints/pixel_generator_ep40.pt", {}),

        ("ae_pixel_direct_lpips_dim64", "best_lpips_ep20", "pixel_direct",
         R / "pixel_generator_lpips/checkpoints/generator_ep20.pt", {}),
        ("ae_pixel_direct_lpips_dim64", "best_mse_ep20", "pixel_direct",
         R / "pixel_generator_lpips/checkpoints/generator_ep20.pt", {}),
        ("ae_pixel_direct_lpips_dim64", "final_ep200", "pixel_direct",
         R / "pixel_generator_lpips/checkpoints/generator_ep200.pt", {}),

        ("ae_pixel_direct_lpips_flagshipz_dim64", "best_lpips_ep20", "pixel_direct",
         R / "pixel_generator_lpips_flagshipz/checkpoints/generator_ep20.pt", {}),
        ("ae_pixel_direct_lpips_flagshipz_dim64", "best_mse_ep20", "pixel_direct",
         R / "pixel_generator_lpips_flagshipz/checkpoints/generator_ep20.pt", {}),
        ("ae_pixel_direct_lpips_flagshipz_dim64", "final_ep200", "pixel_direct",
         R / "pixel_generator_lpips_flagshipz/checkpoints/generator_ep200.pt", {}),

        ("dino_lpips_dim384", "best_lpips_ep10", "pixel_direct",
         RD / "generator_lpips/checkpoints/generator_ep10.pt", {}),
        ("dino_lpips_dim384", "best_mse_ep10", "pixel_direct",
         RD / "generator_lpips/checkpoints/generator_ep10.pt", {}),
        ("dino_lpips_dim384", "final_ep200", "pixel_direct",
         RD / "generator_lpips/checkpoints/generator_ep200.pt", {}),

        ("dino_original_dim384", "best_val_ep10", "pixel_direct",
         RD / "full_pipeline_dim384/checkpoints/generator_ep10.pt", {}),
        ("dino_original_dim384", "final_ep200", "pixel_direct",
         RD / "full_pipeline_dim384/checkpoints/generator_ep200.pt", {}),

        ("lejepa_dim16", "best_lpips_ep10", "pixel_direct",
         RL16 / "full_pipeline/generator/checkpoints/generator_ep10.pt", {}),
        ("lejepa_dim16", "best_mse_ep10", "pixel_direct",
         RL16 / "full_pipeline/generator/checkpoints/generator_ep10.pt", {}),
        ("lejepa_dim16", "final_ep200", "pixel_direct",
         RL16 / "full_pipeline/generator/checkpoints/generator_ep200.pt", {}),

        ("lejepa_dim64", "best_lpips_ep10", "pixel_direct",
         RL64 / "full_pipeline/generator/checkpoints/generator_ep10.pt", {}),
        ("lejepa_dim64", "best_mse_ep10", "pixel_direct",
         RL64 / "full_pipeline/generator/checkpoints/generator_ep10.pt", {}),
        ("lejepa_dim64", "final_ep200", "pixel_direct",
         RL64 / "full_pipeline/generator/checkpoints/generator_ep200.pt", {}),

        ("vae_two_stage_dim256", "final_only", "two_stage_vae",
         RV / "full_pipeline/checkpoints/generator.pt", {}),

        ("pca_no_ae_dim64", "final_only", "pca",
         RP64 / "checkpoints/pca_generator.pt", {"assignment_path": RP64 / "pca_assignment.pt"}),
        ("pca_no_ae_dim128", "final_only", "pca",
         RP128 / "checkpoints/pca_generator.pt", {"assignment_path": RP128 / "pca_assignment.pt"}),

        # "no assignment" ablation: decoder trained directly on raw (pre-assignment)
        # LeJEPA embeddings, but generated from a naive raw N(0,I) prior fed straight
        # in (no persistent Gaussian assignment at all) -- tests whether SIGReg's own
        # soft marginal-gaussianization already makes assignment unnecessary.
        ("lejepa_dim16_no_assignment", "best_val_ep10", "pixel_direct",
         RL16 / "decoder_baseline/checkpoints/decoder_ep10.pt", {}),
        ("lejepa_dim16_no_assignment", "final_ep200", "pixel_direct",
         RL16 / "decoder_baseline/checkpoints/decoder_ep200.pt", {}),
        ("lejepa_dim64_no_assignment", "best_val_ep10", "pixel_direct",
         RL64 / "decoder_baseline/checkpoints/decoder_ep10.pt", {}),
        ("lejepa_dim64_no_assignment", "final_ep200", "pixel_direct",
         RL64 / "decoder_baseline/checkpoints/decoder_ep200.pt", {}),
    ]

    gen_fns = {
        "pixel_direct": lambda p, extra: gen_pixel_direct(p),
        "two_stage_ae": lambda p, extra: gen_two_stage_ae(p),
        "two_stage_vae": lambda p, extra: gen_two_stage_vae(p),
        "pca": lambda p, extra: gen_pca(p, extra["assignment_path"]),
    }

    results_path = Path("results_fid/fid_results.json")
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {(r["method"], r["checkpoint"]) for r in results}
    for method, role, kind, ckpt_path, extra in jobs:
        if (method, role) in done:
            print(f"SKIP {method}/{role}: already computed", flush=True)
            continue
        if not Path(ckpt_path).exists():
            print(f"SKIP {method}/{role}: missing {ckpt_path}", flush=True)
            continue
        print(f"--- {method} / {role} ({ckpt_path}) ---", flush=True)
        imgs = gen_fns[kind](ckpt_path, extra)
        fid = fid_for(imgs, real_mu, real_sigma)
        print(f"    FID = {fid:.3f}", flush=True)
        results.append({"method": method, "checkpoint": role, "path": str(ckpt_path), "fid": fid})
        results_path.write_text(json.dumps(results, indent=2))

    print("\n=== FID summary ===")
    for r in results:
        print(f"{r['method']:28s} {r['checkpoint']:16s} FID={r['fid']:.3f}")


if __name__ == "__main__":
    main()
