#!/usr/bin/env python
"""Encode an HF-parquet 256x256 train split into AE latents = the particle file.

Writes {"h": (N, D) float16, "label": (N,) int64, "encoder": id, "grid": g,
"latent_channels": C, "dataset", "n_particles"} where row i is global parquet
row i (see aag.hf256 for why that order is the identity contract).

--encoder is either a diffusers DC-AE id (mit-han-lab/dc-ae-f32c32-in-1.0-diffusers)
or a path to one of this repo's own AE checkpoints. h is stored as the raw
spatial latent flattened C x g x g -> C*g*g in that order, so run_assignment's
whiten(rotate=False) keeps grid cell j at coordinate j.

Chunked: decodes `--chunk` rows from parquet, encodes, appends. ImageNet at
1.28M rows is ~2.5 GB of fp16 latents at D=2048.
"""
from __future__ import annotations

import argparse, os, time
from pathlib import Path

import torch

from aag.hf256 import DATASETS, load_uint8, manifest, to_float

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=list(DATASETS), required=True)
ap.add_argument("--root", default=os.environ.get("AAG_HF_ROOT", "/data/aag_data/hf"))
ap.add_argument("--encoder", default="mit-han-lab/dc-ae-f32c32-in-1.0-diffusers")
ap.add_argument("--N", type=int, default=0, help="0 = all train rows")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--chunk", type=int, default=16384, help="parquet rows decoded per round")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--amp", action="store_true", help="bf16 autocast for the encoder")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

dev = "cuda"
N_all = manifest(a.root, a.dataset)["offsets"][-1]
N = min(a.N, N_all) if a.N else N_all

if a.encoder.endswith(".pt"):
    from aag.ae import AutoEncoder
    ck = torch.load(a.encoder, map_location=dev, weights_only=False)
    ae = AutoEncoder(ck["latent_dim"], ch=ck["channels"], architecture=ck["architecture"],
                     image_size=ck["image_size"], grid=ck.get("grid", 4)).to(dev).eval()
    ae.load_state_dict(ck["model_state_dict"])
    grid, C = ck.get("grid", 4), ck["latent_dim"] // ck.get("grid", 4) ** 2
    encode = lambda x: ae.enc(x)
else:
    from diffusers import AutoencoderDC
    ae = AutoencoderDC.from_pretrained(a.encoder, torch_dtype=torch.float32).to(dev).eval()
    with torch.no_grad():
        z0 = ae.encode(torch.zeros(1, 3, DATASETS[a.dataset]["image_size"], DATASETS[a.dataset]["image_size"], device=dev)).latent
    C, grid = z0.shape[1], z0.shape[2]
    encode = lambda x: ae.encode(x).latent.flatten(1)
D = C * grid * grid
print(f"encoder {a.encoder}: latent {C}x{grid}x{grid} = {D} dims; encoding {N:,}/{N_all:,} "
      f"{a.dataset} rows, amp={a.amp}", flush=True)

h = torch.empty(N, D, dtype=torch.float16)
labels = torch.empty(N, dtype=torch.int64)
t0 = time.time()
for lo in range(0, N, a.chunk):
    hi = min(N, lo + a.chunk)
    x, y = load_uint8(a.root, a.dataset, lo, hi, workers=a.workers, chunk=max(512, a.chunk // (2 * a.workers)))
    labels[lo:hi] = y
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
        for i in range(0, hi - lo, a.batch):
            xb = to_float(x[i:i + a.batch]).to(dev, non_blocking=True)
            h[lo + i:lo + i + xb.shape[0]] = encode(xb).float().half().cpu()
    done = hi; rate = done / (time.time() - t0)
    print(f"  {done:>9,}/{N:,}  {rate:6.0f} img/s  eta {(N - done) / rate / 60:5.1f} min", flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"h": h, "label": labels, "encoder": a.encoder, "grid": grid, "latent_channels": C,
            "dataset": a.dataset, "root": a.root, "n_particles": N,
            "h_stats": {"mean": h.float().mean().item(), "std": h.float().std().item()}},
           a.out)
print(f"saved {a.out}  h {tuple(h.shape)} fp16  labels {labels.min().item()}..{labels.max().item()}  "
      f"{time.time() - t0:.0f}s", flush=True)
