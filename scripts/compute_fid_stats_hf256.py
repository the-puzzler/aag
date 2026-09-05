#!/usr/bin/env python
"""Reference Inception statistics for FID on the 256x256 HF-parquet datasets.

Protocol, stated once so every number in results_scale256 is comparable:
  * pytorch-fid Inception-v3 (2048-d pool features), the same model aag.fid has
    used for every published AAG number;
  * reference = TRAIN split (ADM's ImageNet-256 protocol uses train-set
    statistics; for CelebA-HQ the train split is the only 28k there is);
  * images are the mirror's center-crop + Lanczos 256 frames. ADM's reference
    batch is resize-then-center-crop -- same region, different resampler --
    so numbers are comparable to the literature only to that approximation,
    and that caveat travels with every FID we report.
Writes mu/sigma as .npz next to the dataset.
"""
from __future__ import annotations

import argparse, os
from pathlib import Path

import numpy as np
import torch

from aag.fid import get_activations, activation_stats
from aag.hf256 import DATASETS, load_uint8, manifest

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=list(DATASETS), required=True)
ap.add_argument("--root", default=os.environ.get("AAG_HF_ROOT", "/data/aag_data/hf"))
ap.add_argument("--split", default="train")
ap.add_argument("--n", type=int, default=50000, help="rows to use (0 = all); a fixed random subset with --seed")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()

N = manifest(a.root, a.dataset, a.split)["offsets"][-1]
n = N if not a.n or a.n >= N else a.n
g = torch.Generator().manual_seed(a.seed)
idx = torch.sort(torch.randperm(N, generator=g)[:n]).values if n < N else torch.arange(N)
print(f"{a.dataset}/{a.split}: {n:,} of {N:,} rows", flush=True)

acts = []
# decode in contiguous windows and pick the sampled rows inside each window
for lo in range(0, N, a.chunk):
    hi = min(N, lo + a.chunk)
    sel = idx[(idx >= lo) & (idx < hi)] - lo
    if sel.numel() == 0:
        continue
    x, _ = load_uint8(a.root, a.dataset, lo, hi, split=a.split, workers=8, chunk=1024)
    x01 = x[sel].permute(0, 3, 1, 2).float().div_(255.0)
    acts.append(get_activations(x01, "cuda", batch=100))
    print(f"  {sum(len(q) for q in acts):>8,}/{n:,}", flush=True)
acts = np.concatenate(acts, 0)
mu, sigma = activation_stats(acts)
out = a.out or Path(a.root).parent / a.dataset / f"fid_stats_{a.split}_{n}.npz"
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, mu=mu, sigma=sigma, n=n, split=a.split, dataset=a.dataset)
print("saved", out, "acts", acts.shape)
