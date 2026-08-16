#!/usr/bin/env python
"""Compute reference Inception activation statistics (mu, sigma) on the full
held-out CelebA `test` split (19962 images, never used for training/assignment
/generator fitting anywhere in this project), at 64x64. Cached once and reused
as the FID reference for every generator/method we evaluate."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/data/hf_cache")

from aag.fid import get_activations, activation_stats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from datasets import load_dataset
    hf = load_dataset("flwrlabs/celeba", "img_align+identity+attr")
    test = hf["test"]
    n = len(test)
    print(f"held-out test split: {n} images", flush=True)

    tf = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),  # [0,1], matches what generators produce after (x+1)/2
    ])
    imgs = torch.empty(n, 3, 64, 64)
    for i in range(n):
        imgs[i] = tf(test[i]["image"].convert("RGB"))
        if (i + 1) % 5000 == 0:
            print(f"  loaded {i+1}/{n}", flush=True)

    print("computing Inception activations ...", flush=True)
    acts = get_activations(imgs, device, batch=200)
    mu, sigma = activation_stats(acts)
    out = Path("results_fid/real_stats.npz")
    out.parent.mkdir(exist_ok=True)
    np.savez(out, mu=mu, sigma=sigma, n=n)
    print(f"saved: {out}  (mu shape {mu.shape}, sigma shape {sigma.shape})")


if __name__ == "__main__":
    main()
