#!/usr/bin/env python
"""FID for the CONDITIONAL CelebA generator.

The existing sweep only handles unconditional models. For a conditional one the
condition must be sampled from the real marginal p(c), and CelebA's 40 attributes
are strongly correlated (Male/No_Beard, Blond_Hair/Male ...), so sampling each bit
independently would fabricate combinations that never occur and inflate FID.
Instead we resample whole real attribute ROWS with replacement, which preserves
the joint distribution exactly.

Reference stats are the held-out test split (19,962 images), the same
results_fid/real_stats.npz every other FID number in this project used.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf_cache")

import numpy as np
import torch

from aag.fid import get_activations, fid_from_stats
from aag.ae import ResidualDecoder as ConvDecoder

N_GEN, SEED = 10000, 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--stats", default="results_fid/real_stats.npz")
    ap.add_argument("--out", type=Path, default=Path("results_fid/fid_conditional.json"))
    a = ap.parse_args()

    ref = np.load(a.stats)
    attrs = torch.load("results_celeba/attrs.pt", map_location=DEV, weights_only=False)
    cond_pool = attrs["attrs"].to(DEV).float()
    n_attrs = cond_pool.shape[1]
    dim_z = torch.load("results_celeba_conditional/assign_4k_alpha0.25/assignment.pt",
                       map_location="cpu", weights_only=False)["z"].shape[1]
    print(f"cond pool {tuple(cond_pool.shape)}, dim_z={dim_z}, {N_GEN} samples/ckpt", flush=True)

    results = {}
    for cp in a.checkpoints:
        ck = torch.load(cp, map_location=DEV, weights_only=False)
        model = ConvDecoder(dim_z + n_attrs, ch=64, image_size=64).to(DEV).eval()
        model.load_state_dict(ck["model_state_dict"])
        torch.manual_seed(SEED)
        imgs = []
        for i in range(0, N_GEN, 500):
            n = min(500, N_GEN - i)
            z = torch.randn(n, dim_z, device=DEV)
            c = cond_pool[torch.randint(cond_pool.shape[0], (n,), device=DEV)]
            imgs.append(((model(torch.cat([z, c], 1)).clamp(-1, 1) + 1) / 2).cpu())
        acts = get_activations(torch.cat(imgs), DEV)
        fid = fid_from_stats(ref["mu"], ref["sigma"], acts)
        ep = ck.get("epoch", "?")
        results[str(ep)] = fid
        print(f"  epoch {ep}: FID = {fid:.3f}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))
    print("saved:", a.out, flush=True)


if __name__ == "__main__":
    main()
