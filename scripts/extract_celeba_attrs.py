#!/usr/bin/env python
"""Extract the 40 binary CelebA attributes for the SAME particle ordering
used everywhere else in this project (aag.celeba_data.celeba_loaders with
n_particles=N, n_train=None -> particles = full train set permuted by
torch.randperm(len(train), generator=Generator().manual_seed(0))).

Saves a (N, 40) bool/float tensor plus the attribute-name list, aligned
index-for-index with every h/z/embeds tensor already produced in this repo.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/data/hf_cache")


def main():
    from datasets import load_dataset
    hf = load_dataset("flwrlabs/celeba", "img_align+identity+attr")
    train = hf["train"]
    n = len(train)
    attr_cols = [k for k in train.features.keys() if k not in ("image", "celeb_id")]
    attr_cols.sort()
    print(f"train size={n}  n_attrs={len(attr_cols)}", flush=True)

    g = torch.Generator().manual_seed(0)
    p_idx = torch.randperm(n, generator=g).tolist()

    cols = {c: train[c] for c in attr_cols}
    attrs = torch.zeros(n, len(attr_cols), dtype=torch.bool)
    for j, c in enumerate(attr_cols):
        attrs[:, j] = torch.tensor(cols[c], dtype=torch.bool)
    attrs = attrs[p_idx]

    out = Path("results_celeba/attrs.pt")
    torch.save({"attrs": attrs, "attr_names": attr_cols, "p_idx": torch.tensor(p_idx)}, out)
    print(f"saved: {out}  shape={tuple(attrs.shape)}")


if __name__ == "__main__":
    main()
