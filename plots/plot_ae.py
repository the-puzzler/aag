#!/usr/bin/env python
"""STANDARD autoencoder view: training curve AND reconstructions in one figure.

  python plots/plot_ae.py <ae_checkpoint.pt> [more.pt ...] --curve <curve.json> --out fig.png

Top row: train/test MSE and LPIPS over epochs (if a curve json is given).
Bottom: real held-out images and each AE's reconstruction of them -- the
reconstruction ceiling any generator on that AE is bounded by.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
from torchvision.utils import make_grid
from aag.ae import AutoEncoder
from aag.datasets import get_loaders, spec as dataset_spec

ap = argparse.ArgumentParser()
ap.add_argument("checkpoints", nargs="+", type=Path)
ap.add_argument("--labels", nargs="*", default=None)
ap.add_argument("--curve", type=Path, default=None)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--dataset", choices=["celeba", "cifar10"], default="celeba")
ap.add_argument("--data", default=None, help="defaults to the dataset's usual root")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
labels = a.labels or [p.stem for p in a.checkpoints]
dev = "cuda" if torch.cuda.is_available() else "cpu"
data_root = a.data or dataset_spec(a.dataset)["default_root"]

models = []
for p, lab in zip(a.checkpoints, labels):
    ck = torch.load(p, map_location=dev, weights_only=False)
    ae = AutoEncoder(ck["latent_dim"], ch=ck["channels"], architecture=ck["architecture"],
                     image_size=ck["image_size"]).to(dev).eval()
    ae.load_state_dict(ck["model_state_dict"]); models.append((lab, ae, ck))

_, _, test_loader, _ = get_loaders(a.dataset, data_root, a.n, n_particles=1,
                                   image_size=models[0][2]["image_size"])
originals = next(iter(test_loader))[0][:a.n].to(dev)

nrow = 1 + len(models)
have_curve = a.curve is not None and a.curve.exists()
fig = plt.figure(figsize=(2.0 * a.n, 2.0 * nrow + (4.0 if have_curve else 0)))
gs = fig.add_gridspec(nrow + (1 if have_curve else 0), 2, height_ratios=([1.6] if have_curve else []) + [1] * nrow)

if have_curve:
    c = json.load(open(a.curve))
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(c["train_epoch"], c["train_mse"], color="#2b7bba", label="train")
    if c.get("test_epoch"): ax.plot(c["test_epoch"], c["test_mse"], color="#e07b39", ls="--", marker="o", ms=3, label="test")
    ax.set_title("AE MSE"); ax.set_xlabel("epoch"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(c["train_epoch"], c["train_lpips"], color="#2b7bba", label="train")
    if c.get("test_epoch"): ax.plot(c["test_epoch"], c["test_lpips"], color="#e07b39", ls="--", marker="o", ms=3, label="test")
    ax.set_title("AE LPIPS"); ax.set_xlabel("epoch"); ax.legend(fontsize=8); ax.grid(alpha=.3)

off = 1 if have_curve else 0
def strip(ax, imgs, title):
    g = make_grid((imgs.clamp(-1, 1) + 1) / 2, nrow=imgs.shape[0]).permute(1, 2, 0).cpu().numpy()
    ax.imshow(g); ax.axis("off"); ax.set_ylabel(title)
    ax.set_title(title, fontsize=10, loc="left")

strip(fig.add_subplot(gs[off, :]), originals, "original (held-out test)")
with torch.no_grad():
    for i, (lab, ae, ck) in enumerate(models):
        strip(fig.add_subplot(gs[off + 1 + i, :]), ae.dec(ae.enc(originals)),
              f"{lab}  (dim={ck['latent_dim']})")
plt.tight_layout(); plt.savefig(a.out, dpi=135); print("saved:", a.out)
