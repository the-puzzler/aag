#!/usr/bin/env python
"""STANDARD unconditional generation view.

  python plots/plot_generations.py <run_dir> [more_dirs ...] --out fig.png

Per run: training curve (train/val MSE + LPIPS) and a 64-sample grid from the
final checkpoint. Samples always use seed=0 so grids are comparable across runs.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
from torchvision.utils import make_grid
from gga.ae import ResidualDecoder as ConvDecoder

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", type=Path)
ap.add_argument("--labels", nargs="*", default=None)
ap.add_argument("--epoch", default="200")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
labels = a.labels or [r.name for r in a.runs]
dev = "cuda" if torch.cuda.is_available() else "cpu"

fig, axes = plt.subplots(len(a.runs), 3, figsize=(16, 4.6 * len(a.runs)),
                         squeeze=False, gridspec_kw={"width_ratios": [1, 1, 1.35]})
for r, (run, lab) in enumerate(zip(a.runs, labels)):
    c = json.load(open(run / "generator_train_curve.json"))
    ax = axes[r]
    ax[0].plot(c["train_epoch"], c["train_mse"], color="#2b7bba", label="train")
    ax[0].plot(c["val_epoch"], c["val_mse"], color="#e07b39", ls="--", marker="o", ms=3, label="val")
    ax[0].set_title("MSE"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_ylabel(lab, fontsize=9)
    ax[1].plot(c["train_epoch"], c["train_lpips"], color="#2b7bba", label="train")
    ax[1].plot(c["val_epoch"], c["val_lpips"], color="#e07b39", ls="--", marker="o", ms=3, label="val")
    ax[1].set_title("LPIPS"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    ck = torch.load(run / f"checkpoints/generator_ep{a.epoch}.pt", map_location=dev, weights_only=False)
    m = ConvDecoder(ck["dim"], ch=ck["ch"], image_size=ck["image_size"]).to(dev).eval()
    m.load_state_dict(ck["model_state_dict"])
    torch.manual_seed(0)
    with torch.no_grad():
        imgs = m(torch.randn(64, ck["dim"], device=dev))
    grid = make_grid((imgs.clamp(-1, 1) + 1) / 2, nrow=8).permute(1, 2, 0).cpu().numpy()
    ax[2].imshow(grid); ax[2].axis("off"); ax[2].set_title(f"samples, epoch {a.epoch} (seed 0)")
    bl = min(range(len(c["val_lpips"])), key=lambda i: c["val_lpips"][i])
    print(f"{lab}: final train_mse {c['train_mse'][-1]:.5f}  val_mse {c['val_mse'][-1]:.5f}  "
          f"val_lpips {c['val_lpips'][-1]:.5f}  (best val_lpips ep{c['val_epoch'][bl]})")
plt.tight_layout(); plt.savefig(a.out, dpi=135); print("saved:", a.out)
