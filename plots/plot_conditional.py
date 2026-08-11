#!/usr/bin/env python
"""STANDARD conditional generation view.

  python plots/plot_conditional.py <run_dir> [more_dirs ...] --out fig.png

Per run, three grids: fixed-condition/varying-z, fixed-z/varying-condition, and
the rare/contradictory-combination tail grid (where residual z-condition
dependence shows up most). Falls back gracefully if a grid is missing.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
import matplotlib.image as mpimg
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", type=Path)
ap.add_argument("--labels", nargs="*", default=None)
ap.add_argument("--epoch", default="200")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
labels = a.labels or [r.name for r in a.runs]
root = Path(__file__).resolve().parent.parent

panels = [("demo_fixed_condition_varying_z.png", "fixed condition, varying z"),
          ("demo_fixed_z_varying_condition.png", "fixed z, varying condition"),
          ("rare_conditions.png", "rare / contradictory combinations")]

fig, axes = plt.subplots(len(a.runs), 3, figsize=(15.5, 5.4 * len(a.runs)), squeeze=False)
for r, (run, lab) in enumerate(zip(a.runs, labels)):
    rare = run / "rare_conditions.png"
    if not rare.exists():
        ck = run / f"checkpoints/generator_ep{a.epoch}.pt"
        if ck.exists():
            subprocess.run([sys.executable, str(root / "demo_rare_conditions.py"),
                            "--checkpoint", str(ck), "--out", str(rare)], check=False)
    for c, (fn, title) in enumerate(panels):
        ax = axes[r][c]; p = run / fn
        if p.exists():
            ax.imshow(mpimg.imread(p))
        else:
            ax.text(.5, .5, f"missing\n{fn}", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=9)
        ax.axis("off"); ax.set_title(f"{lab}\n{title}", fontsize=10)
plt.tight_layout(); plt.savefig(a.out, dpi=125); print("saved:", a.out)
