#!/usr/bin/env python
"""Overlay the generator runs' training curves.

Two panels, MSE and LPIPS, because they are different scales -- never a dual
y-axis. Both are the SINGLE-STEP (step 0) loss, which is the only quantity all
three runs share: gen_seq_lag1 also spent half of every batch on 8-step
sequence prediction, and that extra term is not plotted here because the other
two runs have no counterpart to it.

This is context, not a controlled comparison. Every pair differs in more than
one variable, and both axes are TRAIN loss with no validation split in any run.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"
# reference categorical palette, slots 1-3 in fixed order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

RUNS = [
    ("gen_12d_scratch",
     "AE latents, no rollout  (PRE-fix assignment)",
     "/data/aag_results/results_vpt/gen_12d_scratch/gen_curve.json"),
    ("gen_seq_lag1",
     "AE latents + rollout (seq_prob 0.5)  (lag-fixed)",
     "/data/aag_results/results_vpt/gen_seq_lag1/gen_curve.json"),
    ("gen_pix_lag1",
     "learned 17.7M pixel encoder, no rollout  (lag-fixed)",
     "/data/aag_results/results_vpt/gen_pix_lag1/gen_curve.json"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = []
    for name, desc, path in RUNS:
        p = Path(path)
        if not p.exists():
            print(f"skip {name}: no {path}")
            continue
        data.append((name, desc, json.loads(p.read_text())))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), facecolor=SURFACE)
    for ax, key, title in ((axes[0], "mse", "single-step train MSE"),
                           (axes[1], "lpips", "single-step train LPIPS")):
        ax.set_facecolor(SURFACE)
        for i, (name, desc, c) in enumerate(data):
            ax.plot(c["epoch"], c[key], color=COLORS[i], lw=2,
                    label=f"{name} — {desc}")
            # endpoints only; a number on every point is noise. Staggered
            # vertically because two of these finals differ in the 5th decimal
            # (0.00812 vs 0.00809) and would otherwise print on top of each other.
            ax.annotate(f"{c[key][-1]:.5f}", xy=(c["epoch"][-1], c[key][-1]),
                        xytext=(6, 11 - 11 * i), textcoords="offset points",
                        ha="left", va="center", fontsize=8, color=COLORS[i])
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
        ax.set_xlabel("epoch", color=INK2, fontsize=9)
        ax.set_xlim(0, 46)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5)

    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2,
                   loc="upper right")
    fig.suptitle("VPT generator runs — single-step training loss",
                 color=INK, fontsize=12, x=0.007, ha="left", y=0.99)
    fig.text(0.007, 0.055,
             "TRAIN loss; no validation split in any run. Context, NOT a "
             "controlled comparison: each pair differs in more than one variable "
             "(assignment, context encoder, rollout supervision).",
             color=INK2, fontsize=8, ha="left")
    fig.text(0.007, 0.018,
             "gen_seq_lag1's curve is additionally raised by spending half of "
             "every batch on 8-step sequence prediction, which the other two "
             "never do, so its gap is not attributable to the context encoder.",
             color=INK2, fontsize=8, ha="left")
    fig.tight_layout(rect=(0, 0.095, 1, 0.955))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}\n")
    for name, desc, c in data:
        print(f"{name:18s} {len(c['epoch']):2d} ep  final mse {c['mse'][-1]:.5f}  "
              f"lpips {c['lpips'][-1]:.5f}")


if __name__ == "__main__":
    main()
