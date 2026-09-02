#!/usr/bin/env python
"""Plot generator training curves, keeping the two objectives visually separate.

MSE and LPIPS are different scales, so they get one panel each -- never a dual
y-axis. The plain run (epochs 1-40, single-step) and the rollout fine-tune
(41-60, half the batches predicting 1-3 steps ahead from self-generated context)
are drawn as SEPARATE segments with a divider, deliberately not joined: joining
them would imply the numbers are comparable across the break, and they are not.
That misreading is the whole reason this plot exists.

Both series are train-set losses. There is no validation split in either run, so
these curves show fitting, and cannot by themselves distinguish fitting from
generalising.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2df"
S1 = "#2a78d6"   # reference categorical slot 1
S2 = "#eb6834"   # reference categorical slot 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", type=Path,
                    default=Path("/data/aag_results/results_vpt/gen_12d_scratch/gen_curve.json"))
    ap.add_argument("--rollout", type=Path,
                    default=Path("/data/aag_results/results_vpt/gen_12d_rollout/gen_curve.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    a = json.loads(args.plain.read_text())
    b = json.loads(args.rollout.read_text()) if args.rollout.exists() else None

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURFACE)
    for ax, key, name in ((axes[0], "mse", "train MSE"),
                          (axes[1], "lpips", "train LPIPS")):
        ax.set_facecolor(SURFACE)
        ax.plot(a["epoch"], a[key], color=S1, lw=2, label="plain, single-step (1-40)")
        if b:
            ax.plot(b["epoch"], b[key], color=S2, lw=2,
                    label="rollout fine-tune, k=3 (41-60)")
            ax.axvline(40.5, color=INK2, lw=1, ls=(0, (4, 3)), alpha=0.7)
            # low on the axes, not high: the legend lives upper-right because
            # both curves decay, so an annotation at the top of the divider
            # lands on top of it
            ax.annotate("objective changes", xy=(40.5, 0.30),
                        xycoords=("data", "axes fraction"),
                        xytext=(6, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8, color=INK2)
        # selective direct labels: the two endpoints only, never every point
        ax.annotate(f"{a[key][-1]:.5f}", xy=(a["epoch"][-1], a[key][-1]),
                    xytext=(-6, -14), textcoords="offset points", ha="right",
                    fontsize=8.5, color=INK2)
        if b:
            ax.annotate(f"{b[key][-1]:.5f}", xy=(b["epoch"][-1], b[key][-1]),
                        xytext=(-2, 8), textcoords="offset points", ha="right",
                        fontsize=8.5, color=INK2)
        ax.set_title(name, color=INK, fontsize=11, loc="left", pad=8)
        ax.set_xlabel("epoch", color=INK2, fontsize=9)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")

    fig.suptitle("12-d generator, gen_12d_scratch then gen_12d_rollout",
                 color=INK, fontsize=12, x=0.008, ha="left", y=0.99)
    fig.text(0.008, 0.015,
             "Train-set losses; no validation split in either run. The two "
             "segments are NOT comparable: from epoch 41 half of each batch "
             "predicts 1-3 steps ahead from the model's own output.",
             color=INK2, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")

    # the numbers worth reading off, printed so they are in the log too
    d = a[ "lpips" ]
    print(f"\nplain: lpips {d[0]:.5f} -> {d[-1]:.5f}; "
          f"epochs improving on BOTH metrics: "
          f"{sum(1 for i in range(1, len(d)) if d[i] < d[i-1] and a['mse'][i] < a['mse'][i-1])}"
          f"/{len(d)-1}")
    print(f"plain last 5 epochs lpips: {[round(v,5) for v in d[-5:]]}")
    if b:
        e = b["lpips"]
        print(f"rollout: lpips {e[0]:.5f} -> {e[-1]:.5f}, best {min(e):.5f} "
              f"at epoch {b['epoch'][e.index(min(e))]}")
        print(f"rollout epochs improving on lpips: "
              f"{sum(1 for i in range(1, len(e)) if e[i] < e[i-1])}/{len(e)-1}")


if __name__ == "__main__":
    main()
