#!/usr/bin/env python
"""Plot one finetune run's curves, including the drift profile and GAN terms.

Five panels, one per scale -- never a dual axis. The per-step panel is the one
that matters most: it is the drift profile, and a finetune that only improves
step 0 has not bought persistence no matter what the aggregate says.

Baselines are drawn as dashed reference lines rather than as another series,
because they are single numbers from a different run, not curves.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"


def style(ax, title, xlabel="epoch"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=7)
    ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def end_label(ax, xs, ys, color, dy=0):
    ax.annotate(f"{ys[-1]:.5f}", xy=(xs[-1], ys[-1]), xytext=(5, dy),
                textcoords="offset points", ha="left", va="center",
                fontsize=8, color=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", type=Path, required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--base-mse", type=float, default=0.00809)
    ap.add_argument("--base-lpips", type=float, default=0.09888)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    c = json.loads(args.curve.read_text())
    ep = c["epoch"]

    fig, axes = plt.subplots(1, 5, figsize=(21.5, 4.0), facecolor=SURFACE)

    style(axes[0], "single-step train MSE")
    axes[0].plot(ep, c["mse"], color=C1, lw=2, label="this run")
    axes[0].axhline(args.base_mse, color=INK2, lw=1, ls=(0, (4, 3)))
    axes[0].annotate(f"pre-finetune {args.base_mse:.5f}",
                     xy=(0.02, args.base_mse), xycoords=("axes fraction", "data"),
                     xytext=(0, -11), textcoords="offset points", fontsize=7.5,
                     color=INK2)
    end_label(axes[0], ep, c["mse"], C1)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")

    style(axes[1], "single-step train LPIPS")
    axes[1].plot(ep, c["lpips"], color=C1, lw=2, label="this run")
    axes[1].axhline(args.base_lpips, color=INK2, lw=1, ls=(0, (4, 3)))
    axes[1].annotate(f"pre-finetune {args.base_lpips:.5f}",
                     xy=(0.02, args.base_lpips), xycoords=("axes fraction", "data"),
                     xytext=(0, -11), textcoords="offset points", fontsize=7.5,
                     color=INK2)
    end_label(axes[1], ep, c["lpips"], C1)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")

    style(axes[2], "sequence MSE (mean over 8 rollout steps)")
    sq = [v for v in c.get("seq_mse", []) if v]
    if sq:
        axes[2].plot(ep[-len(sq):], sq, color=C3, lw=2, label="seq_mse")
        end_label(axes[2], ep[-len(sq):], sq, C3)
        axes[2].legend(frameon=False, fontsize=8, labelcolor=INK2,
                       loc="upper right")

    style(axes[3], "adversary: g (generator) and d (critic)")
    if c.get("g"):
        g, d = c["g"], c["d"]
        e2 = ep[-len(g):]
        axes[3].plot(e2, g, color=C2, lw=2, label="g  generator loss")
        axes[3].plot(e2, d, color=C1, lw=2, label="d  critic loss")
        axes[3].axhline(0.693, color=INK2, lw=1, ls=(0, (4, 3)))
        axes[3].annotate("0.693 = critic at chance", xy=(0.02, 0.693),
                         xycoords=("axes fraction", "data"), xytext=(0, 5),
                         textcoords="offset points", fontsize=7.5, color=INK2)
        axes[3].legend(frameon=False, fontsize=8, labelcolor=INK2,
                       loc="center right")
    else:
        axes[3].text(0.5, 0.5, "no adversary in this run", ha="center",
                     va="center", transform=axes[3].transAxes, color=INK2,
                     fontsize=9)

    style(axes[4], "drift profile: MSE by rollout step", xlabel="rollout step")
    ps = [p for p in c.get("per_step", []) if p and any(p)]
    if ps:
        xs = list(range(len(ps[0])))
        axes[4].plot(xs, ps[0], color=C2, lw=2, label=f"first ep with rollout")
        if len(ps) > 1:
            axes[4].plot(xs, ps[-1], color=C3, lw=2, label=f"latest ep")
        axes[4].legend(frameon=False, fontsize=8, labelcolor=INK2,
                       loc="upper left")

    if args.title:
        fig.suptitle(args.title, color=INK, fontsize=12, x=0.005, ha="left",
                     y=0.99)
    fig.text(0.005, 0.02,
             "TRAIN losses, no validation split. LPIPS is a FIDELITY metric "
             "against the true frame, so an adversary is expected to raise it "
             "while improving realism -- a rising LPIPS is not by itself "
             "evidence of degradation. The drift profile is the panel that "
             "answers whether persistence improved.",
             color=INK2, fontsize=8, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=SURFACE)
    print(f"wrote {args.out}")
    print(f"epochs {ep[0]}..{ep[-1]}  mse {c['mse'][0]:.5f}->{c['mse'][-1]:.5f}  "
          f"lpips {c['lpips'][0]:.5f}->{c['lpips'][-1]:.5f}")
    if sq:
        print(f"seq_mse {sq[0]:.5f}->{sq[-1]:.5f}")
    if ps and len(ps) > 1:
        print(f"drift step0 {ps[0][0]:.4f}->{ps[-1][0]:.4f}  "
              f"step7 {ps[0][-1]:.4f}->{ps[-1][-1]:.4f}  "
              f"ratio {ps[0][-1]/ps[0][0]:.2f}->{ps[-1][-1]/ps[-1][0]:.2f}")


if __name__ == "__main__":
    main()
