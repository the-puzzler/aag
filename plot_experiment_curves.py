#!/usr/bin/env python
"""Plot CIFAR-10 training and Gaussian-assignment curves from a run log."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DIM_RE = re.compile(r"\[dim (\d+)\] training autoencoder")
AE_RE = re.compile(r"\[AE\] epoch (\d+)/(\d+)\s+recon_mse=([0-9.eE+-]+)")
ASSIGN_RE = re.compile(
    r"\[assign\] step\s+(\d+)/(\d+)\s+max_proj_W2=([0-9.eE+-]+)"
)
DEC_RE = re.compile(
    r"\[dec\] epoch (\d+)/(\d+)\s+train_mse=([0-9.eE+-]+)\s+"
    r"val_mse=([0-9.eE+-]+)"
)


def parse_log(path: Path):
    curves = {
        "ae": defaultdict(list),
        "assignment": defaultdict(list),
        "decoder_train": defaultdict(list),
        "decoder_val": defaultdict(list),
    }
    dim = None
    for line in path.read_text().splitlines():
        if match := DIM_RE.search(line):
            dim = int(match.group(1))
            continue
        if dim is None:
            continue
        if match := AE_RE.search(line):
            curves["ae"][dim].append((int(match.group(1)), float(match.group(3))))
        elif match := ASSIGN_RE.search(line):
            curves["assignment"][dim].append(
                (int(match.group(1)), float(match.group(3)))
            )
        elif match := DEC_RE.search(line):
            epoch = int(match.group(1))
            curves["decoder_train"][dim].append((epoch, float(match.group(3))))
            curves["decoder_val"][dim].append((epoch, float(match.group(4))))
    return curves


def draw(curves, output: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = plt.cm.viridis_r([0.05, 0.25, 0.45, 0.65, 0.85])
    dims = sorted(curves["ae"])
    color_for = dict(zip(dims, colors))

    for dim in dims:
        points = curves["ae"][dim]
        axes[0].plot(
            [x for x, _ in points], [y for _, y in points],
            label=f"d={dim}", color=color_for[dim], linewidth=2,
        )
    axes[0].set(
        title="Autoencoder training",
        xlabel="Epoch",
        ylabel="Reconstruction MSE",
        yscale="log",
    )
    axes[0].legend(ncols=2, fontsize=9)

    for dim in dims:
        points = curves["assignment"][dim]
        axes[1].plot(
            [x for x, _ in points], [y for _, y in points],
            marker="o", markersize=3, label=f"d={dim}",
            color=color_for[dim], linewidth=2,
        )
    axes[1].set(
        title="Persistent Gaussian assignment",
        xlabel="Assignment step",
        ylabel=r"Max selected projection $W_2^2$",
        yscale="log",
    )

    for dim in dims:
        train = curves["decoder_train"][dim]
        val = curves["decoder_val"][dim]
        axes[2].plot(
            [x for x, _ in train], [y for _, y in train],
            color=color_for[dim], linewidth=2, label=f"d={dim} train",
        )
        axes[2].plot(
            [x for x, _ in val], [y for _, y in val],
            color=color_for[dim], linewidth=1.6, linestyle="--",
            label=f"d={dim} val",
        )
    axes[2].set(
        title="Direct decoder training",
        xlabel="Epoch",
        ylabel="Pair MSE",
        yscale="log",
    )
    axes[2].legend(ncols=2, fontsize=8)

    fig.suptitle("CIFAR-10 persistent Gaussian assignment experiment", fontsize=15)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(parse_log(args.log), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
