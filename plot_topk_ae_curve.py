#!/usr/bin/env python
"""Plot standard-MSE and top-k-loss CIFAR autoencoder training curves."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


STANDARD_RE = re.compile(
    r"\[AE\] epoch (\d+)/(\d+)\s+recon_mse=([0-9.eE+-]+)"
)
TOPK_RE = re.compile(
    r"\[AE\] epoch (\d+)/(\d+)\s+top([0-9.]+)%_mse=([0-9.eE+-]+)"
    r"\s+recon_mse=([0-9.eE+-]+)"
)


def standard_curve(path: Path):
    points = []
    for line in path.read_text().splitlines():
        if match := STANDARD_RE.search(line):
            points.append((int(match.group(1)), float(match.group(3))))
    return points


def topk_curves(path: Path):
    objective = []
    overall = []
    percent = None
    for line in path.read_text().splitlines():
        if match := TOPK_RE.search(line):
            epoch = int(match.group(1))
            percent = float(match.group(3))
            objective.append((epoch, float(match.group(4))))
            overall.append((epoch, float(match.group(5))))
    return percent, objective, overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("standard_log", type=Path)
    parser.add_argument("topk_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    standard = standard_curve(args.standard_log)
    percent, objective, overall = topk_curves(args.topk_log)
    if not standard or not objective:
        raise RuntimeError("could not find the expected AE loss records")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(
        [x for x, _ in standard], [y for _, y in standard],
        linewidth=2.4, label="Standard training: full MSE", color="#31688e",
    )
    ax.plot(
        [x for x, _ in overall], [y for _, y in overall],
        linewidth=2.4, label=f"Top-{percent:g}% training: full MSE",
        color="#35b779",
    )
    ax.plot(
        [x for x, _ in objective], [y for _, y in objective],
        linewidth=2.4, label=f"Top-{percent:g}% training objective",
        color="#d1495b",
    )
    ax.set(
        title="Plain 64D CIFAR-10 autoencoder: top-1% loss",
        xlabel="Epoch",
        ylabel="Mean squared error (log scale)",
        yscale="log",
    )
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
