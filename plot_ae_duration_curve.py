#!/usr/bin/env python
"""Compare autoencoder training curves from runs of different duration."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


LOSS_RE = re.compile(r"\[AE\] epoch (\d+)/(\d+)\s+recon_mse=([0-9.eE+-]+)")


def read_curve(path: Path):
    points = []
    total_epochs = None
    for line in path.read_text().splitlines():
        if match := LOSS_RE.search(line):
            points.append((int(match.group(1)), float(match.group(3))))
            total_epochs = int(match.group(2))
    if not points:
        raise RuntimeError(f"no reconstruction losses found in {path}")
    return total_epochs, points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("short_log", type=Path)
    parser.add_argument("long_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    short_epochs, short = read_curve(args.short_log)
    long_epochs, long = read_curve(args.long_log)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(
        [x for x, _ in short], [y for _, y in short], linewidth=2.4,
        label=f"{short_epochs}-epoch run", color="#31688e",
    )
    ax.plot(
        [x for x, _ in long], [y for _, y in long], linewidth=2.4,
        label=f"{long_epochs}-epoch run", color="#35b779",
    )
    ax.set(
        title="Plain 64D CIFAR-10 autoencoder training duration",
        xlabel="Epoch",
        ylabel="Training reconstruction MSE",
        yscale="log",
    )
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
