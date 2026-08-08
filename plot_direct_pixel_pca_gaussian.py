#!/usr/bin/env python
"""Plot train/validation curves for direct pixel generators at ranks 8 and 12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    colors = {"8": "#31688e", "12": "#35b779"}
    for rank in ("8", "12"):
        history = data["models"][rank]["history"]
        epochs = [row["epoch"] for row in history]
        ax.plot(
            epochs, [row["train_mse"] for row in history],
            color=colors[rank], linewidth=2.3, label=f"rank {rank}: train",
        )
        ax.plot(
            epochs, [row["val_mse"] for row in history],
            color=colors[rank], linewidth=2.3, linestyle="--",
            label=f"rank {rank}: held out",
        )
    ax.set(
        title="Direct CIFAR pixel generators from PCA–Gaussian coordinates",
        xlabel="Epoch",
        ylabel="Pixel MSE",
    )
    ax.legend(ncols=2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
