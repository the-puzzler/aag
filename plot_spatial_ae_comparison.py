#!/usr/bin/env python
"""Plot train/test curves for the spatial 64- and 512-value CIFAR AEs."""

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
    colors = {"64": "#31688e", "512": "#35b779"}
    for dim in ("64", "512"):
        history = data["models"][dim]["history"]
        epochs = [row["epoch"] for row in history]
        ax.plot(
            epochs, [row["train_mse"] for row in history],
            color=colors[dim], linewidth=2.3, label=f"Spatial {dim}: train",
        )
        ax.plot(
            epochs, [row["test_mse"] for row in history],
            color=colors[dim], linewidth=2.3, linestyle="--",
            label=f"Spatial {dim}: test",
        )
    baseline = data["plain_64_baseline"]["final_test_mse"]
    ax.axhline(
        baseline, color="#d1495b", linewidth=2, linestyle=":",
        label=f"Flat 64 baseline: test {baseline:.4f}",
    )
    ax.set(
        title="CIFAR-10 spatial-residual autoencoder comparison",
        xlabel="Epoch",
        ylabel="Reconstruction MSE (log scale)",
        yscale="log",
        xticks=range(1, 11),
    )
    ax.legend(ncols=2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
