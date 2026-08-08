#!/usr/bin/env python
"""Plot the CIFAR spatial-autoencoder rate-distortion curve."""

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
    dims = [64, 128, 256, 512]
    psnr = [data["models"][str(dim)]["psnr_db"] for dim in dims]
    compression = [data["models"][str(dim)]["compression_ratio"] for dim in dims]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(dims, psnr, marker="o", markersize=8, linewidth=2.5, color="#31688e")
    for dim, score, ratio in zip(dims, psnr, compression):
        ax.annotate(
            f"{ratio:g}× compression\n{score:.2f} dB",
            (dim, score), xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=10,
        )
    ax.axvspan(220, 300, color="#35b779", alpha=0.12, label="practical elbow")
    ax.set(
        title="CIFAR-10 spatial AE rate–distortion curve",
        xlabel="Latent values",
        ylabel="Test PSNR (dB)",
        xticks=dims,
    )
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
