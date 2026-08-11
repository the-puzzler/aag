#!/usr/bin/env python
"""STANDARD assignment diagnostic. One figure, everything about an assignment.

  python plots/plot_assignment.py <assignment.pt> [more.pt ...] --out fig.png

Panels (one row per assignment):
  1  transport objective per step, with the N(0,I) noise floor band
     -> the assignment is done when the curve enters the band
  2  first two coords vs a reference N(0,I) cloud
  3  marginal of coordinate 0 vs N(0,1)
  4  radial ||z||^2 vs chi^2(d)
  5  transport displacement ||z - z_whitened||  -> locality cost
Plus, if the file carries a conditional curve, an independence-ratio panel.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
from scipy import stats
from gga.gaussianize import whiten
from gga.diagnostics import transport_objective_floor

ap = argparse.ArgumentParser()
ap.add_argument("assignments", nargs="+", type=Path)
ap.add_argument("--labels", nargs="*", default=None)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--scatter-n", type=int, default=3000)
a = ap.parse_args()
labels = a.labels or [p.parent.name or p.stem for p in a.assignments]

loaded = []
for p, lab in zip(a.assignments, labels):
    D = torch.load(p, map_location="cpu", weights_only=False)
    z = D["z"].float()
    curve = D.get("curve", {})
    h = D.get("h")
    z0 = whiten(h.float().cuda())[0].cpu() if h is not None else None
    loaded.append((lab, z, curve, z0))

d = loaded[0][1].shape[1]
floor, floor_sd = transport_objective_floor(loaded[0][1].shape[0], d)
has_ratio = any("ratio" in c for _, _, c, _ in loaded)
ncol = 6 if has_ratio else 5
fig, axes = plt.subplots(len(loaded), ncol, figsize=(3.5 * ncol, 3.6 * len(loaded)), squeeze=False)
g = torch.Generator().manual_seed(0)

for r, (lab, z, curve, z0) in enumerate(loaded):
    ax = axes[r]
    ax[0].axhspan(floor - floor_sd, floor + floor_sd, color="#4caf93", alpha=.25)
    ax[0].axhline(floor, color="#2e7d5b", ls="--", lw=1.3, label=f"N(0,I) floor {floor:.5f}")
    if "conv_score" in curve:
        w = max(1, len(curve["conv_score"]) // 200)
        y = np.convolve(curve["conv_score"], np.ones(w) / w, mode="valid")
        ax[0].plot(np.array(curve["conv_step"])[w - 1:], y, color="#2b7bba", lw=1.2)
    ax[0].set_yscale("log"); ax[0].set_title("transport objective"); ax[0].legend(fontsize=7)
    ax[0].set_xlabel("step"); ax[0].set_ylabel(lab, fontsize=9)

    idx = torch.randperm(z.shape[0], generator=g)[:a.scatter_n]
    ref = torch.randn(a.scatter_n, 2, generator=g)
    ax[1].scatter(ref[:, 0], ref[:, 1], s=3, alpha=.22, color="gray", label="N(0,I)")
    ax[1].scatter(z[idx, 0], z[idx, 1], s=3, alpha=.35, color="#e07b39", label="assigned")
    ax[1].set_title("z[0] vs z[1]"); ax[1].legend(fontsize=7)

    ax[2].hist(z[:, 0].numpy(), bins=80, density=True, color="#2b7bba", alpha=.75)
    xs = np.linspace(-4, 4, 200); ax[2].plot(xs, stats.norm.pdf(xs), "k", lw=1.4)
    ax[2].set_title("marginal z[0] vs N(0,1)")

    r2 = (z ** 2).sum(1).numpy()
    ax[3].hist(r2, bins=80, density=True, color="#4caf93", alpha=.75)
    xs = np.linspace(r2.min(), r2.max(), 300)
    ax[3].plot(xs, stats.chi2.pdf(xs, df=d), "k", lw=1.4)
    ax[3].set_title(f"radial vs chi^2({d})")

    if z0 is not None:
        disp = (z - z0).norm(dim=1).numpy()
        ax[4].hist(disp, bins=80, density=True, color="#b05fc0", alpha=.8)
        ax[4].axvline(disp.mean(), color="k", ls="--", lw=1.3, label=f"mean {disp.mean():.2f}")
        ax[4].legend(fontsize=7)
    elif "displacement" in curve:
        ax[4].plot(curve["step"], curve["displacement"], color="#b05fc0", lw=1.8)
        ax[4].set_xlabel("step")
    ax[4].set_title("transport displacement")

    if has_ratio:
        if "ratio" in curve:
            ax[5].axhline(1.0, color="k", ls="--", lw=1.4, label="1.0 = independent")
            ax[5].plot(curve["step"], curve["ratio"], color="#08519c", lw=1.6)
            ax[5].legend(fontsize=7); ax[5].set_xlabel("step")
        else:
            ax[5].text(.5, .5, "no conditional steps", ha="center", va="center",
                       transform=ax[5].transAxes, fontsize=9, color="gray")
        ax[5].set_title("independence ratio")
    for x in ax: x.grid(alpha=.3)

plt.tight_layout(); plt.savefig(a.out, dpi=135)
print("saved:", a.out)
for lab, z, curve, z0 in loaded:
    bits = [f"{lab}"]
    if z0 is not None: bits.append(f"displacement {(z - z0).norm(dim=1).mean():.3f}")
    if "ratio" in curve: bits.append(f"independence ratio {curve['ratio'][-1]:.3f}")
    if "proj_over_gauss" in curve: bits.append(f"proj/gauss {curve['proj_over_gauss'][-1]:.3f}")
    print("  " + "  |  ".join(bits))
