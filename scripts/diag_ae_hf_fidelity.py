#!/usr/bin/env python
"""Did the ENCODER retain more detail, or is the DECODER hallucinating it?

This is the only question that matters for AAG, and no metric used so far
separates the two. In AAG, z is a transported copy of h_target = AE_enc(frame),
so z can only carry detail the ENCODER kept. Detail the DECODER invents from a
learned texture prior is worse than useless here: the assignment never had a
handle on it, it was never gaussianised, and at generation time the decoder will
paint generic texture rather than the texture that scene actually had.

An adversary raises high-frequency ENERGY either way, so hf-energy cannot tell
them apart -- a decoder painting plausible grass grain scores exactly like an
encoder that kept the real grain. What separates them is whether the recovered
high-frequency band is in the RIGHT PLACE:

    hf(x) = x - upsample(avgpool(x, 2))     the band a 48x AE loses first

  * CORRELATION of hf(recon) with hf(real), computed per frame then averaged.
    Encoder retention raises it. Hallucination does not -- invented texture is
    uncorrelated with the real texture it replaced, however plausible it looks.
  * MSE of the hf band, which falls only if the detail is both present AND
    positioned correctly.
  * ENERGY RATIO |hf(recon)| / |hf(real)|, for context: 1.0 means the right
    amount of detail, and it can sit at 1.0 while correlation is near 0.

The decisive pattern: energy ratio rising toward 1 while correlation stays flat
or falls means the decoder learned a texture prior and the latent carries no
more than before.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aag.ae import AutoEncoder
from aag.datasets import open_segments


def load_ae(path: Path, dev: str):
    c = torch.load(path, map_location=dev, weights_only=False)
    ae = AutoEncoder(c["latent_dim"], ch=c["channels"],
                     architecture=c["architecture"], image_size=c["image_size"],
                     grid=c.get("grid", 4)).to(dev).eval()
    sd = c["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    ae.load_state_dict(sd)
    return ae, c


def hf(x: torch.Tensor) -> torch.Tensor:
    blur = F.interpolate(F.avg_pool2d(x, 2), size=x.shape[-2:], mode="nearest")
    return x - blur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", type=Path,
                    default=Path("/data/aag_results/results_vpt/assign_12d_lag1/"
                                 "assignment.pt"))
    ap.add_argument("--old-ae", type=Path,
                    default=Path("/data/aag_results/results_vpt/"
                                 "ae_dcae_ch192_dim256_cont/checkpoints/"
                                 "ae_doom_frames_dcae_lpips_ch192_dim256_ep4.pt"))
    ap.add_argument("--new-ae", type=Path, required=True)
    ap.add_argument("--cache", default="/opt/dlami/nvme/vpt_full")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--gui-free", action="store_true", default=True)
    args = ap.parse_args()

    dev = "cuda"
    models = {}
    for nm, p in (("old", args.old_ae), ("new", args.new_ae)):
        m, c = load_ae(p, dev)
        models[nm] = m
        print(f"{nm}: {p.name}  epochs={c.get('epochs')} "
              f"gan_weight={c.get('gan_weight')}")

    A = torch.load(args.assignment, map_location="cpu", weights_only=False)
    ci, fi = A["chunk"].numpy(), A["frame"].numpy()
    segs = open_segments(args.cache)
    rng = np.random.default_rng(0)
    cand = rng.permutation(len(ci))
    if args.gui_free:
        gui = np.load(f"{args.cache}/gui.npy", mmap_mode="r")
        cand = cand[~np.asarray(gui[ci[cand], fi[cand]]).astype(bool)]
    sel = cand[:args.n]

    acc = {nm: {"corr": 0.0, "mse": 0.0, "energy": 0.0} for nm in models}
    real_energy = 0.0
    with torch.no_grad():
        for i in range(0, len(sel), 256):
            b = sel[i:i + 256]
            x = np.stack([np.asarray(segs[int(ci[p])][int(fi[p])]) for p in b])
            x = (torch.from_numpy(x).permute(0, 3, 1, 2).float()
                 .div_(127.5).sub_(1.0).to(dev))
            hx = hf(x)
            hxf = hx.flatten(1)
            real_energy += float(hx.abs().mean()) * len(b)
            for nm, m in models.items():
                r = m.dec(m.enc(x)).clamp(-1, 1)
                hr = hf(r)
                hrf = hr.flatten(1)
                # per-frame Pearson correlation of the high-frequency bands
                a = hxf - hxf.mean(1, keepdim=True)
                c_ = hrf - hrf.mean(1, keepdim=True)
                corr = ((a * c_).sum(1) /
                        (a.norm(dim=1) * c_.norm(dim=1) + 1e-8))
                acc[nm]["corr"] += float(corr.sum())
                acc[nm]["mse"] += float(F.mse_loss(hr, hx)) * len(b)
                acc[nm]["energy"] += float(hr.abs().mean()) * len(b)
    n = len(sel)
    real_energy /= n

    print(f"\nhigh-frequency band, {n:,} gui-free VPT frames")
    print(f"  real |hf| = {real_energy:.5f}\n")
    print(f"  {'model':6s} {'hf corr':>9s} {'hf mse':>10s} {'|hf| ratio':>11s}")
    for nm in ("old", "new"):
        a = acc[nm]
        print(f"  {nm:6s} {a['corr']/n:9.4f} {a['mse']/n:10.6f} "
              f"{a['energy']/n/real_energy:11.3f}")

    co, cn = acc["old"]["corr"] / n, acc["new"]["corr"] / n
    mo, mn = acc["old"]["mse"] / n, acc["new"]["mse"] / n
    print(f"\n  corr   new-old = {cn - co:+.4f}")
    print(f"  hf mse new/old  = {mn / mo:.3f}x")
    print()
    if cn > co + 0.01 and mn < mo:
        print("  ENCODER RETAINED MORE: the recovered detail is correlated with")
        print("  the real detail and better placed, so the latent carries more")
        print("  and the assignment has a handle on it.")
    elif acc["new"]["energy"] > acc["old"]["energy"] and cn <= co + 0.01:
        print("  DECODER HALLUCINATION: more high-frequency energy, but it is no")
        print("  better correlated with the real high-frequency band. The decoder")
        print("  learned a texture prior; the LATENT carries no more than before,")
        print("  so nothing downstream of the encoder -- assignment, generator --")
        print("  gains anything, and z still has no handle on this detail.")
    else:
        print("  MIXED / INCONCLUSIVE: read the three columns directly.")


if __name__ == "__main__":
    main()
