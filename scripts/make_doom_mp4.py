#!/usr/bin/env python
"""Render generated Doom clips as an mp4 collage so they can be judged as video.

Frame-strip PNGs hide temporal artefacts -- flicker, drift, objects popping in
and out -- which are exactly the failure modes a 16-frame clip can have. Each
tile is one generated clip; the grid plays them in parallel, looped, upscaled
with nearest-neighbour so 64x64 detail stays crisp.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from aag.ae import AutoEncoder, AdaLNDecoder3d, ResidualDecoder3d


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True)
    ap.add_argument("--frame-ae", required=True)
    ap.add_argument("--val-cache", default="/data/doom/cache_val")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--loops", type=int, default=6)
    ap.add_argument("--real-strip", action="store_true",
                    help="prepend a row of REAL clips as a reference")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"

    n = a.rows * a.cols
    segs = np.load(f"{a.val_cache}/segments.npy", mmap_mode="r")
    torch.manual_seed(a.seed)
    idx = torch.randperm(len(np.load(f"{a.val_cache}/labels.npy")))[:n].numpy()
    real = torch.from_numpy(np.ascontiguousarray(segs[idx])).to(dev)      # (n,T,H,W,3)
    real_f = real.permute(0, 1, 4, 2, 3).float().div_(255.)               # (n,T,3,H,W)

    fk = torch.load(a.frame_ae, map_location=dev, weights_only=False)
    fae = AutoEncoder(fk["latent_dim"], ch=fk["channels"], architecture=fk["architecture"],
                      image_size=fk["image_size"]).to(dev).eval()
    fae.load_state_dict(fk["model_state_dict"])
    cond = fae.enc(real_f[:, 0] * 2 - 1)                # condition on the val clip's first frame

    g = torch.load(a.generator, map_location=dev, weights_only=False)
    mode = g.get("cond_mode", "adaln")
    if mode == "none":
        m = ResidualDecoder3d(g["dim_z"], ch=64, image_size=64, frames=16).to(dev).eval()
    else:
        m = AdaLNDecoder3d(g["dim_z"], g["cond_dim"], ch=64, image_size=64, frames=16).to(dev).eval()
    m.load_state_dict(g["model_state_dict"])
    z = torch.randn(n, g["dim_z"], device=dev)          # FRESH z
    gen = (m(z) if mode == "none" else m(z, cond)).clamp(-1, 1).add(1).div(2)
    gen = gen.permute(0, 2, 3, 4, 1).cpu().numpy()      # (n,T,H,W,3)
    rl = real_f.permute(0, 1, 3, 4, 2).cpu().numpy()

    T, H, W = gen.shape[1], gen.shape[2], gen.shape[3]
    s, pad = a.scale, 2
    rows = a.rows + (1 if a.real_strip else 0)
    gutter = 96 if a.real_strip else 0          # left margin for the row labels
    canvas_h = rows * (H * s + pad) + pad
    canvas_w = gutter + a.cols * (W * s + pad) + pad
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (canvas_w, canvas_h))

    def tile(img):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = cv2.resize(img, (W * s, H * s), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for _ in range(a.loops):
        for t in range(T):
            canvas = np.full((canvas_h, canvas_w, 3), 24, np.uint8)
            for r in range(rows):
                for c in range(a.cols):
                    k = (r - (1 if a.real_strip else 0)) * a.cols + c
                    if a.real_strip and r == 0:
                        src = rl[c, t]
                    elif 0 <= k < n:
                        src = gen[k, t]
                    else:
                        continue
                    y0 = pad + r * (H * s + pad)
                    x0 = gutter + pad + c * (W * s + pad)
                    canvas[y0:y0 + H * s, x0:x0 + W * s] = tile(src)
            if a.real_strip:
                # row labels in the left gutter: 'real' beside row 0,
                # 'generated' centred over the remaining rows
                y_real = pad + (H * s) // 2 + 5
                cv2.putText(canvas, "real", (10, y_real), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (230, 230, 230), 1, cv2.LINE_AA)
                y_gen = pad + (H * s + pad) + (a.rows * (H * s + pad)) // 2 + 5
                cv2.putText(canvas, "generated", (10, y_gen), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (230, 230, 230), 1, cv2.LINE_AA)
            vw.write(canvas)
    vw.release()
    print(f"wrote {a.out}  ({canvas_w}x{canvas_h}, {T} frames x {a.loops} loops @ {a.fps}fps)")


if __name__ == "__main__":
    main()
