#!/usr/bin/env python
"""Autoregressive rollout to an mp4: several starting contexts, one held action.

The generator predicts PIXELS but is conditioned on AE LATENTS of the preceding
frames, so refeeding means encoding each generated frame back through the AE
encoder -- exactly what the rollout branch of train_generator_vpt.py does, and
the conventions are copied from it deliberately:

  * the stored context is recency-scaled by sqrt(0.95**i) with the newest block
    at weight 1, so it must be DIVIDED by recw to be shifted and multiplied
    again on the way in. Getting this wrong quietly rescales the whole window.
  * the prediction is clamped to [-1, 1] before encoding, because that is the
    range the AE was trained on.
  * a fresh z is drawn EVERY step. That is the honest test: if z still carries
    action or context information, resampling it each frame is what makes the
    rollout argue with the held command, which is the failure this whole
    representation change was aimed at.

Held action is built through encode_live(), the same inference path a live
controller would use, from the act_norm stored in the generator checkpoint --
so this also exercises that contract rather than reimplementing the encoding.

The first ctx_frames of each clip are the REAL frames the context came from, so
the cut from real to generated is visible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from aag.ae import AutoEncoder
from aag.datasets import open_segments
from aag.generator import TransformerGenerator
from aag.vpt_actions import ACTION_NAMES, encode_live


def to_u8(x):
    """(N,3,H,W) in [-1,1] -> (N,H,W,3) uint8 BGR for cv2."""
    x = ((x.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).byte()
    return x.permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--assignment", type=Path, required=True,
                    help="supplies the starting contexts and act_norm")
    ap.add_argument("--ae", type=Path,
                    default=Path("/data/aag_results/results_vpt/ae_dcae_ch192_dim256_cont/"
                                 "checkpoints/ae_doom_frames_dcae_lpips_ch192_dim256_ep4.pt"))
    ap.add_argument("--cache", default="/opt/dlami/nvme/vpt_full")
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--press", default="W",
                    help="comma list of held controls, e.g. 'W' or 'W,shift'")
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--fixed-z", action="store_true",
                    help="draw z ONCE and hold it for the whole rollout instead "
                         "of resampling every frame. Diagnostic: if the rollout "
                         "collapses either way the cause is context drift, not "
                         "z churn.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gui-free", action="store_true", default=True,
                    help="only start from segments with no inventory frame")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    n_frames = int(round(args.seconds * args.fps))

    C = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    dim_z, CTX, DIM = C["dim_z"], C["ctx_frames"], C["ctx_dim"]
    act_dim = C.get("act_dim") or (C["input_dim"] - dim_z - CTX * DIM)
    if C.get("arch") != "transformer":
        raise SystemExit(f"expected a transformer checkpoint, got {C.get('arch')}")
    model = TransformerGenerator(dim_z=dim_z, ctx_frames=CTX, ctx_dim=DIM,
                                 act_dim=act_dim, d_model=C["d_model"],
                                 depth=C["depth"], heads=C["heads"], ch=C["ch"],
                                 image_size=C["image_size"]).to(dev).eval()
    model.load_state_dict(C["model_state_dict"])
    print(f"generator epoch {C['epoch']}  input_dim {C['input_dim']}  "
          f"act_dim {act_dim}  mse {C.get('mse'):.5f} lpips {C.get('lpips'):.5f}",
          flush=True)

    ac = torch.load(args.ae, map_location=dev, weights_only=False)
    ae = AutoEncoder(ac["latent_dim"], ch=ac["channels"],
                     architecture=ac["architecture"], image_size=ac["image_size"],
                     grid=ac.get("grid", 4)).to(dev).eval()
    sd = ac["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    ae.load_state_dict(sd)

    A = torch.load(args.assignment, map_location="cpu", weights_only=False)
    act_norm = C.get("act_norm") or A.get("act_norm")
    if act_norm is None:
        raise SystemExit("no act_norm in checkpoint or assignment -- cannot encode "
                         "a live action the way the generator was trained")
    cond_all, ci_all, fi_all = A["cond"], A["chunk"].numpy(), A["frame"].numpy()
    if cond_all.shape[1] != CTX * DIM:
        raise SystemExit(f"assignment cond is {cond_all.shape[1]} but the "
                         f"generator wants {CTX*DIM}")

    # pick starts, optionally avoiding inventory screens
    rng = np.random.default_rng(args.seed)
    pool = np.arange(len(ci_all))
    if args.gui_free:
        gui = np.load(f"{args.cache}/gui.npy", mmap_mode="r")
        keep = []
        for p in rng.permutation(pool)[:4000]:
            c, t = int(ci_all[p]), int(fi_all[p])
            if not np.asarray(gui[c, t - CTX:t + 1]).any():
                keep.append(p)
            if len(keep) >= args.starts * 4:
                break
        pool = np.array(keep) if keep else pool
    sel = rng.choice(pool, size=args.starts, replace=False)
    print(f"starts: particles {list(map(int, sel))}", flush=True)

    press = [s.strip() for s in args.press.split(",") if s.strip()]
    unknown = [p for p in press if p.lower() not in {n.lower() for n in ACTION_NAMES[:10]}]
    if unknown:
        raise SystemExit(f"unknown controls {unknown}; choose from {ACTION_NAMES[:10]}")
    av = encode_live(press, args.dx, args.dy, act_norm)
    print(f"held action: {press or '[none]'}  dx={args.dx} dy={args.dy}", flush=True)
    print(f"  encoded 12-d: {np.round(av, 3)}", flush=True)
    act = torch.from_numpy(av).float().to(dev).unsqueeze(0).repeat(args.starts, 1)

    recw = torch.tensor(np.sqrt(0.95 ** np.arange(CTX - 1, -1, -1)),
                        dtype=torch.float32, device=dev).view(1, CTX, 1)
    h = cond_all[sel].to(dev).float().view(args.starts, CTX, DIM) / recw

    # the real frames the context came from, so the real->generated cut is visible
    segs = open_segments(args.cache)
    real = np.stack([np.asarray(segs[int(ci_all[p])][int(fi_all[p]) - CTX:int(fi_all[p])])
                     for p in sel])                       # (S, CTX, 64, 64, 3)

    frames = []
    for k in range(CTX):                                  # real context, BGR
        frames.append(real[:, k][..., ::-1].copy())
    with torch.no_grad():
        z_held = torch.randn(args.starts, dim_z, device=dev)
        for _ in range(n_frames):
            z = z_held if args.fixed_z else torch.randn(args.starts, dim_z, device=dev)
            cc = torch.cat([z, (h * recw).reshape(args.starts, -1), act], 1)
            pred = model(cc).clamp(-1, 1)
            frames.append(to_u8(pred))
            hn = ae.enc(pred).view(args.starts, 1, DIM).float()
            h = torch.cat([h[:, 1:], hn], 1)

    S = args.scale
    tileH = frames[0].shape[1] * S
    tileW = frames[0].shape[2] * S
    W, H = tileW * args.starts, tileH
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"avc1"),
                         args.fps, (W, H))
    if not vw.isOpened():                                  # avc1 not always built in
        vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (W, H))
    if not vw.isOpened():
        raise SystemExit(f"could not open a writer for {args.out}")

    stats = []
    for i, fr in enumerate(frames):
        row = [cv2.resize(fr[s], (tileW, tileH), interpolation=cv2.INTER_NEAREST)
               for s in range(args.starts)]
        canvas = np.concatenate(row, 1)
        tag = "REAL" if i < CTX else f"+{i - CTX + 1}"
        cv2.putText(canvas, tag, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(np.ascontiguousarray(canvas))
        if i >= CTX:
            stats.append(fr.reshape(args.starts, -1).astype(np.float32).std(1))
    vw.release()

    st = np.stack(stats)                                   # (n_frames, starts)
    print(f"\nwrote {args.out}  {len(frames)} frames "
          f"({CTX} real + {n_frames} generated) at {args.fps} fps, {W}x{H}",
          flush=True)
    # not a quality claim -- just proof the rollout did not collapse to a
    # constant image or diverge, which is the one thing worth knowing before
    # a human spends time looking at it
    print(f"per-frame pixel std, mean over clips: first {st[0].mean():.1f}  "
          f"mid {st[len(st)//2].mean():.1f}  last {st[-1].mean():.1f}", flush=True)
    print(f"  min over all frames/clips {st.min():.1f}  max {st.max():.1f}",
          flush=True)


if __name__ == "__main__":
    main()
