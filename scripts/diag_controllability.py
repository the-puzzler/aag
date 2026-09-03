#!/usr/bin/env python
"""Does the model actually RESPOND to the controls? The deliverable test.

Every rollout so far held W, which shows only that *something* happens. This
holds the START, the CONTEXT and the ENTIRE z SEQUENCE fixed and varies only the
action, so any difference between the outputs is attributable to the action and
nothing else. Sharing the z sequence across variants is the whole trick: without
it, two rollouts differ because they were handed different noise, and no
conclusion about controllability is available.

Reports, per variant and per step, the mean |pixel| difference from a reference
variant (default: no keys, no mouse). A model that ignores its action input
produces zero. It also reports the W-vs-S pair specifically, because those are
opposites and should diverge more than either does from idle, and dx+ vs dx-,
which is the marginal that has never decorrelated.

Also renders one video per start with the variants tiled left to right, so the
numbers can be checked against what is actually on screen.
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
from aag.vpt_rollout import ContextWindow
from aag.vpt_actions import ACTION_NAMES, encode_live

# (label, pressed-set, dx, dy)
VARIANTS = [
    ("idle", set(), 0.0, 0.0),
    ("W", {"W"}, 0.0, 0.0),
    ("S", {"S"}, 0.0, 0.0),
    ("A", {"A"}, 0.0, 0.0),
    ("D", {"D"}, 0.0, 0.0),
    ("dx+30", set(), 30.0, 0.0),
    ("dx-30", set(), -30.0, 0.0),
    ("attack", {"attack"}, 0.0, 0.0),
]


def to_u8(x):
    x = ((x.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).byte()
    return x.permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--assignment", type=Path, required=True)
    ap.add_argument("--ae", type=Path,
                    default=Path("/data/aag_results/results_vpt/ae_dcae_ch192_dim256_cont/"
                                 "checkpoints/ae_doom_frames_dcae_lpips_ch192_dim256_ep4.pt"))
    ap.add_argument("--cache", default="/opt/dlami/nvme/vpt_full")
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--min-sky", type=float, default=0.5)
    ap.add_argument("--min-alt", type=float, default=62.0)
    ap.add_argument("--scan", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(args.seed)
    C = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    dim_z, CTX, DIM = C["dim_z"], C["ctx_frames"], C["ctx_dim"]
    model = TransformerGenerator(dim_z=dim_z, ctx_frames=CTX, ctx_dim=DIM,
                                 act_dim=C.get("act_dim", 12), d_model=C["d_model"],
                                 depth=C["depth"], heads=C["heads"], ch=C["ch"],
                                 image_size=C["image_size"]).to(dev).eval()
    model.load_state_dict(C["model_state_dict"])
    print(f"generator epoch {C['epoch']}  act_dim {C.get('act_dim')}  "
          f"action_lag {C.get('action_lag')}  seq_len {C.get('seq_len')}", flush=True)

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
    cond_all, ci, fi = A["cond"], A["chunk"].numpy(), A["frame"].numpy()
    segs = open_segments(args.cache)
    gui = np.load(f"{args.cache}/gui.npy", mmap_mode="r")
    pose = np.load(f"{args.cache}/pose.npy", mmap_mode="r")

    rng = np.random.default_rng(args.seed)
    scored = []
    for p in rng.permutation(len(ci))[: args.scan]:
        c, t = int(ci[p]), int(fi[p])
        if np.asarray(gui[c, t - CTX:t + 1]).any():
            continue
        if float(pose[c, t, 3]) < args.min_alt:
            continue
        f = np.asarray(segs[c][t]).astype(np.float32)
        top = f[: f.shape[0] // 4]
        sky = float(((top[..., 2] > top[..., 0] + 8) & (top.mean(2) > 110)).mean())
        if sky >= args.min_sky:
            scored.append((sky, float(f.mean()), int(p)))
    scored.sort(key=lambda r: (-r[0], -r[1]))
    sel = [p for _, _, p in scored[: args.starts]]
    if len(sel) < args.starts:
        raise SystemExit(f"only {len(sel)} outdoor starts found")
    print(f"starts: {sel}", flush=True)

    # the checkpoint says which context it was trained with; honour it
    win = ContextWindow(C, ae, dev, CTX, DIM, image_size=C["image_size"])
    print(f"  {win.describe()}", flush=True)
    V = len(VARIANTS)
    acts = torch.stack([torch.from_numpy(encode_live(pr, dx, dy, act_norm))
                        for _, pr, dx, dy in VARIANTS]).float().to(dev)   # (V,12)

    for si, p in enumerate(sel):
        # ONE z sequence, reused across every variant -- this is what makes the
        # difference attributable to the action
        zs = [torch.randn(1, dim_z, device=dev, generator=None) for _ in range(args.steps)]
        c, t = int(ci[p]), int(fi[p])
        ctx_real = (torch.from_numpy(np.asarray(segs[c][t - CTX:t]).copy())
                    .permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
                    .unsqueeze(0).to(dev))
        cond_row = cond_all[p:p + 1].to(dev).float()
        frames = np.zeros((args.steps, V, 64, 64, 3), np.uint8)
        with torch.no_grad():
            for vi in range(V):
                # re-initialise the window per variant so every variant starts
                # from an identical context, as it must for the comparison
                win.init(cond_row if win.mode == "ae_latent" else None,
                         ctx_real if win.mode != "ae_latent" else None)
                a = acts[vi:vi + 1]
                for s in range(args.steps):
                    inp = torch.cat([zs[s], win.vector(), a], 1)
                    pred = model(inp).clamp(-1, 1)
                    frames[s, vi] = to_u8(pred)[0]
                    win.push(pred)

        f = frames.astype(np.float32)
        ref = f[:, 0]                                       # idle
        print(f"\n=== start {p} : mean |pixel| difference from `idle` ===", flush=True)
        print(f"  {'variant':8s} " + " ".join(f"+{s+1:<4d}" for s in
                                              (0, 4, 9, 19, args.steps - 1)), flush=True)
        for vi, (lab, _, _, _) in enumerate(VARIANTS):
            if vi == 0:
                continue
            d = np.abs(f[:, vi] - ref).mean((1, 2, 3))
            cols = " ".join(f"{d[s]:5.2f}" for s in (0, 4, 9, 19, args.steps - 1))
            print(f"  {lab:8s} {cols}", flush=True)
        iW, iS = 1, 2
        idp, idm = 5, 6
        dws = np.abs(f[:, iW] - f[:, iS]).mean((1, 2, 3))
        ddx = np.abs(f[:, idp] - f[:, idm]).mean((1, 2, 3))
        print(f"  {'W vs S':8s} " + " ".join(f"{dws[s]:5.2f}" for s in
                                             (0, 4, 9, 19, args.steps - 1)), flush=True)
        print(f"  {'dx+/-':8s} " + " ".join(f"{ddx[s]:5.2f}" for s in
                                            (0, 4, 9, 19, args.steps - 1)), flush=True)

        # AUTONOMOUS MOTION. If `idle` moves as much frame-to-frame as `W` does,
        # the model is walking regardless of its input and the action is only
        # modulating a dynamic it would run anyway -- which is exactly what "idle
        # looked like W" means. W is pressed on 33% of frames and the corpus is
        # mostly moving footage, so "moving forward" is the prior the model would
        # fall back on. Compared against the action-induced difference: an action
        # that changes the frame far less than the variant changes on its own is
        # a weak modulator, not a control.
        print(f"\n  --- autonomous motion: mean |frame_(s+1) - frame_s| WITHIN "
              f"each variant ---", flush=True)
        self_mo = np.abs(np.diff(f, axis=0)).mean((2, 3, 4))      # (steps-1, V)
        for vi, (lab, _, _, _) in enumerate(VARIANTS):
            sm = self_mo[:, vi]
            vs_idle = np.abs(f[:, vi] - ref).mean((1, 2, 3)) if vi else None
            extra = (f"   action effect / own motion = "
                     f"{vs_idle.mean()/max(sm.mean(),1e-9):.2f}" if vi else
                     "   (this is the reference)")
            print(f"  {lab:8s} own motion {sm.mean():5.2f}{extra}", flush=True)
        print(f"  a real consecutive frame step is 7.25 on this scale "
              f"(4.12 near-still, 14.80 while turning)", flush=True)

        S = args.scale
        tw, th = 64 * S, 64 * S
        out = args.out_prefix.with_name(args.out_prefix.name + f"_start{si}.mp4")
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (tw * V, th + 18))
        if not vw.isOpened():
            raise SystemExit(f"cannot open writer for {out}")
        for s in range(args.steps):
            row = [cv2.resize(frames[s, vi], (tw, th), interpolation=cv2.INTER_NEAREST)
                   for vi in range(V)]
            canvas = np.zeros((th + 18, tw * V, 3), np.uint8)
            canvas[18:] = np.concatenate(row, 1)
            for vi, (lab, _, _, _) in enumerate(VARIANTS):
                cv2.putText(canvas, lab, (vi * tw + 4, 13), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"+{s+1}", (tw * V - 34, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            vw.write(np.ascontiguousarray(canvas))
        vw.release()
        print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
