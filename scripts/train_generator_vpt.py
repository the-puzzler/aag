#!/usr/bin/env python
"""Action-conditioned frame generator for the Minecraft (VPT) world model.

    z ~ N(0,I) (dim) + h_{t-24..t-1} (24*dim) + one-hot action (81)  ->  frame_t pixels

Direct-to-pixel with MSE+LPIPS, matching train_generator_doom.py and every other
generator in this project. z is the frozen assignment's Gaussian coordinate for
that particle, so at inference a fresh z ~ N(0,I) pairs with any context -- valid
only to the extent the conditional assignment drove the independence ratios
toward 1.0.

Differs from the Doom version in three places, all forced by the VPT cache:
  * frames come from open_segments (102 shards), not a single segments.npy
  * 81 actions, not 18
  * targets stay on the CPU in pinned memory and move per batch. At 512k
    particles a uint8 target tensor is 6.3 GB, and the context tensor is already
    12.6 GB on the GPU -- holding both there is what would break first.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from aag.ae import ResidualDecoder as ConvDecoder
from aag.datasets import open_segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--cache", default="/opt/dlami/nvme/vpt_full")
    ap.add_argument("--n-actions", type=int, default=81)
    ap.add_argument("--ch", type=int, default=192)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--lpips-weight", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--start-epoch", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    A = torch.load(args.assignment, map_location="cpu", weights_only=False)
    for k in ("chunk", "frame"):
        if A.get(k) is None:
            raise SystemExit(
                f"assignment has no '{k}' -- rebuild particles with a "
                f"prepare_vpt_particles.py that saves cache coordinates, then "
                f"re-run the assignment. Without them the pixel targets for each "
                f"particle cannot be located.")
    z, cond, act = A["z"], A["cond"], A["action"]
    N, dim_z = z.shape
    onehot = torch.zeros(N, args.n_actions)
    onehot[torch.arange(N), act] = 1.0
    inp = torch.cat([z, cond, onehot], 1).to(dev)
    print(f"{N:,} particles | generator input dim {inp.shape[1]} "
          f"(z={dim_z} + cond={cond.shape[1]} + action={args.n_actions})", flush=True)
    r = A.get("curve", {}).get("ctx_ratio")
    if r:
        print(f"assignment independence: ctx {r[-1]:.3f}  "
              f"act {A['curve']['act_ratio'][-1]:.3f}  (1.0 = independent)", flush=True)

    # pixel targets: the actual frame each particle points at
    segs = open_segments(args.cache)
    ci, fi = A["chunk"].numpy(), A["frame"].numpy()
    tgt = torch.empty((N, 3, args.image_size, args.image_size), dtype=torch.uint8)
    for s in range(0, N, 4096):
        e = min(s + 4096, N)
        blk = np.stack([np.asarray(segs[int(c)])[int(f)]
                        for c, f in zip(ci[s:e], fi[s:e])])
        tgt[s:e] = torch.from_numpy(blk).permute(0, 3, 1, 2)
        if s % 131072 == 0:
            print(f"  targets {s:,}/{N:,}", flush=True)
    tgt = tgt.pin_memory()
    print(f"targets {tuple(tgt.shape)} uint8 ({tgt.nbytes/1e9:.2f} GB, on CPU)", flush=True)

    model = ConvDecoder(inp.shape[1], ch=args.ch, image_size=args.image_size).to(dev)
    if args.resume is not None:
        rk = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(rk["model_state_dict"])
        print(f"resumed from {args.resume} (epoch {rk.get('epoch')})", flush=True)
    print(f"generator params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    perceptual = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in perceptual.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, N // args.batch))
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.amp else \
          (lambda: torch.enable_grad())

    curve = {"epoch": [], "mse": [], "lpips": []}
    for ep in range(args.start_epoch, args.epochs):
        model.train()
        perm = torch.randperm(N)
        tot_m = tot_l = n = 0
        for i in range(0, N - args.batch + 1, args.batch):
            b = perm[i:i + args.batch]
            x = tgt[b].to(dev, non_blocking=True).float().div_(127.5).sub_(1.0)
            opt.zero_grad(set_to_none=True)
            with amp():
                pred = model(inp[b.to(dev)])
                mse = F.mse_loss(pred, x)
                perc = perceptual(pred.clamp(-1, 1), x).mean()
                loss = mse + args.lpips_weight * perc
            loss.backward()
            opt.step(); sched.step()
            tot_m += mse.item() * len(b); tot_l += perc.item() * len(b); n += len(b)
        curve["epoch"].append(ep + 1)
        curve["mse"].append(tot_m / n); curve["lpips"].append(tot_l / n)
        print(f"epoch {ep+1}/{args.epochs}  mse={tot_m/n:.5f}  lpips={tot_l/n:.5f}",
              flush=True)

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                # fixed context, fresh z: shows what the noise actually controls
                ctx = inp[:8, dim_z:].repeat_interleave(8, 0)
                fresh = torch.randn(64, dim_z, device=dev)
                grid = model(torch.cat([fresh, ctx], 1)).clamp(-1, 1)
            save_image(grid * 0.5 + 0.5, args.out / f"samples_ep{ep+1}.png", nrow=8)
            ck = args.out / "checkpoints"; ck.mkdir(exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep + 1,
                        "input_dim": inp.shape[1], "dim_z": dim_z, "ch": args.ch,
                        "image_size": args.image_size, "n_actions": args.n_actions,
                        "mse": tot_m / n, "lpips": tot_l / n,
                        "assignment": str(args.assignment)},
                       ck / f"gen_vpt_ep{ep+1}.pt")
        json.dump(curve, open(args.out / "gen_curve.json", "w"), indent=1)


if __name__ == "__main__":
    main()
