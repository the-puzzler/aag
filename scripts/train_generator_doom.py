#!/usr/bin/env python
"""Action-conditioned frame generator for the Doom world model.

    z ~ N(0,I) (64) + h_{t-3..t-1} (192) + one-hot action (18)  ->  frame_t pixels

Direct-to-pixel with MSE+LPIPS, matching every other generator in this project
(the two-stage z->h->decode route was abandoned earlier). z is the frozen
assignment's Gaussian coordinate for that particle, so at inference a fresh
z ~ N(0,I) can be paired with any context -- which is only valid to the extent
the conditional assignment drove the independence ratio toward 1.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from aag.ae import ResidualDecoder as ConvDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--cache", default="/data/doom/cache_train")
    ap.add_argument("--n-actions", type=int, default=18)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lpips-weight", type=float, default=0.5)
    ap.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    ap.add_argument("--start-epoch", type=int, default=0, help="epoch number the --resume checkpoint left off at")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.load(args.assignment, map_location="cpu", weights_only=False)
    z, cond, act = A["z"], A["cond"], A["action"]
    N, dim_z = z.shape
    onehot = torch.zeros(N, args.n_actions)
    onehot[torch.arange(N), act] = 1.0
    inp = torch.cat([z, cond, onehot], 1).to(dev)              # (N, 64+192+18)
    print(f"{N:,} particles | generator input dim {inp.shape[1]} "
          f"(z={dim_z} + cond={cond.shape[1]} + action={args.n_actions})", flush=True)
    print(f"assignment independence ratio (final) = "
          f"{A['curve']['ratio'][-1]:.2f}, cond_alpha={A.get('cond_alpha')}", flush=True)

    # pixel targets: the actual frame each particle points at
    segs = np.load(f"{args.cache}/segments.npy", mmap_mode="r")
    ci, fi = A["chunk"].numpy(), A["frame"].numpy()
    tgt = torch.from_numpy(np.ascontiguousarray(segs[ci, fi]))   # (N,H,W,3) uint8
    tgt = tgt.permute(0, 3, 1, 2).contiguous().to(dev)           # uint8 on GPU
    print(f"targets {tuple(tgt.shape)} uint8 ({tgt.nbytes/1e9:.2f} GB)", flush=True)

    model = ConvDecoder(inp.shape[1], ch=args.ch, image_size=args.image_size).to(dev)
    if args.resume is not None:
        rk = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(rk["model_state_dict"])
        print(f"resumed from {args.resume} (was epoch {rk.get('epoch')})", flush=True)
    print(f"generator params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    perceptual = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in perceptual.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, N // args.batch))

    args.out.mkdir(parents=True, exist_ok=True)
    curve = {"epoch": [], "mse": [], "lpips": []}
    for ep in range(args.start_epoch, args.epochs):
        model.train()
        perm = torch.randperm(N, device=dev)
        tot_m = tot_l = n = 0
        for i in range(0, N, args.batch):
            b = perm[i:i + args.batch]
            x = tgt[b].float().div_(127.5).sub_(1.0)
            opt.zero_grad(set_to_none=True)
            pred = model(inp[b])
            mse = F.mse_loss(pred, x)
            perc = perceptual(pred.clamp(-1, 1), x).mean()
            (mse + args.lpips_weight * perc).backward()
            opt.step(); sched.step()
            tot_m += mse.item() * b.numel(); tot_l += perc.item() * b.numel(); n += b.numel()
        curve["epoch"].append(ep + 1)
        curve["mse"].append(tot_m / n); curve["lpips"].append(tot_l / n)
        print(f"epoch {ep+1}/{args.epochs}  mse={tot_m/n:.5f}  lpips={tot_l/n:.5f}", flush=True)

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                # fixed context, fresh z: shows what the noise actually controls
                ctx = inp[:8, dim_z:].repeat_interleave(8, 0)
                fresh = torch.randn(64, dim_z, device=dev)
                grid = model(torch.cat([fresh, ctx], 1))
                real = tgt[:8].float().div_(127.5).sub_(1.0)
                recon = model(inp[:8])
            save_image((grid.clamp(-1,1)+1)/2, args.out / f"samples_ep{ep+1}.png", nrow=8)
            save_image((torch.cat([real, recon]).clamp(-1,1)+1)/2,
                       args.out / f"paired_ep{ep+1}.png", nrow=8)
            _p = args.out / f"generator_ep{ep+1}.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep + 1,
                        "input_dim": inp.shape[1], "dim_z": dim_z}, str(_p) + ".tmp")
            Path(str(_p) + ".tmp").replace(_p)  # atomic: never leaves a truncated file at the real path
            (args.out / "curve.json").write_text(json.dumps(curve, indent=2))
    (args.out / "curve.json").write_text(json.dumps(curve, indent=2))


if __name__ == "__main__":
    main()
