#!/usr/bin/env python
"""First-frame-conditioned video generator.

    z ~ N(0,I) (256)  +  FrameAE(frame 0) (64)   ->   16x64x64 clip

Direct-to-pixel with MSE + per-frame LPIPS, matching every other generator here.
Uses ResidualDecoder3d (fc -> 3D grid) rather than the spatial decoder: the input
is z concatenated with a condition that has no spatial topology, so an fc that
learns the projection is more honest than reshaping the concatenation into a grid.

Sanity check worth watching: the generated frame 0 should resemble the
conditioning frame, since the model is told exactly what it starts from.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F

from aag.ae import ResidualDecoder3d, AdaLNDecoder3d, SpatialCondDecoder3d
from aag.video import lpips_any, save_video_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--cache", default="/data/doom/cache_train")
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lpips-weight", type=float, default=0.5)
    ap.add_argument("--cond-mode", choices=["concat", "adaln", "spatial", "none"], default="spatial",
                     help="concat: c appended to z at the input only. "
                          "adaln: c predicts per-channel scale/shift at every norm layer. "
                          "spatial: c's 4x4 grid tiled over time and concatenated as decoder "
                          "input channels, PLUS adaln -- keeps the layout the frame AE encoded")
    ap.add_argument("--cond-grid", type=int, default=4,
                     help="spatial mode: the condition's spatial grid size (frame AE 'hybrid' = 4)")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--start-epoch", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = "cuda"
    A = torch.load(args.assignment, map_location="cpu", weights_only=False)
    z, cond = A["z"], A["cond"]
    N, dim_z = z.shape
    z, cond = z.to(dev), cond.to(dev)
    inp = torch.cat([z, cond], 1) if args.cond_mode == "concat" else z
    print(f"{N:,} clips | cond_mode={args.cond_mode} | z={dim_z}, first-frame={cond.shape[1]}",
          flush=True)
    print(f"assignment independence ratio (final) = {A['curve']['ratio'][-1]:.2f}", flush=True)

    segs = np.load(f"{args.cache}/segments.npy", mmap_mode="r")
    tgt = torch.from_numpy(np.ascontiguousarray(segs[:N]))          # (N,T,H,W,3) uint8
    tgt = tgt.permute(0, 4, 1, 2, 3).contiguous()                   # (N,3,T,H,W)
    print(f"targets {tuple(tgt.shape)} uint8 ({tgt.nbytes/1e9:.1f} GB, kept on CPU)", flush=True)

    if args.cond_mode == "none":
        # unconditional: z -> clip. The condition is ignored entirely, so no
        # conditional transport scrambling enters the latent at all.
        model = ResidualDecoder3d(dim_z, ch=args.ch, image_size=args.image_size,
                                  frames=args.frames).to(dev)
        fwd = lambda idx: model(z[idx])
    elif args.cond_mode == "concat":
        model = ResidualDecoder3d(inp.shape[1], ch=args.ch, image_size=args.image_size,
                                  frames=args.frames).to(dev)
        fwd = lambda idx: model(inp[idx])
    elif args.cond_mode == "adaln":
        model = AdaLNDecoder3d(dim_z, cond.shape[1], ch=args.ch, image_size=args.image_size,
                               frames=args.frames).to(dev)
        fwd = lambda idx: model(z[idx], cond[idx])
    else:
        grid = args.cond_grid
        c_ch = cond.shape[1] // (grid * grid)
        if c_ch * grid * grid != cond.shape[1]:
            raise ValueError(f"cond dim {cond.shape[1]} is not C x {grid} x {grid}")
        print(f"spatial conditioning: {c_ch} channels on a {grid}x{grid} grid, tiled over time",
              flush=True)
        model = SpatialCondDecoder3d(dim_z, c_ch, ch=args.ch, image_size=args.image_size,
                                     frames=args.frames, cond_grid=grid).to(dev)
        fwd = lambda idx: model(z[idx], cond[idx])
    if args.resume is not None:
        rk = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(rk["model_state_dict"])
        print(f"resumed from {args.resume} (epoch {rk.get('epoch')})", flush=True)
    print(f"generator params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    perceptual = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in perceptual.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, N // args.batch))

    args.out.mkdir(parents=True, exist_ok=True)
    curve = {"epoch": [], "mse": [], "lpips": []}
    for ep in range(args.start_epoch, args.epochs):
        model.train()
        perm = torch.randperm(N)
        tm = tl = n = 0
        for i in range(0, N, args.batch):
            b = perm[i:i + args.batch]
            x = tgt[b].to(dev, non_blocking=True).float().div_(127.5).sub_(1.0)
            opt.zero_grad(set_to_none=True)
            pred = fwd(b.to(dev))
            mse = F.mse_loss(pred, x)
            perc = lpips_any(perceptual, pred.clamp(-1, 1), x).mean()
            (mse + args.lpips_weight * perc).backward()
            opt.step(); sched.step()
            tm += mse.item() * b.numel(); tl += perc.item() * b.numel(); n += b.numel()
        curve["epoch"].append(ep + 1); curve["mse"].append(tm / n); curve["lpips"].append(tl / n)
        print(f"epoch {ep+1}/{args.epochs}  mse={tm/n:.5f}  lpips={tl/n:.5f}", flush=True)

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                # same first frame down each row, fresh z across -> what does z control?
                # same first frame down each row, fresh z across
                zf = torch.randn(32, dim_z, device=dev)
                if args.cond_mode == "none":
                    grid = model(zf)                       # pure unconditional samples
                else:
                    c8 = cond[:8].repeat_interleave(4, 0)
                    grid = (model(torch.cat([zf, c8], 1)) if args.cond_mode == "concat"
                            else model(zf, c8))
                real = tgt[:8].to(dev).float().div_(127.5).sub_(1.0)
                recon = fwd(torch.arange(8, device=dev))
            save_video_grid(grid, args.out / f"samples_ep{ep+1}.png", n_show=8, n_frames=8)
            save_video_grid(real, args.out / f"real_ep{ep+1}.png", n_show=8, n_frames=8)
            save_video_grid(recon, args.out / f"paired_ep{ep+1}.png", n_show=8, n_frames=8)
            _p = args.out / f"generator_ep{ep+1}.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep + 1,
                        "dim_z": dim_z, "cond_dim": cond.shape[1],
                        "cond_mode": args.cond_mode}, str(_p) + ".tmp")
            Path(str(_p) + ".tmp").replace(_p)     # atomic
            (args.out / "curve.json").write_text(json.dumps(curve, indent=2))
    (args.out / "curve.json").write_text(json.dumps(curve, indent=2))


if __name__ == "__main__":
    main()
