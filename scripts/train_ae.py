#!/usr/bin/env python
"""Same as probe_celeba_ae_train_curve.py but with LPIPS perceptual loss
added alongside pixel MSE for the AE's own reconstruction objective --
testing whether the LPIPS win we found for the DINO generator also helps
when training our own AE (encoder+decoder jointly) from scratch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F

from aag.ae import AutoEncoder, VideoAutoEncoder
from aag.datasets import get_loaders, spec


def _lpips_any(perceptual, a, b):
    """LPIPS is 2D. For video (B,C,T,H,W), score every frame and average."""
    if a.dim() == 5:
        B, C, T, H, W = a.shape
        a = a.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        b = b.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    return perceptual(a, b)


def topk_mse(pred, target, frac):
    """MSE over only the hardest-error frac of elements, per sample.

    Plain mean-MSE is dominated by easy, already-correct pixels (background)
    when the count of those vastly outnumbers hard pixels (a moving actor) --
    the average gradient signal for the hard region gets diluted away. Taking
    the top-k highest-error elements per sample keeps gradient focused there.
    """
    if frac >= 1.0:
        return F.mse_loss(pred, target)
    err = (pred - target).pow(2)
    flat = err.reshape(err.shape[0], -1)
    k = max(1, int(flat.shape[1] * frac))
    topk_vals, _ = flat.topk(k, dim=1)
    return topk_vals.mean()


def test_metrics(ae, loader, device, perceptual):
    ae.eval()
    total_mse, total_lpips, count, n_lpips = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            pred = ae(images)
            total_mse += F.mse_loss(pred, images, reduction="sum").item()
            # video yields one LPIPS value per FRAME (B*T), images one per sample (B),
            # so normalise by the number of values, not the batch size
            l = _lpips_any(perceptual, pred.clamp(-1, 1), images)
            total_lpips += l.sum().item()
            n_lpips += l.numel()
            count += images.shape[0]
    ae.train()
    return total_mse / (count * images[0].numel()), total_lpips / max(n_lpips, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["celeba","cifar10","doom","doom_frames"], default="celeba")
    ap.add_argument("--arch", choices=["residual", "spatial", "hybrid", "dcae"], default="residual")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--lpips-weight", type=float, default=0.5)
    ap.add_argument("--topk-frac", type=float, default=1.0,
                     help="fraction of highest-error elements per sample to backprop MSE on (1.0 = plain MSE)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--data", default="/data/hf_cache")
    ap.add_argument("--out", type=Path, default=Path("results_celeba"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loader-workers", type=int, default=4,
                    help="DataLoader workers. The default 4 starves the GPU on the "
                         "sharded VPT cache: each frame is a separate ~12KB random "
                         "read, so queue depth 4 gave only ~900 IOPS and GPU util "
                         "averaged 13% (bursts to 100%, then long stalls)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--t-out", type=int, default=4,
                     help="video only: temporal size of the spatial latent grid")
    ap.add_argument("--resume", type=Path, default=None,
                     help="checkpoint to continue from (model weights only; fresh optimiser)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    train_loader, _, test_loader, n_avail = get_loaders(
        args.dataset, args.data, args.batch, n_particles=1,
        workers=args.loader_workers, image_size=args.image_size)
    print(f"AE trains on {n_avail} {args.dataset} samples at {args.image_size}x{args.image_size}, "
          f"arch={args.arch}, lpips_weight={args.lpips_weight}, topk_frac={args.topk_frac}", flush=True)

    VIDEO = spec(args.dataset).get("video", False)
    if VIDEO:
        # mirror the 2D semantics: 'spatial' keeps the C x t_out x 4 x 4 grid,
        # 'hybrid' is the same grid at residual's channel width.
        v_arch = "spatial" if args.arch in ("spatial", "hybrid") else "residual"
        v_wm = 4 if args.arch == "hybrid" else 2
        ae = VideoAutoEncoder(args.dim, ch=args.ch, image_size=args.image_size,
                              frames=spec(args.dataset)["frames"],
                              architecture=v_arch, t_out=args.t_out,
                              width_mult=v_wm).to(device)
    else:
        ae = AutoEncoder(args.dim, ch=args.ch, architecture=args.arch,
                         image_size=args.image_size).to(device)
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"model params: {n_params:,}", flush=True)

    if args.resume is not None:
        rk = torch.load(args.resume, map_location=device, weights_only=False)
        ae.load_state_dict(rk["model_state_dict"])
        print(f"resumed weights from {args.resume} (was epoch {rk.get('epochs')}, "
              f"test_mse={rk.get('test_mse')})", flush=True)

    perceptual = lpips.LPIPS(net="vgg").to(device).eval()
    for p in perceptual.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(ae.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    curve = {"train_epoch": [], "train_mse": [], "train_lpips": [],
             "test_epoch": [], "test_mse": [], "test_lpips": [], "arch": args.arch}
    ae.train()
    for ep in range(args.epochs):
        running_mse, running_lpips, n = 0.0, 0.0, 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            xr = ae(x)
            mse_full = F.mse_loss(xr, x)
            mse_train = topk_mse(xr, x, args.topk_frac)
            if args.lpips_weight > 0:
                perc = _lpips_any(perceptual, xr.clamp(-1, 1), x).mean()
                loss = mse_train + args.lpips_weight * perc
            else:
                # skip the VGG pass entirely -- it is the dominant per-step cost.
                # test_lpips is still measured at every eval, so checkpoint
                # selection and cross-run comparison stay intact.
                perc = None
                loss = mse_train
            loss.backward()
            opt.step()
            sched.step()
            running_mse += mse_full.item() * x.size(0)
            running_lpips += (perc.item() if perc is not None else 0.0) * x.size(0)
            n += x.size(0)
        curve["train_epoch"].append(ep + 1)
        curve["train_mse"].append(running_mse / n)
        curve["train_lpips"].append(running_lpips / n)

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            tm, tl = test_metrics(ae, test_loader, device, perceptual)
            curve["test_epoch"].append(ep + 1)
            curve["test_mse"].append(tm)
            curve["test_lpips"].append(tl)
            print(f"[{args.arch}] epoch {ep+1}/{args.epochs}  train_mse={running_mse/n:.5f} "
                  f"train_lpips={running_lpips/n:.5f}  test_mse={tm:.5f} test_lpips={tl:.5f}",
                  flush=True)
            ckpt_dir = args.out / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            tag = f"_topk{args.topk_frac}" if args.topk_frac < 1.0 else ""
            ckpt_path = ckpt_dir / f"ae_{args.dataset}_{args.arch}_lpips_ch{args.ch}_dim{args.dim}{tag}_ep{ep+1}.pt"
            _p = ckpt_path
            torch.save({
                "model_state_dict": ae.state_dict(), "latent_dim": args.dim,
                "channels": args.ch, "architecture": args.arch, "image_size": args.image_size,
                "epochs": ep + 1, "test_mse": tm, "test_lpips": tl, "seed": args.seed,
                "t_out": args.t_out, "frames": spec(args.dataset).get("frames"),
            }, str(_p) + ".tmp")
            Path(str(_p) + ".tmp").replace(_p)  # atomic: never leaves a truncated file at the real path
        else:
            print(f"[{args.arch}] epoch {ep+1}/{args.epochs}  train_mse={running_mse/n:.5f} "
                  f"train_lpips={running_lpips/n:.5f}", flush=True)

    best_i = min(range(len(curve["test_lpips"])), key=lambda i: curve["test_lpips"][i])
    best_epoch, best_lpips = curve["test_epoch"][best_i], curve["test_lpips"][best_i]
    print(f"best test LPIPS {best_lpips:.5f} at epoch {best_epoch}", flush=True)

    tag = f"_topk{args.topk_frac}" if args.topk_frac < 1.0 else ""
    out_json = args.out / f"ae_train_curve_{args.arch}_lpips{tag}_{args.epochs}ep.json"
    out_json.write_text(json.dumps(curve, indent=2))
    print("saved curve:", out_json)


if __name__ == "__main__":
    main()
