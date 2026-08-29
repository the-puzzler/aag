#!/usr/bin/env python
"""Same as probe_celeba_ae_train_curve.py but with LPIPS perceptual loss
added alongside pixel MSE for the AE's own reconstruction objective --
testing whether the LPIPS win we found for the DINO generator also helps
when training our own AE (encoder+decoder jointly) from scratch."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F

from aag.ae import AutoEncoder, VideoAutoEncoder
from aag.datasets import get_loaders, spec
from aag.discriminator import (NLayerDiscriminator, adaptive_weight,
                               decoder_head_parameters, g_loss_from,
                               hinge_d_loss, last_conv_weight)


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


def hard_str(running_hard, n):
    """Report the added top-k term only when it is switched on."""
    return f" train_hard={running_hard / n:.5f}" if running_hard else ""


def gan_str(running_g, running_d, running_w, n):
    """Report the adversarial terms only once the discriminator is live."""
    if not running_d:
        return ""
    return (f" g_loss={running_g / n:.4f} d_loss={running_d / n:.4f} "
            f"d_weight={running_w / n:.4f}")


def _lp_res(a, b, factor):
    """Optionally upsample both images before the VGG pass.

    LPIPS' VGG was trained at 224x224. At 64x64 its deeper layers are down to a
    couple of pixels across and contribute almost nothing, so the metric ends up
    dominated by early layers -- exactly the coarse structure the model already
    gets right. Upsampling restores spatial extent to the deep features.
    """
    if factor <= 1:
        return a, b
    up = lambda t: F.interpolate(t, scale_factor=factor, mode="bilinear",
                                 align_corners=False)
    return up(a), up(b)


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
    ap.add_argument("--grid", type=int, default=4,
                    help="Spatial size of the latent grid. Default 4: at 64x64 each "
                         "cell owns a 16x16 pixel region, which is exactly where the "
                         "small-object regression measured on our models collapses "
                         "to ~1%%. grid=8 halves that to 8x8 per cell. Note it also "
                         "removes one down/up block, so match --ch to keep the "
                         "parameter count comparable (grid 8 needs ch 304 to match "
                         "grid 4 at ch 192).")
    ap.add_argument("--lpips-weight", type=float, default=0.5)
    ap.add_argument("--topk-add-frac", type=float, default=0.0,
                    help="If >0, ADD a top-k MSE term over this fraction of the "
                         "hardest elements, on top of the full-mean MSE (unlike "
                         "--topk-frac, which replaces it). At weight 1.0 the mean "
                         "term spreads one unit of gradient mass over all 12288 "
                         "elements while this term spreads one unit over the worst "
                         "0.5%%, so the hardest pixels carry gradient mass equal to "
                         "every other pixel combined. Counters high-frequency "
                         "detail being numerically swamped by the low-frequency "
                         "bulk it is averaged against.")
    ap.add_argument("--topk-add-weight", type=float, default=1.0)
    ap.add_argument("--topk-frac", type=float, default=1.0,
                     help="fraction of highest-error elements per sample to backprop MSE on (1.0 = plain MSE)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--data", default="/data/hf_cache")
    ap.add_argument("--out", type=Path, default=Path("results_celeba"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast for the AE forward and the LPIPS term. bf16 "
                         "not fp16: no GradScaler needed and the tanh output stays "
                         "well within range")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the AE. Needs a fixed batch shape, so the "
                         "loader must drop_last, else every partial batch triggers "
                         "a recompile")
    ap.add_argument("--loader-workers", type=int, default=4,
                    help="DataLoader workers. The default 4 starves the GPU on the "
                         "sharded VPT cache: each frame is a separate ~12KB random "
                         "read, so queue depth 4 gave only ~900 IOPS and GPU util "
                         "averaged 13%% (bursts to 100%%, then long stalls)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--t-out", type=int, default=4,
                     help="video only: temporal size of the spatial latent grid")
    ap.add_argument("--resume", type=Path, default=None,
                     help="checkpoint to continue from (model weights only; fresh optimiser)")
    ap.add_argument("--gan-weight", type=float, default=0.0,
                    help="If >0, add DC-AE's adversarial refinement phase (arXiv "
                         "2410.10733 sec 3.2). Scales the adaptively-balanced "
                         "discriminator term; SD-VAE uses 0.5. Off by default.")
    ap.add_argument("--gan-start-epoch", type=int, default=1,
                    help="Epoch (1-based) at which the discriminator switches on. "
                         "Earlier epochs are reconstruction-only, so the decoder "
                         "is already sane before it is critiqued.")
    ap.add_argument("--gan-layers", type=int, default=2,
                    help="PatchGAN depth. 2 gives a 34px receptive field at "
                         "64x64; the usual 3 gives 70px, larger than the frame.")
    ap.add_argument("--gan-ndf", type=int, default=64)
    ap.add_argument("--gan-lr", type=float, default=4.5e-5)
    ap.add_argument("--gan-head-modules", type=int, default=0,
                    help="If >0, freeze everything except the last N modules of "
                         "the decoder stack -- DC-AE phase 3 tunes only the "
                         "decoder head. Keeps the encoder, and therefore every "
                         "latent the particle pipeline was built from, unchanged.")
    ap.add_argument("--lpips-upsample", type=int, default=1,
                    help="Compute LPIPS on images upsampled by this factor. VGG "
                         "at 64x64 leaves its deeper layers almost no spatial "
                         "extent to measure; 2 restores it. Affects the training "
                         "term only -- reported test_lpips stays at native "
                         "resolution so it is comparable across every run.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    train_loader, _, test_loader, n_avail = get_loaders(
        args.dataset, args.data, args.batch, n_particles=1,
        workers=args.loader_workers, image_size=args.image_size)
    print(f"AE trains on {n_avail} {args.dataset} samples at {args.image_size}x{args.image_size}, "
          f"arch={args.arch}, lpips_weight={args.lpips_weight}, topk_frac={args.topk_frac}, "
          f"topk_add_frac={args.topk_add_frac}, topk_add_weight={args.topk_add_weight}, "
          f"grid={args.grid}x{args.grid}", flush=True)

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
                         image_size=args.image_size, grid=args.grid).to(device)
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"model params: {n_params:,}", flush=True)

    if args.resume is not None:
        rk = torch.load(args.resume, map_location=device, weights_only=False)
        _sd = rk["model_state_dict"]
        # checkpoints written before the compile fix carry torch.compile's
        # "_orig_mod." prefix on every key
        if any(k.startswith("_orig_mod.") for k in _sd):
            _sd = {k.replace("_orig_mod.", "", 1): v for k, v in _sd.items()}
        ae.load_state_dict(_sd)
        print(f"resumed weights from {args.resume} (was epoch {rk.get('epochs')}, "
              f"test_mse={rk.get('test_mse')})", flush=True)

    if args.compile:
        if args.gan_weight > 0:
            # alternating D/G steps and the adaptive-weight autograd.grad probes
            # graph-break repeatedly; the refinement phase is short enough that
            # compilation would not pay for itself anyway.
            print("ignoring --compile: not supported with --gan-weight", flush=True)
        else:
            ae = torch.compile(ae)
            print("torch.compile enabled (first steps pay compilation)", flush=True)
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.amp \
        else (lambda: contextlib.nullcontext())
    if args.amp:
        print("bf16 autocast enabled", flush=True)

    perceptual = lpips.LPIPS(net="vgg").to(device).eval()
    for p in perceptual.parameters():
        p.requires_grad_(False)

    disc = opt_d = last_layer = None
    if args.gan_weight > 0:
        disc = NLayerDiscriminator(3, args.gan_ndf, args.gan_layers).to(device)
        opt_d = torch.optim.Adam(disc.parameters(), lr=args.gan_lr, betas=(0.5, 0.9))
        last_layer = last_conv_weight(ae)
        print(f"adversarial refinement on: weight={args.gan_weight}, "
              f"start_epoch={args.gan_start_epoch}, layers={args.gan_layers}, "
              f"disc params {sum(p.numel() for p in disc.parameters()):,}", flush=True)

    if args.gan_head_modules > 0:
        train_params = decoder_head_parameters(ae, args.gan_head_modules)
        print(f"frozen to decoder head: training {sum(p.numel() for p in train_params):,} "
              f"of {n_params:,} params (last {args.gan_head_modules} decoder modules); "
              f"encoder unchanged, so existing latents stay valid", flush=True)
    else:
        train_params = list(ae.parameters())

    opt = torch.optim.Adam(train_params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    curve = {"train_epoch": [], "train_mse": [], "train_lpips": [],
             "test_epoch": [], "test_mse": [], "test_lpips": [], "arch": args.arch}
    ae.train()
    for ep in range(args.epochs):
        running_mse, running_lpips, running_hard, n = 0.0, 0.0, 0.0, 0
        running_g, running_d, running_w = 0.0, 0.0, 0.0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                xr = ae(x)
                mse_full = F.mse_loss(xr, x)
                mse_train = topk_mse(xr, x, args.topk_frac)
                if args.topk_add_frac > 0:
                    hard = topk_mse(xr, x, args.topk_add_frac)
                    mse_train = mse_train + args.topk_add_weight * hard
                    running_hard += hard.item() * x.size(0)
                if args.lpips_weight > 0:
                    perc = _lpips_any(perceptual, *_lp_res(xr.clamp(-1, 1), x,
                                                           args.lpips_upsample)).mean()
                    loss = mse_train + args.lpips_weight * perc
                else:
                    # skip the VGG pass entirely -- it is the dominant per-step
                    # cost. test_lpips is still measured at every eval, so
                    # checkpoint selection and cross-run comparison stay intact.
                    perc = None
                    loss = mse_train

            gan_on = disc is not None and (ep + 1) >= args.gan_start_epoch
            if gan_on:
                # generator step: the adaptive weight is measured against the
                # reconstruction gradient at the output conv, so the adversarial
                # term stays a fixed fraction of it however the two drift.
                with amp_ctx():
                    g = g_loss_from(disc(xr))
                # the grad-norm probe stays outside autocast so the ratio is fp32
                w = adaptive_weight(loss, g, last_layer) * args.gan_weight
                loss = loss + w * g
                running_g += g.item() * x.size(0)
                running_w += w.item() * x.size(0)

            loss.backward()
            opt.step()
            sched.step()

            if gan_on:
                opt_d.zero_grad(set_to_none=True)
                with amp_ctx():
                    d_loss = hinge_d_loss(disc(x), disc(xr.detach()))
                d_loss.backward()
                opt_d.step()
                running_d += d_loss.item() * x.size(0)

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
                  f"train_lpips={running_lpips/n:.5f}{hard_str(running_hard, n)}"
                  f"{gan_str(running_g, running_d, running_w, n)}  "
                  f"test_mse={tm:.5f} test_lpips={tl:.5f}",
                  flush=True)
            ckpt_dir = args.out / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            tag = f"_topk{args.topk_frac}" if args.topk_frac < 1.0 else ""
            ckpt_path = ckpt_dir / f"ae_{args.dataset}_{args.arch}_lpips_ch{args.ch}_dim{args.dim}{tag}_ep{ep+1}.pt"
            _p = ckpt_path
            # torch.compile wraps the module, so ae.state_dict() prefixes every
            # key with "_orig_mod." and a plain AutoEncoder cannot load it. Save
            # the unwrapped weights so checkpoints stay portable to the particle
            # builder, the generator and any later run without --compile.
            sd = (ae._orig_mod if hasattr(ae, "_orig_mod") else ae).state_dict()
            torch.save({
                "model_state_dict": sd, "latent_dim": args.dim,
                "channels": args.ch, "architecture": args.arch, "image_size": args.image_size,
                "grid": args.grid,
                "epochs": ep + 1, "test_mse": tm, "test_lpips": tl, "seed": args.seed,
                "topk_add_frac": args.topk_add_frac, "topk_add_weight": args.topk_add_weight,
                "gan_weight": args.gan_weight, "gan_head_modules": args.gan_head_modules,
                "lpips_upsample": args.lpips_upsample,
                # separate key: every loader in the pipeline reads model_state_dict
                # and must keep seeing plain AE weights and nothing else
                "disc_state_dict": disc.state_dict() if disc is not None else None,
                "t_out": args.t_out, "frames": spec(args.dataset).get("frames"),
            }, str(_p) + ".tmp")
            Path(str(_p) + ".tmp").replace(_p)  # atomic: never leaves a truncated file at the real path

            # Divergence guard. A dcae run at ch=112 saturated its decoder tanh
            # at epoch 3, froze at a constant output with gradient norm exactly
            # 0, and then burned 17 more epochs printing identical numbers. Abort
            # instead: nothing after a dead gradient is recoverable.
            #
            # Under an adversarial phase the MSE-based limbs do not apply: the
            # GAN term trades pixel error for detail on purpose, so a rising
            # test_mse is the intended behaviour rather than a divergence. The
            # non-finite check still stands -- NaN is never intended.
            hist = curve["test_mse"]
            if not math.isfinite(tm):
                print(f"DIVERGED: test_mse is {tm} at epoch {ep+1} -- aborting", flush=True)
                break
            if args.gan_weight > 0:
                continue
            best_so_far = min(hist)
            if len(hist) >= 2 and tm > 3.0 * best_so_far:
                print(f"DIVERGED: test_mse {tm:.5f} is >3x the best {best_so_far:.5f} "
                      f"at epoch {ep+1} -- aborting", flush=True)
                break
            if len(hist) >= 3 and len(set(f"{v:.7f}" for v in hist[-3:])) == 1:
                print(f"FROZEN: test_mse identical for 3 epochs ({tm:.5f}) at epoch "
                      f"{ep+1} -- gradient is probably dead, aborting", flush=True)
                break
        else:
            print(f"[{args.arch}] epoch {ep+1}/{args.epochs}  train_mse={running_mse/n:.5f} "
                  f"train_lpips={running_lpips/n:.5f}{hard_str(running_hard, n)}"
                  f"{gan_str(running_g, running_d, running_w, n)}", flush=True)

    best_i = min(range(len(curve["test_lpips"])), key=lambda i: curve["test_lpips"][i])
    best_epoch, best_lpips = curve["test_epoch"][best_i], curve["test_lpips"][best_i]
    print(f"best test LPIPS {best_lpips:.5f} at epoch {best_epoch}", flush=True)

    tag = f"_topk{args.topk_frac}" if args.topk_frac < 1.0 else ""
    out_json = args.out / f"ae_train_curve_{args.arch}_lpips{tag}_{args.epochs}ep.json"
    out_json.write_text(json.dumps(curve, indent=2))
    print("saved curve:", out_json)


if __name__ == "__main__":
    main()
