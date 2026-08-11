#!/usr/bin/env python
"""Conditional generator, direct-to-pixel version: (z, condition) -> image,
skipping the frozen AE decoder entirely (contrast with
train_conditional_generator.py's two-stage z,cond -> h -> ae.dec route).
Same z/h/cond sources (particle-order aligned), same MSE+LPIPS+grad-clip
recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from gga.ae import ResidualDecoder as ConvDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-source", type=Path, default=Path("results_celeba_conditional/interleaved_every4.pt"))
    ap.add_argument("--attrs", type=Path, default=Path("results_celeba/attrs.pt"))
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--lpips-weight", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--data", default="/data/hf_cache")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def log(*a):
        print(*a, flush=True)

    z_state = torch.load(args.z_source, map_location=device, weights_only=False)
    z = z_state["z"].to(device)
    attrs = torch.load(args.attrs, map_location=device, weights_only=False)
    cond = attrs["attrs"].to(device).float()
    attr_names = attrs["attr_names"]
    N, dim_z = z.shape
    n_attrs = cond.shape[1]
    log(f"z: {tuple(z.shape)}  cond: {tuple(cond.shape)} ({n_attrs} attrs)")
    assert z.shape[0] == cond.shape[0], "particle-order mismatch between z/cond"

    log("collecting target images (same particle order as z/cond) ...")
    from gga.celeba_data import celeba_loaders
    _, enc_loader, _, n_avail = celeba_loaders(
        args.data, args.batch, n_particles=N, image_size=args.image_size,
    )
    imgs = []
    for x, _ in enc_loader:
        imgs.append(x)
    targets = torch.cat(imgs, 0).to(device)
    log(f"targets: {tuple(targets.shape)}")

    log("loading LPIPS (VGG) ...")
    perceptual = lpips.LPIPS(net="vgg").to(device).eval()
    for p in perceptual.parameters():
        p.requires_grad_(False)

    perm = torch.randperm(N)
    n_val = int(args.val_frac * N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    zc = torch.cat([z, cond], dim=1)
    model = ConvDecoder(dim_z + n_attrs, ch=args.ch, image_size=args.image_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"conditional pixel generator params: {n_params:,}  lpips_weight={args.lpips_weight}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = (tr_idx.numel() + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    def run(idx, train):
        model.train(train)
        order = idx[torch.randperm(idx.numel())] if train else idx
        total_mse, total_lpips, n = 0.0, 0.0, 0
        for i in range(0, order.numel(), args.batch):
            b = order[i:i + args.batch]
            with torch.set_grad_enabled(train):
                pred = model(zc[b])
                mse = F.mse_loss(pred, targets[b])
                perc = perceptual(pred.clamp(-1, 1), targets[b]).mean()
                loss = mse + args.lpips_weight * perc
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
                    sched.step()
            total_mse += mse.item() * b.numel()
            total_lpips += perc.item() * b.numel()
            n += b.numel()
        return total_mse / n, total_lpips / n

    curve = {"train_epoch": [], "train_mse": [], "train_lpips": [],
             "val_epoch": [], "val_mse": [], "val_lpips": []}
    for ep in range(args.epochs):
        tr_mse, tr_lpips = run(tr_idx, True)
        curve["train_epoch"].append(ep + 1)
        curve["train_mse"].append(tr_mse)
        curve["train_lpips"].append(tr_lpips)
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            vl_mse, vl_lpips = run(val_idx, False)
            curve["val_epoch"].append(ep + 1)
            curve["val_mse"].append(vl_mse)
            curve["val_lpips"].append(vl_lpips)
            log(f"epoch {ep+1}/{args.epochs}  train_mse={tr_mse:.5f} train_lpips={tr_lpips:.5f}  "
                f"val_mse={vl_mse:.5f} val_lpips={vl_lpips:.5f}")
            torch.save({
                "model_state_dict": model.state_dict(), "dim_z": dim_z, "n_attrs": n_attrs,
                "attr_names": attr_names, "ch": args.ch, "image_size": args.image_size,
                "epoch": ep + 1, "val_mse": vl_mse, "val_lpips": vl_lpips,
            }, ckpt_dir / f"generator_ep{ep+1}.pt")
        else:
            log(f"epoch {ep+1}/{args.epochs}  train_mse={tr_mse:.5f} train_lpips={tr_lpips:.5f}")

    (args.out / "generator_train_curve.json").write_text(json.dumps(curve, indent=2))

    model.eval()
    name_to_idx = {n: i for i, n in enumerate(attr_names)}
    torch.manual_seed(0)
    n_cols = 8
    conds_demo, labels = [], []
    for want_on in [[], ["Male"], ["Smiling"], ["Blond_Hair"], ["Male", "Eyeglasses"]]:
        c = torch.zeros(n_attrs, device=device)
        for a in want_on:
            c[name_to_idx[a]] = 1.0
        conds_demo.append(c)
        labels.append("+".join(want_on) if want_on else "(all attrs off)")
    with torch.no_grad():
        rows = []
        for c in conds_demo:
            zs = torch.randn(n_cols, dim_z, device=device)
            cc = c.unsqueeze(0).repeat(n_cols, 1)
            imgs = model(torch.cat([zs, cc], dim=1))
            rows.append(imgs)
        grid = torch.cat(rows, dim=0)
    save_image((grid.clamp(-1, 1) + 1) / 2, args.out / "demo_fixed_condition_varying_z.png", nrow=n_cols)
    log("saved: demo_fixed_condition_varying_z.png  rows=" + " | ".join(labels))

    toggle_attrs = ["Male", "Smiling", "Blond_Hair", "Eyeglasses", "Wearing_Hat", "Young", "No_Beard"]
    torch.manual_seed(1)
    n_z_rows = 4
    with torch.no_grad():
        rows = []
        for _ in range(n_z_rows):
            z_fixed = torch.randn(1, dim_z, device=device)
            base_c = torch.zeros(1, n_attrs, device=device)
            ccs = [base_c]
            for a in toggle_attrs:
                c = base_c.clone()
                c[0, name_to_idx[a]] = 1.0
                ccs.append(c)
            cc = torch.cat(ccs, dim=0)
            zz = z_fixed.repeat(cc.shape[0], 1)
            imgs = model(torch.cat([zz, cc], dim=1))
            rows.append(imgs)
        grid = torch.cat(rows, dim=0)
    save_image((grid.clamp(-1, 1) + 1) / 2, args.out / "demo_fixed_z_varying_condition.png",
              nrow=len(toggle_attrs) + 1)
    log("saved: demo_fixed_z_varying_condition.png  cols=(none)," + ",".join(toggle_attrs))


if __name__ == "__main__":
    main()
