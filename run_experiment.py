#!/usr/bin/env python
"""CIFAR-10 realization of the Persistent Global Gaussian Assignment study.

Implements the two experiments the report (Section 8.2) names as the most
important next tests on real image latents:

  A. Bottleneck-dimension sweep at fixed N.  For each d: train an AE, encode a
     fixed N-image subset, measure the intrinsic dimension of those latents,
     build the persistent Gaussian assignment, then train a direct decoder and
     evaluate fresh-prior one-pass generation.  (Report Sections 3-4, Table 2.)

  B. Effective-rank prior sweep.  Fix the representation width, vary the sampled
     prior rank k, and compare full-prior vs rank-k-prior generation.
     (Report Section 5.1, Table 4.)

Because CIFAR has no exact source density, "valid mass" is measured against a
kNN density model fit to the real AE latents (see gga/decoder.py).

Usage:
    python run_experiment.py --quick          # fast smoke test
    python run_experiment.py                   # default full run
    python run_experiment.py --dims 4 8 16 32  # custom sweep
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gga.ae import AutoEncoder, encode_all, train_autoencoder
from gga.data import cifar_loaders
from gga.decoder import (decode_samples, fresh_prior_metrics, train_decoder)
from gga.diagnostics import assignment_diagnostics, intrinsic_dimension_twonn
from gga.gaussianize import AssignConfig, build_assignment


def save_grid(imgs, path, nrow=8):
    from torchvision.utils import save_image
    save_image((imgs.clamp(-1, 1) + 1) / 2, path, nrow=nrow)


def build_ae_and_latents(d, ae_loader, enc_loader, args, device, log):
    log(f"[dim {d}] training autoencoder ...")
    ae = AutoEncoder(d, ch=args.ae_ch, architecture=args.ae_arch).to(device)
    train_autoencoder(ae, ae_loader, epochs=args.ae_epochs, lr=args.ae_lr,
                      device=device, topk_percent=args.ae_topk_percent, log=log)
    h = encode_all(ae, enc_loader, device).to(device)
    return ae, h


def run_pipeline(d, ae, h, args, device, out, log):
    """Assignment + decoder + full-prior metrics for one bottleneck dim."""
    row = {"dim": d, "N": h.shape[0]}

    idim = intrinsic_dimension_twonn(h.cpu())
    row["intrinsic_dim"] = idim
    row["N_pow_1_over_idim"] = float(h.shape[0] ** (1.0 / max(idim, 1e-6)))
    log(f"[dim {d}] intrinsic dim (TwoNN) = {idim:.2f}")

    log(f"[dim {d}] building persistent Gaussian assignment ...")
    cfg = AssignConfig(steps=args.assign_steps, search_subset=args.search_subset,
                       seed=args.seed)
    res = build_assignment(h, cfg, log=log)
    z = res["z"]

    row["assignment"] = assignment_diagnostics(z, d=d, seed=args.seed)
    log(f"[dim {d}] assignment diagnostics: {row['assignment']}")

    log(f"[dim {d}] training direct decoder ...")
    model, held = train_decoder(z, h, dim=d, epochs=args.dec_epochs, lr=args.dec_lr,
                                batch=args.dec_batch, device=device, log=log)
    row["held_out_pair_mse"] = held

    m = fresh_prior_metrics(model, h, dim=d, n_samples=args.n_eval, device=device)
    row.update(m)
    log(f"[dim {d}] fresh-prior (full N(0,I_{d})): {m}")

    # image samples for eyeballing
    with torch.no_grad():
        imgs = decode_samples(model, ae, dim=d, n=64, mean=res["mean"],
                              W_inv=res["W_inv"], device=device)
    save_grid(imgs.cpu(), out / f"samples_dim{d}.png")
    return row, model, res


def effective_rank_sweep(d, ae, h, model, res, args, device, out, log):
    """Report Section 5.1: fix width d, vary sampled prior rank k."""
    rows = []
    for k in args.ranks:
        if k > d:
            continue
        full = fresh_prior_metrics(model, h, dim=d, n_samples=args.n_eval,
                                   rank=None, device=device)
        rk = fresh_prior_metrics(model, h, dim=d, n_samples=args.n_eval,
                                 rank=k, device=device)
        rows.append({
            "rank_k": k,
            "N_pow_1_over_k": float(h.shape[0] ** (1.0 / k)),
            "full_prior_valid@1%": full["valid@1%"],
            "rank_k_valid@1%": rk["valid@1%"],
            "rank_k_valid@5%": rk["valid@5%"],
            "rank_k_frechet": rk["latent_frechet"],
        })
        log(f"[rank sweep d={d}] k={k}: full={full['valid@1%']:.3f} "
            f"rank-k={rk['valid@1%']:.3f}")
        with torch.no_grad():
            imgs = decode_samples(model, ae, dim=d, n=64, mean=res["mean"],
                                  W_inv=res["W_inv"], rank=k, device=device)
        save_grid(imgs.cpu(), out / f"samples_dim{d}_rank{k}.png")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--rank-sweep-dim", type=int, default=64,
                    help="fixed representation width for the effective-rank sweep")
    ap.add_argument("--N", type=int, default=6000, help="persistent particle count")
    ap.add_argument("--n-eval", type=int, default=6000)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ae-ch", type=int, default=64)
    ap.add_argument("--ae-arch", choices=["plain", "residual", "spatial"],
                    default="plain")
    ap.add_argument("--ae-topk-percent", type=float, default=100.0,
                    help="percent of largest per-image squared errors used by AE loss")
    ap.add_argument("--ae-epochs", type=int, default=30)
    ap.add_argument("--ae-lr", type=float, default=2e-3)
    ap.add_argument("--assign-steps", type=int, default=400)
    ap.add_argument("--search-subset", type=int, default=2048)
    ap.add_argument("--dec-epochs", type=int, default=200)
    ap.add_argument("--dec-lr", type=float, default=1e-3)
    ap.add_argument("--dec-batch", type=int, default=512)
    ap.add_argument("--n-train", type=int, default=None,
                    help="limit AE training images (None=full 50k)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke config: overrides epochs/steps/dims")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.dims = [4, 16]
        args.ranks = [2, 4, 8]
        args.rank_sweep_dim = 16
        args.N = 2000
        args.n_eval = 2000
        args.ae_epochs = 2
        args.assign_steps = 40
        args.dec_epochs = 20
        args.n_train = 5000

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    def log(*a):
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    log(f"device={device}  dims={args.dims}  N={args.N}")
    ae_loader, enc_loader, _, n_avail = cifar_loaders(
        args.data, args.batch, args.N, n_train=args.n_train)
    log(f"AE trains on {n_avail} images; {args.N} persistent particles.")

    results = {"config": vars(args), "sweep": [], "rank_sweep": []}
    cache = {}  # dim -> (ae, h, model, res) so the rank sweep can reuse dim=64

    for d in args.dims:
        t0 = time.time()
        ae, h = build_ae_and_latents(d, ae_loader, enc_loader, args, device, log)
        row, model, res = run_pipeline(d, ae, h, args, device, out, log)
        row["seconds"] = time.time() - t0
        results["sweep"].append(row)
        cache[d] = (ae, h, model, res)
        (out / "results.json").write_text(json.dumps(results, indent=2))
        log(f"[dim {d}] done in {row['seconds']:.1f}s")

    # effective-rank sweep on the fixed-width representation
    rd = args.rank_sweep_dim
    if rd not in cache:
        ae, h = build_ae_and_latents(rd, ae_loader, enc_loader, args, device, log)
        _, model, res = run_pipeline(rd, ae, h, args, device, out, log)
    else:
        ae, h, model, res = cache[rd]
    results["rank_sweep"] = effective_rank_sweep(
        rd, ae, h, model, res, args, device, out, log)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    log("ALL DONE. Results in " + str(out / "results.json"))
    _print_tables(results, log)


def _print_tables(results, log):
    log("\n=== Bottleneck-dimension sweep (analogue of Table 2) ===")
    log(f"{'d':>4} {'idim':>6} {'N^1/idim':>9} {'pairMSE':>9} "
        f"{'valid@1%':>9} {'valid@5%':>9} {'frechet':>9}")
    for r in results["sweep"]:
        log(f"{r['dim']:>4} {r['intrinsic_dim']:>6.2f} "
            f"{r['N_pow_1_over_idim']:>9.2f} {r['held_out_pair_mse']:>9.5f} "
            f"{r['valid@1%']:>9.3f} {r['valid@5%']:>9.3f} {r['latent_frechet']:>9.2f}")
    log("\n=== Effective-rank prior sweep (analogue of Table 4) ===")
    log(f"{'k':>4} {'N^1/k':>9} {'full@1%':>9} {'rankk@1%':>9} {'rankk@5%':>9}")
    for r in results["rank_sweep"]:
        log(f"{r['rank_k']:>4} {r['N_pow_1_over_k']:>9.2f} "
            f"{r['full_prior_valid@1%']:>9.3f} {r['rank_k_valid@1%']:>9.3f} "
            f"{r['rank_k_valid@5%']:>9.3f}")


if __name__ == "__main__":
    main()
