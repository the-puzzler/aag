#!/usr/bin/env python
"""Step 2 for CelebA, v2: same as run_celeba_gaussianize.py but generalized
for any AE checkpoint (arch/dim/ch inferred from the checkpoint itself) and
defaulting to a much larger assign-steps budget, since 400 steps proved
insufficient to converge at full N=162770 (see results_celeba/assignment)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gga.ae import AutoEncoder
from gga.celeba_data import celeba_loaders
from gga.diagnostics import assignment_diagnostics, intrinsic_dimension_twonn
from gga.gaussianize import AssignConfig, build_assignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--N", type=int, default=162770, help="persistent particle count")
    ap.add_argument("--data", default="/data/hf_cache")
    ap.add_argument("--out", type=Path, default=Path("results_celeba_full"))
    ap.add_argument("--assign-steps", type=int, default=4000)
    ap.add_argument("--search-subset", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    def log(*a):
        print(*a, flush=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ae = AutoEncoder(ckpt["latent_dim"], ch=ckpt["channels"],
                     architecture=ckpt["architecture"],
                     image_size=ckpt["image_size"]).to(device)
    ae.load_state_dict(ckpt["model_state_dict"])
    ae.eval()
    log(f"loaded {args.checkpoint}: dim={ckpt['latent_dim']} ch={ckpt['channels']} "
        f"arch={ckpt['architecture']} image_size={ckpt['image_size']} "
        f"trained {ckpt['epochs']} epochs, test_mse={ckpt.get('test_mse')}")

    _, enc_loader, _, n_avail = celeba_loaders(
        args.data, args.batch, n_particles=args.N, image_size=ckpt["image_size"],
    )
    log(f"encoding {args.N} persistent particles out of {n_avail} available")

    hs = []
    with torch.no_grad():
        for x, _ in enc_loader:
            hs.append(ae.enc(x.to(device)).cpu())
    h = torch.cat(hs, 0).to(device)
    log(f"encoded latents: {tuple(h.shape)}")

    idim = intrinsic_dimension_twonn(h.cpu())
    log(f"intrinsic dim (TwoNN) = {idim:.2f}")

    cfg = AssignConfig(steps=args.assign_steps, search_subset=args.search_subset,
                       seed=args.seed)
    res = build_assignment(h, cfg, log=log)
    z = res["z"]

    diag = assignment_diagnostics(z, d=ckpt["latent_dim"], seed=args.seed)
    log(f"assignment diagnostics: {diag}")

    torch.save({
        "z": z.cpu(), "h": h.cpu(), "mean": res["mean"].cpu(), "W": res["W"].cpu(),
        "W_inv": res["W_inv"].cpu(), "intrinsic_dim": idim, "diagnostics": diag,
        "N": args.N, "checkpoint": str(args.checkpoint), "assign_steps": args.assign_steps,
    }, args.out / "assignment.pt")

    (args.out / "assignment_summary.json").write_text(json.dumps({
        "N": args.N, "n_avail": n_avail, "intrinsic_dim": idim, "diagnostics": diag,
        "assign_steps": args.assign_steps,
    }, indent=2))
    log("saved: " + str(args.out / "assignment.pt"))


if __name__ == "__main__":
    main()
