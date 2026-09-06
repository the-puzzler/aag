#!/usr/bin/env python
"""AAG2 assignment on real encoder particles: joint K=d ridge blocks, stop at the matched
finite-Gaussian floor measured on frozen held-out directions (user spec 2026-09-06).

Writes the same checkpoint format as run_assignment_classes.py (z, h, label, mean, W, W_inv,
metadata) so train_generator256_ddp.py / assign_surrogate_score.py consume it unchanged.
Checkpoints are saved every --save-every blocks and at the floor crossing (…_floor.pt); the run
continues to --max-blocks so over-transport can be compared.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from aag.gaussianize import whiten, aag2_block_step, aag2_defect, aag2_floor, _rand_unit

ap = argparse.ArgumentParser()
ap.add_argument("--particles", required=True)
ap.add_argument("--ridge", type=float, default=0.02)
ap.add_argument("--K", type=int, default=0, help="directions per block; 0 = d")
ap.add_argument("--rotate", type=int, default=1, help="1 = PCA whitening (spec); 0 = per-coordinate scaling (AAG1 runs)")
ap.add_argument("--eval-dirs", type=int, default=256)
ap.add_argument("--eval-subset", type=int, default=0, help="0 = all particles")
ap.add_argument("--floor-clouds", type=int, default=4)
ap.add_argument("--max-blocks", type=int, default=400)
ap.add_argument("--save-every", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args(); dev = "cuda"

d0 = torch.load(a.particles, map_location="cpu", weights_only=False)
h = d0["h"].float().to(dev); labels = d0["label"]; N, D = h.shape
z, mean, W, W_inv = whiten(h, rotate=bool(a.rotate)); z = z.contiguous()
gen = torch.Generator(device=dev).manual_seed(a.seed)
egen = torch.Generator(device=dev).manual_seed(a.seed + 1000)
dirs = _rand_unit(a.eval_dirs, D, dev, z.dtype)               # frozen, never used for updates
sub = a.eval_subset or None
floor, floor_sd = aag2_floor(N, D, dirs, n_clouds=a.floor_clouds, subset=sub, gen=egen, device=dev)
K = a.K or D
print(f"{N:,} particles dim={D}  K={K} ridge={a.ridge} rotate={a.rotate}  eval dirs={a.eval_dirs} subset={sub or N}  "
      f"floor G={floor:.6f} +/- {floor_sd:.6f}", flush=True)
meta = {k: v for k, v in d0.items() if k not in ("h", "label")}
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
curve = {"block": [], "G": [], "ratio": [], "req": [], "disp": []}
def save(path, blocks, crossed):
    torch.save({"z": z.cpu(), "h": d0["h"], "label": labels, "mean": mean.cpu(), "W": W.cpu(), "W_inv": W_inv.cpu(),
                "curve": curve, "steps": blocks, "levels": [], "method": "aag2", "K": K, "ridge": a.ridge, "rotate": a.rotate,
                "floor": floor, "floor_crossed_at": crossed, "particles": a.particles, **{k: meta[k] for k in ("encoder", "grid", "latent_channels", "dataset", "root", "n_particles") if k in meta}}, path)
z0 = z.clone(); t0 = time.time(); crossed = None
G = aag2_defect(z, dirs, subset=sub, gen=egen)
print(f"block {0:4d}  G={G:.6f}  G/floor={G / floor:.3f}", flush=True)
for b in range(1, a.max_blocks + 1):
    req = aag2_block_step(z, ridge=a.ridge, gen=gen, K=K)
    G = aag2_defect(z, dirs, subset=sub, gen=egen); r = G / floor
    disp = float((z - z0).norm(dim=1).mean() / D ** 0.5)
    for k_, v_ in zip(curve, (b, G, r, req, disp)): curve[k_].append(v_)
    if b % 5 == 0 or r <= 1.0 and crossed is None:
        print(f"block {b:4d}  G={G:.6f}  G/floor={r:.3f}  req={req:.4f}  disp={disp:.2f}  [{(time.time() - t0) / b:.2f} s/block]", flush=True)
    if crossed is None and r <= 1.0:
        crossed = b; save(str(out).replace(".pt", "_floor.pt"), b, crossed); print(f"  floor crossed at block {b} -> saved _floor.pt", flush=True)
    if b % a.save_every == 0:
        save(str(out).replace(".pt", f"_block{b}.pt"), b, crossed)
save(str(out), a.max_blocks, crossed)
Path(str(out).replace(".pt", ".curve.json")).write_text(json.dumps(curve))
print(f"saved -> {out}  floor crossed at {crossed}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
