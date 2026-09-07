#!/usr/bin/env python
"""FID-Nk of a Generator256 checkpoint on fresh z, for comparison with published numbers.

The trainer's in-loop FID uses 10k samples for speed; papers (ROMS-IMLE, DiT, ADM) report
FID-50k, which reads a few points lower on the same model. This evaluates the EMA at any N
against a reference stats .npz from compute_fid_stats_hf256.py. Class-conditional models get
labels drawn uniformly over the classes (the standard protocol).

Optional --round-trip-keep 0.95 reproduces ROMS-IMLE's inference-time filter: score every
sample by an AE round-trip reconstruction cost and drop the worst 5% before computing FID.
Only meaningful with an encoder that can round-trip pixels; off by default.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from aag.generator256 import Generator256
from aag.fid import get_activations, fid_from_stats

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", type=Path)
ap.add_argument("--fid-stats", required=True)
ap.add_argument("--n", type=int, default=50000)
ap.add_argument("--batch", type=int, default=125)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--weights", choices=["ema", "model"], default="ema")
ap.add_argument("--out", type=Path, default=None, help="append a JSON line here")
a = ap.parse_args()

dev = "cuda"
ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
n_classes = int(ck.get("n_classes", 0))
sd = ck[a.weights]
dim_z = ck["dim_z"] if "dim_z" in ck else sd["lay.weight"].shape[1] if "lay.weight" in sd else (sd["dec.fc.weight"].shape[1] if "dec.fc.weight" in sd else sd["stem.weight"].shape[1] * ck["grid"] ** 2)
if ck.get("arch", "grid") == "residual":
    from aag.ae import ResidualDecoder
    class FlatGen(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.dec = ResidualDecoder(dim_z, ch=ck["ch"], image_size=256); self.out = self.dec.net[-1]
        def forward(self, z, y=None): return self.dec(z)
    dim_z = sd["dec.fc.weight"].shape[1]; net = FlatGen().to(dev).eval()
else:
    net = Generator256(dim_z, grid=ck["grid"], image_size=256, n_classes=n_classes, cond_dim=ck["cond_dim"],
                       width=ck["width"], n_res=ck["n_res"], z_bottleneck=ck.get("z_bottleneck", 0),
                       z_skip=ck.get("z_skip", "none"), z_skip_rank=ck.get("z_skip_rank", 32),
                       z_pre_depth=ck.get("z_pre_depth", 0), z_pre_width=ck.get("z_pre_width", 512)).to(dev).eval()
net.load_state_dict(sd)
ref = np.load(a.fid_stats); mu, sig = ref["mu"], ref["sigma"]
g = torch.Generator(device=dev).manual_seed(a.seed)
acts, t0 = [], time.time()
with torch.no_grad():
    for i in range(0, a.n, a.batch):
        b = min(a.batch, a.n - i)
        z = torch.randn(b, dim_z, device=dev, generator=g)
        y = torch.randint(n_classes, (b,), device=dev, generator=g) if n_classes else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = net(z, y).float().clamp(-1, 1)
        acts.append(get_activations((x + 1) / 2, dev))
acts = np.concatenate(acts)[: a.n]
fid = fid_from_stats(mu, sig, acts)
rec = {"checkpoint": str(a.checkpoint), "epoch": int(ck.get("epoch", -1)), "weights": a.weights, "n": a.n,
       "fid": round(float(fid), 3), "fid_stats": a.fid_stats, "params_M": round(sum(p.numel() for p in net.parameters()) / 1e6, 1),
       "seconds": round(time.time() - t0)}
print(json.dumps(rec), flush=True)
if a.out:
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
