#!/usr/bin/env python
"""Rare / contradictory attribute-combination grid for a trained conditional
generator. Rationale: if z is not independent of the condition, p(z|c) deviates
from p(z) most for UNUSUAL c, so sampling z ~ N(0,I) and pairing it with a rare
combination lands furthest off-manifold. Typical-condition grids will not show
this; the tail is the diagnostic. Works for both the two-stage (through frozen
AE decoder) and direct-to-pixel conditional generators -- detected from the
checkpoint keys."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image

COMBOS = [
    ["Male", "Heavy_Makeup"],
    ["Male", "Wearing_Lipstick"],
    ["Bald", "Wearing_Lipstick"],
    ["Male", "Blond_Hair", "Heavy_Makeup"],
    ["Bald", "Eyeglasses", "Wearing_Hat"],
    ["Young", "Gray_Hair"],
    ["Chubby", "Bald", "Eyeglasses"],
    ["Male", "Arched_Eyebrows", "Rosy_Cheeks"],
]

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n-cols", type=int, default=8)
ap.add_argument("--seed", type=int, default=3)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
names = ck["attr_names"]; nz = ck["dim_z"]; na = ck["n_attrs"]
idx = {n: i for i, n in enumerate(names)}

if "ae_checkpoint" in ck:                       # two-stage: (z,c) -> h -> ae.dec
    from gga.ae import AutoEncoder
    from gga.decoder import ResidualDecoder as Gen
    aeck = torch.load(ck["ae_checkpoint"], map_location=dev, weights_only=False)
    ae = AutoEncoder(aeck["latent_dim"], ch=aeck["channels"],
                     architecture=aeck["architecture"],
                     image_size=aeck["image_size"]).to(dev).eval()
    ae.load_state_dict(aeck["model_state_dict"])
    net = Gen(nz + na, out_dim=ck["dim_h"]).to(dev).eval()
    net.load_state_dict(ck["model_state_dict"])
    render = lambda zc: ae.dec(net(zc))
    kind = "two-stage"
else:                                            # direct-to-pixel: (z,c) -> image
    from gga.ae import ResidualDecoder as ConvDecoder
    net = ConvDecoder(nz + na, ch=ck["ch"], image_size=ck["image_size"]).to(dev).eval()
    net.load_state_dict(ck["model_state_dict"])
    render = lambda zc: net(zc)
    kind = "direct-to-pixel"

torch.manual_seed(args.seed)
Z = torch.randn(args.n_cols, nz, device=dev)     # SAME z across every row
rows, spreads = [], []
with torch.no_grad():
    for combo in COMBOS:
        c = torch.zeros(1, na, device=dev)
        for a in combo:
            c[0, idx[a]] = 1.0
        imgs = render(torch.cat([Z, c.repeat(args.n_cols, 1)], 1))
        rows.append(imgs)
        cl = imgs.clamp(-1, 1)
        spreads.append(((cl.amax(1) - cl.amin(1)).mean().item(), "+".join(combo)))
save_image((torch.cat(rows).clamp(-1, 1) + 1) / 2, args.out, nrow=args.n_cols)
print(f"[{kind}] saved: {args.out}")
print("  rows (top->bottom), with channel-spread (high => garish/off-manifold):")
for s, lab in spreads:
    print(f"    {s:.3f}  {lab}")
