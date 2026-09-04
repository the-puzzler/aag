#!/usr/bin/env python
"""Original / old AE / new AE reconstructions side by side.

WHY THIS IS THE DELIVERABLE AND THE NUMBERS ARE CONTEXT. The whole point of
adding an adversary is detail that MSE resolves to a conditional mean, and MSE
therefore cannot score it -- a sharper reconstruction that puts texture in
almost-but-not-exactly the right place scores WORSE on L2 than the blur it
replaced. LPIPS is better but is still a VGG-feature L2, and this project has
already been burned three times by frame-difference statistics standing in for
looking at frames. So the comparison that decides whether the adversary bought
anything is visual.

Frame selection is deliberately not random. An adversary's effect lives in fine
texture, so most rows are drawn from the highest-detail frames available --
ranked by high-frequency energy, measured as the residual after a 2x2 box blur,
which is exactly the band an AE at 48x compression loses first. Control rows are
taken from the MIDDLE of that ranking rather than the tail, because the
least-detailed frames are flat sky where every AE looks identical and the row
would say nothing either way.

INVENTORY FRAMES ARE EXCLUDED (gui.npy), and this is not cosmetic. Ranking the
raw pool by high-frequency energy returns almost nothing but open inventory
screens: a dense grid of item icons and 1px text is the highest-frequency
content in the corpus by a wide margin. Those frames say nothing about whether
the AE reconstructs WORLD texture -- leaves, gravel, stone, wood grain -- which
is what the generator has to render. The first version of this script picked
four GUI screens out of five detail rows.

Each row is also shown as a zoomed centre crop, because at 64x64 a side-by-side
of whole frames is too small to judge and the eye needs the detail at usable
size.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from aag.ae import AutoEncoder
from aag.datasets import open_segments


def load_ae(path: Path, dev: str):
    c = torch.load(path, map_location=dev, weights_only=False)
    ae = AutoEncoder(c["latent_dim"], ch=c["channels"],
                     architecture=c["architecture"], image_size=c["image_size"],
                     grid=c.get("grid", 4)).to(dev).eval()
    sd = c["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    ae.load_state_dict(sd)
    return ae, c


def hf_energy(x: torch.Tensor) -> torch.Tensor:
    """High-frequency energy: residual after a 2x2 box blur, per frame.

    The band an AE loses first at this compression, and the band an adversary is
    supposed to restore -- so it is the right axis to rank frames on. Note this
    measures how much high-frequency content is PRESENT, not whether it is in
    the right place: a model hallucinating texture scores as high as one
    reproducing it. Use it to select frames and to check the adversary is adding
    detail at all, never as a quality score.
    """
    blur = F.avg_pool2d(x, 2)
    blur = F.interpolate(blur, size=x.shape[-2:], mode="nearest")
    return (x - blur).abs().mean(dim=(1, 2, 3))


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(-1, 1).add(1).mul(127.5)
            .permute(1, 2, 0).cpu().numpy().round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", type=Path,
                    default=Path("/data/aag_results/results_vpt/assign_12d_lag1/"
                                 "assignment.pt"))
    ap.add_argument("--old-ae", type=Path,
                    default=Path("/data/aag_results/results_vpt/"
                                 "ae_dcae_ch192_dim256_cont/checkpoints/"
                                 "ae_doom_frames_dcae_lpips_ch192_dim256_ep4.pt"))
    ap.add_argument("--new-ae", type=Path, required=True)
    ap.add_argument("--cache", default="/opt/dlami/nvme/vpt_full")
    ap.add_argument("--pool", type=int, default=768,
                    help="frames to scan when ranking by detail")
    ap.add_argument("--gui-free", action="store_true", default=True,
                    help="exclude open-inventory frames via gui.npy. On by "
                         "default because they dominate a high-frequency "
                         "ranking and say nothing about world texture")
    ap.add_argument("--allow-gui", dest="gui_free", action="store_false")
    ap.add_argument("--detail-rows", type=int, default=5)
    ap.add_argument("--typical-rows", type=int, default=2,
                    help="control rows from the middle of the detail ranking, "
                         "so the picture is not cherry-picked to the metric")
    ap.add_argument("--zoom", type=int, default=4, help="upscale for the full frame")
    ap.add_argument("--crop", type=int, default=32,
                    help="centre-crop size, shown at 2x --zoom")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = "cuda"
    old, oc = load_ae(args.old_ae, dev)
    new, nc = load_ae(args.new_ae, dev)
    print(f"old: {args.old_ae.name}  epochs={oc.get('epochs')} "
          f"gan_weight={oc.get('gan_weight')}")
    print(f"new: {args.new_ae.name}  epochs={nc.get('epochs')} "
          f"gan_weight={nc.get('gan_weight')}")

    A = torch.load(args.assignment, map_location="cpu", weights_only=False)
    ci, fi = A["chunk"].numpy(), A["frame"].numpy()
    segs = open_segments(args.cache)
    rng = np.random.default_rng(0)

    cand = rng.permutation(len(ci))
    if args.gui_free:
        gui = np.load(f"{args.cache}/gui.npy", mmap_mode="r")
        keep = ~np.asarray(gui[ci[cand], fi[cand]]).astype(bool)
        n_drop = int((~keep).sum())
        cand = cand[keep]
        print(f"gui-free: dropped {n_drop:,} of {n_drop + len(cand):,} candidates "
              f"({100.0 * n_drop / (n_drop + len(cand)):.1f}% are inventory frames)")
    pool = cand[:args.pool]

    def fetch(idx):
        x = np.stack([np.asarray(segs[int(ci[p])][int(fi[p])]) for p in idx])
        return (torch.from_numpy(x).permute(0, 3, 1, 2).float()
                .div_(127.5).sub_(1.0).to(dev))

    with torch.no_grad():
        e = torch.cat([hf_energy(fetch(pool[i:i + 128]))
                       for i in range(0, len(pool), 128)])
    order = torch.argsort(e, descending=True).cpu().numpy()
    detail = pool[order[:args.detail_rows]]
    mid = pool[order[len(order) // 2: len(order) // 2 + args.typical_rows]]
    sel = np.concatenate([detail, mid])
    labels = ([f"detail {i+1}" for i in range(len(detail))] +
              [f"typical {i+1}" for i in range(len(mid))])

    with torch.no_grad():
        x = fetch(sel)
        r_old = old.dec(old.enc(x))
        r_new = new.dec(new.enc(x))

    Z, C, ZC = args.zoom, args.crop, args.zoom * 2
    S = 64
    cell = S * Z
    crop_w = C * ZC
    pad, hdr, lbl = 8, 28, 78
    widths = [cell] * 3 + [crop_w] * 3
    total_w = lbl + sum(widths) + pad * (len(widths) + 1)
    total_h = hdr + len(sel) * (cell + pad) + pad
    canvas = Image.new("RGB", (total_w, total_h), (18, 18, 20))
    d = ImageDraw.Draw(canvas)

    heads = ["original", f"old AE (ep{oc.get('epochs')})",
             f"new AE +GAN (ep{nc.get('epochs')})",
             "original zoom", "old AE zoom", "new AE zoom"]
    xs, cx = [], lbl + pad
    for w in widths:
        xs.append(cx)
        cx += w + pad
    for xo, h in zip(xs, heads):
        d.text((xo + 4, 9), h, fill=(235, 235, 235))

    for r in range(len(sel)):
        y = hdr + r * (cell + pad)
        d.text((6, y + cell // 2 - 6), labels[r], fill=(170, 170, 175))
        lo = (S - C) // 2
        for c, t in enumerate((x[r], r_old[r], r_new[r])):
            im = Image.fromarray(to_u8(t))
            canvas.paste(im.resize((cell, cell), Image.NEAREST), (xs[c], y))
            cr = im.crop((lo, lo, lo + C, lo + C))
            canvas.paste(cr.resize((crop_w, crop_w), Image.NEAREST), (xs[3 + c], y))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"\nwrote {args.out}  ({total_w}x{total_h})")

    with torch.no_grad():
        print(f"\non these {len(sel)} frames only -- far too few to rank models, "
              f"printed so the image and the numbers describe the same frames:")
        for nm, r in (("old", r_old), ("new", r_new)):
            print(f"   {nm}: mse {F.mse_loss(r.clamp(-1,1), x):.5f}   "
                  f"hf-energy {hf_energy(r.clamp(-1,1)).mean():.5f}")
        print(f"   real: hf-energy {hf_energy(x).mean():.5f}  "
              f"<- what the adversary is meant to restore; closer is better, but "
              f"hf-energy cannot tell restored detail from hallucinated detail, "
              f"so the image decides")


if __name__ == "__main__":
    main()
