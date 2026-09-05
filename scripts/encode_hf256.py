#!/usr/bin/env python
"""Encode an HF-parquet 256x256 train split into AE latents = the particle file.

Writes {"h": (N, D) float16, "label": (N,) int64, "encoder": id, "grid": g,
"latent_channels": C, "dataset", "n_particles"} where row i is global parquet
row i (see aag.hf256 for why that order is the identity contract).

--encoder is either a diffusers DC-AE id (mit-han-lab/dc-ae-f32c32-in-1.0-diffusers)
or a path to one of this repo's own AE checkpoints. h is stored as the raw
spatial latent flattened C x g x g -> C*g*g in that order, so run_assignment's
whiten(rotate=False) keeps grid cell j at coordinate j.

Chunked: decodes `--chunk` rows from parquet, encodes, appends. ImageNet at
1.28M rows is ~2.5 GB of fp16 latents at D=2048.
"""
from __future__ import annotations

import argparse, os, time
from pathlib import Path

import torch

from aag.hf256 import DATASETS, load_uint8, manifest, to_float

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=list(DATASETS), required=True)
ap.add_argument("--root", default=os.environ.get("AAG_HF_ROOT", "/data/aag_data/hf"))
ap.add_argument("--encoder", default="mit-han-lab/dc-ae-f32c32-in-1.0-diffusers")
ap.add_argument("--N", type=int, default=0, help="0 = all train rows")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--chunk", type=int, default=16384, help="parquet rows decoded per round")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--amp", action="store_true", help="bf16 autocast for the encoder")
ap.add_argument("--flip", action="store_true", help="encode horizontally flipped images (pair-doubling augmentation)")
ap.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)),
                help="shard index: encode only rows [rank*N/world, (rank+1)*N/world) and write <out>.shard<rank>")
ap.add_argument("--world", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
ap.add_argument("--merge", action="store_true",
                help="instead of encoding, concatenate <out>.shard0..<world-1> into <out> (rank 0 after a sharded run)")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

if a.merge:
    shards = [torch.load(f"{a.out}.shard{r}", map_location="cpu", weights_only=False) for r in range(a.world)]
    m = dict(shards[0]); m["h"] = torch.cat([s["h"] for s in shards]); m["label"] = torch.cat([s["label"] for s in shards])
    m["n_particles"] = m["h"].shape[0]
    assert m["n_particles"] == sum(s["n_particles"] for s in shards)
    torch.save(m, a.out)
    for r in range(a.world):
        os.remove(f"{a.out}.shard{r}")
    print(f"merged {a.world} shards -> {a.out}  h {tuple(m['h'].shape)}", flush=True)
    raise SystemExit(0)

dev = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"   # set_device needs an index, not bare "cuda"
torch.cuda.set_device(dev)

N_all = manifest(a.root, a.dataset)["offsets"][-1]
N_total = min(a.N, N_all) if a.N else N_all
from aag.hf256 import rank_slice
lo0, hi0 = rank_slice(N_total, a.rank, a.world)     # this shard's global rows
N = hi0 - lo0

if a.encoder.endswith(".pt"):
    from aag.ae import AutoEncoder
    ck = torch.load(a.encoder, map_location=dev, weights_only=False)
    ae = AutoEncoder(ck["latent_dim"], ch=ck["channels"], architecture=ck["architecture"],
                     image_size=ck["image_size"], grid=ck.get("grid", 4)).to(dev).eval()
    # checkpoints written under torch.compile carry an "_orig_mod." prefix on every key
    ae.load_state_dict({k.replace("_orig_mod.", "", 1): v for k, v in ck["model_state_dict"].items()})
    grid, C = ck.get("grid", 4), ck["latent_dim"] // ck.get("grid", 4) ** 2
    encode = lambda x: ae.enc(x)
elif a.encoder.startswith("titok:"):
    # ByteDance TiTok VAE (continuous 1-D tokenizer): 32/64/128 tokens x 16 dims, no spatial grid.
    # Chosen for CelebA-HQ because independent-N/d matters (METHOD.md s5): 28k/2048 = 14 with
    # DC-AE, 28k/512 = 55 with TiTok-LL-32 -- at the price of reconstruction (MSE 0.026 vs 0.010).
    # The posterior MEAN is the latent (deterministic encoder); TiTok takes images in [0, 1].
    import sys; sys.path.insert(0, os.environ.get("TITOK_REPO", "/data/tmp/1d-tokenizer"))
    from modeling.titok import TiTok
    tok = TiTok.from_pretrained(a.encoder[len("titok:"):]).to(dev).eval(); tok.requires_grad_(False)
    with torch.no_grad():
        z0 = tok.quantize(tok.encoder(pixel_values=torch.zeros(1, 3, 256, 256, device=dev), latent_tokens=tok.latent_tokens)).mode()
    C, grid = z0.flatten(1).shape[1], 0        # grid=0: flat latent, the generator uses a learned stem
    encode = lambda x: tok.quantize(tok.encoder(pixel_values=(x + 1) / 2, latent_tokens=tok.latent_tokens)).mode().flatten(1)
else:
    from diffusers import AutoencoderDC
    ae = AutoencoderDC.from_pretrained(a.encoder, torch_dtype=torch.float32).to(dev).eval()
    with torch.no_grad():
        z0 = ae.encode(torch.zeros(1, 3, DATASETS[a.dataset]["image_size"], DATASETS[a.dataset]["image_size"], device=dev)).latent
    C, grid = z0.shape[1], z0.shape[2]
    encode = lambda x: ae.encode(x).latent.flatten(1)
D = C * grid * grid if grid else C
print(f"encoder {a.encoder}: latent {C}x{grid}x{grid} = {D} dims; shard {a.rank}/{a.world}: rows [{lo0:,}, {hi0:,}) "
      f"of {N_total:,} {a.dataset} rows, amp={a.amp}", flush=True)

h = torch.empty(N, D, dtype=torch.float16)
labels = torch.empty(N, dtype=torch.int64)
t0 = time.time()
for lo in range(0, N, a.chunk):
    hi = min(N, lo + a.chunk)
    x, y = load_uint8(a.root, a.dataset, lo0 + lo, lo0 + hi, workers=a.workers, chunk=max(512, a.chunk // (2 * a.workers)))
    labels[lo:hi] = y
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
        for i in range(0, hi - lo, a.batch):
            xb = to_float(x[i:i + a.batch]).to(dev, non_blocking=True)
            if a.flip: xb = xb.flip(-1)                  # (B,3,H,W): flip W
            h[lo + i:lo + i + xb.shape[0]] = encode(xb).float().half().cpu()
    done = hi; rate = done / (time.time() - t0)
    print(f"  {done:>9,}/{N:,}  {rate:6.0f} img/s  eta {(N - done) / rate / 60:5.1f} min", flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
out_path = Path(f"{a.out}.shard{a.rank}") if a.world > 1 else a.out
torch.save({"h": h, "label": labels, "encoder": a.encoder, "grid": grid, "latent_channels": C,
            "dataset": a.dataset, "root": a.root, "n_particles": N, "shard": (a.rank, a.world, lo0, hi0),
            "h_stats": {"mean": h.float().mean().item(), "std": h.float().std().item()}},
           out_path)
print(f"saved {out_path}  h {tuple(h.shape)} fp16  labels {labels.min().item()}..{labels.max().item()}  "
      f"{time.time() - t0:.0f}s", flush=True)
