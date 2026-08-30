#!/usr/bin/env python
"""Build world-model particles from the VPT cache.

A particle is one target frame plus the conditioning needed to generate it:

    h_target   (dim,)            what the generator must produce
    h_context  (ctx*dim,)        the ctx preceding frame latents, recency-weighted
    action     (int)             the 81-way categorical, kept for exact grouping
    action_vec (A,)              continuous action features, for a DISTANCE on
                                 the action side instead of an exact match

Design decisions and why:

  * Frame AE only. Encoding the action into the latent was considered and
    rejected: the action is already known exactly, so pushing it through a
    reconstruction bottleneck can only lose fidelity, and it would bury the
    visual-vs-behavioural weighting of the conditioning metric inside encoder
    weights where it cannot be tuned. Appending it keeps that weight explicit.

  * ctx=24 frames (1.2s at 20fps). 64 was the first plan; at 64x64 latents that
    is a 4096-dim cond, which is 213 GB resident at 13M particles and does not
    fit the GPU. 24 frames is 1536 dims, 3.9 GB at the Doom run's 630k particles.

  * Recency weighting folds into the vector, not the metric: block i is scaled by
    sqrt(gamma**i), so a plain L2 on the result IS the recency-weighted L2. That
    means no change to cond_distance and no new metric code.

  * action_vec is signed-log1p then standardised. Raw, dx alone is 81% of the L2
    variance and the key bits are ~0% -- a nearest-k on raw actions would be a
    nearest-k on dx. Mouse is then upweighted by sqrt(MOUSE_W) because mouse
    magnitude explains 16.4% of frame-to-frame pixel variance against 2.1% for
    all keys combined. The E column is dropped: it only fired on inventory-open
    frames, so it is near-constant and standardising it manufactures noise.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from aag.ae import AutoEncoder
from aag.datasets import open_segments

KEY_COLS = list(range(7))          # W A S D space shift ctrl -- drop E (col 7)
MOUSE_W = 8.0                      # mouse:key variance ratio from the pixel-change fit


def build_action_vec(keys, mouse):
    """(N,7) keys + (N,2) mouse -> standardised, mouse-upweighted features."""
    k = keys[:, KEY_COLS].astype(np.float32)
    m = np.sign(mouse).astype(np.float32) * np.log1p(np.abs(mouse.astype(np.float32)))
    v = np.concatenate([k, m], 1)
    sd = v.std(0)
    sd[sd < 1e-6] = 1.0
    v = (v - v.mean(0)) / sd
    v[:, -2:] *= np.sqrt(MOUSE_W)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/opt/dlami/nvme/vpt_stage")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--context", type=int, default=24)
    ap.add_argument("--gamma", type=float, default=0.95,
                    help="recency decay; block i scaled by sqrt(gamma**i) so plain "
                         "L2 equals the recency-weighted L2")
    ap.add_argument("--per-chunk", type=int, default=1,
                    help="target frames sampled per chunk. Chunks are 80 frames so "
                         "80-context targets are available; 1 keeps particles "
                         "decorrelated, which is the UCF-101 lesson")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--limit-chunks", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    ae = AutoEncoder(ck["latent_dim"], ch=ck["channels"],
                     architecture=ck["architecture"],
                     image_size=ck["image_size"]).to(dev).eval()
    ae.load_state_dict(ck["model_state_dict"])
    dim = ck["latent_dim"]
    print(f"AE dim={dim} arch={ck['architecture']} ch={ck['channels']} "
          f"ep={ck.get('epochs')} test_mse={ck.get('test_mse'):.5f}", flush=True)

    R = args.cache
    segs = open_segments(R)
    n_valid = len(np.load(f"{R}/labels.npy"))
    frames = segs.shape[1]
    C = args.context
    if frames <= C:
        raise SystemExit(f"chunk is {frames} frames, need > context {C}")
    acts = np.load(f"{R}/action_seqs.npy", mmap_mode="r")
    keys = np.load(f"{R}/keys.npy", mmap_mode="r")
    mouse = np.load(f"{R}/mouse.npy", mmap_mode="r")
    clip = np.load(f"{R}/clip_ids.npy")

    n_chunks = n_valid if args.limit_chunks is None else min(n_valid, args.limit_chunks)
    rng = np.random.default_rng(args.seed)
    # Which chunks, not just how many. The cache is written in download order
    # across four separate VPT index files, so taking a prefix would quietly
    # weight the particle set toward whichever contractor batches were fetched
    # first. Sample the subset instead; reading is random-access either way.
    chunk_ids = (np.arange(n_valid) if n_chunks == n_valid
                 else np.sort(rng.choice(n_valid, size=n_chunks, replace=False)))
    print(f"{n_chunks:,} chunks x {args.per_chunk} = "
          f"{n_chunks*args.per_chunk:,} particles, cond_dim {C*dim}", flush=True)

    w = np.sqrt(args.gamma ** np.arange(C - 1, -1, -1)).astype(np.float32)  # oldest..newest
    print(f"recency weights sqrt(gamma^i): newest {w[-1]:.3f} ... oldest {w[0]:.3f}",
          flush=True)

    # Preallocate rather than append-then-stack. Accumulating 1.66M separate
    # (6144,) arrays in a Python list costs per-object overhead on top of the
    # data, and np.stack then duplicates the whole 40.9 GB block transiently --
    # measured at 23.5 GB anonymous by 400k particles, projecting to ~139 GB
    # peak against 124 GB of RAM. Writing into preallocated arrays removes both.
    N_total = n_chunks * min(args.per_chunk, frames - C)
    H_t = np.empty((N_total, dim), dtype=np.float32)
    H_c = np.empty((N_total, C * dim), dtype=np.float32)
    print(f"preallocated h_target {H_t.nbytes/1e9:.1f} GB + h_context "
          f"{H_c.nbytes/1e9:.1f} GB for {N_total:,} particles", flush=True)
    A_i, A_v, EP = [], [], []
    n_written = 0
    buf_frames, buf_meta = [], []

    def flush():
        nonlocal n_written
        if not buf_frames:
            return
        x = torch.from_numpy(np.concatenate(buf_frames, 0))
        x = x.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0).to(dev)
        with torch.no_grad():
            h = torch.cat([ae.enc(x[i:i + args.batch])
                           for i in range(0, len(x), args.batch)]).cpu().numpy()
        off = 0
        for (ci, t, nf) in buf_meta:
            block = h[off:off + nf]
            off += nf
            H_t[n_written] = block[-1]                              # target = frame t
            ctx = block[:-1] * w[:, None]                            # recency-scaled
            H_c[n_written] = ctx.reshape(-1)
            n_written += 1
            A_i.append(int(acts[ci, t]))
            A_v.append((int(ci), int(t)))
            EP.append(int(clip[ci]))
        buf_frames.clear(); buf_meta.clear()

    for n_done, ci in enumerate(chunk_ids):
        ci = int(ci)
        ts = rng.choice(np.arange(C, frames), size=min(args.per_chunk, frames - C),
                        replace=False)
        seg = np.asarray(segs[ci])
        for t in ts:
            t = int(t)
            buf_frames.append(seg[t - C:t + 1])                      # C context + target
            buf_meta.append((ci, t, C + 1))
        if len(buf_meta) >= 256:
            flush()
        if n_done and n_done % 20000 == 0:
            print(f"  {n_done:,}/{n_chunks:,} chunks, {n_written:,} particles", flush=True)
    flush()

    kv = np.stack([np.asarray(keys[c, t]) for c, t in A_v])
    mv = np.stack([np.asarray(mouse[c, t]) for c, t in A_v])
    act_vec = build_action_vec(kv, mv)

    out = {
        # zero-copy views of the preallocated blocks, trimmed to what was filled
        "h_target": torch.from_numpy(H_t[:n_written]),
        "h_context": torch.from_numpy(H_c[:n_written]),
        "action": torch.tensor(A_i, dtype=torch.long),
        "action_vec": torch.from_numpy(act_vec.astype(np.float32)),
        "episode": torch.tensor(EP, dtype=torch.long),
        # (chunk, frame) for every particle, in particle order. The generator
        # needs these to fetch its pixel targets out of the cache -- without
        # them a particle file cannot be turned into training data at all.
        "chunk": torch.tensor([c for c, _ in A_v], dtype=torch.long),
        "frame": torch.tensor([t for _, t in A_v], dtype=torch.long),
        "dim": dim, "context": C, "gamma": args.gamma,
        "cache": str(R), "checkpoint": str(args.checkpoint),
        "n_segments_snapshot": int(n_valid),      # freeze: order IS identity
        "chunk_ids": torch.from_numpy(chunk_ids),  # which chunks were sampled
        "mouse_weight": MOUSE_W, "key_cols": KEY_COLS,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"\nsaved {n_written:,} particles -> {args.out}")
    print(f"  h_target  {tuple(out['h_target'].shape)}")
    print(f"  h_context {tuple(out['h_context'].shape)}")
    print(f"  action_vec {tuple(out['action_vec'].shape)}  "
          f"distinct actions {len(set(A_i))}/81")
    print(f"  episodes  {len(set(EP)):,}")


if __name__ == "__main__":
    main()
