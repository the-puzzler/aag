#!/usr/bin/env python
"""Build a uint8 segment cache from the p-doom/doom-dataset ArrayRecord files.

One 16-frame segment per record, so every particle comes from a DISTINCT
episode -- this dataset was chosen precisely because UCF-101's 87,953 segments
came from only 9,537 clips (~9 near-duplicates each), which capped the usable
latent dimension. Here N == number of independent episodes.

Per-record handling, all verified against the data rather than assumed:
  * sequence_length varies (40 or 160); byte length agrees with it.
  * frames 0-12 are a deterministic green episode-start flash -- dropped.
  * rows 52-59 are the static HUD (temporal std ~1 vs ~24) -- cropped.
  * the play area is then centre-cropped to a square and resized to 64x64,
    so nothing is aspect-distorted.
"""
from __future__ import annotations
import argparse, glob, json, pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from array_record.python.array_record_data_source import ArrayRecordDataSource

RAW_H, RAW_W = 60, 80
FLASH_FRAMES = 13      # green start-of-episode flash
HUD_ROW = 52           # first HUD row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/doom/p-doom/train")
    ap.add_argument("--out", type=Path, default=Path("/data/doom/cache_train"))
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--per-record", type=int, default=3,
                     help="CONSECUTIVE chunks per episode; adjacency is what makes "
                          "(previous chunk, action) conditioning possible")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.src}/*.array_record"))
    if args.limit_files:
        files = files[:args.limit_files]
    print(f"{len(files)} array_record files", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cap = len(files) * 100 * args.per_record
    segments = np.lib.format.open_memmap(
        args.out / "segments.npy", mode="w+", dtype=np.uint8,
        shape=(cap, args.frames, args.size, args.size, 3))
    acts, act_seqs, rec_ids, chunk_idx = [], [], [], []
    w = 0

    for fi, f in enumerate(files):
        src = ArrayRecordDataSource([f])
        for ri in range(len(src)):
            d = pickle.loads(src[ri])
            n = int(d["sequence_length"])
            usable = n - FLASH_FRAMES
            if usable < args.frames:
                continue
            v = np.frombuffer(d["raw_video"], np.uint8).reshape(n, RAW_H, RAW_W, 3)
            a = np.asarray(d["actions"])
            n_take = min(args.per_record, usable // args.frames)
            # CONSECUTIVE, from a random aligned offset: chunk k is immediately
            # followed by chunk k+1, so (record_id, chunk_idx) recovers the
            # temporal ordering the conditioning needs.
            span = n_take * args.frames
            base = FLASH_FRAMES + int(rng.integers(0, usable - span + 1))
            for k in range(n_take):
                s = base + k * args.frames
                clip = v[s:s + args.frames]                       # (T,60,80,3)
                clip = clip[:, :HUD_ROW]                          # drop HUD
                ch, cw = clip.shape[1], clip.shape[2]   # NOT h, w -- w is the write counter
                side = min(ch, cw)
                y0, x0 = (ch - side) // 2, (cw - side) // 2
                clip = clip[:, y0:y0 + side, x0:x0 + side]        # square crop
                t = torch.from_numpy(clip.copy()).permute(0, 3, 1, 2).float()
                t = F.interpolate(t, size=(args.size, args.size),
                                  mode="bilinear", align_corners=False)
                segments[w] = (t.round_().clamp_(0, 255).to(torch.uint8)
                                .permute(0, 2, 3, 1).numpy())       # (T,S,S,3)
                w += 1
                seq = a[s:s + args.frames].astype(np.uint8)
                act_seqs.append(seq)                              # per-frame, for
                acts.append(int(np.bincount(seq).argmax()))       # world-model use
                rec_ids.append(fi * 100 + ri)
                chunk_idx.append(k)
        if (fi + 1) % 25 == 0:
            print(f"  {fi+1}/{len(files)} files -> {w:,} chunks", flush=True)

    segments.flush()
    assert w == len(acts) == len(rec_ids) == len(chunk_idx), (
        f"write counter {w} disagrees with metadata lengths "
        f"{len(acts)}/{len(rec_ids)}/{len(chunk_idx)}")
    labels = np.asarray(acts, dtype=np.int64)
    record_ids = np.asarray(rec_ids, dtype=np.int64)
    np.save(args.out / "labels.npy", labels)
    np.save(args.out / "chunk_idx.npy", np.asarray(chunk_idx, dtype=np.int64))
    np.save(args.out / "clip_ids.npy", record_ids)
    # full action sequence per segment: needed for action-conditioned
    # next-frame prediction, where the condition is (past frames, action)
    # rather than a single discrete class.
    np.save(args.out / "action_seqs.npy", np.stack(act_seqs))
    meta = dict(n_segments=int(w), capacity=int(cap), frames=args.frames,
                size=args.size, n_records=int(len(np.unique(record_ids))),
                n_actions=int(labels.max()) + 1, per_record=args.per_record,
                flash_frames=FLASH_FRAMES, hud_row=HUD_ROW,
                gb=float(segments.nbytes / 1e9))
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
