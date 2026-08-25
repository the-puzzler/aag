#!/usr/bin/env python
"""Convert a single-file VPT segment cache into shards, in place.

Preserves work already downloaded when switching layouts. Copies only the VALID
prefix (len(labels)), not the preallocated capacity, and reconciles done.jsonl to
that prefix -- an unclean death leaves videos marked done whose segments were
never covered by a sidecar flush, and those must be re-fetched rather than
silently skipped.

Flushes and unmaps after every shard so the conversion itself does not recreate
the unbounded-dirty-pages pattern it exists to fix.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

import numpy as np


def mem() -> str:
    try:
        d = {}
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            if k in ("Dirty", "Writeback"):
                d[k] = int(v.split()[0]) / 1024
        return f"dirty {d.get('Dirty', 0):.0f}M wb {d.get('Writeback', 0):.0f}M"
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--shard-segments", type=int, default=8192)
    ap.add_argument("--keep-original", action="store_true",
                    help="leave segments.npy in place (uses 2x disk; note the "
                         "loaders prefer the single file when both exist)")
    args = ap.parse_args()

    src = args.cache / "segments.npy"
    if not src.exists():
        raise SystemExit(f"{src} not found -- already sharded?")
    labels = np.load(args.cache / "labels.npy")
    n = int(labels.shape[0])
    a = np.load(src, mmap_mode="r")
    print(f"source {a.shape} ({a.shape[0]:,} capacity), valid prefix {n:,} segments")
    per = int(np.prod(a.shape[1:]))
    print(f"copying {n * per / 1e9:.0f} GB into shards of {args.shard_segments:,}")

    n_shards = (n + args.shard_segments - 1) // args.shard_segments
    for si in range(n_shards):
        lo = si * args.shard_segments
        hi = min(n, lo + args.shard_segments)
        p = args.cache / f"segments_{si:05d}.npy"
        if p.exists():
            existing = np.load(p, mmap_mode="r")
            if existing.shape[0] == hi - lo:
                print(f"  shard {si:05d} exists ({hi-lo:,}) -- skipping")
                del existing
                continue
            del existing
        out = np.lib.format.open_memmap(
            p, mode="w+", dtype=a.dtype, shape=(hi - lo,) + a.shape[1:])
        # chunk the copy so neither side holds a huge dirty window
        step = 512
        for j in range(lo, hi, step):
            k = min(step, hi - j)
            out[j - lo:j - lo + k] = a[j:j + k]
        out.flush()
        del out
        print(f"  shard {si:05d}: {hi-lo:,} segments  [{mem()}]", flush=True)

    # reconcile done.jsonl to the flushed prefix
    dp = args.cache / "done.jsonl"
    if dp.exists():
        recs = [json.loads(l) for l in open(dp) if l.strip()]
        keep, run = [], 0
        for r in recs:
            run += int(r.get("chunks", 0))
            if run > n:
                break
            keep.append(r)
        if len(keep) != len(recs):
            print(f"reconciling done.jsonl: {len(recs)} -> {len(keep)} videos "
                  f"({len(recs)-len(keep)} to be re-fetched)")
            with open(dp, "w") as fh:
                for r in keep:
                    fh.write(json.dumps(r) + "\n")

    meta_p = args.cache / "meta.json"
    meta = json.load(open(meta_p)) if meta_p.exists() else {}
    meta["shard_size"] = args.shard_segments
    meta["n_segments"] = n
    json.dump(meta, open(meta_p, "w"), indent=2)

    del a
    if not args.keep_original:
        os.remove(src)
        print(f"removed {src}")
    print(f"done: {n_shards} shards, {n:,} segments")


if __name__ == "__main__":
    main()
