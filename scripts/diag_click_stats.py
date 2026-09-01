#!/usr/bin/env python
"""Press rates and run-length structure of the recovered mouse buttons.

Two questions this answers before an assignment is launched on the new
representation:

  * Are attack/use frequent enough to be worth a marginal of their own, and not
    so rare that the 0/1 scaling argument for E applies to them too?
  * How autocorrelated are they?  attack is HELD for the seconds it takes to
    break a block, so its per-frame rate and its per-EVENT rate are different
    quantities, and a long hold means a segment tends to be all-attack or
    no-attack.  That decides whether a binary grouping is the right partition or
    whether onset (newButtons-like) deserves its own.

Only rows whose clip actually appears in clicks_done.jsonl as ok are counted --
every other row is still the zero-fill from array creation, which would read as
"the player never clicks" and quietly bias every rate downward.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np


def runs_of_ones(a: np.ndarray) -> np.ndarray:
    """Lengths of maximal runs of 1 along the last axis, flattened."""
    p = np.zeros((a.shape[0], a.shape[1] + 2), np.int8)
    p[:, 1:-1] = a
    d = np.diff(p.astype(np.int8), axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    if len(starts) == 0:
        return np.zeros(0, np.int64)
    return (ends[:, 1] - starts[:, 1]).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("/data/vpt/cache_train"))
    ap.add_argument("--sample-clips", type=int, default=4000)
    args = ap.parse_args()

    C = args.cache
    done = [json.loads(l) for l in (C / "done.jsonl").read_text().splitlines() if l.strip()]
    ok_rel = set()
    cd = C / "clicks_done.jsonl"
    if cd.exists():
        for l in cd.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                if d.get("ok"):
                    ok_rel.add(d["relpath"])
    ok_clip = np.array([done[i]["relpath"] in ok_rel for i in range(len(done))])
    clip_ids = np.load(C / "clip_ids.npy")
    n_valid = len(clip_ids)
    row_ok = ok_clip[clip_ids]
    print(f"clips patched {ok_clip.sum():,}/{len(done):,}   "
          f"segments covered {row_ok.sum():,}/{n_valid:,} "
          f"({100.0*row_ok.mean():.2f}%)", flush=True)

    rows = np.where(row_ok)[0]
    if len(rows) == 0:
        raise SystemExit("no covered rows yet")
    rng = np.random.default_rng(0)
    if len(rows) > args.sample_clips:
        rows = np.sort(rng.choice(rows, args.sample_clips, replace=False))

    clicks = np.load(C / "clicks.npy", mmap_mode="r")
    keys = np.load(C / "keys.npy", mmap_mode="r")
    hotbar = np.load(C / "hotbar.npy", mmap_mode="r")
    dwheel = np.load(C / "dwheel.npy", mmap_mode="r")
    Cl = np.asarray(clicks[rows]).astype(np.int8)          # (n,80,2)
    K = np.asarray(keys[rows]).astype(np.int8)
    HB = np.asarray(hotbar[rows])
    DW = np.asarray(dwheel[rows]).astype(np.float32)
    n, F, _ = Cl.shape
    print(f"\nsampled {n:,} segments x {F} frames = {n*F:,} frames\n", flush=True)

    for j, nm in ((0, "attack (LMB)"), (1, "use (RMB)")):
        a = Cl[:, :, j]
        r = runs_of_ones(a)
        seg_any = (a.max(1) > 0).mean()
        seg_all = (a.min(1) > 0).mean()
        print(f"{nm}", flush=True)
        print(f"  frame press rate     {a.mean():.5f}", flush=True)
        print(f"  segments with any    {seg_any:.4f}", flush=True)
        print(f"  segments all-held    {seg_all:.4f}", flush=True)
        if len(r):
            print(f"  hold runs            n={len(r):,} mean {r.mean():.1f} fr "
                  f"({r.mean()/20:.2f}s) median {np.median(r):.0f} "
                  f"p90 {np.percentile(r,90):.0f} max {r.max()} "
                  f"(80 = whole segment)", flush=True)
            print(f"  runs reaching 80 fr  {(r>=F).mean():.4f}", flush=True)
        # lag-1 autocorrelation within a segment
        x = a[:, :-1].ravel().astype(np.float32); y = a[:, 1:].ravel().astype(np.float32)
        if x.std() > 0 and y.std() > 0:
            print(f"  lag-1 autocorr       {np.corrcoef(x, y)[0,1]:.4f}", flush=True)
        print(flush=True)

    both = ((Cl[:, :, 0] > 0) & (Cl[:, :, 1] > 0)).mean()
    either = ((Cl[:, :, 0] > 0) | (Cl[:, :, 1] > 0)).mean()
    print(f"attack AND use same frame {both:.5f}   either {either:.5f}", flush=True)
    print(f"\nfor comparison, keyboard frame rates:", flush=True)
    for i, nm in enumerate(["W", "A", "S", "D", "space", "shift", "ctrl", "E"]):
        print(f"  {nm:6s} {K[:,:,i].mean():.5f}", flush=True)
    print(f"\nhotbar slot distribution: "
          f"{np.bincount(HB.ravel(), minlength=9)[:9] / HB.size}", flush=True)
    print(f"dwheel nonzero rate {np.mean(DW != 0):.5f}", flush=True)


if __name__ == "__main__":
    main()
