#!/usr/bin/env python
"""Re-encode an existing particle file's action side to the 12-d representation.

No AE re-encode and no GPU: a particle file already stores `chunk` and `frame`
for every particle, and the action side was only ever a lookup into the cache
side arrays at those coordinates.  So swapping the 81-way + 9-d condition for
the 12-d one (aag/vpt_actions) is a few minutes of numpy, not hours.

Adds to the particle dict:
    action_raw  (N,12) float32   physical vector: 10 binaries in {0,1}, dx, dy px
    action_vec  (N,12) float32   the same, mapped for use as a k-NN distance
    act_norm    dict             normalisation constants -- MUST travel with the
                                 particles, or live inference encodes differently
                                 than training did
    act_names   list[str]

`action` (the 81-way int) is left untouched so old assignments still load, but it
is no longer generator input.  As a consistency check the 81-way index is
recomputed from action_raw and compared against the cached action_seqs: they must
agree, which proves the (chunk, frame) lookup landed on the same ticks the
builder used.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from aag.vpt_actions import (A_DIM, ACTION_NAMES, action_index_from_raw,
                             action_marginals, apply_action_norm,
                             build_action_raw, fit_action_norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("/data/vpt/cache_train"))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: overwrite --particles in place (atomic via .tmp)")
    ap.add_argument("--allow-partial-clicks", action="store_true",
                    help="Proceed even if some particles sit on clips whose "
                         "clicks were never recovered. Off by default and it "
                         "should stay off: an unpatched row is the zero-fill "
                         "from array creation, which is indistinguishable from "
                         "'the player never clicked'. Fitting act_norm over "
                         "those rows biases every press rate downward and "
                         "teaches the generator that a third of frames have no "
                         "button down when they do.")
    args = ap.parse_args()

    C = args.cache
    P = torch.load(args.particles, map_location="cpu", weights_only=False)
    ch = P["chunk"].numpy()
    fr = P["frame"].numpy()
    N = len(ch)
    print(f"{N:,} particles from {args.particles}", flush=True)

    keys = np.load(C / "keys.npy", mmap_mode="r")
    mouse = np.load(C / "mouse.npy", mmap_mode="r")
    acts = np.load(C / "action_seqs.npy", mmap_mode="r")
    clicks_p = C / "clicks.npy"
    if not clicks_p.exists():
        raise SystemExit(f"{clicks_p} missing -- run scripts/patch_vpt_clicks.py first")
    clicks = np.load(clicks_p, mmap_mode="r")

    # Only rows whose clip was successfully patched carry real clicks; every
    # other row is still the zero-fill, which would read as "never clicks".
    # Refuse to build a condition on top of that silently.
    done = [json.loads(l) for l in (C / "done.jsonl").read_text().splitlines() if l.strip()]
    ok_rel = set()
    cd = C / "clicks_done.jsonl"
    if cd.exists():
        for l in cd.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                if d.get("ok"):
                    ok_rel.add(d["relpath"])
    clip_ids = np.load(C / "clip_ids.npy")
    ok_clip = np.array([done[i]["relpath"] in ok_rel for i in range(len(done))])
    row_ok = ok_clip[clip_ids[ch]]
    n_bad = int((~row_ok).sum())
    print(f"clicks coverage: {N - n_bad:,}/{N:,} particles "
          f"({100.0*(N-n_bad)/max(N,1):.2f}%) from {len(ok_rel):,}/{len(done):,} clips",
          flush=True)
    if n_bad and not args.allow_partial_clicks:
        raise SystemExit(
            f"{n_bad:,}/{N:,} particles sit on clips whose clicks were never "
            f"recovered; their attack/use are the zero-fill, not observations. "
            f"Re-run scripts/patch_vpt_clicks.py (it resumes, so it only retries "
            f"failures), or pass --allow-partial-clicks if you have decided the "
            f"bias is acceptable.")

    kv = np.stack([np.asarray(keys[c, t]) for c, t in zip(ch, fr)])
    mv = np.stack([np.asarray(mouse[c, t]) for c, t in zip(ch, fr)])
    cv = np.stack([np.asarray(clicks[c, t]) for c, t in zip(ch, fr)])
    raw = build_action_raw(kv, mv, cv)

    # --- consistency: the recomputed 81-way must match what the builder cached
    idx_new = action_index_from_raw(raw)
    idx_old = np.stack([np.asarray(acts[c, t]) for c, t in zip(ch, fr)]).astype(np.int64)
    agree = int((idx_new == idx_old).sum())
    print(f"81-way index recomputed from action_raw agrees with action_seqs on "
          f"{agree:,}/{N:,} ({100.0*agree/max(N,1):.3f}%)", flush=True)
    if agree != N:
        bad = np.where(idx_new != idx_old)[0][:5]
        for b in bad:
            print(f"  mismatch particle {b}: new {idx_new[b]} old {idx_old[b]} "
                  f"raw {raw[b]}", flush=True)
        raise SystemExit("action index mismatch -- the (chunk, frame) lookup is wrong")

    norm = fit_action_norm(raw)
    vec = apply_action_norm(raw, norm)
    print(f"\naction_vec {vec.shape}  (was {tuple(P['action_vec'].shape)})", flush=True)
    print("press rates:", flush=True)
    for i, nm in enumerate(ACTION_NAMES[:10]):
        print(f"  {nm:7s} {raw[:, i].mean():.5f}", flush=True)
    print(f"  dx  |.| p50 {np.percentile(np.abs(raw[:,10]),50):.1f} "
          f"p90 {np.percentile(np.abs(raw[:,10]),90):.1f} "
          f"p99 {np.percentile(np.abs(raw[:,10]),99):.1f}", flush=True)
    print(f"  dy  |.| p50 {np.percentile(np.abs(raw[:,11]),50):.1f} "
          f"p90 {np.percentile(np.abs(raw[:,11]),90):.1f} "
          f"p99 {np.percentile(np.abs(raw[:,11]),99):.1f}", flush=True)
    bv = vec[:, :10].var(0).sum(); mvv = vec[:, 10:].var(0).sum()
    print(f"  variance: binaries {bv:.3f}  mouse {mvv:.3f}  ratio {mvv/max(bv,1e-9):.2f} "
          f"(target {norm['mouse_w']:.1f})", flush=True)
    mg = action_marginals(raw)
    print(f"  marginals to transport: "
          f"{ {k: int(g.max())+1 for k, g in mg.items()} }", flush=True)

    P["action_raw"] = torch.from_numpy(raw)
    P["action_vec"] = torch.from_numpy(vec.astype(np.float32))
    P["act_norm"] = norm
    P["act_names"] = list(ACTION_NAMES)
    P["act_dim"] = A_DIM
    P["clicks_coverage"] = float((N - n_bad) / max(N, 1))

    out = args.out or args.particles
    tmp = out.with_suffix(out.suffix + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(P, tmp)
    tmp.replace(out)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
