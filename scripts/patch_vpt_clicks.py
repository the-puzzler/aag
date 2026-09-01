#!/usr/bin/env python
"""Patch mouse buttons (attack/use), hotbar and dwheel into an existing VPT cache.

prepare_vpt_cache.py wrote keys/mouse/pose/gui side arrays but its KEYS list is
keyboard-only, so attack (LMB) and use (RMB) were never stored -- and the jsonl
was deleted after transcode.  This recovers them by re-downloading ONLY the jsonl
sidecars.  Frames, AE latents and assignment particles are untouched.

Why this is cheap and safe:

  * chunk_idx.npy records the frame offset actually cached for every segment, so
    pick selection is NOT recomputed -- the tick index is exactly
    chunk_idx * frames + off, the same expression the builder used
    (`recs[picks[ci] + off]`, with picks[ci] == chunk_idx[row] * frames).  This
    matters because the cache was built by a DIFFERENT version of the builder
    than the one on disk: meta.json says skip_gui_chunks=false with
    gui_filtered_before=153413, and measured, segments 0..153413 are GUI-free
    while the rest carry a 13.1% GUI frame rate.  Recomputing `cands` would
    therefore disagree with the cache; reading chunk_idx cannot.

  * Every jsonl line also carries the 8 keyboard keys, dx/dy and isGuiOpen, all
    of which are already cached.  So each video is self-checking: the recovered
    rows must match keys.npy/mouse.npy/gui.npy EXACTLY before its clicks are
    written.  A tick<->frame off-by-one or any join drift fails loudly, per
    video, rather than silently poisoning the conditioning.

  * Only ~350 of each ~1.1 kB line is needed (mouse/keyboard/isGuiOpen/pose all
    precede "inventory"), so lines are truncated at the inventory marker before
    json.loads -- ~3x less parse for an identical result.

Writes clicks.npy (N,F,2) uint8 [attack, use], hotbar.npy (N,F) uint8,
dwheel.npy (N,F) float16 at cache capacity, plus clicks_done.jsonl for resume.
"""
from __future__ import annotations

import argparse, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

import numpy as np

INV = b',"inventory"'
KEYS = ["key.keyboard.w", "key.keyboard.a", "key.keyboard.s", "key.keyboard.d",
        "key.keyboard.space", "key.keyboard.left.shift", "key.keyboard.left.control",
        "key.keyboard.e"]
BASEDIR = "https://openaipublic.blob.core.windows.net/minecraft-rl/"


def fetch_bytes(url: str, tries: int = 4, timeout: int = 300):
    for t in range(tries):
        try:
            with urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if t == tries - 1:
                return None
            time.sleep(1.5 * (t + 1))
    return None


def parse_line(raw: bytes) -> dict:
    """json.loads only the prefix up to "inventory" -- still valid JSON."""
    i = raw.find(INV)
    if i != -1:
        raw = raw[:i] + b"}"
    return json.loads(raw)


def process_one(job):
    """One video: fetch jsonl, extract the cached ticks, verify against the cache.

    Returns (relpath, None, reason) on failure, else
    (relpath, (row0, row1, clicks, hotbar, dwheel), reason).
    """
    relpath, chunks, row0, row1, frames, ck, cm, cg = job
    url = BASEDIR + relpath[:-4] + ".jsonl"
    blob = fetch_bytes(url)
    if blob is None:
        return relpath, None, "fetch_failed"
    lines = [l for l in blob.split(b"\n") if l.strip()]
    n = len(lines)
    K = len(chunks)
    clicks = np.zeros((K, frames, 2), np.uint8)
    hotbar = np.zeros((K, frames), np.uint8)
    dwheel = np.zeros((K, frames), np.float16)
    bad_key = bad_mouse = bad_gui = 0
    for ci in range(K):
        base = int(chunks[ci]) * frames
        if base + frames > n:
            return relpath, None, "short_jsonl"
        for off in range(frames):
            try:
                r = parse_line(lines[base + off])
            except Exception:
                return relpath, None, "parse_error"
            m = r["mouse"]
            b = m.get("buttons") or []
            clicks[ci, off, 0] = 1 if 0 in b else 0
            clicks[ci, off, 1] = 1 if 1 in b else 0
            hotbar[ci, off] = int(r.get("hotbar") or 0)
            dwheel[ci, off] = np.float16(m.get("dwheel") or 0.0)
            # --- self-check against what the builder already cached ---
            kset = set(r["keyboard"]["keys"])
            kk = np.fromiter((1 if nm in kset else 0 for nm in KEYS), np.uint8, 8)
            if not np.array_equal(kk, ck[ci, off]):
                bad_key += 1
            mm = np.array([m["dx"], m["dy"]], np.float16)
            if not np.array_equal(mm, cm[ci, off]):
                bad_mouse += 1
            if (1 if r.get("isGuiOpen") else 0) != cg[ci, off]:
                bad_gui += 1
    tot = K * frames
    if bad_key or bad_mouse or bad_gui:
        return relpath, None, f"verify_fail k={bad_key} m={bad_mouse} g={bad_gui} of {tot}"
    return relpath, (row0, row1, clicks, hotbar, dwheel), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("/data/vpt/cache_train"))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="pilot: N videos only")
    ap.add_argument("--shuffle-pilot", action="store_true",
                    help="pilot samples across vintages instead of the head")
    ap.add_argument("--dry-run", action="store_true", help="verify, do not write arrays")
    args = ap.parse_args()

    C = args.cache
    meta = json.loads((C / "meta.json").read_text())
    F, cap = meta["frames"], meta["capacity"]
    done_recs = [json.loads(l) for l in (C / "done.jsonl").read_text().splitlines() if l.strip()]
    clip_ids = np.load(C / "clip_ids.npy")
    chunk_idx = np.load(C / "chunk_idx.npy")
    N = len(clip_ids)
    assert bool(np.all(np.diff(clip_ids) >= 0)), "clip_ids not non-decreasing"
    keys = np.load(C / "keys.npy", mmap_mode="r")
    mouse = np.load(C / "mouse.npy", mmap_mode="r")
    gui = np.load(C / "gui.npy", mmap_mode="r")

    nclip = len(done_recs)
    starts = np.searchsorted(clip_ids, np.arange(nclip), "left")
    ends = np.searchsorted(clip_ids, np.arange(nclip), "right")

    resume = C / "clicks_done.jsonl"
    already = set()
    if resume.exists():
        for l in resume.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                if d.get("ok"):
                    already.add(d["relpath"])
    print(f"{nclip} clips, {N} segments, {len(already)} already patched", flush=True)

    todo = [i for i in range(nclip) if done_recs[i]["relpath"] not in already]
    if args.limit:
        if args.shuffle_pilot:
            import random
            random.Random(11).shuffle(todo)
        todo = todo[: args.limit]
    print(f"{len(todo)} clips to fetch, workers={args.workers}", flush=True)

    clicks_a = hotbar_a = dwheel_a = None
    if not args.dry_run:
        def arr(name, shape, dt):
            p = C / name
            if p.exists():
                a = np.load(p, mmap_mode="r+")
                assert a.shape == shape and a.dtype == dt, f"{name} shape/dtype mismatch"
                return a
            return np.lib.format.open_memmap(p, mode="w+", dtype=dt, shape=shape)
        clicks_a = arr("clicks.npy", (cap, F, 2), np.uint8)
        hotbar_a = arr("hotbar.npy", (cap, F), np.uint8)
        dwheel_a = arr("dwheel.npy", (cap, F), np.float16)

    def gen_jobs():
        for i in todo:
            s, e = int(starts[i]), int(ends[i])
            yield (done_recs[i]["relpath"], np.asarray(chunk_idx[s:e]), s, e, F,
                   np.asarray(keys[s:e]), np.asarray(mouse[s:e]), np.asarray(gui[s:e]))

    t0 = time.time()
    n_ok = n_fail = 0
    reasons: dict[str, int] = {}
    # a dry run must NOT record progress, or the real run would skip these clips
    fh = open("/dev/null", "a") if args.dry_run else open(resume, "a")
    total = len(todo)
    done_n = 0
    it = gen_jobs()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        # bounded in-flight window: never hold all 29,582 jobs' cached arrays in RAM
        inflight = {}
        for _ in range(args.workers * 3):
            try:
                j = next(it)
            except StopIteration:
                break
            inflight[ex.submit(process_one, j)] = 1
        while inflight:
            for fut in as_completed(list(inflight), timeout=None):
                inflight.pop(fut, None)
                relpath, payload, reason = fut.result()
                done_n += 1
                if payload is None:
                    n_fail += 1
                    tag = reason.split()[0]
                    reasons[tag] = reasons.get(tag, 0) + 1
                    fh.write(json.dumps({"relpath": relpath, "ok": False, "why": reason}) + "\n")
                else:
                    r0, r1, cl, hb, dw = payload
                    if clicks_a is not None:
                        clicks_a[r0:r1] = cl
                        hotbar_a[r0:r1] = hb
                        dwheel_a[r0:r1] = dw
                    n_ok += 1
                    fh.write(json.dumps({"relpath": relpath, "ok": True}) + "\n")
                try:
                    inflight[ex.submit(process_one, next(it))] = 1
                except StopIteration:
                    pass
                if done_n % 200 == 0 or done_n == total:
                    fh.flush()
                    el = time.time() - t0
                    rate = done_n / max(el, 1e-9)
                    print(f"[{done_n}/{total}] ok={n_ok} fail={n_fail} "
                          f"{rate:.2f} clip/s eta={(total-done_n)/max(rate,1e-9)/60:.1f}min "
                          f"reasons={reasons}", flush=True)
                break
    fh.close()
    if clicks_a is not None:
        clicks_a.flush(); hotbar_a.flush(); dwheel_a.flush()
    print(f"DONE ok={n_ok} fail={n_fail} reasons={reasons} "
          f"in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
