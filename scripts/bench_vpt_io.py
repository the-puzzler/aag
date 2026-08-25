#!/usr/bin/env python
"""Measure where prepare_vpt_cache.py actually spends its time.

Three questions:
  1. What is the network ceiling? We measured 246 MB/s at 12 concurrent streams,
     but that may not be the cap.
  2. How long does a FULL sequential decode of one 6001-frame video take?
  3. How much cheaper is seek-then-decode, given we keep only 560 of 6001 frames?

Answers decide whether to (a) raise worker count, (b) split download from decode
into a producer/consumer pipeline, or (c) stop decoding frames we discard.
"""
from __future__ import annotations

import json, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

BASE = "https://openaipublic.blob.core.windows.net/minecraft-rl/"
IDX = BASE + "snapshots/all_9xx_Jun_29.json"
SCRATCH = Path("/opt/dlami/nvme/bench")


def head_size(url):
    try:
        import urllib.request
        r = urllib.request.Request(url, method="HEAD")
        with urlopen(r, timeout=30) as h:
            return int(h.headers.get("Content-Length", 0))
    except Exception:
        return 0


def get(url, dest=None):
    try:
        with urlopen(url, timeout=300) as r:
            if dest is None:
                n = 0
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    n += len(b)
                return n
            with open(dest, "wb") as f:
                shutil.copyfileobj(r, f, length=4 << 20)
            return dest.stat().st_size
    except Exception:
        return 0


def bench_net(urls, par):
    with ThreadPoolExecutor(max_workers=par) as ex:
        t = time.time()
        got = sum(ex.map(get, urls))
        el = time.time() - t
    return got / 1e6, el, got / 1e6 / el


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with urlopen(IDX, timeout=120) as r:
        d = json.load(r)
    rel = [x for x in d["relpaths"] if x.endswith(".mp4")]

    print("=== 1. network ceiling (threads, discard bytes) ===", flush=True)
    off = 300
    for par in (12, 24, 40):
        urls = [BASE + x for x in rel[off:off + par]]
        off += par
        mb, el, rate = bench_net(urls, par)
        print(f"  {par:2d} streams: {mb:7.0f} MB in {el:5.1f}s -> {rate:6.0f} MB/s", flush=True)

    print("\n=== 2/3. decode cost on one real video ===", flush=True)
    target = None
    for x in rel[500:540]:
        p = SCRATCH / "probe.mp4"
        if get(BASE + x, p) > 1_000_000:
            target = p
            break
    if target is None:
        sys.exit("could not fetch a probe video")
    print(f"  probe: {target.stat().st_size/1e6:.0f} MB", flush=True)

    F, K, SIZE = 80, 7, 64
    cap = cv2.VideoCapture(str(target), cv2.CAP_FFMPEG)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    picks = [int(i * (n - F) / K) // F * F for i in range(K)]
    print(f"  {n} frames; keeping {K}x{F}={K*F} ({100*K*F/n:.0f}%); chunk starts {picks}", flush=True)

    # (a) full sequential decode up to the last needed frame
    cap = cv2.VideoCapture(str(target), cv2.CAP_FFMPEG)
    want = {}
    for ci, s in enumerate(picks):
        for f in range(s, s + F):
            want[f] = ci
    last = max(want)
    t = time.time()
    fi = kept = 0
    while fi <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if fi in want:
            cv2.resize(fr[:320, 160:480], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            kept += 1
        fi += 1
    full = time.time() - t
    cap.release()
    print(f"  (a) full sequential : {full:6.2f}s  decoded {fi} frames, kept {kept}", flush=True)

    # (b) seek to each chunk, decode only F frames
    cap = cv2.VideoCapture(str(target), cv2.CAP_FFMPEG)
    t = time.time()
    kept2 = 0
    for s in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        for _ in range(F):
            ok, fr = cap.read()
            if not ok:
                break
            cv2.resize(fr[:320, 160:480], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            kept2 += 1
    seek = time.time() - t
    cap.release()
    print(f"  (b) seek-then-decode: {seek:6.2f}s  kept {kept2}", flush=True)
    print(f"  -> seek is {full/max(seek,1e-9):.2f}x {'faster' if seek<full else 'SLOWER'}", flush=True)

    # verify seek is frame-accurate: compare one chunk both ways
    cap = cv2.VideoCapture(str(target), cv2.CAP_FFMPEG)
    s0 = picks[len(picks) // 2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
    ok, a = cap.read()
    cap.release()
    cap = cv2.VideoCapture(str(target), cv2.CAP_FFMPEG)
    for i in range(s0 + 1):
        ok2, b = cap.read()
    cap.release()
    if ok and ok2:
        print(f"  seek accuracy at frame {s0}: identical={np.array_equal(a,b)} "
              f"meanabsdiff={np.abs(a.astype(np.int16)-b.astype(np.int16)).mean():.2f}", flush=True)
    try:
        target.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
