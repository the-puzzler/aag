#!/usr/bin/env python
"""Can we avoid downloading 137 MB per video to keep 560 frames?

The network is the hard ceiling (~250 MB/s regardless of concurrency), so the
only way to go faster is to move fewer bytes. Two things to test:

  1. Does the CDN honour HTTP Range? (Azure blob normally does.)
  2. Can ffmpeg/cv2 open the URL directly and seek, fetching only the byte
     ranges it needs, instead of us pulling the whole mp4?

If yes, per-video bytes drop toward (7 chunks x keyframe-aligned span + moov),
and the run becomes bandwidth-cheap enough to widen episode coverage.

Also times the jsonl parse, which is 24 MB of line-delimited JSON per video and
a plausible hidden cost.
"""
from __future__ import annotations

import json, re, shutil, time, urllib.request
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

BASE = "https://openaipublic.blob.core.windows.net/minecraft-rl/"
IDX = BASE + "snapshots/all_9xx_Jun_29.json"
SCRATCH = Path("/opt/dlami/nvme/bench")
F, K = 80, 7


def pick_video():
    with urlopen(IDX, timeout=120) as r:
        d = json.load(r)
    for x in (v for v in d["relpaths"] if v.endswith(".mp4")):
        req = urllib.request.Request(BASE + x, method="HEAD")
        try:
            with urlopen(req, timeout=30) as h:
                if int(h.headers.get("Content-Length", 0)) > 50_000_000:
                    return x, int(h.headers["Content-Length"]), h.headers.get("Accept-Ranges")
        except Exception:
            continue
    raise SystemExit("no suitable video")


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rel, size, ranges = pick_video()
    url = BASE + rel
    print(f"video: {size/1e6:.0f} MB  Accept-Ranges: {ranges!r}")

    # 1. does Range actually work?
    req = urllib.request.Request(url, headers={"Range": "bytes=0-1048575"})
    with urlopen(req, timeout=60) as r:
        got = len(r.read())
        print(f"1. Range request -> HTTP {r.status}, got {got/1e6:.2f} MB "
              f"({'RANGE HONOURED' if r.status == 206 else 'IGNORED - full body'})")

    # 2. remote seek+decode via ffmpeg, counting bytes actually pulled
    #    cv2 gives no byte counter, so measure with /proc net counters delta.
    def rx_bytes():
        tot = 0
        for line in open("/proc/net/dev"):
            p = line.split()
            if len(p) > 9 and p[0].rstrip(":") not in ("lo",):
                try:
                    tot += int(p[1])
                except ValueError:
                    pass
        return tot

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("2. remote open FAILED - ffmpeg could not open the URL")
        return
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"2. remote open OK, reports {n} frames")
    picks = [int(i * (max(n - F, 1)) / K) // F * F for i in range(K)]
    r0 = rx_bytes(); t = time.time(); kept = 0
    for s in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        for _ in range(F):
            ok, fr = cap.read()
            if not ok:
                break
            cv2.resize(fr[:320, 160:480], (64, 64), interpolation=cv2.INTER_AREA)
            kept += 1
    el = time.time() - t; rx = rx_bytes() - r0
    cap.release()
    print(f"   remote seek+decode: {el:.2f}s, kept {kept} frames, "
          f"~{rx/1e6:.0f} MB pulled ({100*rx/size:.0f}% of the file)")

    # 3. baseline: full download then local seek
    p = SCRATCH / "full.mp4"
    r0 = rx_bytes(); t = time.time()
    with urlopen(url, timeout=300) as r, open(p, "wb") as f:
        shutil.copyfileobj(r, f, length=4 << 20)
    dl = time.time() - t; rx_dl = rx_bytes() - r0
    cap = cv2.VideoCapture(str(p), cv2.CAP_FFMPEG)
    t = time.time(); kept2 = 0
    for s in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        for _ in range(F):
            ok, fr = cap.read()
            if not ok:
                break
            cv2.resize(fr[:320, 160:480], (64, 64), interpolation=cv2.INTER_AREA)
            kept2 += 1
    dec = time.time() - t
    cap.release()
    print(f"3. full download {dl:.2f}s ({rx_dl/1e6:.0f} MB) + local seek {dec:.2f}s "
          f"= {dl+dec:.2f}s, kept {kept2}")

    # 4. jsonl: full json.loads of every line vs cheap substring scan
    jp = SCRATCH / "full.jsonl"
    with urlopen(url[:-4] + ".jsonl", timeout=300) as r, open(jp, "wb") as f:
        shutil.copyfileobj(r, f, length=4 << 20)
    raw = open(jp).read().splitlines()
    print(f"4. jsonl {jp.stat().st_size/1e6:.0f} MB, {len(raw)} lines")
    t = time.time(); recs = [json.loads(l) for l in raw if l.strip()]
    print(f"   full json.loads of all lines : {time.time()-t:.2f}s")
    t = time.time(); flags = [('"isGuiOpen":true' in l) for l in raw]
    print(f"   substring scan for isGuiOpen : {time.time()-t:.3f}s  "
          f"(agrees: {sum(flags) == sum(1 for r in recs if r.get('isGuiOpen'))})")
    t = time.time(); sub = [json.loads(raw[i]) for i in range(0, min(len(raw), K * F))]
    print(f"   json.loads of only kept ticks: {time.time()-t:.2f}s ({len(sub)} of {len(raw)})")
    for q in (p, jp):
        try:
            q.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
