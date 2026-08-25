#!/usr/bin/env python
"""Build a uint8 segment cache from the OpenAI VPT contractor data.

Emits exactly the layout aag/datasets.py already reads for Doom
(segments/action_seqs/labels/clip_ids/chunk_idx/meta.json), so the existing AE
and generator scripts run unmodified with --dataset doom / doom_frames.

Design choices, all measured against the data rather than assumed:

  * Source is 640x360 @ 20fps, 6001 frames (300.1s) per segment file.
  * The HUD is bottom-CENTRE and semi-transparent, so Doom's trick of finding
    near-zero-variance rows does not work.  Two independent measurements on the
    centre columns (180:460) agree: the temporal-std ratio against the outer
    columns holds 1.00 through row 320 then breaks (0.82 at 321, 0.65 at 322),
    and mean red-channel excess jumps -13.6 at row 320 -> +45.9 at 322 -> +64.3
    at 324, which is the health bar.  Rows 331-335 are the hotbar (ratio 0.37).
    So the HUD starts at 321 and HUD_ROW = 320.  Cutting at 324 -- the hotbar
    edge alone -- leaves a visible sliver of hearts in every cached frame.
  * 5.1% of ticks have isGuiOpen (inventory/crafting overlays the world).  Any
    chunk containing one is skipped -- a world model should not be asked to
    render inventory screens.
  * FEW chunks from MANY videos, not many from few.  One video is one
    independent episode, and it is episode-level N that bounds usable latent
    dim -- the UCF-101 lesson (87,953 segments from 9,537 clips).  Download is
    ~246 MB/s at 12-way, so episode diversity is nearly free: prefer many
    videos x few chunks over few videos x many chunks.
  * --frames 80 with --chunks-per-video 7 gives 560 frames/video and, crucially,
    16 target frames per chunk that each have a full 64-frame contiguous history.
    Chunks are NOT adjacent to each other, so a context longer than one chunk is
    impossible -- frames must be >= context + targets.  Measured: pixel MSE
    between neighbouring frames inside a chunk is 751 vs 7,465 across chunks.
  * Nothing is written to the root filesystem: TMPDIR is pinned to --tmp, and a
    disk guard aborts the run before root or the output volume runs dry.
  * mp4+jsonl are deleted immediately after transcode, so peak disk is the
    cache plus a few GB in flight, not the ~5 TB of source pulled.

The categorical condition is movement(9) x turn(3) x pitch(3) = 81 classes.
Camera motion dominates what the frame looks like (|dx| p50=2, p90=36), so it
belongs in the grouping; attack/use/jump/sprint barely move the image and are
kept in the side arrays instead.  Those side arrays (keys/mouse/pose/gui) cost
~350 MB per 10M frames and mean richer conditioning never needs a re-download.
"""
from __future__ import annotations

import argparse, json, os, random, shutil, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np

RAW_H, RAW_W = 360, 640
HUD_ROW = 320                 # first HUD row, measured (see module docstring)
FPS = 20

INDEX_URLS = {                # 6.x omitted: ~50% of its entries 404
    "7xx":  "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_7xx_Apr_6.json",
    "8xx":  "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_8xx_Jun_29.json",
    "9xx":  "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_9xx_Jun_29.json",
    "10xx": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_10xx_Jun_29.json",
}

# side-array key order; index into KEYS is the column in keys.npy
KEYS = ["key.keyboard.w", "key.keyboard.a", "key.keyboard.s", "key.keyboard.d",
        "key.keyboard.space", "key.keyboard.left.shift", "key.keyboard.left.control",
        "key.keyboard.e"]
DX_DEADZONE = 5.0             # |dx| below this counts as "not turning" (p50=2)
DY_DEADZONE = 5.0

N_ACTIONS = 81                # 9 movement x 3 turn x 3 pitch


def move_class(w: bool, a: bool, s: bool, d: bool) -> int:
    """9-way: 0 none, then F,B,L,R,FL,FR,BL,BR. Opposing pairs cancel."""
    fb = (1 if w else 0) - (1 if s else 0)
    lr = (1 if d else 0) - (1 if a else 0)
    return {(0, 0): 0, (1, 0): 1, (-1, 0): 2, (0, -1): 3, (0, 1): 4,
            (1, -1): 5, (1, 1): 6, (-1, -1): 7, (-1, 1): 8}[(fb, lr)]


def action_index(rec: dict) -> int:
    keys = set(rec["keyboard"]["keys"])
    m = move_class("key.keyboard.w" in keys, "key.keyboard.a" in keys,
                   "key.keyboard.s" in keys, "key.keyboard.d" in keys)
    dx = float(rec["mouse"]["dx"]); dy = float(rec["mouse"]["dy"])
    turn = 0 if abs(dx) < DX_DEADZONE else (1 if dx > 0 else 2)
    tilt = 0 if abs(dy) < DY_DEADZONE else (1 if dy > 0 else 2)
    return m * 9 + turn * 3 + tilt


def side_row(rec: dict):
    keys = set(rec["keyboard"]["keys"])
    k = np.array([1 if n in keys else 0 for n in KEYS], np.uint8)
    mouse = np.array([rec["mouse"]["dx"], rec["mouse"]["dy"]], np.float16)
    pose = np.array([rec["yaw"], rec["pitch"], rec["xpos"], rec["ypos"], rec["zpos"]],
                    np.float32)
    return k, mouse, pose, np.uint8(1 if rec.get("isGuiOpen") else 0)


def crop_resize(frame: np.ndarray, size: int) -> np.ndarray:
    """HUD-crop, centre square crop (no aspect distortion), resize. BGR->RGB."""
    f = frame[:HUD_ROW]                                  # (320,640,3)
    h, w = f.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    f = f[y0:y0 + side, x0:x0 + side]                    # (320,320,3)
    f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
    return f[:, :, ::-1]                                 # BGR -> RGB


def fetch(url: str, dest: Path, tries: int = 3) -> bool:
    for t in range(tries):
        try:
            with urlopen(url, timeout=180) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, length=4 << 20)
            return True
        except HTTPError as e:
            # 22% of indexed entries are simply gone. A 404/410 is permanent, so
            # retrying it just burns 4.5s of backoff in a worker slot -- which at
            # a 23% failure rate was costing ~13% of total throughput.
            if e.code in (404, 410):
                return False
            if dest.exists():
                dest.unlink()
            if t == tries - 1:
                return False
            time.sleep(1.5 * (t + 1))
        except Exception:
            if dest.exists():
                dest.unlink()
            if t == tries - 1:
                return False
            time.sleep(1.5 * (t + 1))
    return False


def process_one(job):
    """Download one segment, transcode K chunks, delete the source.

    Returns (relpath, segs, acts, labels, keys, mouse, pose, gui) or (relpath, None).
    Runs in a worker process; the parent does all cache writing so segments stay
    contiguous (aag/datasets.py takes len(labels) as the valid prefix).
    """
    relpath, basedir, tmpdir, k_chunks, frames, size, seed = job
    tmp = Path(tmpdir)
    mp4 = tmp / (relpath.replace("/", "_"))
    jsonl = mp4.with_suffix(".jsonl")
    try:
        if not fetch(basedir + relpath, mp4):
            return relpath, None
        if not fetch(basedir + relpath[:-4] + ".jsonl", jsonl):
            return relpath, None

        recs = []
        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        break
        if len(recs) < frames * 2:
            return relpath, None

        cap = cv2.VideoCapture(str(mp4), cv2.CAP_FFMPEG)
        # Hard decode timeouts. Without these a single wedged read blocks a worker
        # slot indefinitely -- the first run logged stalls of 90s-500s on LOCAL
        # files once the root disk filled and I/O started thrashing.
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 20_000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 20_000)
        n_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = min(n_v, len(recs))
        if n < frames * 2:
            cap.release()
            return relpath, None

        gui_flag = np.array([1 if r.get("isGuiOpen") else 0 for r in recs[:n]], np.uint8)

        # aligned candidate offsets spread across the episode, GUI-free only
        n_slots = n // frames
        cands = [i * frames for i in range(n_slots)
                 if not gui_flag[i * frames:(i + 1) * frames].any()]
        if not cands:
            cap.release()
            return relpath, None
        # crc32, NOT hash(): str hashing is salted per process, so hash() would
        # make chunk selection irreproducible across runs
        rng = random.Random(zlib.crc32(relpath.encode()) ^ seed)
        if len(cands) > k_chunks:                        # even spread, jittered
            step = len(cands) / k_chunks
            picks = sorted({cands[min(len(cands) - 1, int(i * step + rng.random() * step))]
                            for i in range(k_chunks)})
        else:
            picks = cands
        want = {}
        for ci, s in enumerate(picks):
            for f in range(s, s + frames):
                want[f] = (ci, f - s)

        K = len(picks)
        segs = np.zeros((K, frames, size, size, 3), np.uint8)
        got = np.zeros((K, frames), bool)
        last = max(want) if want else -1

        # Decode strategy is chosen by coverage, because the two costs cross over.
        # Measured on a 6001-frame video keeping 560 frames: one sequential pass
        # is 1.18s, seek-then-decode is 0.32s (3.7x faster, and frame-accurate --
        # verified byte-identical against the sequential result). But each seek
        # pays a keyframe-rewind, so once the chunks cover a large fraction of the
        # file the single pass wins again.
        sparse = (K * frames) < 0.25 * max(last + 1, 1)
        if sparse:
            for ci, s in enumerate(picks):
                cap.set(cv2.CAP_PROP_POS_FRAMES, s)
                for off in range(frames):
                    ok, fr = cap.read()
                    if not ok:
                        break
                    if fr.shape[0] == RAW_H and fr.shape[1] == RAW_W:
                        segs[ci, off] = crop_resize(fr, size)
                        got[ci, off] = True
        else:
            fi = 0
            while fi <= last:
                ok, fr = cap.read()
                if not ok:
                    break
                hit = want.get(fi)
                if hit is not None and fr.shape[0] == RAW_H and fr.shape[1] == RAW_W:
                    ci, off = hit
                    segs[ci, off] = crop_resize(fr, size)
                    got[ci, off] = True
                fi += 1
        cap.release()

        keep = got.all(1)
        if not keep.any():
            return relpath, None
        picks = [p for p, kp in zip(picks, keep) if kp]
        segs = segs[keep]

        K = len(picks)
        acts = np.zeros((K, frames), np.uint8)
        keys = np.zeros((K, frames, len(KEYS)), np.uint8)
        mouse = np.zeros((K, frames, 2), np.float16)
        pose = np.zeros((K, frames, 5), np.float32)
        gui = np.zeros((K, frames), np.uint8)
        for ci, s in enumerate(picks):
            for off in range(frames):
                r = recs[s + off]
                acts[ci, off] = action_index(r)
                keys[ci, off], mouse[ci, off], pose[ci, off], gui[ci, off] = side_row(r)
        labels = np.array([np.bincount(a, minlength=N_ACTIONS).argmax() for a in acts],
                          np.int64)
        return relpath, (segs, acts, labels, keys, mouse, pose, gui, np.array(picks))
    except Exception as e:
        return relpath, None
    finally:
        for p in (mp4, jsonl):
            try:
                p.unlink()
            except OSError:
                pass


def mem_snapshot() -> str:
    """Dirty/Writeback/MemAvailable, in MB/GB. We had to guess at the cause of two
    hard hangs precisely because nothing recorded these while the run was live."""
    want = ("Dirty", "Writeback", "MemAvailable")
    got = {}
    try:
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            if k in want:
                got[k] = int(v.split()[0])          # kB
    except OSError:
        return "mem n/a"
    return (f"dirty {got.get('Dirty',0)/1024:.0f}M wb {got.get('Writeback',0)/1024:.0f}M "
            f"avail {got.get('MemAvailable',0)/1e6:.0f}G")


def free_gb(path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def disk_guard(out_path, min_root_gb=8.0, min_out_gb=25.0):
    """Abort rather than wedge the box.

    The first run died when the ROOT filesystem filled and the OS locked up.
    Root is small (194G, mostly /home) and unrelated to where this writes, but
    anything using tempfile lands in /tmp on root, so it has to be watched too.
    Returns a reason string when the run should stop, else None.
    """
    r, o = free_gb("/"), free_gb(out_path)
    if r < min_root_gb:
        return f"root filesystem down to {r:.1f} GB free (floor {min_root_gb}) -- stopping"
    if o < min_out_gb:
        return f"output filesystem down to {o:.1f} GB free (floor {min_out_gb}) -- stopping"
    return None


def flush_sidecars(args, labels, clip_ids, chunk_idx, w, n_ok, n_fail, complete=False):
    """Write labels/clip_ids/chunk_idx + meta, replacing atomically.

    Called periodically, not just at the end: aag/datasets.py takes len(labels)
    as the valid prefix, so flushing makes a partial cache trainable during the
    ~9h build instead of only after it.
    """
    for name, data in (("labels", labels), ("clip_ids", clip_ids),
                       ("chunk_idx", chunk_idx)):
        # ".tmp.npy", NOT ".npy.tmp": np.save appends ".npy" unless the name
        # already ends in it, so the latter writes labels.npy.tmp.npy and the
        # os.replace below fails with FileNotFoundError.
        t = args.out / (name + ".tmp.npy")
        np.save(t, np.array(data, np.int64))
        os.replace(t, args.out / (name + ".npy"))
    meta = dict(n_segments=w, frames=args.frames, size=args.size,
                shard_size=args.shard_segments,
                n_records=n_ok, per_record=args.chunks_per_video,
                n_actions=N_ACTIONS, hud_row=HUD_ROW, source="vpt",
                subsets=args.subsets, keys=KEYS, complete=complete,
                dx_deadzone=DX_DEADZONE, dy_deadzone=DY_DEADZONE,
                gb=w * args.frames * args.size ** 2 * 3 / 1e9, failed=n_fail)
    t = args.out / "meta.json.tmp"
    json.dump(meta, open(t, "w"), indent=2)
    os.replace(t, args.out / "meta.json")


def load_index(url_or_path: str):
    if url_or_path.startswith("http"):
        with urlopen(url_or_path, timeout=120) as r:
            d = json.load(r)
    else:
        d = json.load(open(url_or_path))
    return d["basedir"], d["relpaths"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/data/vpt/cache_train"))
    ap.add_argument("--tmp", type=Path, default=Path("/data/vpt/inflight"))
    ap.add_argument("--subsets", nargs="+", default=["7xx", "8xx", "9xx", "10xx"])
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--chunks-per-video", type=int, default=17,
                    help="few chunks from many videos -- episode-level N is what bounds "
                         "usable latent dim")
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8,
                    help="8 is enough: the network caps at ~250 MB/s and 8 streams "
                         "reach it, while fewer concurrent readers/writers cut the "
                         "I/O pressure that wedged this box twice")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capacity-factor", type=float, default=0.82,
                    help="fraction of n_total*chunks to preallocate; observed yield "
                         "is ~0.79 (82%% of videos resolve, ~96%% of chunks kept)")
    ap.add_argument("--min-root-gb", type=float, default=8.0)
    ap.add_argument("--shard-segments", type=int, default=8192,
                    help="segments per shard file; one shard mapping is live at a "
                         "time, which bounds the dirty page set (8192 x 80 frames "
                         "at 64px = 8.1 GB per shard)")
    ap.add_argument("--min-out-gb", type=float, default=25.0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.tmp.mkdir(parents=True, exist_ok=True)
    # Keep every temp file off the root filesystem. tempfile defaults to /tmp,
    # which lives on root (194G, already ~92%% full); the first run's OS wedge
    # came from root exhaustion, so this is set before anything else runs.
    os.environ.setdefault("TMPDIR", str(args.tmp))

    cands = []
    for s in args.subsets:
        src = INDEX_URLS.get(s, s)
        basedir, relpaths = load_index(src)
        cands += [(r, basedir) for r in relpaths if r.endswith(".mp4")]
        print(f"{s}: {len(relpaths)} entries", flush=True)
    random.Random(args.seed).shuffle(cands)              # interleave subsets
    if args.limit_videos:
        cands = cands[:args.limit_videos]
    n_total = len(cands)                                 # BEFORE the done filter

    done_path = args.out / "done.jsonl"
    # done.jsonl is appended per video but the sidecars flush every 200, so an
    # unclean death leaves videos marked done whose segments sit BEYOND
    # len(labels). Those would be skipped on resume while their data gets
    # overwritten -- a silent loss (4.16% when this box hard-hung at 10:33).
    # Reconcile: keep only the done prefix whose cumulative chunk count is
    # covered by the last flushed labels.npy, and re-fetch the rest.
    done_recs = []
    if done_path.exists():
        for line in open(done_path):
            line = line.strip()
            if not line:
                continue
            try:
                done_recs.append(json.loads(line))
            except Exception:
                pass
    flushed = 0
    lab_path = args.out / "labels.npy"
    if lab_path.exists():
        flushed = int(np.load(lab_path).shape[0])
    keep, run = [], 0
    for rec in done_recs:
        run += int(rec.get("chunks", 0))
        if run > flushed:
            break
        keep.append(rec)
    dropped = len(done_recs) - len(keep)
    if dropped:
        print(f"reconciling done.jsonl: dropping {dropped} videos ({run - flushed}+ "
              f"segments) written after the last sidecar flush -- they will be re-fetched",
              flush=True)
        with open(done_path, "w") as fh:
            for rec in keep:
                fh.write(json.dumps(rec) + "\n")
    done = {r["relpath"] for r in keep}
    cands = [c for c in cands if c[0] not in done]
    print(f"{len(cands)} videos to process ({len(done)} already done)", flush=True)

    # resume: done.jsonl says WHICH videos are cached, labels.npy says how many
    # segments are already written.  Both are needed -- restoring only the first
    # would overwrite the cache from index 0.  This MUST run before cap_n is
    # sized: cap_n counts only the videos still to do, so a resumed run whose
    # write cursor already exceeds it would trip the overflow break on its first
    # result and silently write nothing.
    labels, clip_ids, chunk_idx = [], [], []
    if done and (args.out / "labels.npy").exists():
        labels = np.load(args.out / "labels.npy").tolist()
        clip_ids = np.load(args.out / "clip_ids.npy").tolist()
        chunk_idx = np.load(args.out / "chunk_idx.npy").tolist()
        assert len(labels) == len(clip_ids) == len(chunk_idx), "cache sidecars disagree"

    w = len(labels)
    n_ok = (clip_ids[-1] + 1) if clip_ids else 0
    n_fail = 0
    if w:
        print(f"resuming at segment {w} ({n_ok} episodes already cached)", flush=True)

    # Size from the pre-filter total, NOT the remaining list: capacity must be
    # identical across resumes of the same run, or the existing arrays look too
    # small and get reallocated (which silently zeroes everything written so far).
    # Capacity is a SPARSE preallocation, so `du --apparent-size` and `ls -l`
    # report the full figure while real usage tracks what has been written.
    # Sized to the observed yield (~82% of videos resolve, ~96% of chunks kept)
    # rather than the theoretical maximum, so the apparent number is not wildly
    # misleading; arr() grows by copy if the estimate is ever exceeded.
    cap_n = int(n_total * args.chunks_per_video * args.capacity_factor) + 16
    F, S = args.frames, args.size
    print(f"capacity {cap_n:,} segments = "
          f"{cap_n * args.frames * args.size ** 2 * 3 / 1e9:.0f} GB apparent (sparse); "
          f"free: root {free_gb('/'):.0f}G, out {free_gb(args.out):.0f}G", flush=True)
    stop = disk_guard(args.out, args.min_root_gb, args.min_out_gb)
    if stop:
        sys.exit(f"refusing to start: {stop}")

    def arr(name, shape, dtype):
        p = args.out / name
        if p.exists():
            a = np.load(p, mmap_mode="r+")
            if a.shape[0] >= shape[0]:
                return a
            # Too small anyway (e.g. --limit-videos raised): grow by COPYING the
            # existing prefix. Reallocating in place would destroy the cache.
            print(f"growing {name}: {a.shape[0]} -> {shape[0]} (copying prefix)", flush=True)
            tmp = args.out / (name[:-4] + ".grow.npy")
            b = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=shape)
            b[:a.shape[0]] = a[:]
            b.flush()
            del a, b
            os.replace(tmp, p)
            return np.load(p, mmap_mode="r+")
        return np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=shape)

    class ShardWriter:
        """Write segments into fixed-size shards, holding ONE mapping open.

        A single 770 GB mapping accumulates dirty pages across the whole run and
        leaves writeback entirely to the kernel. This box hard-hung twice under
        that pattern (journald went silent, local reads timed out at 40-117s, then
        a silent panic -- nvme_core.io_timeout is infinite here and panic=-1, so an
        I/O stall wedges forever and reboots without a trace).

        One live shard bounds the dirty set by construction, and each shard is
        msync'd and unmapped at rollover instead of lingering.
        """

        def __init__(self, out: Path, shard: int, frames: int, size: int):
            self.out, self.shard = out, shard
            self.frames, self.size = frames, size
            self.i = -1
            self.a = None

        def path(self, i):
            return self.out / f"segments_{i:05d}.npy"

        def _open(self, i):
            self.close()
            p = self.path(i)
            full = (self.shard, self.frames, self.size, self.size, 3)
            if p.exists():
                a = np.load(p, mmap_mode="r+")
                if a.shape[0] < self.shard:
                    # A partially-filled tail shard, e.g. the last one written by
                    # shard_vpt_cache.py, which sizes it to the valid prefix. It
                    # must be grown to full size before we can append into it --
                    # otherwise the write lands on an EMPTY slice and numpy raises
                    # a broadcast error deep in the parent loop.
                    print(f"growing shard {i:05d}: {a.shape[0]} -> {self.shard}",
                          flush=True)
                    keep = np.array(a)
                    del a
                    b = np.lib.format.open_memmap(p, mode="w+", dtype=np.uint8,
                                                  shape=full)
                    b[:len(keep)] = keep
                    b.flush()
                    del b, keep
                    a = np.load(p, mmap_mode="r+")
                self.a = a
            else:
                self.a = np.lib.format.open_memmap(p, mode="w+", dtype=np.uint8,
                                                   shape=full)
            self.i = i

        def close(self):
            if self.a is not None:
                self.a.flush()
                del self.a
                self.a = None
                self.i = -1

        def write(self, start, block):
            """Write `block` at global segment offset `start`, splitting on shard
            boundaries. Returns nothing; raises only on programmer error."""
            off = 0
            while off < len(block):
                gi = start + off
                si, so = divmod(gi, self.shard)
                if si != self.i:
                    self._open(si)
                take = min(len(block) - off, self.shard - so)
                self.a[so:so + take] = block[off:off + take]
                off += take

    segments = ShardWriter(args.out, args.shard_segments, F, S)
    action_seqs = arr("action_seqs.npy", (cap_n, F), np.uint8)
    keys_a = arr("keys.npy", (cap_n, F, len(KEYS)), np.uint8)
    mouse_a = arr("mouse.npy", (cap_n, F, 2), np.float16)
    pose_a = arr("pose.npy", (cap_n, F, 5), np.float32)
    gui_a = arr("gui.npy", (cap_n, F), np.uint8)
    t0 = time.time()
    jobs = [(r, b, str(args.tmp), args.chunks_per_video, F, S, args.seed)
            for r, b in cands]
    # NOT a `with` block: on an exception in the consume loop, __exit__ runs
    # shutdown(wait=True), which waits for every queued job. That turned one
    # broadcast error into six minutes of workers downloading at 149 MB/s and
    # discarding the results, with the traceback withheld until the whole queue
    # drained. Shut down with cancel_futures instead, and surface the error now.
    ex = ProcessPoolExecutor(max_workers=args.workers)
    fatal = None
    with open(done_path, "a") as dh:
      try:
          futs = {ex.submit(process_one, j): j[0] for j in jobs}
          for fut in as_completed(futs):
              relpath, res = fut.result()
              if res is None:
                  n_fail += 1
                  continue
              segs, acts, labs, ks, ms, ps, gi, picks = res
              k = len(segs)
              if w + k > cap_n:
                  break
              segments.write(w, segs)
              action_seqs[w:w + k] = acts
              keys_a[w:w + k] = ks
              mouse_a[w:w + k] = ms
              pose_a[w:w + k] = ps
              gui_a[w:w + k] = gi
              labels.extend(labs.tolist())
              clip_ids.extend([n_ok] * k)                   # one id per EPISODE
              chunk_idx.extend((picks // F).tolist())
              w += k
              n_ok += 1
              dh.write(json.dumps({"relpath": relpath, "chunks": int(k)}) + "\n")
              dh.flush()
              if n_ok % 200 == 0:
                  # msync everything, so dirty pages are bounded by the flush
                  # interval rather than left entirely to kernel writeback
                  for _m in (action_seqs, keys_a, mouse_a, pose_a, gui_a):
                      _m.flush()
                  segments.close()
                  flush_sidecars(args, labels, clip_ids, chunk_idx, w, n_ok, n_fail)
              if n_ok % 25 == 0:
                  el = time.time() - t0
                  gb = w * F * S * S * 3 / 1e9
                  print(f"{n_ok} videos ok / {n_fail} failed | {w} segs "
                        f"({w*F/1e6:.2f}M frames, {gb:.1f} GB) | "
                        f"{n_ok/el*3600:.0f} vid/h | "
                        f"root {free_gb('/'):.0f}G out {free_gb(args.out):.0f}G | "
                        f"{mem_snapshot()}",
                        flush=True)
                  stop = disk_guard(args.out, args.min_root_gb, args.min_out_gb)
                  if stop:
                      print(f"DISK GUARD: {stop}", flush=True)
                      flush_sidecars(args, labels, clip_ids, chunk_idx, w, n_ok, n_fail)
                      break
      except BaseException as e:                       # noqa: BLE001
        import traceback
        fatal = e
        print("FATAL in consume loop -- cancelling queued work so it does not "
              "keep downloading:", flush=True)
        traceback.print_exc()
    ex.shutdown(wait=False, cancel_futures=True)

    segments.close()
    for _m in (action_seqs, keys_a, mouse_a, pose_a, gui_a):
        _m.flush()
    flush_sidecars(args, labels, clip_ids, chunk_idx, w, n_ok, n_fail, complete=True)
    meta = dict(n_segments=w, capacity=int(cap_n), frames=F, size=S, complete=True,
                shard_size=args.shard_segments,
                n_records=n_ok, per_record=args.chunks_per_video,
                n_actions=N_ACTIONS, hud_row=HUD_ROW, source="vpt",
                subsets=args.subsets, keys=KEYS,
                dx_deadzone=DX_DEADZONE, dy_deadzone=DY_DEADZONE,
                gb=w * F * S * S * 3 / 1e9, failed=n_fail)
    json.dump(meta, open(args.out / "meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))
    print(f"\n{w} segments = {w*F/1e6:.2f}M frames from {n_ok} episodes "
          f"({n_fail} videos failed)", flush=True)
    if fatal is not None:
        raise fatal


if __name__ == "__main__":
    main()
