"""HF-parquet image datasets at 256x256 as GPU-resident uint8 tensors.

Both scale-up datasets are Parquet repacks on the Hub:
  benjamin-paine/imagenet-1k-256x256   1,281,167 train  center-crop + Lanczos 256
  korexyz/celeba-hq-256x256               28,000 train  (+2,000 validation)

The particle identity x_i <-> z_i is the GLOBAL ROW INDEX over the train
parquet files in sorted filename order. Every consumer (encoder, assignment,
generator, per-rank shards) indexes by that integer, so the only way to
misalign is to change the file order -- which is why the order is sorted here
and asserted against the manifest, never taken from a directory listing.

Under DDP each rank decodes only its contiguous slice [lo, hi) and keeps it on
its own GPU: ImageNet at 256x256x3 uint8 is 252 GB, i.e. 31.5 GB per B200, and
after that one decode there is no I/O path in the epoch loop at all -- the
VPT trainers' GPU-resident pattern, so batches are z[idx] and images[idx].
"""
from __future__ import annotations

import io, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

DATASETS = {
    "imagenet256": dict(hub="benjamin-paine/imagenet-1k-256x256", dirname="imagenet-1k-256x256",
                        n_classes=1000, image_size=256),
    "celebahq256": dict(hub="korexyz/celeba-hq-256x256", dirname="celeba-hq-256x256",
                        n_classes=2, image_size=256),   # label = female/male; unconditional by default
}


def parquet_files(root: str | Path, dataset: str, split: str = "train") -> list[Path]:
    d = Path(root) / DATASETS[dataset]["dirname"] / "data"
    files = sorted(d.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet under {d}")
    return files


def manifest(root, dataset, split="train") -> dict:
    """Row counts per file, cached next to the data. Row offsets define the
    global particle index, so this is the identity contract."""
    files = parquet_files(root, dataset, split)
    mpath = files[0].parent / f"_manifest_{split}.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        if m["files"] == [f.name for f in files]:
            return m
    import pyarrow.parquet as pq
    rows = [pq.ParquetFile(f).metadata.num_rows for f in files]
    m = {"files": [f.name for f in files], "rows": rows,
         "offsets": [int(x) for x in np.cumsum([0] + rows)]}
    mpath.write_text(json.dumps(m))
    return m


def _decode_rows(args):
    """Decode rows [a, b) of one parquet file into a uint8 (n, H, W, 3) array."""
    path, a, b, size = args
    import pyarrow.parquet as pq
    from PIL import Image
    pf = pq.ParquetFile(path)
    out = np.empty((b - a, size, size, 3), dtype=np.uint8)
    labels = np.empty(b - a, dtype=np.int16)
    k, row0 = 0, 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        lo, hi = max(a, row0), min(b, row0 + n)
        if lo < hi:
            t = pf.read_row_group(rg, columns=["image", "label"])
            imgs = t.column("image").to_pylist()
            labs = t.column("label").to_pylist()
            for i in range(lo - row0, hi - row0):
                im = Image.open(io.BytesIO(imgs[i]["bytes"])).convert("RGB")
                if im.size != (size, size):
                    im = im.resize((size, size), Image.LANCZOS)
                out[k] = np.asarray(im)
                labels[k] = -1 if labs[i] is None else labs[i]
                k += 1
        row0 += n
        if row0 >= b:
            break
    return out, labels


def load_uint8(root, dataset, lo=0, hi=None, split="train", workers=None, chunk=4096):
    """Decode global rows [lo, hi) -> (uint8 tensor (n,H,W,3) CPU, int64 labels).

    Work is split into ~`chunk`-row pieces across processes; ImageNet's full
    1.28M rows decode in about a minute on 96 cores.
    """
    spec = DATASETS[dataset]
    m = manifest(root, dataset, split)
    files = parquet_files(root, dataset, split)
    N = m["offsets"][-1]
    hi = N if hi is None else min(hi, N)
    assert 0 <= lo < hi <= N, (lo, hi, N)
    jobs = []
    for f, off, n in zip(files, m["offsets"][:-1], m["rows"]):
        a, b = max(lo, off), min(hi, off + n)
        for s in range(a, b, chunk):
            jobs.append((str(f), s - off, min(s + chunk, b) - off, spec["image_size"]))
    workers = workers or min(len(jobs), os.cpu_count() or 8)
    parts, labs = [], []
    with ProcessPoolExecutor(workers) as ex:
        for out, lab in ex.map(_decode_rows, jobs):
            parts.append(out); labs.append(lab)
    x = torch.from_numpy(np.concatenate(parts, 0))
    y = torch.from_numpy(np.concatenate(labs, 0).astype(np.int64))
    assert x.shape[0] == hi - lo
    return x, y


def to_float(x_u8: torch.Tensor) -> torch.Tensor:
    """(B,H,W,3) uint8 -> (B,3,H,W) float in [-1,1], the AE/generator convention."""
    return x_u8.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


def rank_slice(N: int, rank: int, world: int) -> tuple[int, int]:
    per = (N + world - 1) // world
    return rank * per, min(N, (rank + 1) * per)
