#!/usr/bin/env python
"""Build world-model particles from the Doom chunk cache.

A particle is one frame t together with its conditioning context:
    target   h_t                      (dim,)      what the generator must produce
    context  [h_{t-3}, h_{t-2}, h_{t-1}]  (3*dim,) continuous part of the condition
    action   a_t                      int         exact 18-way categorical part

Frames must be CONSECUTIVE, so this reads whole chunks (the AE was trained on
stride-4 frames only to avoid near-duplicates; adjacency is needed here).

Quadruples are sampled a few per EPISODE rather than densely per chunk: the
190k chunks come from only 70k episodes, and it is effective (episode-level)
N that bounds the usable latent dimension -- the lesson UCF-101 taught.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch

from aag.ae import AutoEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/data/doom/cache_train")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--context", type=int, default=3, help="past frames in the condition")
    ap.add_argument("--per-episode", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    ae = AutoEncoder(ck["latent_dim"], ch=ck["channels"], architecture=ck["architecture"],
                     image_size=ck["image_size"]).to(dev).eval()
    ae.load_state_dict(ck["model_state_dict"])
    dim = ck["latent_dim"]
    print(f"AE dim={dim} arch={ck['architecture']} (ep{ck.get('epochs')}, "
          f"test_mse={ck.get('test_mse'):.5f})", flush=True)

    segs = np.load(f"{args.cache}/segments.npy", mmap_mode="r")
    acts = np.load(f"{args.cache}/action_seqs.npy")
    rec = np.load(f"{args.cache}/clip_ids.npy")
    n_chunks, T = len(rec), segs.shape[1]
    C = args.context

    # sample: per episode, pick `per_episode` (chunk, t) pairs with t >= C
    rng = np.random.default_rng(args.seed)
    order = np.argsort(rec, kind="stable")
    picks = []
    for ep_start, ep_end in _runs(rec[order]):
        rows = order[ep_start:ep_end]
        for _ in range(args.per_episode):
            c = int(rng.choice(rows))
            t = int(rng.integers(C, T))
            picks.append((c, t))
    picks = np.array(picks)
    print(f"{len(picks):,} particles from {len(np.unique(rec)):,} episodes "
          f"({args.per_episode}/episode)", flush=True)

    # encode: every particle needs frames t-C..t of its chunk
    need = np.stack([picks[:, 1] - k for k in range(C, -1, -1)], 1)     # (N, C+1)
    H = torch.empty(len(picks), C + 1, dim)
    with torch.no_grad():
        for i in range(0, len(picks), args.batch):
            sl = slice(i, i + args.batch)
            c_i, f_i = picks[sl, 0], need[sl]
            frames = segs[c_i[:, None], f_i]                            # (b,C+1,H,W,3)
            x = torch.from_numpy(np.ascontiguousarray(frames)).to(dev)
            b, m = x.shape[0], x.shape[1]
            x = x.reshape(b * m, *x.shape[2:]).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
            H[sl] = ae.enc(x).reshape(b, m, dim).cpu()
            if i % (args.batch * 40) == 0:
                print(f"  encoded {i:,}/{len(picks):,}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "h_target": H[:, -1].contiguous(),                 # (N, dim)
        "h_context": H[:, :-1].reshape(len(picks), -1).contiguous(),   # (N, C*dim)
        "action": torch.from_numpy(acts[picks[:, 0], picks[:, 1]].astype(np.int64)),
        "chunk": torch.from_numpy(picks[:, 0]), "frame": torch.from_numpy(picks[:, 1]),
        "episode": torch.from_numpy(rec[picks[:, 0]]),
        "dim": dim, "context": C, "checkpoint": args.checkpoint,
    }, args.out / "particles.pt")
    meta = dict(n=len(picks), dim=dim, context=C, cond_dim=C * dim,
                n_episodes=int(len(np.unique(rec))), per_episode=args.per_episode)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


def _runs(sorted_ids):
    """Yield (start, end) index ranges of equal consecutive ids."""
    b = np.flatnonzero(np.diff(sorted_ids)) + 1
    edges = np.concatenate([[0], b, [len(sorted_ids)]])
    return zip(edges[:-1], edges[1:])


if __name__ == "__main__":
    main()
