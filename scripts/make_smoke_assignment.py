#!/usr/bin/env python
"""Build a tiny 12-d assignment so the generator's load path can be smoke tested.

The point is to fail fast. The real assignment takes ~13 h, and if the
generator's conditioning block has a typo the chained run would die instantly at
the end of that wait. This splices the 12-d action fields onto an existing
assignment's z/cond/h and subsets it, giving something with real shapes and real
pixel-target coordinates that a 1-epoch run can chew through in minutes.

Not for training anything that matters: the z comes from an assignment built
against the OLD 9-d condition.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", type=Path, required=True)
    ap.add_argument("--particles", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    A = torch.load(args.assignment, map_location="cpu", weights_only=False)
    P = torch.load(args.particles, map_location="cpu", weights_only=False)
    N = A["z"].shape[0]
    if P["action_raw"].shape[0] != N:
        raise SystemExit(f"particle/assignment size mismatch {P['action_raw'].shape[0]} vs {N}")
    # sanity: the assignment and the particles must describe the same rows
    if not torch.equal(A["chunk"], P["chunk"]) or not torch.equal(A["frame"], P["frame"]):
        raise SystemExit("chunk/frame differ -- not the same particle set")

    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(N, generator=g)[: args.n].sort().values

    out = dict(A)
    for k in ("z", "h", "cond", "action", "chunk", "frame", "episode"):
        if out.get(k) is not None:
            out[k] = A[k][idx].clone()
    out["action_raw"] = P["action_raw"][idx].clone()
    out["action_vec"] = P["action_vec"][idx].clone()
    out["act_norm"] = P["act_norm"]
    out["act_names"] = P["act_names"]
    out["act_dim"] = P["act_dim"]
    out["clicks_coverage"] = P.get("clicks_coverage")
    out["partial"] = True
    out["smoke"] = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"{args.n:,} particles -> {args.out}")
    print(f"  z {tuple(out['z'].shape)}  cond {tuple(out['cond'].shape)}  "
          f"action_vec {tuple(out['action_vec'].shape)}")


if __name__ == "__main__":
    main()
