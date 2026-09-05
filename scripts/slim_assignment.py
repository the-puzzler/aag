#!/usr/bin/env python
"""Write a generator-only copy of an assignment: z (fp16) + label + metadata. The trainer reads just
A["z"], A["label"], grid, levels, steps (train_generator256_ddp.py:107-117); h/z_ref/W are 4/5 of the file."""
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
d = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
slim = {k: v for k, v in d.items() if k not in ("z", "h", "z_ref", "W", "W_inv", "mean", "curve")}
slim["z"] = d["z"].half().contiguous(); slim["label"] = d["label"]; slim["slim_from"] = src
torch.save(slim, dst); print("saved", dst, f"z {tuple(slim['z'].shape)} fp16")
