#!/usr/bin/env python
"""Concatenate particle files (encode_hf256.py outputs) row-wise: originals first, then e.g. the --flip copies.
The generator trainer's --aug-hflip expects exactly [originals | flipped] in that order."""
import sys, torch
outs = [torch.load(p, map_location="cpu", weights_only=False) for p in sys.argv[1:-1]]
m = dict(outs[0]); m["h"] = torch.cat([o["h"] for o in outs]); m["label"] = torch.cat([o["label"] for o in outs])
m["n_particles"] = m["h"].shape[0]; m["merged_from"] = sys.argv[1:-1]; m["aug"] = "hflip" if len(outs) == 2 else f"x{len(outs)}"
torch.save(m, sys.argv[-1]); print("saved", sys.argv[-1], tuple(m["h"].shape))
