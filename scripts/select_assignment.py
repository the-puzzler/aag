#!/usr/bin/env python
"""Pick an assignment: low G first, then lowest I, with R_impact as tie-break.

G is primary -- a non-Gaussian z means fresh samples land in the wrong
distribution at all -- so candidates are first filtered to within --g-tol of the
best G. Among those, lower I is preferred: I dominated on the one pair where we
have held-out FID (Doom video 4k vs 20k), where R_impact ranked it backwards.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch

from aag.diagnostics import assignment_diagnostics, r_rel, f_z

ap = argparse.ArgumentParser()
ap.add_argument("candidates", nargs="+")
ap.add_argument("--g-tol", type=float, default=1.25)
ap.add_argument("--out", default="results_doom/assignment_choice.json")
a = ap.parse_args()

rows = []
for p in a.candidates:
    D = torch.load(p, map_location="cuda", weights_only=False)
    z, h, c = D["z"].cuda().float(), D["h"].cuda().float(), D["cond"].cuda().float()
    g = torch.Generator(device="cuda").manual_seed(0)
    G = assignment_diagnostics(z, d=z.shape[1])["proj_over_gauss"]
    rc = r_rel(z, h, cond=c, gen=g)
    fz = f_z(h, c, gen=g)
    curve = D.get("curve", {}) or {}
    I = curve.get("ratio", [float("nan")])[-1] if curve.get("ratio") else float("nan")
    rows.append(dict(path=p, G=G, I=I, R_rel_c=rc, f_z=fz, R_impact=fz * rc,
                     steps=D.get("steps"), cps=D.get("cond_per_step")))
    print(f"{Path(p).parent.name:<28} G={G:6.2f}  I={I:6.2f}  R_imp={fz*rc:5.3f}", flush=True)

gbest = min(r["G"] for r in rows)
ok = [r for r in rows if r["G"] <= gbest * a.g_tol]
ok.sort(key=lambda r: (r["I"], r["R_impact"]))
pick = ok[0]
print(f"\nbest G={gbest:.2f}; accepting G <= {gbest*a.g_tol:.2f} -> {len(ok)}/{len(rows)} candidates")
print(f"CHOSEN {Path(pick['path']).parent.name}: G={pick['G']:.2f} I={pick['I']:.2f} "
      f"R_impact={pick['R_impact']:.3f}", flush=True)
json.dump({"rows": rows, "chosen": pick["path"]}, open(a.out, "w"), indent=1)
print(pick["path"])
