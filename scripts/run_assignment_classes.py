#!/usr/bin/env python
"""Gaussian assignment for a CATEGORICAL condition with a hierarchy, or none.

Used for ImageNet-256 (class-conditional) and CelebA-HQ-256 (unconditional).
Two of the user's rules shape it:

  1. "More Gaussian is more good." The transport budget is generous, the run
     saves step-stamped checkpoints (--keep-checkpoints) and the step count is
     chosen AFTERWARDS by the generator trained on each -- never by the
     transport objective, which sits inside its own noise floor after a few
     thousand steps and cannot see further progress.

  2. Gaussianising a group is not gaussianising its marginals. Independence
     from the 1000-way class does not buy independence from "is a dog": each
     per-class transport moves ~1281 particles and cannot resolve a shift
     shared by all 116 dog classes, while one transport over those 148k
     particles removes it. So group transport is INTERLEAVED across every
     hierarchy level (--levels), size-weighted so touches per particle are
     equalised across groups of very different sizes, and the readout reports
     the ratio at EVERY level -- one number against the joint class is exactly
     the reading that hid the VPT action leak for days.

Global steps are the published recipe (greedy rank transport + slab cleanup +
radial chi calibration). Whitening is rotate=False so the spatial grid of the
AE latent survives into z (coordinate j of z is still grid cell j of h).
"""
from __future__ import annotations

import argparse, gc, json, time
from pathlib import Path

import torch

from aag.diagnostics import group_w2, random_subset_w2, transport_objective_floor
from aag.gaussianize import (greedy_rank_transport_step, group_rank_transport_step,
                             offset_slab_cleanup_step, radial_chi_calibration, whiten)

ap = argparse.ArgumentParser()
ap.add_argument("--particles", required=True, help="output of encode_hf256.py")
ap.add_argument("--groups", default=None, help="class_groups.pt from imagenet_class_groups.py")
ap.add_argument("--levels", default="",
                help="comma list of hierarchy levels to transport against, e.g. "
                     "'joint,depth7,depth6,depth5,depth4,living,animal,dog'. Empty = unconditional.")
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--search-subset", type=int, default=2048)
ap.add_argument("--n-dirs", type=int, default=64)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--grp-per-step", type=int, default=8,
                help="group-transport firings per global step, split evenly over levels")
ap.add_argument("--cond-alpha", type=float, default=0.5)
ap.add_argument("--max-group", type=int, default=32768,
                help="particles per group firing; a coarse group like 'artifact' has "
                     "660k members and 32k already puts quantile noise at 0.6%%")
ap.add_argument("--cleanup-every", type=int, default=2)
ap.add_argument("--chi-every", type=int, default=20)
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--eval-k", type=int, default=4096)
ap.add_argument("--save-every", type=int, default=0)
ap.add_argument("--keep-checkpoints", action="store_true")
ap.add_argument("--resume-z", default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = "cuda"
P = torch.load(a.particles, map_location="cpu", weights_only=False)
h = P["h"].to(dev).float()
labels = P["label"].to(dev)
N, D = h.shape
keep = {k: P[k] for k in ("encoder", "grid", "latent_channels", "dataset", "root", "n_particles") if k in P}
del P; gc.collect()

levels = [x.strip() for x in a.levels.split(",") if x.strip()]
grp_ids = {}
if levels:
    G = torch.load(a.groups, map_location="cpu", weights_only=False)
    for lv in levels:
        if lv not in G["levels"]:
            raise SystemExit(f"--levels: unknown '{lv}', choose from {sorted(G['levels'])}")
        grp_ids[lv] = G["levels"][lv].to(dev)[labels]        # per-particle group id at that level
    grp_per = max(1, a.grp_per_step // len(levels))
print(f"{N:,} particles  dim={D}  levels={levels or 'none (unconditional)'}"
      + (f"  {grp_per} firings/level/step, max_group={a.max_group}, cond_alpha={a.cond_alpha}" if levels else ""),
      flush=True)

step0 = 0
if a.resume_z:
    R = torch.load(a.resume_z, map_location="cpu", weights_only=False)
    z = R["z"].to(dev).float().contiguous(); mean, W, W_inv = R["mean"].to(dev), R["W"].to(dev), R["W_inv"].to(dev)
    z_ref = R["z_ref"].to(dev) if "z_ref" in R else None
    step0 = int(R.get("steps", 0)); curve = R["curve"]
    print(f"resumed z from {a.resume_z} at step {step0:,}", flush=True)
    del R; gc.collect()
else:
    z, mean, W, W_inv = whiten(h, rotate=False)
    z = z.contiguous(); z_ref = None
    curve = {"step": [], "G": [], "disp": [], "floor": [], "ratio": {lv: [] for lv in levels}}
d = z.shape[1]
if z_ref is None:
    z_ref = z.clone()
gen = torch.Generator(device=dev).manual_seed(a.seed + step0)
fl_mean, fl_std = transport_objective_floor(N, d, search_subset=a.search_subset, n_dirs=a.n_dirs, device=dev, gen=gen)
print(f"transport objective noise floor at (N={N:,}, d={d}): {fl_mean:.5f} +/- {fl_std:.5f}  "
      f"(the objective is uninformative once inside this band; keep going anyway -- rule 1)", flush=True)
zrad = float(d) ** 0.5


def gdefect(t):
    dirs = torch.randn(64, t.shape[1], device=dev, generator=gen); dirs /= dirs.norm(dim=1, keepdim=True)
    s, _ = torch.sort(t @ dirs.T, dim=0)
    q = torch.special.ndtri((torch.arange(len(t), device=dev, dtype=t.dtype) + .5) / len(t)).unsqueeze(1)
    return ((s - q) ** 2).mean().item()


def save(path, partial):
    torch.save({"z": z.cpu(), "h": h.half().cpu(), "label": labels.cpu(), "mean": mean.cpu(),
                "W": W.cpu(), "W_inv": W_inv.cpu(), "z_ref": z_ref.cpu(), "curve": curve,
                "steps": step0 + step, "levels": levels,
                "cond_alpha": a.cond_alpha, "max_group": a.max_group, "grp_per_step": a.grp_per_step,
                "particles": a.particles, "groups": a.groups, "partial": partial, **keep},
               str(path) + ".tmp")
    Path(str(path) + ".tmp").replace(path)


t0 = time.time()
for step in range(1, a.steps + 1):
    obj = greedy_rank_transport_step(z, search_subset=a.search_subset, n_dirs=a.n_dirs, alpha=a.alpha, gen=gen)
    if a.cleanup_every and step % a.cleanup_every == 0:
        offset_slab_cleanup_step(z, search_subset=a.search_subset, n_slabs=32, eps=0.5, alpha=1.0, gen=gen)
    for lv in levels:
        for _ in range(grp_per):
            group_rank_transport_step(z, grp_ids[lv], n_dirs=a.n_dirs, alpha=a.cond_alpha, gen=gen,
                                      max_group=a.max_group, size_weighted=True)
    if a.chi_every and step % a.chi_every == 0:
        radial_chi_calibration(z, d=d, alpha_r=1.0)

    if step % a.eval_every == 0 or step == 1:
        G = gdefect(z); disp = float((z - z_ref).norm(dim=1).mean())
        floor = random_subset_w2(z, k=a.eval_k, n_eval=20, gen=gen)
        curve["step"].append(step0 + step); curve["G"].append(G); curve["disp"].append(disp); curve["floor"].append(floor)
        parts = []
        for lv in levels:
            gw, kk = group_w2(z, grp_ids[lv], n_eval=20, gen=gen, max_group=a.eval_k)
            fl = floor if kk == a.eval_k else random_subset_w2(z, k=kk, n_eval=20, gen=gen)
            r = gw / max(fl, 1e-12); curve["ratio"][lv].append(r); parts.append(f"{lv}={r:.2f}")
        rate = step / (time.time() - t0)
        print(f"step {step0 + step:6d}  obj={obj:.5f} (floor {fl_mean:.5f})  G={G:.5f}  "
              f"disp={disp:.2f} ({100 * disp / zrad:.0f}% of ||z||)  " + "  ".join(parts)
              + f"   [{rate:.1f} step/s, eta {(a.steps - step) / rate / 60:.0f} min]", flush=True)

    if a.save_every and step % a.save_every == 0 and step < a.steps:
        p = Path(a.out)
        if a.keep_checkpoints:
            p = p.with_name(f"{p.stem}_step{step0 + step}{p.suffix}")
        p.parent.mkdir(parents=True, exist_ok=True)
        save(p, partial=True); gc.collect()
        print(f"  [checkpoint {p.name}]", flush=True)

out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
save(out, partial=False)
(out.with_suffix(".curve.json")).write_text(json.dumps(curve, indent=1))
print(f"saved -> {out}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
