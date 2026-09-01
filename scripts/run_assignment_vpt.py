#!/usr/bin/env python
"""Gaussianise VPT world-model particles, with the action side as a DISTANCE.

Mirrors run_assignment_doom_ctx.py's budget split -- mostly context-only steps
plus some action-conditioned ones -- but the action steps use
action_dist_knn_transport_step (action-nearest-k_act, then context-nearest-k)
instead of an exact categorical filter, because the 81-way index discards mouse
magnitude and magnitude explains more frame-to-frame change than all keys
combined.

Reports, every --eval-every steps:
  G  global gaussian defect
  I_ctx     context-nearest-k W2 / random-subset floor   -> 1.0
  I_act     action-nearest-k  W2 / random-subset floor   -> 1.0
  per-action worst classes, so one bad action is visible rather than averaged
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from aag.gaussianize import (continuous_knn_transport_batch, group_rank_transport_step, whiten, greedy_rank_transport_step,
                             continuous_knn_transport_step,
                             action_dist_knn_transport_step)
from aag.diagnostics import (random_subset_w2, continuous_knn_w2,
                             action_dist_knn_w2, per_action_w2)

ap = argparse.ArgumentParser()
ap.add_argument("--particles", required=True)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--search-subset", type=int, default=2048)
ap.add_argument("--n-dirs", type=int, default=64)
ap.add_argument("--ctx-per-step", type=int, default=8)
ap.add_argument("--act-per-step", type=int, default=8)
ap.add_argument("--grp-per-step", type=int, default=0,
                help="Exact-action-class transport firings per step. The other "
                     "action step decorrelates z from the CONTINUOUS act_vec, but "
                     "the generator conditions on a discrete 81-way one-hot -- a "
                     "different partition. Measured on the 16k run: z sits 2.49x "
                     "further off-centre within an action class than chance, "
                     "against 1.03x for context, which is why a fresh z degrades. "
                     "This transports the true p(z | class).")
ap.add_argument("--max-group", type=int, default=8192,
                help="cap per class so one huge class (a0 has 207k members) does "
                     "not dominate the step cost")
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--cond-alpha", type=float, default=0.25)
ap.add_argument("--k", type=int, default=2048)
ap.add_argument("--k-act", type=int, default=8192)
ap.add_argument("--eval-k", type=int, default=2048)
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save-every", type=int, default=0,
                help="If >0, write the assignment every N steps as well as at the "
                     "end. A long run that only saves on completion loses "
                     "everything to a crash, and cannot be stopped early to take "
                     "what it has.")
ap.add_argument("--keep-checkpoints", action="store_true",
                help="Write step-stamped checkpoints instead of overwriting one "
                     "file, so the step count can be chosen after the fact by "
                     "the fresh-z MSE of a generator trained on each. The "
                     "optimum is non-monotonic and the transport objective "
                     "cannot see it: CelebA's 4k assignment fit 1.6-2.2x BETTER "
                     "than its 60k one, because mean displacement keeps growing "
                     "(15% of ||z|| at 4k, 43% at 60k) and particles moved that "
                     "far no longer have neighbouring z mapping to similar "
                     "images. Costs ~2.6 GB per checkpoint at 512k/dim256.")
ap.add_argument("--act-groups", default="joint",
                help="Comma list of action groupings for the exact-class "
                     "transport, INTERLEAVED within every step. The 81-way "
                     "index is a product, action = move*9 + turn*3 + tilt, so "
                     "independence from the joint class does not imply "
                     "independence from move alone or turn alone -- the same "
                     "aggregate-hides-the-marginal error as the context side. "
                     "Measured on the 3-frame assignment: joint 81-way mean "
                     "0.593, but 'turning vs not' 1.272 and 'turn only' 1.043. "
                     "z still carried whether the player was turning, so a "
                     "fresh z argues with the turn command, which is what weak "
                     "action following looks like. Legacy 81-way choices: "
                     "joint, move, turn, tilt, moveturn, turntilt, moving, "
                     "turning. With the 12-d representation (action_raw "
                     "present) each control is also a grouping in its own "
                     "right: w, a, s, d, space, shift, ctrl, e, attack, use, "
                     "anyclick, plus dxsign/dysign (3 each, cache deadzones) "
                     "and dxmag/dymag (deadzone + terciles). Prefer these -- "
                     "they are exactly the marginals of what the generator now "
                     "sees, whereas the 81-way ones are marginals of an index "
                     "it no longer receives.")
ap.add_argument("--grp-uniform", action="store_true",
                help="Restore the old uniform-over-classes sampling for the "
                     "exact-class action transport. The default now samples a "
                     "class with probability proportional to max(1, n/"
                     "max_group), which equalises transport touches PER "
                     "PARTICLE rather than per class. Uniform sampling gave "
                     "VPT's a9 (n=74,291) 0.022 touches per member per step "
                     "against a46 (n=371) at 0.198 -- a 9x disparity biased "
                     "against the commonest actions, which are exactly the "
                     "ones used at inference. It showed up as corr(log n, "
                     "per-action ratio) = +0.75 with the big classes worst "
                     "decorrelated. Only for reproducing pre-fix runs.")
ap.add_argument("--ctx-scales", default="",
                help="Comma list of context lengths to transport against, "
                     "INTERLEAVED within every global step (e.g. '1,3,5,12'). "
                     "Overrides --ctx-frames for the context transport. "
                     "Motivation: decorrelating z against one context length "
                     "leaves it correlated with shorter ones. Measured at "
                     "n_eval=200, the 24-frame assignment reads 0.897 against "
                     "its own unweighted context but 1.162 / 3.030 / 4.779 / "
                     "8.253 against the newest 12 / 5 / 3 / 1 -- a monotone "
                     "escalation as the probe shortens. Shortening the "
                     "transport to 3 frames moved newest-1 from 8.253 to "
                     "2.139, but the same gap reappeared one scale down (its "
                     "own metric read 1.246). Interleaving all scales in one "
                     "run addresses every scale at once, and unlike a staged "
                     "coarse-to-fine schedule no later stage can undo an "
                     "earlier one. The ctx budget is split evenly across "
                     "scales. The FULL context is saved, so the generator "
                     "still sees all 24 frames.")
ap.add_argument("--ctx-frames", type=int, default=0,
                help="Transport against only the newest N of the stored context "
                     "blocks (0 = all of them). Measured on the 24-frame cosine "
                     "assignment at n_eval=200: the saved z reaches I_ctx 0.954 "
                     "against the full weighted context and 0.897 unweighted, "
                     "but 3.030 against the newest 5 frames and 8.253 against "
                     "the newest 1. Independence from the whole trajectory is a "
                     "much weaker condition than independence from where the "
                     "camera is right now -- two particles sharing a last frame "
                     "but differing in history sit far apart in 6144-d context "
                     "space, so they never share a neighbourhood and transport "
                     "never decorrelates z across them. A one-step predictor "
                     "needs the latter. Doom, which worked, used 3 frames.")
ap.add_argument("--ctx-metric", choices=["l2", "cosine"], default="l2",
                help="Distance defining the CONTEXT neighbourhood. The doom run "
                     "that reached I 0.860 used cosine; VPT has been using l2. "
                     "cond_distance's rationale: AE latent magnitude tracks "
                     "brightness/contrast rather than content, so under l2 a "
                     "neighbourhood groups frames by overall brightness as much "
                     "as by scene -- and Minecraft spans day, night, caves and "
                     "biomes. Applied to the eval too: measuring l2 independence "
                     "while transporting cosine neighbourhoods would be "
                     "meaningless.")
ap.add_argument("--resume-z", default=None,
                help="Continue transport from a saved assignment's z instead of "
                     "re-whitening from scratch. The 512k run looked plateaued "
                     "at I_ctx 7.5 around step 5000 and still reached 1.035 by "
                     "16000, so an apparent plateau is not evidence of a floor "
                     "-- this makes testing that cheap rather than a restart.")
a = ap.parse_args()

dev = "cuda"
import gc
P = torch.load(a.particles, map_location="cpu", weights_only=False)
h = P["h_target"].to(dev).float()
cond = P["h_context"].to(dev).float()
scales = [int(x) for x in a.ctx_scales.split(",") if x.strip()]
# The two compose: --ctx-frames sets what the generator will SEE (cond is sliced
# before saving), --ctx-scales sets what the transport decorrelates AGAINST
# within that. "--ctx-frames 3 --ctx-scales 1,3" keeps a 3-frame model while
# closing the newest-1 gap, which a plain 3-frame transport leaves at 2.139.
if a.ctx_frames:
    _C = int(P["context"])
    if not 0 < a.ctx_frames <= _C:
        raise SystemExit(f"--ctx-frames must be in 1..{_C}")
    # cond is recency-scaled with the newest block at weight 1, so a suffix
    # slice is exactly what a shorter-context particle build would produce.
    _blk = cond.shape[1] // _C
    cond = cond[:, (_C - a.ctx_frames) * _blk:].contiguous()
    P["context"] = a.ctx_frames
act = P["action"].to(dev)
_m, _t, _p = act // 9, (act // 3) % 3, act % 3
# Legacy groupings, derived from the 81-way index. Retained so pre-12d
# assignments reproduce; the index itself is no longer generator input.
_ACT_GROUPINGS = {"joint": act, "move": _m, "turn": _t, "tilt": _p,
                  "moveturn": _m * 3 + _t, "turntilt": _t * 3 + _p,
                  "moving": (_m > 0).long(), "turning": (_t > 0).long()}
# The 12-d representation's marginals: one per binary control, mouse direction
# and mouse magnitude per axis. These are the low-dimensional functions of the
# condition the generator can actually key on now that it sees the 12-d vector
# and nothing else, and the project's standing lesson is that independence from
# the vector AS A WHOLE does not imply independence from any of them.
if "action_raw" in P:
    from aag.vpt_actions import action_marginals
    _raw = P["action_raw"].numpy()
    for _nm, _g in action_marginals(_raw).items():
        _ACT_GROUPINGS[_nm] = torch.from_numpy(_g).to(dev)
    del _raw
av = P["action_vec"].to(dev).float()
N, d = h.shape
# h_context is 40.9 GB at 1.66M particles and is dead once cond is on the GPU.
# Holding it alongside the GPU copy AND the save-time cond.cpu() copy is what
# OOM-killed the first 32k attempt -- earlyoom fired at VmRSS 117 GB.
_keep = {k: P[k] for k in ("chunk", "frame", "episode", "cache", "checkpoint",
                           "context", "gamma",
                           # the 12-d action side: act_norm is the inference
                           # contract (live input must be encoded with the same
                           # constants), action_raw is what the diagnostics
                           # rebuild the marginals from. 24 MB at 512k.
                           "action_raw", "act_norm", "act_names", "act_dim",
                           "clicks_coverage") if k in P}
P.clear(); P.update(_keep)
gc.collect()
print(f"{N:,} particles  dim={d}  cond_dim={cond.shape[1]}  "
      f"context={P['context']} gamma={P['gamma']} ctx_metric={a.ctx_metric}",
      flush=True)
cond_scales = []
if scales:
    _C = int(P["context"])          # already reduced if --ctx-frames sliced
    _blk = cond.shape[1] // _C
    for n in scales:
        if not 0 < n <= _C:
            raise SystemExit(f"--ctx-scales entries must be in 1..{_C}")
        # contiguous copies, not views: the transport does a matmul against the
        # whole tensor every firing, and the cosine path caches a normalised
        # copy per scale. 1/3/5/12/24 at 512k particles is 23.6 GB together.
        cond_scales.append(cond[:, (_C - n) * _blk:].contiguous())
    per = max(1, a.ctx_per_step // len(scales))
    print(f"multi-scale context transport: scales {scales}, "
          f"{per} firings each per step "
          f"({[c.shape[1] for c in cond_scales]} dims)", flush=True)
act_groups = [x.strip() for x in a.act_groups.split(",") if x.strip()]
for _g in act_groups:
    if _g not in _ACT_GROUPINGS:
        raise SystemExit(f"--act-groups: unknown '{_g}', choose from "
                         f"{sorted(_ACT_GROUPINGS)}")
grp_ids = [_ACT_GROUPINGS[g] for g in act_groups]
grp_per = max(1, a.grp_per_step // len(grp_ids))
if len(grp_ids) > 1:
    print(f"action groupings interleaved: {act_groups}, {grp_per} firings each "
          f"per step", flush=True)
print(f"budget: {a.steps} global x ({a.ctx_per_step} ctx + {a.act_per_step} act "
      f"+ {a.grp_per_step} grp) = "
      f"{a.steps*(a.ctx_per_step+a.act_per_step+a.grp_per_step):,} conditional firings",
      flush=True)
print("NOTE: these ratios do NOT select a good assignment. Audited\n       2026-08-31: the doom assignment that actually won logged\n       joint 1.14 / 1.02 / 0.86 / 1.12 (the last two the SAME step),\n       and VPT's runs live at 0.92-1.17 -- the same regime on a\n       +/-0.13 instrument. The 0.860 and 0.778 figures previously\n       quoted as targets were single noisy evals. Read neighbouring\n       evals before quoting any value, and judge an assignment by the\n       fresh-z MSE of a generator trained on it, which is what the\n       doom run selected on.", flush=True)

# rotate=False keeps coordinate j of z meaning coordinate j of h, which matters
# for a spatial AE latent whose grid topology a PCA rotation would destroy.
step0 = 0
if a.resume_z:
    R = torch.load(a.resume_z, map_location="cpu", weights_only=False)
    if R["z"].shape != h.shape:
        raise SystemExit(f"resume z is {tuple(R['z'].shape)} but these particles "
                         f"are {tuple(h.shape)} -- different particle file")
    z = R["z"].to(dev).float().contiguous()
    mean = R["mean"].to(dev); W = R["W"].to(dev); W_inv = R["W_inv"].to(dev)
    step0 = int(R.get("steps", 0))
    print(f"resumed z from {a.resume_z} at step {step0:,} "
          f"(I_ctx was {R['curve']['ctx_ratio'][-1]:.3f}, "
          f"I_act {R['curve']['act_ratio'][-1]:.3f})", flush=True)
    _curve0 = {k: list(v) for k, v in R["curve"].items()}
    R.clear(); R["curve"] = _curve0    # its cond/h were a third 40.9 GB copy
    gc.collect()
else:
    z, mean, W, W_inv = whiten(h, rotate=False)
    z = z.contiguous()
gen = torch.Generator(device=dev).manual_seed(a.seed)

curve = ({k: list(v) for k, v in R["curve"].items()} if a.resume_z
         else {"step": [], "ctx_ratio": [], "act_ratio": [], "floor": [], "G": []})


# Mean transport displacement ||z - z_whitened||, the quantity that PREDICTED
# generator fit degradation when the per-step objective could not see it. On
# CelebA the 4k assignment moved particles 15% of their own radius and the 60k
# one 43%, and the 60k z fit 1.6-2.2x WORSE at matched epochs: particles moved so
# far that neighbouring z no longer map to similar images, so the z->image map
# loses locality. The greedy objective is inside its own noise band long before
# this stops growing, so it cannot be used as the stopping signal. Logging this
# is what makes "it has moved too much" a measurement rather than an intuition.
z_ref = z.clone()
_zrad = float(z.shape[1]) ** 0.5          # typical ||z|| for N(0,I_d)


def displacement():
    return float((z - z_ref).norm(dim=1).mean())


def gdefect(t):
    dirs = torch.randn(64, t.shape[1], device=t.device, generator=gen)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    s, _ = torch.sort(t @ dirs.T, dim=0)
    q = torch.special.ndtri(
        (torch.arange(len(t), device=t.device, dtype=t.dtype) + .5) / len(t)).unsqueeze(1)
    return ((s - q) ** 2).mean().item()


for step in range(1, a.steps + 1):
    greedy_rank_transport_step(z, search_subset=a.search_subset, n_dirs=a.n_dirs,
                               alpha=a.alpha, gen=gen)
    # one pass over cond for all ctx firings instead of one per firing --
    # bit-identical, and cond is 12.6 GB at 512k particles
    if cond_scales:
        for _cs in cond_scales:
            continuous_knn_transport_batch(z, _cs, k=a.k, n_dirs=a.n_dirs,
                                           alpha=a.cond_alpha, gen=gen,
                                           n_fire=per, metric=a.ctx_metric)
    else:
        continuous_knn_transport_batch(z, cond, k=a.k, n_dirs=a.n_dirs,
                                       alpha=a.cond_alpha, gen=gen,
                                       n_fire=a.ctx_per_step, metric=a.ctx_metric)
    for _ in range(a.act_per_step):
        action_dist_knn_transport_step(z, cond, av, k=a.k, k_act=a.k_act,
                                       n_dirs=a.n_dirs, alpha=a.cond_alpha, gen=gen)
    for _gids in grp_ids:
        for _ in range(grp_per):
            group_rank_transport_step(z, _gids, n_dirs=a.n_dirs,
                                      alpha=a.cond_alpha, gen=gen,
                                      max_group=a.max_group,
                                      size_weighted=not a.grp_uniform)

    if step % a.eval_every == 0 or step == 1:
        floor = random_subset_w2(z, k=a.eval_k, n_eval=20, gen=gen)
        ctx = continuous_knn_w2(z, cond, k=a.eval_k, n_eval=20, gen=gen,
                                metric=a.ctx_metric)
        actw = action_dist_knn_w2(z, cond, av, k=a.eval_k, k_act=a.k_act,
                                  n_eval=20, gen=gen)
        G = gdefect(z)
        disp = displacement()
        curve["step"].append(step0 + step)
        curve["floor"].append(floor); curve["G"].append(G)
        curve["ctx_ratio"].append(ctx / max(floor, 1e-12))
        curve["act_ratio"].append(actw / max(floor, 1e-12))
        curve.setdefault("disp", []).append(disp)
        print(f"step {step0 + step:5d}  G={G:.5f}  floor={floor:.5f}  "
              f"I_ctx={ctx/max(floor,1e-12):.3f}  I_act={actw/max(floor,1e-12):.3f}"
              f"  disp={disp:.3f} ({100*disp/_zrad:.0f}% of ||z||)",
              flush=True)

    if a.save_every and step % a.save_every == 0 and step < a.steps:
        # --keep-checkpoints writes step-stamped files instead of overwriting one.
        # The number of transport steps is a real hyperparameter with a KNOWN
        # non-monotonic optimum -- CelebA's 4k assignment beat its 60k one by
        # 1.6-2.2x on generator fit because displacement costs z->image locality
        # -- and the transport objective saturates long before it can be read off.
        # Keeping the intermediates makes the step count selectable afterwards by
        # fresh-z MSE, which is the only thing that has ever selected here.
        # Overwriting one file forces the choice before the evidence exists.
        _p = Path(a.out)
        if a.keep_checkpoints:
            _p = _p.with_name(f"{_p.stem}_step{step0 + step}{_p.suffix}")
        _p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"z": z.cpu(), "h": h.cpu(), "cond": cond.cpu(), "action": act.cpu(),
                    "chunk": P.get("chunk"), "frame": P.get("frame"),
                    "episode": P.get("episode"), "cache": P.get("cache"),
                    "ae_checkpoint": P.get("checkpoint"),
                    "action_vec": av.cpu(), "mean": mean.cpu(), "W": W.cpu(),
                    "action_raw": P.get("action_raw"), "act_norm": P.get("act_norm"),
                    "act_names": P.get("act_names"), "act_dim": P.get("act_dim"),
                    "clicks_coverage": P.get("clicks_coverage"),
                    "W_inv": W_inv.cpu(), "curve": curve, "per_action": None,
                    "steps": step0 + step, "k": a.k, "k_act": a.k_act,
                    "ctx_per_step": a.ctx_per_step, "act_per_step": a.act_per_step,
                    "cond_alpha": a.cond_alpha, "ctx_metric": a.ctx_metric,
                    "context": P["context"],
                    "gamma": P["gamma"], "particles": a.particles,
                    "partial": True}, str(_p) + ".tmp")
        Path(str(_p) + ".tmp").replace(_p)   # atomic: never a truncated file
        gc.collect()
        print(f"  [checkpoint at step {step0 + step}]", flush=True)

pa = per_action_w2(z, act, gen=gen)
floor = random_subset_w2(z, k=a.eval_k, n_eval=40, gen=gen)
rows = sorted(((r, k, n) for k, (r, n, _, _) in pa.items()), reverse=True)
print(f"\nper-action W2 / SIZE-MATCHED floor  ({len(rows)} classes with >=256 members)")
print(f"  worst 6: " + "  ".join(f"a{k}:{r:.2f}(n={n})" for r, k, n in rows[:6]))
print(f"  best  6: " + "  ".join(f"a{k}:{r:.2f}(n={n})" for r, k, n in rows[-6:]))
rs = np.array([r for r, _, _ in rows]); ns = np.array([n for _, _, n in rows])
print(f"  mean {rs.mean():.3f}  median {np.median(rs):.3f}  max {rs.max():.3f}")
print(f"  corr(log n, ratio) {np.corrcoef(np.log(ns), rs)[0,1]:+.2f} "
      f"(near 0 = the floor match is working)")

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
# the assignment permutes nothing -- row i of z is still particle i -- so the
# cache coordinates carry through unchanged, and the generator can index frames
torch.save({"z": z.cpu(), "h": h.cpu(), "cond": cond.cpu(), "action": act.cpu(),
            "chunk": P.get("chunk"), "frame": P.get("frame"),
            "episode": P.get("episode"), "cache": P.get("cache"),
            "ae_checkpoint": P.get("checkpoint"),
            "action_vec": av.cpu(), "mean": mean.cpu(), "W": W.cpu(), "W_inv": W_inv.cpu(),
            "action_raw": P.get("action_raw"), "act_norm": P.get("act_norm"),
            "act_names": P.get("act_names"), "act_dim": P.get("act_dim"),
            "clicks_coverage": P.get("clicks_coverage"),
            "curve": curve, "per_action": pa,
            "steps": step0 + a.steps, "k": a.k, "k_act": a.k_act,
            "ctx_per_step": a.ctx_per_step, "act_per_step": a.act_per_step,
            "cond_alpha": a.cond_alpha, "grp_per_step": a.grp_per_step,
            "ctx_metric": a.ctx_metric, "grp_uniform": a.grp_uniform,
            "ctx_scales": scales, "act_groups": act_groups,
            "context": P["context"], "gamma": P["gamma"],
            "ctx_frames": a.ctx_frames,
            "particles": a.particles}, out)
print(f"\nsaved -> {out}")
