"""Context-split assignment with FREQUENCY-BALANCED action steps.

action_knn_transport_step picks its query particle uniformly at random, so each
action receives conditional transport in proportion to how often it occurs. On
this data that is badly skewed: Forward+Turn+Left variants hold 11-12% of
particles each while Attack holds 2.8%. The measured consequence is a per-action
independence ratio of 1.05 for Forward but 1.67 for Attack -- so a fresh z still
carries "attack-ness" and can inject a muzzle flash into a Forward rollout.

Here the action step samples the ACTION uniformly first, then a query within that
group, so every action gets 1/18 of the conditional budget regardless of size.
Attack goes from 2.8% to 5.6% of the action steps. Groups are precomputed rather
than re-derived per call, which also removes a full 630k scan from every step.
"""
import argparse, json, os
from pathlib import Path

import torch

from aag.gaussianize import (whiten, greedy_rank_transport_step,
                             continuous_knn_transport_step,
                             _rand_unit, _gaussian_quantiles)
from aag.diagnostics import (knn_preservation, assignment_diagnostics, action_knn_w2,
                             random_subset_w2, continuous_knn_w2, group_w2)


def action_step_balanced(z, cond, groups, *, k, n_dirs, alpha, gen):
    """One action-filtered k-NN transport step with the action sampled uniformly."""
    n_act = len(groups)
    a_id = int(torch.randint(n_act, (1,), device=z.device, generator=gen))
    same = groups[a_id]
    if same.numel() < 64:
        return 0.0
    qi = same[int(torch.randint(same.numel(), (1,), device=z.device, generator=gen))]
    dist = torch.cdist(cond[qi:qi + 1], cond[same]).squeeze(0)
    kk = min(k, same.numel())
    idx = same[torch.topk(dist, kk, largest=False).indices]
    zs = z[idx]
    dirs = _rand_unit(n_dirs, z.shape[1], z.device, z.dtype)
    s, _ = torch.sort(zs @ dirs.T, dim=0)
    q = _gaussian_quantiles(kk, z.device, z.dtype).unsqueeze(1)
    scores = ((s - q) ** 2).mean(0)
    best = int(torch.argmax(scores))
    av = dirs[best]
    proj = zs @ av
    order = torch.argsort(proj)
    target = torch.empty_like(proj)
    target[order] = _gaussian_quantiles(kk, z.device, z.dtype)
    z[idx] = zs + alpha * (target - proj).unsqueeze(1) * av.unsqueeze(0)
    return float(scores[best])


ap = argparse.ArgumentParser()
ap.add_argument("--particles", required=True)
ap.add_argument("--steps", type=int, default=16000)
ap.add_argument("--search-subset", type=int, default=2048)
ap.add_argument("--n-dirs", type=int, default=64)
ap.add_argument("--ctx-per-step", type=int, default=112)
ap.add_argument("--act-per-step", type=int, default=16)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--cond-alpha", type=float, default=0.25)
ap.add_argument("--k", type=int, default=4096)
ap.add_argument("--eval-k", type=int, default=4096)
ap.add_argument("--eval-every", type=int, default=800)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
P = torch.load(a.particles, map_location=dev, weights_only=False)
h, cond, act = P["h_target"].to(dev), P["h_context"].to(dev), P["action"].to(dev)
N, d = h.shape
n_act = int(act.max()) + 1
groups = [(act == i).nonzero(as_tuple=True)[0] for i in range(n_act)]
print(f"{N:,} particles, dim={d}, cond_dim={cond.shape[1]}, {n_act} actions", flush=True)
print("group sizes:", [int(g.numel()) for g in groups], flush=True)
print(f"{a.ctx_per_step} context-only + {a.act_per_step} BALANCED action steps per global step",
      flush=True)

z, mean, W, W_inv = whiten(h)
z0 = z.clone()
gen = torch.Generator(device=dev).manual_seed(a.seed)
curve = {"step": [], "ratio": [], "cond_w2": [], "floor_w2": [], "displacement": [],
         "proj_over_gauss": [], "knn": [], "frames_ratio": [], "action_ratio": []}

for s in range(1, a.steps + 1):
    greedy_rank_transport_step(z, search_subset=a.search_subset, n_dirs=a.n_dirs,
                               alpha=a.alpha, gen=gen)
    for _ in range(a.ctx_per_step):
        continuous_knn_transport_step(z, cond, k=a.k, n_dirs=a.n_dirs,
                                      alpha=a.cond_alpha, gen=gen, metric="cosine")
    for _ in range(a.act_per_step):
        action_step_balanced(z, cond, groups, k=a.k, n_dirs=a.n_dirs,
                             alpha=a.cond_alpha, gen=gen)
    if s % a.eval_every == 0 or s == a.steps:
        cw = action_knn_w2(z, cond, act, k=a.eval_k, gen=gen)
        fw = random_subset_w2(z, k=a.eval_k, gen=gen)
        fr = continuous_knn_w2(z, cond, k=a.eval_k, gen=gen)
        ar, _ = group_w2(z, act, gen=gen, max_group=a.eval_k)
        diag = assignment_diagnostics(z, d=d)
        disp = float((z - z0).norm(dim=1).mean())
        knn = knn_preservation(z0, z, k=10, gen=gen)
        curve["step"].append(s); curve["ratio"].append(cw / max(fw, 1e-12))
        curve["cond_w2"].append(cw); curve["floor_w2"].append(fw)
        curve["displacement"].append(disp)
        curve["proj_over_gauss"].append(diag["proj_over_gauss"])
        curve["knn"].append(knn)
        curve["frames_ratio"].append(fr / max(fw, 1e-12))
        curve["action_ratio"].append(ar / max(fw, 1e-12))
        print(f"  [{s}/{a.steps}] joint={cw/max(fw,1e-12):.2f} frames={fr/max(fw,1e-12):.2f} "
              f"action={ar/max(fw,1e-12):.2f} disp={disp:.3f} "
              f"G={diag['proj_over_gauss']:.2f} knn={knn:.4f}", flush=True)

diag = assignment_diagnostics(z, d=d)
print("assignment diagnostics:", diag, flush=True)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
torch.save({"z": z.cpu(), "h": h.cpu(), "cond": cond.cpu(), "action": act.cpu(),
            "chunk": P["chunk"], "frame": P["frame"], "episode": P["episode"],
            "mean": mean.cpu(), "W": W.cpu(), "W_inv": W_inv.cpu(),
            "diagnostics": diag, "curve": curve, "N": N, "alpha": a.alpha,
            "cond_alpha": a.cond_alpha, "steps": a.steps, "k": a.k, "eval_k": a.eval_k,
            "ctx_per_step": a.ctx_per_step, "act_per_step": a.act_per_step,
            "balanced_actions": True}, out / "assignment.pt")
(out / "curve.json").write_text(json.dumps(curve, indent=2))
print("saved:", out / "assignment.pt", flush=True)
print("BAL_ASSIGN_DONE", flush=True)
