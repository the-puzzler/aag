"""AAG3: learned worst-direction pursuit + AAG1 single-direction exact rank transport (user spec 2026-09-16).

Each step: optimise a unit vector a to MAXIMISE the 1-D Gaussian defect L(a) = mean_i ((Za)_(i) - q_i)^2 over the
sorted projections of ALL particles (gradient ascent, renormalised each step, warm-started from the previous worst
direction, plus random restarts), pick the candidate with the highest defect, and apply the exact AAG1 rank transport
along that single direction. The frozen held-out projection diagnostic R_G = G(Z)/G_floor(N, d) (aag2_defect /
aag2_floor on frozen directions) is evaluated every --eval-every steps: the assignment is saved at the first crossing
R_G <= 1 (…_floor.pt) and the run continues until the held-out defect plateaus -- operationally, when the running best
G has not improved by more than --min-rel-improve over the last --patience evaluations (…_plateau.pt). Checkpoint
format matches run_assignment_classes.py / run_assignment_aag2.py.
"""
import argparse, json, math, time
from pathlib import Path
import torch
from aag.gaussianize import whiten, refine_direction, rank_transport_along, aag2_defect, aag2_floor, _rand_unit, _gaussian_quantiles, _group_members

ap = argparse.ArgumentParser()
ap.add_argument("--particles", required=True)
ap.add_argument("--rotate", type=int, default=1, help="1 = PCA whitening (REQUIRED: with per-coordinate scaling the worst direction is just the top principal axis and single-direction transport raises the held-out defect); 0 = per-coordinate scaling")
ap.add_argument("--ascent-steps", type=int, default=20)
ap.add_argument("--ascent-lr", type=float, default=0.05)
ap.add_argument("--restarts", type=int, default=4, help="K fresh random initialisations per step (in addition to the warm start when --warm 1)")
ap.add_argument("--warm", type=int, default=1, help="1: also warm-start from the previous worst direction; 0: pure form -- fresh random init(s) only "
                                                     "(the exact update zeroes the defect along the chosen direction, so it is rarely tomorrow's worst)")
ap.add_argument("--alpha", type=float, default=1.0, help="transport fraction along the chosen direction (1 = exact)")
ap.add_argument("--eval-dirs", type=int, default=256)
ap.add_argument("--eval-every", type=int, default=10)
ap.add_argument("--floor-clouds", type=int, default=4)
ap.add_argument("--patience", type=int, default=100, help="evaluations without a meaningful improvement of the running-best G -> plateau")
ap.add_argument("--min-rel-improve", type=float, default=0.005)
ap.add_argument("--max-steps", type=int, default=200000)
ap.add_argument("--save-every", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--groups", default=None, help="class_groups.pt (ImageNet): enables class-respecting transport")
ap.add_argument("--levels", default="", help="comma list of hierarchy levels for group firings, e.g. joint,depth7,depth6,depth5,depth4,living,animal,dog")
ap.add_argument("--grp-per-step", type=int, default=8, help="group firings per global step, split over levels (each: learned worst direction WITHIN the group, then group rank transport)")
ap.add_argument("--cond-alpha", type=float, default=0.5)
ap.add_argument("--max-group", type=int, default=32768)
ap.add_argument("--cond-eval-classes", type=int, default=16, help="classes sampled for the conditional-vs-random-subset defect diagnostic at each eval")
ap.add_argument("--stop-after-floor", type=int, default=0, help=">0: stop this many steps after the floor crossing (skip the plateau search)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda"; torch.manual_seed(a.seed)
d0 = torch.load(a.particles, map_location="cpu", weights_only=False)
h = d0["h"].float().to(dev); labels = d0["label"]; N, D = h.shape
z, mean, W, W_inv = whiten(h, rotate=bool(a.rotate)); z = z.contiguous()
egen = torch.Generator(device=dev).manual_seed(1234 + a.seed)
dirs = _rand_unit(a.eval_dirs, D, dev, z.dtype)  # frozen held-out diagnostic directions (never used for transport)
floor, floor_sd = aag2_floor(N, D, dirs, n_clouds=a.floor_clouds, gen=egen, device=dev)
q = _gaussian_quantiles(N, dev, z.dtype)
labels_dev = torch.as_tensor(labels).to(dev)
levels = [x.strip() for x in a.levels.split(",") if x.strip()]; grp_ids = {}
if a.groups and levels:
    Gm = torch.load(a.groups, map_location="cpu", weights_only=False)
    for lv in levels:
        grp_ids[lv] = Gm["levels"][lv].to(dev)[labels_dev]
    grp_per = max(1, a.grp_per_step // len(levels))
    print(f"class-respecting transport: levels={levels}, {grp_per} firings/level/step, cond_alpha={a.cond_alpha}, max_group={a.max_group}", flush=True)
ggen = torch.Generator(device=dev).manual_seed(777 + a.seed)
def group_fire(gid):
    idx = _group_members(gid, a.max_group, ggen)
    if idx.numel() > a.max_group: idx = idx[torch.randperm(idx.numel(), device=dev, generator=ggen)[:a.max_group]]
    if idx.numel() < 64: return
    zs = z[idx]
    u = refine_direction(zs, _rand_unit(1, D, dev, z.dtype)[0], steps=a.ascent_steps, lr=a.ascent_lr)
    proj = zs @ u; order = torch.argsort(proj); target = torch.empty_like(proj); target[order] = _gaussian_quantiles(idx.numel(), dev, z.dtype)
    z[idx] += a.cond_alpha * (target - proj).unsqueeze(1) * u.unsqueeze(0)
def cond_ratio():
    """user diagnostic: mean class-conditional defect / mean random-subset defect of the same size (1.0 = z independent of class)"""
    if not grp_ids: return float("nan")
    cls = grp_ids.get("joint", labels_dev); rs = []
    for c in torch.randint(int(cls.max()) + 1, (a.cond_eval_classes,), device=dev, generator=egen).tolist():
        idx = (cls == c).nonzero(as_tuple=True)[0]
        if idx.numel() < 64: continue
        m = idx.numel(); dc = aag2_defect(z[idx], dirs); dr = aag2_defect(z[torch.randperm(N, device=dev, generator=egen)[:m]], dirs); rs.append(dc / max(dr, 1e-12))
    return float(sum(rs) / max(len(rs), 1))
def defect_along(u):
    s, _ = torch.sort(z @ u); return float(((s - q) ** 2).mean())
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
meta = {k: v for k, v in d0.items() if k not in ("h", "label")}
curve = {"step": [], "G": [], "ratio": [], "L_chosen": [], "L_random_mean": []}
def save(path, step, tag):
    torch.save({"z": z.cpu(), "h": d0["h"], "label": labels, "mean": mean.cpu(), "W": W.cpu(), "W_inv": W_inv.cpu(), "steps": step,
                "method": "aag3", "tag": tag, "rotate": a.rotate, "ascent_steps": a.ascent_steps, "restarts": a.restarts, "warm": a.warm,
                "floor": floor, "curve": curve, "levels": levels, "cond_alpha": a.cond_alpha, "grp_per_step": a.grp_per_step, "max_group": a.max_group, **meta}, path)
    json.dump(curve, open(str(out).replace(".pt", ".curve.json"), "w"))
G = aag2_defect(z, dirs); r = G / floor
print(f"{N:,} particles dim={D} rotate={a.rotate}  ascent {a.ascent_steps} steps x ({'warm + ' if a.warm else ''}{max(a.restarts,1)} fresh random init(s))  eval dirs={a.eval_dirs}  "
      f"floor G={floor:.6f} +/- {floor_sd:.6f}  start G={G:.6f} R_G={r:.3f}", flush=True)
prev = _rand_unit(1, D, dev, z.dtype)[0]; crossed = None; best = G; best_step = 0; since = 0; t0 = time.time()
for step in range(1, a.max_steps + 1):
    cands = [refine_direction(z, prev, steps=a.ascent_steps, lr=a.ascent_lr)] if a.warm else []
    rnd = _rand_unit(max(a.restarts, 1), D, dev, z.dtype)
    for k in range(max(a.restarts, 1)):
        cands.append(refine_direction(z, rnd[k], steps=a.ascent_steps, lr=a.ascent_lr))
    Ls = [defect_along(c) for c in cands]
    j = max(range(len(cands)), key=lambda i: Ls[i]); u = cands[j]; L = Ls[j]
    L_rand = float(((torch.sort(z @ rnd.T, dim=0)[0] - q.unsqueeze(1)) ** 2).mean())
    rank_transport_along(z, u, alpha=a.alpha); prev = u
    for lv in levels:
        for _ in range(grp_per): group_fire(grp_ids[lv])
    if step % a.eval_every == 0 or step == 1:
        G = aag2_defect(z, dirs); r = G / floor
        curve["step"].append(step); curve["G"].append(G); curve["ratio"].append(r); curve["L_chosen"].append(L); curve["L_random_mean"].append(L_rand)
        if crossed is None and r <= 1.0:
            crossed = step; save(str(out).replace(".pt", "_floor.pt"), step, "floor"); print(f"  floor crossed at step {step} (R_G={r:.3f}) -> saved _floor.pt", flush=True)
        if G < best * (1 - a.min_rel_improve): best, best_step, since = G, step, 0
        else: since += 1
        if step % (a.eval_every * 10) == 0:
            print(f"step {step:6d}  G={G:.6f}  R_G={r:.3f}  L(chosen)={L:.5f}  L(random)={L_rand:.5f}  best={best:.6f}@{best_step}  since={since}"
                  + (f"  cond/random={cond_ratio():.3f}" if grp_ids else "") + f"  [{(time.time()-t0)/step*1000:.0f} ms/step]", flush=True)
        if a.stop_after_floor > 0 and crossed is not None and step >= crossed + a.stop_after_floor:
            print(f"  stopping {a.stop_after_floor} steps after the floor crossing (step {step}, R_G={r:.3f})", flush=True); break
        if since >= a.patience and crossed is not None:
            save(str(out).replace(".pt", "_plateau.pt"), step, "plateau"); print(f"  plateau at step {step}: best G={best:.6f} (R_G={best/floor:.3f}) @ step {best_step}, no >{a.min_rel_improve*100:.1f}% improvement in {a.patience} evals -> saved _plateau.pt", flush=True)
            break
    if step % a.save_every == 0: save(str(out).replace(".pt", f"_step{step}.pt"), step, "periodic")
save(str(out), step, "final"); print(f"done: {step} steps, final G={G:.6f} R_G={r:.3f}, floor crossed at {crossed}", flush=True)
