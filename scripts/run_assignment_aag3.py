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
from aag.gaussianize import whiten, refine_direction, rank_transport_along, aag2_defect, aag2_floor, _rand_unit, _gaussian_quantiles

ap = argparse.ArgumentParser()
ap.add_argument("--particles", required=True)
ap.add_argument("--rotate", type=int, default=0, help="0 = per-coordinate scaling (as the AAG1 200k baseline); 1 = PCA whitening")
ap.add_argument("--ascent-steps", type=int, default=20)
ap.add_argument("--ascent-lr", type=float, default=0.05)
ap.add_argument("--restarts", type=int, default=4, help="random restarts per step, in addition to the warm start")
ap.add_argument("--alpha", type=float, default=1.0, help="transport fraction along the chosen direction (1 = exact)")
ap.add_argument("--eval-dirs", type=int, default=256)
ap.add_argument("--eval-every", type=int, default=10)
ap.add_argument("--floor-clouds", type=int, default=4)
ap.add_argument("--patience", type=int, default=100, help="evaluations without a meaningful improvement of the running-best G -> plateau")
ap.add_argument("--min-rel-improve", type=float, default=0.005)
ap.add_argument("--max-steps", type=int, default=200000)
ap.add_argument("--save-every", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
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
def defect_along(u):
    s, _ = torch.sort(z @ u); return float(((s - q) ** 2).mean())
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
meta = {k: v for k, v in d0.items() if k not in ("h", "label")}
curve = {"step": [], "G": [], "ratio": [], "L_chosen": [], "L_random_mean": []}
def save(path, step, tag):
    torch.save({"z": z.cpu(), "h": d0["h"], "label": labels, "mean": mean.cpu(), "W": W.cpu(), "W_inv": W_inv.cpu(), "steps": step,
                "method": "aag3", "tag": tag, "rotate": a.rotate, "ascent_steps": a.ascent_steps, "restarts": a.restarts,
                "floor": floor, "curve": curve, **meta}, path)
    json.dump(curve, open(str(out).replace(".pt", ".curve.json"), "w"))
G = aag2_defect(z, dirs); r = G / floor
print(f"{N:,} particles dim={D} rotate={a.rotate}  ascent {a.ascent_steps} steps x (warm + {a.restarts} restarts)  eval dirs={a.eval_dirs}  "
      f"floor G={floor:.6f} +/- {floor_sd:.6f}  start G={G:.6f} R_G={r:.3f}", flush=True)
prev = _rand_unit(1, D, dev, z.dtype)[0]; crossed = None; best = G; best_step = 0; since = 0; t0 = time.time()
for step in range(1, a.max_steps + 1):
    cands = [refine_direction(z, prev, steps=a.ascent_steps, lr=a.ascent_lr)]
    rnd = _rand_unit(a.restarts, D, dev, z.dtype)
    for k in range(a.restarts):
        cands.append(refine_direction(z, rnd[k], steps=a.ascent_steps, lr=a.ascent_lr))
    Ls = [defect_along(c) for c in cands]
    j = max(range(len(cands)), key=lambda i: Ls[i]); u = cands[j]; L = Ls[j]
    L_rand = float(((torch.sort(z @ rnd.T, dim=0)[0] - q.unsqueeze(1)) ** 2).mean())
    rank_transport_along(z, u, alpha=a.alpha); prev = u
    if step % a.eval_every == 0 or step == 1:
        G = aag2_defect(z, dirs); r = G / floor
        curve["step"].append(step); curve["G"].append(G); curve["ratio"].append(r); curve["L_chosen"].append(L); curve["L_random_mean"].append(L_rand)
        if crossed is None and r <= 1.0:
            crossed = step; save(str(out).replace(".pt", "_floor.pt"), step, "floor"); print(f"  floor crossed at step {step} (R_G={r:.3f}) -> saved _floor.pt", flush=True)
        if G < best * (1 - a.min_rel_improve): best, best_step, since = G, step, 0
        else: since += 1
        if step % (a.eval_every * 10) == 0:
            print(f"step {step:6d}  G={G:.6f}  R_G={r:.3f}  L(chosen)={L:.5f}  L(random)={L_rand:.5f}  best={best:.6f}@{best_step}  since={since}  [{(time.time()-t0)/step*1000:.0f} ms/step]", flush=True)
        if since >= a.patience and crossed is not None:
            save(str(out).replace(".pt", "_plateau.pt"), step, "plateau"); print(f"  plateau at step {step}: best G={best:.6f} (R_G={best/floor:.3f}) @ step {best_step}, no >{a.min_rel_improve*100:.1f}% improvement in {a.patience} evals -> saved _plateau.pt", flush=True)
            break
    if step % a.save_every == 0: save(str(out).replace(".pt", f"_step{step}.pt"), step, "periodic")
save(str(out), step, "final"); print(f"done: {step} steps, final G={G:.6f} R_G={r:.3f}, floor crossed at {crossed}", flush=True)
