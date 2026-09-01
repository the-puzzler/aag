#!/usr/bin/env python
"""Does z add action information BEYOND the context?  The conditional question.

Earlier probes asked "can you read the action out of z".  That is the wrong
question for turn, and the h control proves it: an MLP extracts +16.1 points of
"moving vs still" from h, the AE latent of the true target frame, but +0.1 of
"turning vs not" from the same h.  A turn is only visible as a DIFFERENCE
between consecutive frames -- no single image carries a mark saying a turn just
happened -- so no probe on one latent can see it, in z or anywhere else.  The
same is expected to hold for a click: attack shows up as a crack overlay and a
hand swing, which is a difference, not a state.  This is why the W2-per-marginal
instrument must not be used to clear the action side: it nearly hid the turn fix.

The quantity that actually matters is conditional.  The generator is given the
context and the action; the failure mode is z arriving with its own opinion that
overrides the action.  So:

    A = accuracy of  cond          -> action marginal
    B = accuracy of  cond + z      -> action marginal
    B - A = action information z carries that the context did not already have

B - A near zero is what conditional independence should deliver.  Anything
positive is exactly the leak that lets a fresh z contradict the action input.

Reported for EVERY marginal of the 12-d representation -- each of the ten binary
controls, mouse direction and mouse magnitude per axis -- because the standing
lesson of this project is that independence from the condition as a whole says
nothing about independence from any low-dimensional function of it.  The joint
81-way class read 0.593 while "turning vs not" read 1.272.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from aag.vpt_actions import action_marginals


def probe(xtr, xte, ytr, yte, k, dev, epochs, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], 512), torch.nn.GELU(),
        torch.nn.Linear(512, 512), torch.nn.GELU(),
        torch.nn.Linear(512, k)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    B = 16384
    for _ in range(epochs):
        idx = torch.randperm(len(xtr), device=dev)
        for s0 in range(0, len(xtr), B):
            j = idx[s0:s0 + B]
            loss = torch.nn.functional.cross_entropy(net(xtr[j]), ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = torch.cat([(net(xte[i:i + B]).argmax(1) == yte[i:i + B])
                         for i in range(0, len(xte), B)]).float().mean().item()
    del net, opt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", type=Path, required=True)
    ap.add_argument("--particles", type=Path, default=None,
                    help="Take action_raw from this particle file instead of "
                         "from the assignment. Lets a PRE-12d assignment be "
                         "probed against the new marginals, which is the only "
                         "way to get a before/after rather than a lone number. "
                         "Particle order is identity, so this is only valid for "
                         "the particle file the assignment was actually built "
                         "from -- the N check below is the guard.")
    ap.add_argument("--control", action="store_true",
                    help="Replace z with fresh N(0,I) noise of the same shape "
                         "and report the same table. This is the NULL, and it is "
                         "not optional when reading small numbers: appending 256 "
                         "uninformative dims to a 768-dim probe input costs the "
                         "probe some accuracy, so 'z adds' has a negative bias "
                         "whose size is unknown until measured. A true "
                         "assignment's z is marginally N(0,I), so noise of the "
                         "same shape is exactly the right null -- any row where "
                         "z scores no higher than the control carries no "
                         "detectable action information, and the control's own "
                         "spread IS the detection threshold.")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--subset", type=int, default=0,
                    help="probe on N particles instead of all (0 = all)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.load(args.assignment, map_location="cpu", weights_only=False)
    z, cond = A["z"].float(), A["cond"].float()
    N = z.shape[0]
    print(f"{N:,} particles from {args.assignment.parent.name}  "
          f"z {z.shape[1]}  cond {cond.shape[1]}", flush=True)

    raw = None
    if A.get("action_raw") is not None:
        raw = A["action_raw"].numpy()
        cov = A.get("clicks_coverage")
    elif args.particles is not None:
        # a pre-12d assignment: take the marginals from the particle file it was
        # built from. Particle order IS identity, so this is only valid for that
        # file -- N is the guard.
        Pp = torch.load(args.particles, map_location="cpu", weights_only=False)
        if Pp.get("action_raw") is None:
            raise SystemExit(f"{args.particles} has no action_raw -- patch it "
                             f"with scripts/patch_vpt_particle_actions.py first")
        if Pp["action_raw"].shape[0] != N:
            raise SystemExit(
                f"{args.particles} has {Pp['action_raw'].shape[0]:,} particles "
                f"but the assignment has {N:,} -- different particle set, so the "
                f"row-for-row join would be meaningless")
        raw = Pp["action_raw"].numpy()
        cov = Pp.get("clicks_coverage")
        print(f"action_raw taken from {args.particles.name}", flush=True)
        del Pp

    targets = []
    if raw is not None:
        for nm, g in action_marginals(raw).items():
            targets.append((nm, torch.from_numpy(g).long(), int(g.max()) + 1))
        if cov is not None:
            print(f"clicks coverage {100*cov:.2f}% of particles", flush=True)
    else:
        print("NOTE no action_raw -- falling back to the legacy 81-way marginals. "
              "Patch the particles with scripts/patch_vpt_particle_actions.py to "
              "probe what the 12-d generator actually sees.", flush=True)
        act = A["action"].long()
        move, turn, tilt = act // 9, (act // 3) % 3, act % 3
        targets = [("joint 81-way", act, 81), ("move only", move, 9),
                   ("turn only", turn, 3), ("tilt only", tilt, 3),
                   ("moving vs still", (move > 0).long(), 2),
                   ("turning vs not", (turn > 0).long(), 2)]

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    if args.subset:
        perm = perm[: args.subset]
    n = len(perm)
    ntr = int(n * 0.9)
    tr, te = perm[:ntr], perm[ntr:]

    ctr, cte = cond[tr].to(dev), cond[te].to(dev)
    ztr, zte = z[tr].to(dev), z[te].to(dev)
    czt, czv = torch.cat([ctr, ztr], 1), torch.cat([cte, zte], 1)
    if args.control:
        g2 = torch.Generator(device=dev).manual_seed(1234)
        ntr_ = torch.randn(ztr.shape, device=dev, generator=g2)
        nte_ = torch.randn(zte.shape, device=dev, generator=g2)
        cnt, cnv = torch.cat([ctr, ntr_], 1), torch.cat([cte, nte_], 1)

    hdr = (f"\n{'marginal':14s} {'groups':>6s} {'chance':>8s} {'cond':>8s} "
           f"{'cond+z':>8s} {'z adds':>8s}")
    if args.control:
        hdr += f" {'noise':>8s} {'z-noise':>8s}"
    print(hdr, flush=True)
    rows = []
    for name, y, k in targets:
        ytr, yte = y[tr].to(dev), y[te].to(dev)
        base = torch.bincount(yte, minlength=k).max().item() / len(yte)
        a = probe(ctr, cte, ytr, yte, k, dev, args.epochs)
        b = probe(czt, czv, ytr, yte, k, dev, args.epochs)
        line = (f"{name:14s} {k:6d} {100*base:7.2f}% {100*a:7.2f}% "
                f"{100*b:7.2f}% {100*(b-a):+7.2f}")
        excess = b - a
        if args.control:
            c = probe(cnt, cnv, ytr, yte, k, dev, args.epochs)
            excess = b - c          # against the null, not against cond alone
            line += f" {100*(c-a):+7.2f} {100*(b-c):+7.2f}"
        rows.append((name, excess))
        print(line, flush=True)

    rows.sort(key=lambda r: -r[1])
    key = "z above the N(0,I) null" if args.control else "z above cond"
    print(f"\nworst leaks ({key}): "
          + ", ".join(f"{n} {100*v:+.2f}" for n, v in rows[:5]), flush=True)
    if args.control:
        vals = np.array([v for _, v in rows])
        print(f"null-referenced excess: mean {100*vals.mean():+.2f} "
              f"sd {100*vals.std():.2f} -> anything under ~{100*2*vals.std():.2f} "
              f"points is inside this instrument's band", flush=True)
    print("\n'z adds' is action information present in z that the context did not\n"
          "already carry. Near zero is what conditional independence should give;\n"
          "positive is what lets a fresh z override the action it was handed.",
          flush=True)


if __name__ == "__main__":
    main()
