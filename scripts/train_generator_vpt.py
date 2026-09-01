#!/usr/bin/env python
"""Action-conditioned frame generator for the Minecraft (VPT) world model.

    z ~ N(0,I) (dim) + h_{t-24..t-1} (24*dim) + one-hot action (81)  ->  frame_t pixels

Direct-to-pixel with MSE+LPIPS, matching train_generator_doom.py and every other
generator in this project. z is the frozen assignment's Gaussian coordinate for
that particle, so at inference a fresh z ~ N(0,I) pairs with any context -- valid
only to the extent the conditional assignment drove the independence ratios
toward 1.0.

Differs from the Doom version in three places, all forced by the VPT cache:
  * frames come from open_segments (102 shards), not a single segments.npy
  * 81 actions, not 18
  * targets stay on the CPU in pinned memory and move per batch. At 512k
    particles a uint8 target tensor is 6.3 GB, and the context tensor is already
    12.6 GB on the GPU -- holding both there is what would break first.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from aag.ae import AutoEncoder, ResidualDecoder as ConvDecoder
from aag.generator import TransformerGenerator
from aag.datasets import open_segments
from aag.discriminator import (NLayerDiscriminator, adaptive_weight, paired_batch,
                               paired_d_loss, paired_g_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--cache", default="/opt/dlami/nvme/vpt_full")
    ap.add_argument("--n-actions", type=int, default=81)
    ap.add_argument("--act-vec", action="store_true",
                    help="No-op, kept so old command lines still run. The "
                         "continuous action vector is now the ONLY action "
                         "conditioning and is always used.")
    ap.add_argument("--action-onehot", action="store_true",
                    help="Legacy: also feed the 81-way one-hot alongside the "
                         "12-d vector, reproducing the pre-12d condition. Off "
                         "by default. The index is a deterministic function of "
                         "a subset of the vector, so it adds no information; it "
                         "cannot express attack/use/E; and it quantises every "
                         "mouse magnitude past a 5 px deadzone into 3 classes, "
                         "which is what made a turn command read as a tilt. "
                         "Only for reproducing pre-12d runs.")
    ap.add_argument("--z-dims", type=int, default=0,
                    help="Use only the first N dims of the assigned z (0 = all). "
                         "Tests 'z explains too much, so context becomes "
                         "optional' without the confound of retraining a "
                         "lower-dim AE, which would also raise the "
                         "reconstruction floor -- measured at 0.00764 MSE, "
                         "already 60%% of the assigned-z error. A marginal of a "
                         "Gaussianised z is still Gaussian, so the assignment "
                         "stays valid on the kept dims.")
    ap.add_argument("--ctx-frames", type=int, default=0,
                    help="Keep only the newest N context blocks (0 = all of "
                         "whatever the assignment holds; its own --ctx-frames "
                         "may already have shortened it, and the saved cond is "
                         "the sliced one). The "
                         "stored cond is recency-scaled by sqrt(gamma^i) with "
                         "the newest block at weight 1, so a suffix slice is "
                         "exactly what a shorter-context particle build would "
                         "produce -- no rebuild needed.")
    ap.add_argument("--arch", choices=["conv", "transformer"], default="conv",
                    help="conv: one Linear absorbs the whole flat context, which "
                         "is 74%% of its 107.5M params and gives z no way to "
                         "query the history. transformer: the 24 context frames "
                         "become 24 tokens, z and the action one each, "
                         "self-attention lets z interrogate the context, and only "
                         "the z token is decoded.")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ch", type=int, default=192)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--lpips-weight", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--start-epoch", type=int, default=0)
    ap.add_argument("--gan-weight", type=float, default=0.0,
                    help="If >0, add a PAIRED adversarial term. Every generated "
                         "frame has an exact corresponding real one (same z, "
                         "context and action), so the critic is handed the pair "
                         "in randomised order and asked which is the true "
                         "continuation -- better posed than judging images "
                         "independently, and it needs no conditioning input "
                         "because the pairing carries it. Targets the measured "
                         "failure: off its training pairs the generator reverts "
                         "to a conditional mean, losing a quarter of its "
                         "high-frequency energy, and a rollout collapses to a "
                         "flat field within ~1.5s.")
    ap.add_argument("--gan-start-epoch", type=int, default=3,
                    help="reconstruction-only until here, so the critic is not "
                         "asked to judge noise")
    ap.add_argument("--gan-layers", type=int, default=2)
    ap.add_argument("--gan-ndf", type=int, default=64)
    ap.add_argument("--gan-lr", type=float, default=4.5e-5)
    ap.add_argument("--rollout-k", type=int, default=0,
                    help="If >0, train on the model's OWN output as context for "
                         "up to this many steps before scoring. The generator "
                         "otherwise only ever sees REAL context, so it has no "
                         "incentive to be stable under its own errors -- and a "
                         "rollout collapses to a flat field within ~1.5s, right "
                         "as the last real frame leaves the 1.2s window. Classic "
                         "exposure bias. The rollout steps run under no_grad and "
                         "only the final step is supervised, which gives the "
                         "model experience of its own drift without paying to "
                         "backprop through k generations.")
    ap.add_argument("--rollout-prob", type=float, default=0.5,
                    help="fraction of batches that roll out; the rest stay on "
                         "real context so reconstruction is not forgotten")
    ap.add_argument("--ae", default="/data/aag_results/results_vpt/"
                                    "ae_dcae_ch192_dim256_cont/checkpoints/"
                                    "ae_doom_frames_dcae_lpips_ch192_dim256_ep4.pt",
                    help="encoder used to turn generated frames back into context")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    DIM = 256
    A = torch.load(args.assignment, map_location="cpu", weights_only=False)
    # The assignment saves the cond it actually transported against, which its
    # own --ctx-frames may have shortened, so 24 is not safe to assume.
    CTX = A["cond"].shape[1] // DIM
    for k in ("chunk", "frame"):
        if A.get(k) is None:
            raise SystemExit(
                f"assignment has no '{k}' -- rebuild particles with a "
                f"prepare_vpt_particles.py that saves cache coordinates, then "
                f"re-run the assignment. Without them the pixel targets for each "
                f"particle cannot be located.")
    z, cond, act = A["z"], A["cond"], A["action"]
    if args.z_dims:
        if not 0 < args.z_dims <= z.shape[1]:
            raise SystemExit(f"--z-dims must be in 1..{z.shape[1]}")
        z = z[:, :args.z_dims].contiguous()
    if args.ctx_frames and args.ctx_frames != CTX:
        if not 0 < args.ctx_frames <= CTX:
            raise SystemExit(f"--ctx-frames must be in 1..{CTX}")
        cond = cond[:, (CTX - args.ctx_frames) * DIM:].contiguous()
        CTX = args.ctx_frames
    N, dim_z = z.shape
    # Action conditioning is the 12-d vector -- 10 binary controls + dx/dy --
    # and by default NOTHING else. The 81-way one-hot it replaces was a
    # deterministic function of a SUBSET of this vector, so it added no
    # information, while being unable to express attack/use/E at all and
    # collapsing every mouse magnitude past a 5 px deadzone into one class.
    # Feeding both also invites the network to key on the coarse discrete signal
    # and underuse magnitude, which is the authority split we already paid for.
    # 90 dims (81 + 9) -> 12.
    parts = [z, cond]
    a_desc = ""
    if args.action_onehot:
        onehot = torch.zeros(N, args.n_actions)
        onehot[torch.arange(N), act] = 1.0
        parts.append(onehot)
        a_desc = f"onehot={args.n_actions}"
    av = A.get("action_vec")
    if av is None:
        raise SystemExit("assignment has no 'action_vec' -- patch the particles "
                         "with scripts/patch_vpt_particle_actions.py, then re-run "
                         "the assignment")
    if not args.action_onehot and av.shape[1] < 12:
        raise SystemExit(
            f"action_vec is {av.shape[1]}-d, expected 12 (W A S D space shift "
            f"ctrl E attack use dx dy). This is a pre-clicks particle set: run "
            f"scripts/patch_vpt_clicks.py then "
            f"scripts/patch_vpt_particle_actions.py, or pass --action-onehot to "
            f"reproduce the old condition.")
    parts.append(av.float())
    a_desc += (" + " if a_desc else "") + f"act_vec={av.shape[1]}"
    act_norm = A.get("act_norm")
    if act_norm is None and not args.action_onehot:
        print("WARNING assignment carries no act_norm -- live inference cannot "
              "encode controller input with the same constants", flush=True)
    inp = torch.cat(parts, 1).to(dev)
    n_in = inp.shape[1]
    # Free every CPU copy now the input lives on the GPU. At 1.66M particles
    # A["cond"] is 40.9 GB and the cat another 43.2 GB; holding both alongside
    # the 20.4 GB target buffer peaks at 124.9 GB against 124 GB of RAM.
    del parts, cond
    for _k in ("cond", "h", "z", "action_vec", "mean", "W", "W_inv"):
        A.pop(_k, None)
    gc.collect()
    print(f"{N:,} particles | generator input dim {n_in} "
          f"(z={dim_z} + cond={CTX * DIM} + {a_desc})", flush=True)
    r = A.get("curve", {}).get("ctx_ratio")
    if r:
        print(f"assignment independence: ctx {r[-1]:.3f}  "
              f"act {A['curve']['act_ratio'][-1]:.3f}  (1.0 = independent)", flush=True)

    # pixel targets: the actual frame each particle points at
    segs = open_segments(args.cache)
    ci, fi = A["chunk"].numpy(), A["frame"].numpy()
    # allocate pinned up front: .pin_memory() afterwards would duplicate 20.4 GB
    tgt = torch.empty((N, 3, args.image_size, args.image_size), dtype=torch.uint8,
                      pin_memory=True)
    for s in range(0, N, 4096):
        e = min(s + 4096, N)
        blk = np.stack([np.asarray(segs[int(c)])[int(f)]
                        for c, f in zip(ci[s:e], fi[s:e])])
        tgt[s:e] = torch.from_numpy(blk).permute(0, 3, 1, 2)
        if s % 131072 == 0:
            print(f"  targets {s:,}/{N:,}", flush=True)
    print(f"targets {tuple(tgt.shape)} uint8 ({tgt.nbytes/1e9:.2f} GB, pinned on CPU)",
          flush=True)

    # Frames the rollout lands on. Rolling j steps from frame f must be scored
    # against the REAL frame f+j, so those have to be resident too. Particles
    # whose segment ends before f+k are excluded rather than clamped -- scoring
    # against a repeated last frame would teach the model that motion stops.
    tgt_fwd, roll_ok = [], torch.zeros(N, dtype=torch.bool)
    if args.rollout_k > 0:
        fmax = segs.shape[1]
        roll_ok = torch.from_numpy((fi + args.rollout_k) < fmax)
        print(f"rollout targets: {int(roll_ok.sum()):,}/{N:,} particles have "
              f"{args.rollout_k} real frames ahead", flush=True)
        for j in range(1, args.rollout_k + 1):
            tj = torch.empty((N, 3, args.image_size, args.image_size),
                             dtype=torch.uint8, pin_memory=True)
            for st in range(0, N, 4096):
                en = min(st + 4096, N)
                blk = np.stack([np.asarray(segs[int(c)])[min(int(f) + j, fmax - 1)]
                                for c, f in zip(ci[st:en], fi[st:en])])
                tj[st:en] = torch.from_numpy(blk).permute(0, 3, 1, 2)
            tgt_fwd.append(tj)
            print(f"  +{j} frame targets loaded ({tj.nbytes/1e9:.2f} GB)", flush=True)

    ae = None
    if args.rollout_k > 0:
        ac = torch.load(args.ae, map_location=dev, weights_only=False)
        ae = AutoEncoder(ac["latent_dim"], ch=ac["channels"],
                         architecture=ac["architecture"],
                         image_size=ac["image_size"],
                         grid=ac.get("grid", 4)).to(dev).eval()
        _sd = ac["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in _sd):
            _sd = {k.replace("_orig_mod.", "", 1): v for k, v in _sd.items()}
        ae.load_state_dict(_sd)
        for _p in ae.parameters():
            _p.requires_grad_(False)
        print(f"rollout training: k<={args.rollout_k}, prob={args.rollout_prob}, "
              f"encoder from AE epoch {ac['epochs']}", flush=True)

    if args.arch == "transformer":
        model = TransformerGenerator(
            dim_z=dim_z, ctx_frames=CTX, ctx_dim=DIM,
            act_dim=n_in - dim_z - CTX * DIM,
            d_model=args.d_model, depth=args.depth, heads=args.heads,
            ch=args.ch, image_size=args.image_size).to(dev)
    else:
        model = ConvDecoder(n_in, ch=args.ch, image_size=args.image_size).to(dev)
    resume_ck = None
    if args.resume is not None:
        resume_ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(resume_ck["model_state_dict"])
        print(f"resumed from {args.resume} (epoch {resume_ck.get('epoch')})",
              flush=True)
    print(f"generator params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    perceptual = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in perceptual.parameters():
        p.requires_grad_(False)
    disc = opt_d = None
    if args.gan_weight > 0:
        disc = NLayerDiscriminator(6, args.gan_ndf, args.gan_layers).to(dev)
        opt_d = torch.optim.Adam(disc.parameters(), lr=args.gan_lr, betas=(0.5, 0.9))
        print(f"paired adversary on: weight={args.gan_weight}, "
              f"start_epoch={args.gan_start_epoch}, "
              f"disc params {sum(p.numel() for p in disc.parameters()):,}", flush=True)
    dgen = torch.Generator(device=dev).manual_seed(args.seed)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, N // args.batch))
    if resume_ck is not None:
        # Model weights alone are not a resume. Without the Adam moments the
        # first steps after a restart are effectively un-preconditioned, and
        # without the critic the generator is briefly scored by a random one.
        if resume_ck.get("opt_state_dict"):
            opt.load_state_dict(resume_ck["opt_state_dict"])
        if disc is not None and resume_ck.get("disc_state_dict") is not None:
            disc.load_state_dict(resume_ck["disc_state_dict"])
            if opt_d is not None and resume_ck.get("opt_d_state_dict"):
                opt_d.load_state_dict(resume_ck["opt_d_state_dict"])
            print("  critic and its optimiser restored", flush=True)
        # One continuous cosine over the NEW horizon rather than a warm restart
        # at peak LR: fast-forward the schedule to where this resume begins.
        for _ in range(args.start_epoch * max(1, N // args.batch)):
            sched.step()
        print(f"  resumed at epoch {args.start_epoch}, lr now "
              f"{opt.param_groups[0]['lr']:.2e} on a {args.epochs}-epoch cosine",
              flush=True)
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.amp else \
          (lambda: torch.enable_grad())

    CTXDIM = CTX * DIM
    recw = torch.tensor(np.sqrt(0.95 ** np.arange(CTX - 1, -1, -1)),
                        dtype=torch.float32, device=dev).view(1, CTX, 1)
    curve = {"epoch": [], "mse": [], "lpips": []}
    for ep in range(args.start_epoch, args.epochs):
        model.train()
        perm = torch.randperm(N)
        tot_m = tot_l = n = 0
        tot_g = tot_d = tot_w = 0.0
        gan_on = disc is not None and (ep + 1) >= args.gan_start_epoch
        for i in range(0, N - args.batch + 1, args.batch):
            b = perm[i:i + args.batch]
            x = tgt[b].to(dev, non_blocking=True).float().div_(127.5).sub_(1.0)
            bd = b.to(dev)
            row = inp[bd]
            n_roll = 0
            if ae is not None and torch.rand(1, generator=dgen, device=dev).item() < args.rollout_prob:
                # walk the model forward on its OWN output, then supervise the
                # step that lands back on a frame we have. offsets beyond the
                # segment are skipped by construction (roll_ok below).
                ok = roll_ok[b].to(dev)   # roll_ok lives on CPU
                if ok.any():
                    n_roll = int(torch.randint(1, args.rollout_k + 1, (1,),
                                               generator=dgen, device=dev).item())
                    h = row[:, dim_z:dim_z + CTXDIM].view(-1, CTX, DIM) / recw
                    with torch.no_grad():
                        for _s in range(n_roll):
                            zz = torch.randn(len(b), dim_z, device=dev)
                            cc = torch.cat([zz, (h * recw).reshape(len(b), -1),
                                            row[:, dim_z + CTXDIM:]], 1)
                            with amp():
                                yy = model(cc).clamp(-1, 1)
                                hn = ae.enc(yy).view(-1, 1, DIM)
                            # back to fp32: h is fp32 and mixing dtypes in the
                            # context would silently downcast the whole window
                            h = torch.cat([h[:, 1:], hn.float()], 1)
                    row = torch.cat([row[:, :dim_z], (h * recw).reshape(len(b), -1),
                                     row[:, dim_z + CTXDIM:]], 1)
                    x = torch.where(ok.view(-1, 1, 1, 1),
                                    tgt_fwd[n_roll - 1][b].to(dev, non_blocking=True)
                                    .float().div_(127.5).sub_(1.0), x)
            opt.zero_grad(set_to_none=True)
            with amp():
                pred = model(row)
                mse = F.mse_loss(pred, x)
                perc = perceptual(pred.clamp(-1, 1), x).mean()
                loss = mse + args.lpips_weight * perc

            if gan_on:
                with amp():
                    pair, lab = paired_batch(x, pred, dgen)
                    g = paired_g_loss(disc(pair), lab)
                # grad-norm probe in fp32, against the output conv
                _last = (model.dec.net[-1] if args.arch == "transformer"
                         else model.net[-1]).weight
                w = adaptive_weight(loss, g, _last) * args.gan_weight
                loss = loss + w * g
                tot_g += g.item() * len(b); tot_w += w.item() * len(b)

            loss.backward()
            opt.step(); sched.step()

            if gan_on:
                opt_d.zero_grad(set_to_none=True)
                with amp():
                    pair, lab = paired_batch(x, pred.detach(), dgen)
                    d = paired_d_loss(disc(pair), lab)
                d.backward()
                opt_d.step()
                tot_d += d.item() * len(b)

            tot_m += mse.item() * len(b); tot_l += perc.item() * len(b); n += len(b)
        curve["epoch"].append(ep + 1)
        curve["mse"].append(tot_m / n); curve["lpips"].append(tot_l / n)
        gstr = (f"  g={tot_g/n:.4f} d={tot_d/n:.4f} w={tot_w/n:.4f}"
                if tot_d else "")
        print(f"epoch {ep+1}/{args.epochs}  mse={tot_m/n:.5f}  lpips={tot_l/n:.5f}{gstr}",
              flush=True)

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                # fixed context, fresh z: shows what the noise actually controls
                ctx = inp[:8, dim_z:].repeat_interleave(8, 0)
                fresh = torch.randn(64, dim_z, device=dev)
                grid = model(torch.cat([fresh, ctx], 1)).clamp(-1, 1)
            save_image(grid * 0.5 + 0.5, args.out / f"samples_ep{ep+1}.png", nrow=8)
            ck = args.out / "checkpoints"; ck.mkdir(exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep + 1,
                        "input_dim": n_in, "dim_z": dim_z, "ch": args.ch,
                        "ctx_frames": CTX, "ctx_dim": DIM,
                        "opt_state_dict": opt.state_dict(),
                        "opt_d_state_dict": (opt_d.state_dict()
                                             if opt_d is not None else None),
                        "image_size": args.image_size, "n_actions": args.n_actions,
                        # act_norm and act_names are the inference contract: a
                        # live controller must be encoded with the SAME
                        # constants the generator trained on, so they ship
                        # inside the checkpoint rather than beside it.
                        "act_norm": act_norm, "act_names": A.get("act_names"),
                        "act_dim": int(av.shape[1]),
                        "action_onehot": args.action_onehot,
                        "act_vec": args.act_vec, "gan_weight": args.gan_weight,
                        "arch": args.arch, "d_model": args.d_model,
                        "depth": args.depth, "heads": args.heads,
                        "disc_state_dict": disc.state_dict() if disc is not None else None,
                        "mse": tot_m / n, "lpips": tot_l / n,
                        "assignment": str(args.assignment)},
                       ck / f"gen_vpt_ep{ep+1}.pt")
        json.dump(curve, open(args.out / "gen_curve.json", "w"), indent=1)


if __name__ == "__main__":
    main()
