#!/usr/bin/env python
"""Generator training with TRUE-SEQUENCE rollout supervision.

The difference from the rollout branch in train_generator_vpt.py, and the reason
this exists as its own trainer:

  * that branch walked n_roll steps while REUSING the action at t for every
    step, then supervised against the real frame at t+n_roll -- which in the data
    was produced by a_t, a_{t+1}, ... Measured on this cache, the full 12-d
    action is identical across a 3-step window only 28.2% of the time (the mouse
    moves, median 4 px), so 71.8% of rollout batches were trained toward a
    target the conditioning could not reach. Being pushed at unreachable targets
    is a hedging pressure, and hedging over possible futures looks exactly like
    the blur-and-drift the rollouts showed.

  * here, step s is conditioned on the ACTUAL action for that step and
    supervised against the ACTUAL frame for that step. Every generated frame in
    the sequence gets a loss, which is the user's specification: roll out over a
    real action sequence and require the generated frames to match the true ones.

Indexing, which is the whole game (C = context frames, lag = ACTION_LAG = 1):

    particle (c, t):  context frames t-C .. t-1      target frame t
    step s:           context ends at t-1+s          target frame t+s
                      action  a_(t-1+s)              <-- lag applied, see below
    so actions come from frames t-1 .. t-2+L  and targets from t .. t+L-1
    requires t-1 >= 0 and t+L-1 <= F-1

The action lag is not cosmetic. VPT stores (observation, action) per tick, so the
action at tick t is chosen after seeing frame t and produces frame t+1; the
action that produces frame t is a_(t-1). Measured: the signed horizontal image
shift into frame t correlates -0.588 with dx_(t-1) against -0.477 with dx_t, and
agrees in sign 69.4% vs 61.7%. Step s=0 of a sequence is therefore exactly the
ordinary single-step problem, and that identity is asserted at startup.

Gradient: the context is DETACHED between steps, so the L per-step graphs are
independent and their summed loss needs one backward. No backprop through the AE
encoder -- that would cost L times the activations for credit assignment not yet
shown to be needed. --bptt keeps the graph if you want to try it.

Data: actions and targets are gathered per batch from the cache mmaps rather
than preloaded. A particle's targets are one contiguous slice segs[c][t:t+L], the
cache sits in page cache, and preloading L target buffers would cap L by RAM at
about 8. Gathering lets L be chosen on merit.
"""
from __future__ import annotations

import argparse, gc, json, time
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F

from aag.ae import AutoEncoder
from aag.datasets import open_segments
from aag.generator import PixelContextEncoder, TransformerGenerator
from aag.vpt_actions import (ACTION_LAG, apply_action_norm, build_action_raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", type=Path, required=True)
    ap.add_argument("--cache", default="/opt/dlami/nvme/vpt_full",
                    help="segments live here (pixel targets)")
    ap.add_argument("--side-cache", default="/data/vpt/cache_train",
                    help="keys/mouse/clicks live here; must be the same segment "
                         "ordering as --cache, which is asserted at startup")
    ap.add_argument("--ae", default="/data/aag_results/results_vpt/"
                                    "ae_dcae_ch192_dim256_cont/checkpoints/"
                                    "ae_doom_frames_dcae_lpips_ch192_dim256_ep4.pt")
    ap.add_argument("--seq-len", type=int, default=8,
                    help="L: rollout steps per sequence batch. Step 0 is the "
                         "ordinary single-step problem, so this is a superset of "
                         "single-step training.")
    ap.add_argument("--seq-prob", type=float, default=0.5,
                    help="fraction of batches that are sequence batches")
    ap.add_argument("--seq-warmup", type=int, default=3,
                    help="epochs of pure single-step before sequence batches "
                         "start. Rolling a randomly-initialised model on its own "
                         "output supervises against noise.")
    ap.add_argument("--pixel-context", action="store_true",
                    help="Condition on the context FRAMES through a learned "
                         "encoder instead of the frozen AE latents. The "
                         "assignment is untouched -- it still supplies z, which "
                         "is all it is needed for; conditioning at generator "
                         "train time is free. The encoder emits ctx_dim per "
                         "frame so it drops into the existing emb_ctx with no "
                         "architectural change. Motivation: one AE encode-decode "
                         "destroys 6.55 mean |pixel| against a real frame step of "
                         "7.16, and velocity lives in the DIFFERENCE between "
                         "consecutive context vectors. It also removes the AE "
                         "from the rollout loop entirely -- generated pixels go "
                         "straight back in, through an encoder trained on that "
                         "exact path.")
    ap.add_argument("--pix-ch", type=int, default=64,
                    help="base width of the pixel context encoder")
    ap.add_argument("--finetune-ae-enc", action="store_true",
                    help="Like --pixel-context but reuse the AE's OWN encoder as "
                         "the context encoder and fine-tune it, instead of "
                         "training a fresh one. Better on two counts: it starts "
                         "from features that already work, and it STAYS NEAR the "
                         "space the assignment decorrelated z against, so it "
                         "shrinks the independence risk a from-scratch encoder "
                         "simply accepts. What it fixes is exactly the AE's "
                         "inability to cope with its own generated input: the "
                         "encoder gets gradient from the NEXT step's loss, i.e. "
                         "'encode generated frames so the generator can predict "
                         "well from them'. Only the encoder moves -- the AE "
                         "decoder is not in the generator path at all. Mutually "
                         "exclusive with --pixel-context.")
    ap.add_argument("--ae-lr", type=float, default=3e-5,
                    help="LR for the fine-tuned AE encoder. Deliberately ~1/10 of "
                         "--lr: how far this encoder drifts is how far the "
                         "generator's context moves from what z was decorrelated "
                         "against, so it is a knob trading robustness against the "
                         "validity of that independence.")
    ap.add_argument("--bptt", action="store_true",
                    help="keep the graph across rollout steps (backprop through "
                         "the AE encoder). Costs L x activations; off by default.")
    ap.add_argument("--ch", type=int, default=192)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lpips-weight", type=float, default=0.5)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "checkpoints").mkdir(exist_ok=True)

    A = torch.load(args.assignment, map_location="cpu", weights_only=False)
    for k in ("chunk", "frame", "action_vec", "act_norm"):
        if A.get(k) is None:
            raise SystemExit(f"assignment lacks '{k}'")
    z = A["z"].float()
    cond = A["cond"].float()
    ci = A["chunk"].numpy()
    fi = A["frame"].numpy()
    act_norm = A["act_norm"]
    N, dim_z = z.shape
    DIM = int(A.get("dim") or 256)
    CTX = cond.shape[1] // DIM
    lag = int(A.get("action_lag", ACTION_LAG))
    print(f"{N:,} particles  z {dim_z}  cond {cond.shape[1]} = {CTX} x {DIM}  "
          f"action_lag {lag}", flush=True)
    if lag != ACTION_LAG:
        print(f"  WARNING assignment was built with action_lag={lag} but "
              f"ACTION_LAG={ACTION_LAG}; the sequence actions use {lag} to stay "
              f"consistent with the z it was transported against", flush=True)

    segs = open_segments(args.cache)
    F_seg = segs.shape[1]
    S = args.side_cache
    keys = np.load(f"{S}/keys.npy", mmap_mode="r")
    mouse = np.load(f"{S}/mouse.npy", mmap_mode="r")
    clicks = np.load(f"{S}/clicks.npy", mmap_mode="r")
    acts81 = np.load(f"{S}/action_seqs.npy", mmap_mode="r")

    # --- the two caches must share segment ordering, or every action is wrong.
    # The particles recorded `action` from --cache at build time; the side arrays
    # come from --side-cache. Compare them at the lagged tick on a sample.
    if A.get("action") is not None:
        pr = np.random.default_rng(0).choice(N, 4096, replace=False)
        got = np.stack([np.asarray(acts81[int(ci[p]), int(fi[p]) - lag]) for p in pr])
        want = A["action"].numpy()[pr]
        # the particle's stored `action` is the UNLAGGED tick, so compare against
        # that tick rather than assuming; this checks ORDERING, not lag
        got0 = np.stack([np.asarray(acts81[int(ci[p]), int(fi[p])]) for p in pr])
        if not (got0 == want).all():
            raise SystemExit(
                "side cache disagrees with the particles' stored action at the "
                "same tick -- the two caches are not in the same segment order, "
                "so every gathered action would be wrong")
        print(f"  cache ordering check: side arrays agree with the particles' "
              f"stored 81-way action on {len(pr):,}/{len(pr):,} sampled ticks",
              flush=True)
        del got, got0, want

    def gather_actions(idx, L):
        """(n, L, 12) action_vec for steps 0..L-1 of each particle in idx."""
        out = np.empty((len(idx), L, 12), np.float32)
        for j, p in enumerate(idx):
            c, t = int(ci[p]), int(fi[p])
            f0 = t - lag                       # action for step 0
            k = np.asarray(keys[c, f0:f0 + L])
            m = np.asarray(mouse[c, f0:f0 + L])
            cl = np.asarray(clicks[c, f0:f0 + L])
            out[j] = apply_action_norm(build_action_raw(k, m, cl), act_norm)
        return out

    def gather_targets(idx, L):
        """(n, L, 3, H, W) float targets in [-1,1] for frames t..t+L-1."""
        out = np.empty((len(idx), L, args.image_size, args.image_size, 3), np.uint8)
        for j, p in enumerate(idx):
            c, t = int(ci[p]), int(fi[p])
            out[j] = np.asarray(segs[c][t:t + L])
        x = torch.from_numpy(out).permute(0, 1, 4, 2, 3).float()
        return x.div_(127.5).sub_(1.0)

    def gather_ctx_pixels(idx):
        """(n, CTX, 3, H, W) in [-1,1]: the REAL context frames t-CTX .. t-1."""
        out = np.empty((len(idx), CTX, args.image_size, args.image_size, 3), np.uint8)
        for j, p in enumerate(idx):
            c, t = int(ci[p]), int(fi[p])
            out[j] = np.asarray(segs[c][t - CTX:t])
        x = torch.from_numpy(out).permute(0, 1, 4, 2, 3).float()
        return x.div_(127.5).sub_(1.0)

    # particles that can support a full L-step sequence, as a SUBSET not a mask:
    # masking inside a batch would waste the slots
    L = args.seq_len
    ok = (fi - lag >= 0) & (fi + L - 1 <= F_seg - 1) & (fi - lag + L - 1 <= F_seg - 1)
    seq_pool = np.where(ok)[0]
    print(f"  sequence-capable particles: {len(seq_pool):,}/{N:,} "
          f"({100.0*len(seq_pool)/N:.1f}%) for L={L}", flush=True)
    if len(seq_pool) < args.batch:
        raise SystemExit(f"only {len(seq_pool)} particles support L={L}")

    ac = torch.load(args.ae, map_location=dev, weights_only=False)
    ae = AutoEncoder(ac["latent_dim"], ch=ac["channels"],
                     architecture=ac["architecture"], image_size=ac["image_size"],
                     grid=ac.get("grid", 4)).to(dev).eval()
    _sd = ac["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in _sd):
        _sd = {k.replace("_orig_mod.", "", 1): v for k, v in _sd.items()}
    ae.load_state_dict(_sd)
    for p in ae.parameters():
        p.requires_grad_(False)

    model = TransformerGenerator(dim_z=dim_z, ctx_frames=CTX, ctx_dim=DIM,
                                 act_dim=12, d_model=args.d_model,
                                 depth=args.depth, heads=args.heads, ch=args.ch,
                                 image_size=args.image_size).to(dev)
    print(f"generator params: {sum(p.numel() for p in model.parameters()):,}",
          flush=True)
    if args.pixel_context and args.finetune_ae_enc:
        raise SystemExit("--pixel-context and --finetune-ae-enc both replace the "
                         "context encoder; pick one")
    enc_pix = None
    enc_groups = []
    params = list(model.parameters())
    if args.pixel_context:
        enc_pix = PixelContextEncoder(ctx_dim=DIM, ch=args.pix_ch,
                                      image_size=args.image_size).to(dev)
        params += list(enc_pix.parameters())
        print(f"pixel context encoder (fresh): "
              f"{sum(p.numel() for p in enc_pix.parameters()):,} params; the "
              f"frozen AE is now used ONLY by the assignment, and not at all in "
              f"the rollout loop", flush=True)
    elif args.finetune_ae_enc:
        # The AE's own encoder becomes the context encoder, and trains. Only the
        # encoder: ae.dec is not in the generator path at all.
        enc_pix = ae.enc
        for prm in enc_pix.parameters():
            prm.requires_grad_(True)
        enc_groups = [{"params": list(enc_pix.parameters()), "lr": args.ae_lr}]
        print(f"fine-tuning the AE encoder as the context encoder: "
              f"{sum(p.numel() for p in enc_pix.parameters()):,} params at lr "
              f"{args.ae_lr:.1e} (generator at {args.lr:.1e}). Starts from "
              f"working features and stays near the space z was decorrelated "
              f"against.", flush=True)
    def encode_ctx(frames):
        """(B, T, 3, H, W) -> (B, T*DIM), for either encoder kind.

        PixelContextEncoder takes the time axis itself; the AE's encoder is a
        plain image encoder, so the frames are folded into the batch and unfolded
        after. Its per-image output is (latent_channels, grid, grid), which
        flattens to exactly DIM.
        """
        B, T = frames.shape[:2]
        if args.pixel_context:
            return enc_pix(frames).reshape(B, -1)
        out = enc_pix(frames.reshape(B * T, *frames.shape[2:]))
        return out.reshape(B, -1)

    perceptual = lpips.LPIPS(net="vgg").to(dev).eval()
    opt = torch.optim.Adam([{"params": params, "lr": args.lr}] + enc_groups)
    spe = max(1, N // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * spe)
    amp = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.amp
           else (lambda: torch.enable_grad()))

    recw = torch.tensor(np.sqrt(0.95 ** np.arange(CTX - 1, -1, -1)),
                        dtype=torch.float32, device=dev).view(1, CTX, 1)

    # --- assert step 0 IS the single-step problem: the gathered action for step
    # 0 must equal the assignment's own action_vec, and the gathered target for
    # step 0 must be the frame the particle points at.
    _probe = seq_pool[:256]
    _a = gather_actions(_probe, L)
    _av = A["action_vec"].numpy()[_probe]
    if not np.allclose(_a[:, 0], _av, atol=1e-5):
        raise SystemExit(
            "step-0 gathered action != the assignment's action_vec. The lag or "
            "the cache is inconsistent; every sequence action would be shifted.")
    _varies = float((np.abs(_a[:, 1:] - _a[:, :1]).max(2) > 0).any(1).mean())
    print(f"  step-0 action matches the assignment exactly; the action CHANGES "
          f"within the window for {100*_varies:.1f}% of probed particles",
          flush=True)
    if _varies < 0.2:
        print("  WARNING actions barely vary across the window -- the whole "
              "point of sequence rollout is that they do; check the lag",
              flush=True)
    del _a, _av

    z = z.to(dev)
    cond = cond.to(dev)
    A.pop("cond", None); A.pop("z", None); A.pop("h", None)
    gc.collect()

    curve = {"epoch": [], "mse": [], "lpips": [], "seq_mse": [], "per_step": []}
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        model.train()
        if enc_pix is not None:
            enc_pix.train()
        seq_on = ep >= args.seq_warmup
        perm = rng.permutation(N)
        tot_m = tot_l = n_b = 0.0
        tot_sm = n_sb = 0.0
        step_acc = np.zeros(L); step_n = np.zeros(L)
        t0 = time.time()
        for bi in range(spe):
            b = perm[bi * args.batch:(bi + 1) * args.batch]
            if len(b) == 0:
                continue
            do_seq = seq_on and (rng.random() < args.seq_prob)
            if not do_seq:
                # ordinary single-step: exactly step 0 of a sequence
                bt = torch.from_numpy(b).to(dev)
                x = gather_targets(b, 1)[:, 0].to(dev, non_blocking=True)
                a0 = torch.from_numpy(gather_actions(b, 1)[:, 0]).to(dev)
                if enc_pix is not None:
                    px = gather_ctx_pixels(b).to(dev, non_blocking=True)
                    cvec = encode_ctx(px)
                else:
                    cvec = cond[bt]
                inp = torch.cat([z[bt], cvec, a0], 1)
                opt.zero_grad(set_to_none=True)
                with amp():
                    pred = model(inp)
                    m = F.mse_loss(pred, x)
                    l = perceptual(pred.clamp(-1, 1), x).mean()
                    loss = m + args.lpips_weight * l
                loss.backward(); opt.step(); sched.step()
                tot_m += float(m); tot_l += float(l); n_b += 1
                continue

            # ---- sequence batch ----
            bs = rng.choice(seq_pool, size=min(args.batch, len(seq_pool)),
                            replace=False)
            bt = torch.from_numpy(bs).to(dev)
            acts = torch.from_numpy(gather_actions(bs, L)).to(dev)      # (n,L,12)
            tgts = gather_targets(bs, L).to(dev, non_blocking=True)     # (n,L,3,H,W)
            if enc_pix is not None:
                # pixel context: keep a WINDOW OF FRAMES and slide generated
                # frames into it. The AE never appears -- which is the point,
                # since with AE context every refeed step passed through an
                # encoder never trained on generated images.
                ctx_px = gather_ctx_pixels(bs).to(dev, non_blocking=True)
                h = None
            else:
                ctx_px = None
                h = cond[bt].view(len(bs), CTX, DIM) / recw
            opt.zero_grad(set_to_none=True)
            seq_loss = 0.0
            for s in range(L):
                zz = torch.randn(len(bs), dim_z, device=dev)
                if enc_pix is not None:
                    cvec = encode_ctx(ctx_px)
                else:
                    cvec = (h * recw).reshape(len(bs), -1)
                inp = torch.cat([zz, cvec, acts[:, s]], 1)
                with amp():
                    pred = model(inp)
                    m = F.mse_loss(pred, tgts[:, s])
                    l = perceptual(pred.clamp(-1, 1), tgts[:, s]).mean()
                    seq_loss = seq_loss + m + args.lpips_weight * l
                step_acc[s] += float(m); step_n[s] += 1
                tot_sm += float(m)
                if s + 1 < L and enc_pix is not None:
                    # Detach the PIXELS, never the encoder. The window holds
                    # frames and enc_pix is re-run on it every step, so the next
                    # step's loss still reaches the encoder's weights -- which is
                    # the whole point when that encoder is being fine-tuned. The
                    # detach only removes backprop into the generator through its
                    # own earlier prediction, i.e. BPTT.
                    nxt = pred.clamp(-1, 1).float().unsqueeze(1)
                    if args.bptt:
                        ctx_px = torch.cat([ctx_px[:, 1:], nxt], 1)
                    else:
                        ctx_px = torch.cat([ctx_px[:, 1:].detach(),
                                            nxt.detach()], 1)
                elif s + 1 < L:
                    # .float() is not optional: under autocast `pred` is
                    # bfloat16 and the AE's conv weights are fp32, and this call
                    # sits OUTSIDE the autocast block, so it would raise
                    # "Input type (c10::BFloat16) and bias type (float) should
                    # be the same". Encoding in fp32 also keeps the context
                    # window in one dtype -- mixing would silently downcast it.
                    nxt = pred.clamp(-1, 1).float()
                    if args.bptt:
                        hn = ae.enc(nxt).view(len(bs), 1, DIM).float()
                        h = torch.cat([h[:, 1:], hn], 1)
                    else:
                        with torch.no_grad():
                            hn = ae.enc(nxt.detach()).view(len(bs), 1, DIM).float()
                        h = torch.cat([h[:, 1:].detach(), hn], 1)
            (seq_loss / L).backward()
            opt.step(); sched.step()
            n_sb += 1

        el = time.time() - t0
        mse_e = tot_m / max(n_b, 1); lp_e = tot_l / max(n_b, 1)
        per_step = (step_acc / np.maximum(step_n, 1)).tolist()
        curve["epoch"].append(ep + 1); curve["mse"].append(mse_e)
        curve["lpips"].append(lp_e)
        curve["seq_mse"].append(tot_sm / max(n_sb * L, 1))
        curve["per_step"].append(per_step)
        ps = "  ".join(f"{v:.4f}" for v in per_step) if seq_on else "(warmup)"
        print(f"epoch {ep+1}/{args.epochs}  step0_mse={mse_e:.5f} "
              f"lpips={lp_e:.5f}  seq_mse={curve['seq_mse'][-1]:.5f}  "
              f"[{el/60:.1f}m]\n  per-step mse: {ps}", flush=True)
        (args.out / "gen_curve.json").write_text(json.dumps(curve))
        torch.save({"model_state_dict": model.state_dict(), "epoch": ep + 1,
                    "input_dim": dim_z + CTX * DIM + 12, "dim_z": dim_z,
                    "ch": args.ch, "ctx_frames": CTX, "ctx_dim": DIM,
                    "arch": "transformer", "d_model": args.d_model,
                    "depth": args.depth, "heads": args.heads,
                    "image_size": args.image_size, "act_dim": 12,
                    "action_onehot": False, "gan_weight": 0.0,
                    "act_norm": act_norm, "act_names": A.get("act_names"),
                    "action_lag": lag, "seq_len": L, "seq_prob": args.seq_prob,
                    "pixel_context": args.pixel_context, "pix_ch": args.pix_ch,
                    "finetune_ae_enc": args.finetune_ae_enc, "ae_lr": args.ae_lr,
                    "ae_checkpoint": str(args.ae),
                    "enc_pix_state_dict": (enc_pix.state_dict()
                                           if enc_pix is not None else None),
                    "opt_state_dict": opt.state_dict(),
                    "mse": mse_e, "lpips": lp_e,
                    "assignment": str(args.assignment)},
                   args.out / "checkpoints" / f"gen_seq_ep{ep+1}.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
