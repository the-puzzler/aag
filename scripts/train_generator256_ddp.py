#!/usr/bin/env python
"""DDP generator training at 256x256 from a persistent assignment.

    torchrun --standalone --nproc_per_node=8 scripts/train_generator256_ddp.py \
        --assignment .../assign.pt --dataset imagenet256 --out .../gen

Data is GPU-RESIDENT per rank: rank r decodes global rows [lo_r, hi_r) of the
train parquet straight into a uint8 tensor on its own GPU (31.5 GB per B200
for ImageNet) and takes z[lo_r:hi_r] from the assignment. After that one
decode the epoch loop touches no I/O at all. Identity is asserted at startup:
the labels stored in the assignment (which came through encode_hf256.py) must
equal the labels decoded from parquet for the same rows -- a misaligned
particle order once looked exactly like mode collapse in this project.

Loss is MSE + 0.5 * LPIPS(VGG), the project's standing recipe; bf16 autocast;
EMA weights are what get evaluated and sampled. Held-out pairs (a fixed seeded
1% of particles, never trained) give the assignment-selection metric: the MSE /
LPIPS of the generator on assigned (z_i, x_i) it has not seen. FID on fresh z is
computed against --fid-stats when given (compute_fid_stats_hf256.py), sharded
over ranks; sample grids are written at every eval for the user's eyes.
"""
from __future__ import annotations

import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.utils import save_image

from aag.generator256 import Generator256
from aag.hf256 import DATASETS, load_uint8, manifest, rank_slice, to_float

ap = argparse.ArgumentParser()
ap.add_argument("--assignment", required=True)
ap.add_argument("--dataset", choices=list(DATASETS), required=True)
ap.add_argument("--root", default=os.environ.get("AAG_HF_ROOT", "/data/aag_data/hf"))
ap.add_argument("--n-particles", type=int, default=0, help="0 = all rows in the assignment")
ap.add_argument("--n-classes", type=int, default=-1,
                help="-1 = infer (labels.max()+1) when the assignment was class-conditional (has levels), "
                     "else 0; 0 forces unconditional")
ap.add_argument("--width", type=float, default=1.0)
ap.add_argument("--n-res", type=int, default=2)
ap.add_argument("--cond-dim", type=int, default=512)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=32, help="per GPU")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--warmup", type=int, default=2000)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--grad-clip", type=float, default=1.0)
ap.add_argument("--lpips-weight", type=float, default=0.5)
ap.add_argument("--ema", type=float, default=0.9995)
ap.add_argument("--val-frac", type=float, default=0.01)
ap.add_argument("--eval-every", type=int, default=1, help="epochs")
ap.add_argument("--fid-stats", default=None, help=".npz from compute_fid_stats_hf256.py")
ap.add_argument("--fid-n", type=int, default=10000)
ap.add_argument("--log-every", type=int, default=200)
ap.add_argument("--compile", action="store_true")
ap.add_argument("--no-amp", action="store_true")
ap.add_argument("--decode-workers", type=int, default=0, help="0 = cpu_count // world_size")
ap.add_argument("--resume", default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

# ---------------------------------------------------------------- distributed
ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
if ddp:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
else:
    rank, world, local = 0, 1, 0
torch.cuda.set_device(local)
dev = torch.device("cuda", local)
is_main = rank == 0
torch.manual_seed(a.seed + rank)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
amp = not a.no_amp


def log(*s):
    if is_main:
        print(*s, flush=True)


# ---------------------------------------------------------------- assignment + data
A = torch.load(a.assignment, map_location="cpu", weights_only=False)
z_all, lab_all = A["z"].float(), A["label"]
N_avail = z_all.shape[0]
N = min(a.n_particles, N_avail) if a.n_particles else N_avail
dim_z, grid = z_all.shape[1], int(A.get("grid", 8))
levels = A.get("levels", [])
n_classes = a.n_classes if a.n_classes >= 0 else (int(lab_all.max()) + 1 if levels else 0)
lo, hi = rank_slice(N, rank, world)
z = z_all[lo:hi].to(dev); y = lab_all[lo:hi].to(dev)
del z_all
log(f"assignment {a.assignment}: N={N:,}/{N_avail:,} dim_z={dim_z} grid={grid} levels={levels} "
    f"steps={A.get('steps')} -> n_classes={n_classes} ({'class-conditional' if n_classes else 'unconditional'})")

workers = a.decode_workers or max(1, (os.cpu_count() or 8) // world)
t = time.time()
x_u8, y_dec = load_uint8(a.root, a.dataset, lo, hi, workers=workers, chunk=2048)
x_u8 = x_u8.to(dev)                                      # (n, H, W, 3) uint8, GPU-resident
y_dec = y_dec.to(dev)
if not torch.equal(y_dec, y):
    raise SystemExit(f"rank {rank}: labels decoded from parquet rows [{lo},{hi}) differ from the assignment's "
                     f"labels -- particle order mismatch, refusing to train")
n_local = hi - lo
log(f"rank shards: {n_local:,} images each, {x_u8.numel() / 1e9:.1f} GB uint8 per GPU, decoded in {time.time() - t:.0f}s "
    f"with {workers} workers/rank; identity check (labels) passed")

g = torch.Generator().manual_seed(a.seed)               # same on every rank
val_mask_all = torch.zeros(N, dtype=torch.bool)
val_mask_all[torch.randperm(N, generator=g)[:int(a.val_frac * N)]] = True
val_mask = val_mask_all[lo:hi].to(dev)
tr_idx = (~val_mask).nonzero(as_tuple=True)[0]
va_idx = val_mask.nonzero(as_tuple=True)[0]
n_train_total = int((~val_mask_all).sum())
log(f"held-out pairs: {int(val_mask_all.sum()):,} total ({a.val_frac:.1%}), {va_idx.numel():,} on rank 0")

# ---------------------------------------------------------------- model
model = Generator256(dim_z, grid=grid, image_size=DATASETS[a.dataset]["image_size"], n_classes=n_classes,
                     cond_dim=a.cond_dim, width=a.width, n_res=a.n_res).to(dev)
n_params = sum(p.numel() for p in model.parameters())
ema = Generator256(dim_z, grid=grid, image_size=DATASETS[a.dataset]["image_size"], n_classes=n_classes,
                   cond_dim=a.cond_dim, width=a.width, n_res=a.n_res).to(dev).eval()
ema.load_state_dict(model.state_dict())
for p in ema.parameters():
    p.requires_grad_(False)
import lpips
perceptual = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
for p in perceptual.parameters():
    p.requires_grad_(False)

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=a.wd)
steps_per_epoch = math.ceil(tr_idx.numel() / a.batch)
total_steps = steps_per_epoch * a.epochs
def lr_at(s):
    if s < a.warmup:
        return a.lr * (s + 1) / a.warmup
    p = (s - a.warmup) / max(1, total_steps - a.warmup)
    return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
start_epoch, gstep = 0, 0
curve = {"epoch": [], "train_mse": [], "train_lpips": [], "val_mse": [], "val_lpips": [], "fid": []}
if a.resume:
    R = torch.load(a.resume, map_location=dev, weights_only=False)
    model.load_state_dict(R["model"]); ema.load_state_dict(R["ema"]); opt.load_state_dict(R["opt"])
    start_epoch, gstep, curve = R["epoch"], R["gstep"], R["curve"]
    log(f"resumed {a.resume}: epoch {start_epoch}, step {gstep:,}, last val_mse {curve['val_mse'][-1] if curve['val_mse'] else None}")
raw = model
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
fwd = torch.compile(model) if a.compile else model

log(f"generator: {n_params / 1e6:.1f}M params  width={a.width} n_res={a.n_res}  batch {a.batch}x{world}={a.batch * world}  "
    f"{steps_per_epoch:,} steps/epoch x {a.epochs} epochs  lr {a.lr} warmup {a.warmup}  ema {a.ema}")
log(f"precision: {'bf16 autocast' if amp else 'fp32'}  compile: {a.compile}  lpips_weight {a.lpips_weight}  "
    f"fid: {'every eval, n=' + str(a.fid_n) + ' vs ' + a.fid_stats if a.fid_stats else 'off'}")

if is_main:
    a.out.mkdir(parents=True, exist_ok=True); (a.out / "checkpoints").mkdir(exist_ok=True)
    (a.out / "args.json").write_text(json.dumps(vars(a), indent=1, default=str))
fixed_gen = torch.Generator(device=dev).manual_seed(1234)
z_fixed = torch.randn(64, dim_z, device=dev, generator=fixed_gen)
y_fixed = (torch.arange(64, device=dev) * (n_classes // 64 if n_classes >= 64 else 1)) % max(n_classes, 1) if n_classes else None


def all_reduce_mean(v: float, n: int) -> float:
    t_ = torch.tensor([v * n, n], dtype=torch.float64, device=dev)
    if ddp:
        dist.all_reduce(t_)
    return float(t_[0] / max(t_[1], 1))


@torch.no_grad()
def evaluate(net):
    net.eval()
    tm, tl, n = 0.0, 0.0, 0
    for i in range(0, va_idx.numel(), a.batch):
        b = va_idx[i:i + a.batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = net(z[b], y[b] if n_classes else None)
        tgt = to_float(x_u8[b])
        tm += F.mse_loss(pred.float(), tgt).item() * b.numel()
        tl += perceptual(pred.float().clamp(-1, 1), tgt).mean().item() * b.numel()
        n += b.numel()
    return all_reduce_mean(tm / max(n, 1), n), all_reduce_mean(tl / max(n, 1), n)


@torch.no_grad()
def fid_fresh(net):
    from aag.fid import get_activations, fid_from_stats
    ref = np.load(a.fid_stats)
    per = math.ceil(a.fid_n / world)
    gg = torch.Generator(device=dev).manual_seed(999 + rank)
    acts = []
    for i in range(0, per, a.batch):
        m = min(a.batch, per - i)
        zz = torch.randn(m, dim_z, device=dev, generator=gg)
        yy = torch.randint(0, n_classes, (m,), device=dev, generator=gg) if n_classes else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            img = net(zz, yy)
        acts.append(torch.from_numpy(get_activations((img.float().clamp(-1, 1) + 1) / 2, dev, batch=m)))
    acts = torch.cat(acts).to(dev)
    if ddp:
        gathered = [torch.empty_like(acts) for _ in range(world)]
        dist.all_gather(gathered, acts); acts = torch.cat(gathered)
    return fid_from_stats(ref["mu"], ref["sigma"], acts[:a.fid_n].cpu().numpy()) if is_main else None


t0 = time.time()
for epoch in range(start_epoch, a.epochs):
    model.train()
    perm = tr_idx[torch.randperm(tr_idx.numel(), device=dev)]
    run_mse, run_lp, run_n = 0.0, 0.0, 0
    for i in range(0, perm.numel(), a.batch):
        b = perm[i:i + a.batch]
        for pg in opt.param_groups:
            pg["lr"] = lr_at(gstep)
        tgt = to_float(x_u8[b])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = fwd(z[b], y[b] if n_classes else None)
        pred = pred.float()
        mse = F.mse_loss(pred, tgt)
        perc = perceptual(pred.clamp(-1, 1), tgt).mean() if a.lpips_weight > 0 else torch.zeros((), device=dev)
        loss = mse + a.lpips_weight * perc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), a.grad_clip)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), raw.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        gstep += 1
        run_mse += mse.item() * b.numel(); run_lp += perc.item() * b.numel(); run_n += b.numel()
        if is_main and gstep % a.log_every == 0:
            el = time.time() - t0
            print(f"  ep {epoch + 1} step {gstep:,}/{total_steps:,}  mse {run_mse / run_n:.5f}  lpips {run_lp / run_n:.4f}  "
                  f"lr {lr_at(gstep):.2e}  {gstep * a.batch * world / el:,.0f} img/s  eta {(total_steps - gstep) * el / max(gstep - start_epoch * steps_per_epoch, 1) / 3600:.1f} h",
                  flush=True)
    tr_mse, tr_lp = all_reduce_mean(run_mse / max(run_n, 1), run_n), all_reduce_mean(run_lp / max(run_n, 1), run_n)
    curve["epoch"].append(epoch + 1); curve["train_mse"].append(tr_mse); curve["train_lpips"].append(tr_lp)
    if (epoch + 1) % a.eval_every == 0 or epoch + 1 == a.epochs:
        vm, vl = evaluate(ema)
        fid = fid_fresh(ema) if a.fid_stats else None
        curve["val_mse"].append(vm); curve["val_lpips"].append(vl); curve["fid"].append(fid)
        log(f"epoch {epoch + 1}/{a.epochs}  train_mse={tr_mse:.5f} train_lpips={tr_lp:.4f}  "
            f"heldout_mse={vm:.5f} heldout_lpips={vl:.4f}" + (f"  fid{a.fid_n // 1000}k={fid:.2f}" if fid is not None else "")
            + f"  [{(time.time() - t0) / 3600:.2f} h]")
        if is_main:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                imgs = ema(z_fixed, y_fixed).float()
            save_image((imgs.clamp(-1, 1) + 1) / 2, a.out / f"samples_ep{epoch + 1:03d}.png", nrow=8)
            b = va_idx[:16]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                rec = ema(z[b], y[b] if n_classes else None).float()
            pair = torch.stack([to_float(x_u8[b]), rec.clamp(-1, 1)], 1).flatten(0, 1)
            save_image((pair + 1) / 2, a.out / f"heldout_pairs_ep{epoch + 1:03d}.png", nrow=8)
            ck = a.out / "checkpoints" / f"gen_ep{epoch + 1:03d}.pt"
            torch.save({"model": raw.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "epoch": epoch + 1, "gstep": gstep, "curve": curve, "args": vars(a), "dim_z": dim_z,
                        "grid": grid, "n_classes": n_classes, "width": a.width, "n_res": a.n_res,
                        "cond_dim": a.cond_dim, "assignment": a.assignment}, str(ck) + ".tmp")
            Path(str(ck) + ".tmp").replace(ck)
            (a.out / "curve.json").write_text(json.dumps(curve, indent=1))
    else:
        log(f"epoch {epoch + 1}/{a.epochs}  train_mse={tr_mse:.5f} train_lpips={tr_lp:.4f}  [{(time.time() - t0) / 3600:.2f} h]")
    if ddp:
        dist.barrier()
log("done")
if ddp:
    dist.destroy_process_group()
