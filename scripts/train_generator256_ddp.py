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
ap.add_argument("--aug-hflip", action="store_true",
                help="assignment rows [n_img, 2 n_img) are the horizontally flipped copies of rows [0, n_img) "
                     "(particles from encode_hf256.py --flip, merged by merge_particles.py)")
ap.add_argument("--n-res", type=int, default=2)
ap.add_argument("--cond-dim", type=int, default=512)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=32, help="per GPU")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--warmup", type=int, default=2000)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--grad-clip", type=float, default=1.0)
ap.add_argument("--lpips-weight", type=float, default=0.5)
ap.add_argument("--mse-weight", type=float, default=1.0, help="pixel MSE weight (ROMS-IMLE pixel recipe: 0.1 with LPIPS 1.0 and DINO 1.0)")
ap.add_argument("--dino-weight", type=float, default=0.0,
                help="0 = off; L2 between DINOv2 ViT-B/14 features (CLS + patch tokens, 224x224 input) of prediction and target. "
                     "Loaded via torch.hub from $TORCH_HOME (mirror at /mnt/shared/aag/torch_hub on the cluster).")
ap.add_argument("--ema", type=float, default=0.9995)
ap.add_argument("--val-frac", type=float, default=0.01)
ap.add_argument("--eval-every", type=int, default=1, help="epochs")
ap.add_argument("--fid-stats", default=None, help=".npz from compute_fid_stats_hf256.py")
ap.add_argument("--fid-n", type=int, default=10000)
ap.add_argument("--log-every", type=int, default=200)
ap.add_argument("--compile", action="store_true")
ap.add_argument("--no-amp", action="store_true")
ap.add_argument("--decode-workers", type=int, default=0, help="0 = cpu_count // world_size")
ap.add_argument("--resume", default=None, help="checkpoint path, or 'auto' = latest gen_ep*.pt under --out")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--arch", choices=["grid", "residual"], default="grid",
                help="grid = Generator256 (reshape/stem onto a spatial grid, GroupNorm, DC-AE shortcuts); "
                     "residual = the published 64x64 recipe scaled to 256: flat z -> Linear -> 4x4 -> BatchNorm residual up-blocks (aag.ae.ResidualDecoder)")
ap.add_argument("--ch", type=int, default=128, help="--arch residual: base channels of ResidualDecoder")
# --- pairwise adversary (user, 2026-09-05): a finetune on the converged plain generator.
# The critic sees real and generated images for the SAME z stacked on the channel axis in a
# random order and says which side is real (aag.discriminator.paired_*). No AE adversary: a
# linear probe showed an AE adversary changes only the decoder, which AAG throws away.
ap.add_argument("--fresh-gan-weight", type=float, default=0.0,
                help="0 = off; UNPAIRED critic on real images vs G(z) at FRESH z ~ N(0,I) (user idea 2026-09-06): acts only where "
                     "the pairs give no supervision. Adaptive weight (adversarial grad = this fraction of the supervised grad at "
                     "the last conv) unless --fresh-gan-fixed; keep it weak relative to the supervised objective.")
ap.add_argument("--fresh-gan-fixed", action="store_true", help="use --fresh-gan-weight as a fixed multiplier instead of x adaptive")
ap.add_argument("--fresh-gan-layers", type=int, default=3)
ap.add_argument("--fresh-gan-ndf", type=int, default=64)
ap.add_argument("--gan-weight", type=float, default=0.0, help="0 = off; scales the adaptively balanced adversarial term")
ap.add_argument("--gan-lr", type=float, default=4.5e-5)
ap.add_argument("--gan-ndf", type=int, default=64)
ap.add_argument("--gan-layers", type=int, default=3, help="3 = 70px receptive field, 2 = 34px")
ap.add_argument("--gan-fixed", action="store_true", help="use --gan-weight as a fixed weight instead of adaptive x weight")
ap.add_argument("--reset-schedule", action="store_true",
                help="with --resume: fresh warmup+cosine over the remaining epochs (finetune) instead of continuing the old one")
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
if a.aug_hflip:
    n_img = manifest(a.root, a.dataset)["offsets"][-1]
    if N_avail != 2 * n_img:
        raise SystemExit(f"--aug-hflip: assignment has {N_avail:,} rows, expected 2 x {n_img:,}")
    parts, labs = [], []
    if lo < n_img:                                       # original rows in this shard
        xa, ya = load_uint8(a.root, a.dataset, lo, min(hi, n_img), workers=workers, chunk=2048); parts.append(xa); labs.append(ya)
    if hi > n_img:                                       # flipped rows: image (r - n_img), W-flipped
        xb, yb = load_uint8(a.root, a.dataset, max(lo, n_img) - n_img, hi - n_img, workers=workers, chunk=2048)
        parts.append(xb.flip(2)); labs.append(yb)         # (n,H,W,3): dim 2 is W
    x_u8, y_dec = torch.cat(parts), torch.cat(labs)
else:
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
if a.arch == "residual":
    from aag.ae import ResidualDecoder
    assert n_classes == 0, "--arch residual is unconditional"
    class FlatGen(torch.nn.Module):          # same forward(z, y) interface as Generator256
        def __init__(self):
            super().__init__(); self.dec = ResidualDecoder(dim_z, ch=a.ch, image_size=DATASETS[a.dataset]["image_size"])
            self.out = self.dec.net[-1]      # last conv, for the adaptive adversarial weight
        def forward(self, z, y=None):
            return self.dec(z)
    make = lambda: FlatGen().to(dev)
else:
    make = lambda: Generator256(dim_z, grid=grid, image_size=DATASETS[a.dataset]["image_size"], n_classes=n_classes,
                                cond_dim=a.cond_dim, width=a.width, n_res=a.n_res).to(dev)
model = make()
n_params = sum(p.numel() for p in model.parameters())
ema = make().eval()
ema.load_state_dict(model.state_dict())
for p in ema.parameters():
    p.requires_grad_(False)
import lpips
perceptual = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
for p in perceptual.parameters():
    p.requires_grad_(False)


class DinoFeatures(torch.nn.Module):
    """DINOv2 ViT-B/14 feature distance: images in [-1,1] -> 224x224, ImageNet-normalised -> CLS + patch tokens.
    Weights are frozen; the repo/checkpoint come from torch.hub's cache (offline on the cluster: TORCH_HOME mirror)."""
    def __init__(self):
        super().__init__()
        self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False, trust_repo=True, skip_validation=True).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):
        x = F.interpolate((x + 1) / 2, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        out = self.net.forward_features((x - self.mean) / self.std)
        return out["x_norm_clstoken"], out["x_norm_patchtokens"]
    def distance(self, pred, tgt):
        cp, pp = self(pred)
        with torch.no_grad():
            ct, pt = self(tgt)
        return F.mse_loss(cp, ct) + F.mse_loss(pp, pt)
dino = DinoFeatures().to(dev) if a.dino_weight > 0 else None

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=a.wd)
adv = a.gan_weight > 0
fadv = a.fresh_gan_weight > 0
disc = opt_d = fdisc = opt_fd = None
if adv or fadv:
    from aag.discriminator import NLayerDiscriminator, paired_batch, paired_d_loss, paired_g_loss, adaptive_weight, hinge_d_loss, g_loss_from
if adv:
    disc = NLayerDiscriminator(6, a.gan_ndf, a.gan_layers).to(dev)
    opt_d = torch.optim.Adam(disc.parameters(), lr=a.gan_lr, betas=(0.5, 0.9))
    pair_gen = torch.Generator(device=dev).manual_seed(4321 + rank)
if fadv:
    fdisc = NLayerDiscriminator(3, a.fresh_gan_ndf, a.fresh_gan_layers).to(dev)
    opt_fd = torch.optim.Adam(fdisc.parameters(), lr=a.gan_lr, betas=(0.5, 0.9))
    fresh_gen = torch.Generator(device=dev).manual_seed(8765 + rank)
# Every rank must run the SAME number of steps per epoch: each backward is an NCCL
# all-reduce, and the held-out mask removes a Binomial number of rows per shard, so
# ceil(len(tr_idx)/batch) differs by one across ranks. A rank that leaves the loop
# early deadlocks the others at their next gradient all-reduce. Use the minimum.
n_local_steps = math.ceil(tr_idx.numel() / a.batch)
_cnt = torch.tensor([n_local_steps], device=dev)
if ddp:
    _all = [torch.zeros_like(_cnt) for _ in range(world)]
    dist.all_gather(_all, _cnt)
    steps_per_epoch = int(min(int(c) for c in _all))
    log(f"per-rank train batches {[int(c) for c in _all]} -> every rank runs {steps_per_epoch} steps/epoch (the min)")
else:
    steps_per_epoch = n_local_steps
total_steps = steps_per_epoch * a.epochs
def lr_at(s):
    if s < a.warmup:
        return a.lr * (s + 1) / a.warmup
    p = (s - a.warmup) / max(1, total_steps - a.warmup)
    return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
start_epoch, gstep = 0, 0
curve = {"epoch": [], "train_mse": [], "train_lpips": [], "val_mse": [], "val_lpips": [], "fid": []}
if a.resume == "auto":
    # latest checkpoint under --out, or a fresh start if there is none: lets a
    # preempted/relaunched job continue without editing the config
    _cks = sorted((a.out / "checkpoints").glob("gen_ep*.pt")) if (a.out / "checkpoints").exists() else []
    a.resume = str(_cks[-1]) if _cks else None
    log(f"--resume auto -> {a.resume or 'no checkpoint found, starting fresh'}")
sched0 = 0
if a.resume:
    R = torch.load(a.resume, map_location=dev, weights_only=False)
    model.load_state_dict(R["model"]); ema.load_state_dict(R["ema"])
    if a.reset_schedule:
        # finetune: keep the weights, drop the old optimizer state and start a fresh schedule
        sched0 = R["gstep"]
        total_steps = steps_per_epoch * (a.epochs - R["epoch"])
        def lr_at(s):
            s = s - sched0
            if s < a.warmup:
                return a.lr * (s + 1) / a.warmup
            p = (s - a.warmup) / max(1, total_steps - a.warmup)
            return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    else:
        opt.load_state_dict(R["opt"])
    if adv and "disc" in R:
        disc.load_state_dict(R["disc"]); opt_d.load_state_dict(R["opt_d"])
    if fadv and "fdisc" in R:
        fdisc.load_state_dict(R["fdisc"]); opt_fd.load_state_dict(R["opt_fd"])
    start_epoch, gstep, curve = R["epoch"], R["gstep"], R["curve"]
    log(f"resumed {a.resume}: epoch {start_epoch}, step {gstep:,}, last val_mse {curve['val_mse'][-1] if curve['val_mse'] else None}"
        + (f"; fresh schedule: lr {a.lr}, warmup {a.warmup}, {total_steps:,} steps" if a.reset_schedule else ""))
raw = model
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    if adv:
        disc = torch.nn.parallel.DistributedDataParallel(disc, device_ids=[local])
    if fadv:
        fdisc = torch.nn.parallel.DistributedDataParallel(fdisc, device_ids=[local])
if fadv:
    log(f"fresh-z adversary on: weight={a.fresh_gan_weight} ({'fixed' if a.fresh_gan_fixed else 'x adaptive'}), critic ndf={a.fresh_gan_ndf} "
        f"n_layers={a.fresh_gan_layers} ({sum(p.numel() for p in fdisc.parameters()) / 1e6:.1f}M), lr={a.gan_lr}; real batch vs G(N(0,I)) batch")
if adv:
    log(f"pairwise adversary on: weight={a.gan_weight} ({'fixed' if a.gan_fixed else 'x adaptive'}), critic ndf={a.gan_ndf} "
        f"n_layers={a.gan_layers} ({sum(p.numel() for p in disc.parameters()) / 1e6:.1f}M), lr={a.gan_lr}")
fwd = torch.compile(model) if a.compile else model

log(f"generator: {n_params / 1e6:.1f}M params  arch={a.arch}{' ch=' + str(a.ch) if a.arch == 'residual' else ''} width={a.width} n_res={a.n_res}  batch {a.batch}x{world}={a.batch * world}  "
    f"{steps_per_epoch:,} steps/epoch x {a.epochs} epochs  lr {a.lr} warmup {a.warmup}  ema {a.ema}")
log(f"precision: {'bf16 autocast' if amp else 'fp32'}  compile: {a.compile}  mse_weight {a.mse_weight}  lpips_weight {a.lpips_weight}  dino_weight {a.dino_weight}  "
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
gstep_run0 = gstep    # throughput/ETA count from here, whatever step the resume landed on
for epoch in range(start_epoch, a.epochs):
    model.train()
    perm = tr_idx[torch.randperm(tr_idx.numel(), device=dev)]
    run_mse, run_lp, run_n = 0.0, 0.0, 0
    run_dn = 0.0
    run_g = run_d = run_w = 0.0
    run_gf = run_df = run_wf = 0.0
    for i in range(steps_per_epoch):
        b = perm[i * a.batch:(i + 1) * a.batch]
        for pg in opt.param_groups:
            pg["lr"] = lr_at(gstep)
        tgt = to_float(x_u8[b])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = fwd(z[b], y[b] if n_classes else None)
        pred = pred.float()
        mse = F.mse_loss(pred, tgt)
        perc = perceptual(pred.clamp(-1, 1), tgt).mean() if a.lpips_weight > 0 else torch.zeros((), device=dev)
        if dino is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                dn = dino.distance(pred.clamp(-1, 1), tgt).float()
        else:
            dn = torch.zeros((), device=dev)
        loss = a.mse_weight * mse + a.lpips_weight * perc + a.dino_weight * dn
        opt.zero_grad(set_to_none=True)
        total = loss
        if adv:
            # generator step: fool the pair critic; the adaptive weight measures the adversarial
            # gradient against the reconstruction gradient at the generator's last conv
            pair, lab = paired_batch(tgt, pred.clamp(-1, 1), pair_gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                g_adv = paired_g_loss(disc(pair).float(), lab)
            w = torch.tensor(a.gan_weight, device=dev) if a.gan_fixed else adaptive_weight(loss, g_adv, raw.out.weight) * a.gan_weight
            total = total + w * g_adv
        if fadv:
            # fresh-z term: samples from N(0,I) (labels drawn from the batch when conditional) judged by an
            # unpaired critic against the real batch -- supervision exactly where the pairs give none
            zf = torch.randn(b.numel(), dim_z, device=dev, generator=fresh_gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pred_f = fwd(zf, y[b] if n_classes else None).float().clamp(-1, 1)
                g_adv_f = g_loss_from(fdisc(pred_f).float())
            wf = torch.tensor(a.fresh_gan_weight, device=dev) if a.fresh_gan_fixed else adaptive_weight(loss, g_adv_f, raw.out.weight) * a.fresh_gan_weight
            total = total + wf * g_adv_f
        total.backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), a.grad_clip)
        opt.step()
        if adv:
            # critic step on the same pairs with the generator output detached
            pair_d, lab_d = paired_batch(tgt, pred.detach().clamp(-1, 1), pair_gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                d_loss = paired_d_loss(disc(pair_d).float(), lab_d)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            run_g += g_adv.item() * b.numel(); run_d += d_loss.item() * b.numel(); run_w += float(w) * b.numel()
        if fadv:
            # ONE critic forward on real||fake: under DDP every forward re-broadcasts the BatchNorm buffers
            # in place, so two forwards before one backward corrupt the saved tensors of the first
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                lg = fdisc(torch.cat([tgt, pred_f.detach()], 0)).float()
            d_loss_f = hinge_d_loss(lg[: tgt.shape[0]], lg[tgt.shape[0]:])
            opt_fd.zero_grad(set_to_none=True)
            d_loss_f.backward()
            opt_fd.step()
            run_gf += g_adv_f.item() * b.numel(); run_df += d_loss_f.item() * b.numel(); run_wf += float(wf) * b.numel()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), raw.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        gstep += 1
        run_mse += mse.item() * b.numel(); run_lp += perc.item() * b.numel(); run_n += b.numel(); run_dn += dn.item() * b.numel()
        if is_main and gstep % a.log_every == 0:
            el = time.time() - t0
            print(f"  ep {epoch + 1} step {gstep - sched0:,}/{total_steps:,}  mse {run_mse / run_n:.5f}  lpips {run_lp / run_n:.4f}  "
                  + (f"g {run_g / run_n:.3f} d {run_d / run_n:.3f} w {run_w / run_n:.3g}  " if adv else "")
                  + (f"gf {run_gf / run_n:.3f} df {run_df / run_n:.3f} wf {run_wf / run_n:.3g}  " if fadv else "") +
                  f"lr {lr_at(gstep):.2e}  {(gstep - gstep_run0) * a.batch * world / el:,.0f} img/s  "
                  f"eta {(total_steps - (gstep - sched0)) * el / max(gstep - gstep_run0, 1) / 3600:.1f} h",
                  flush=True)
    tr_mse, tr_lp = all_reduce_mean(run_mse / max(run_n, 1), run_n), all_reduce_mean(run_lp / max(run_n, 1), run_n)
    curve["epoch"].append(epoch + 1); curve["train_mse"].append(tr_mse); curve["train_lpips"].append(tr_lp)
    if (a.eval_every > 0 and (epoch + 1) % a.eval_every == 0) or epoch + 1 == a.epochs:
        vm, vl = evaluate(ema)
        fid = fid_fresh(ema) if a.fid_stats else None
        curve["val_mse"].append(vm); curve["val_lpips"].append(vl); curve["fid"].append(fid)
        log(f"epoch {epoch + 1}/{a.epochs}  train_mse={tr_mse:.5f} train_lpips={tr_lp:.4f}" + (f" train_dino={run_dn / max(run_n, 1):.4f}" if dino is not None else "") + "  "
            f"heldout_mse={vm:.5f} heldout_lpips={vl:.4f}" + (f"  fid@{a.fid_n}={fid:.2f}" if fid is not None else "")
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
                        **({"disc": (disc.module if ddp else disc).state_dict(), "opt_d": opt_d.state_dict()} if adv else {}),
                        **({"fdisc": (fdisc.module if ddp else fdisc).state_dict(), "opt_fd": opt_fd.state_dict()} if fadv else {}),
                        "epoch": epoch + 1, "gstep": gstep, "curve": curve, "args": vars(a), "dim_z": dim_z,
                        "grid": grid, "n_classes": n_classes, "width": a.width, "n_res": a.n_res,
                        "cond_dim": a.cond_dim, "assignment": a.assignment, "arch": a.arch, "ch": a.ch}, str(ck) + ".tmp")
            Path(str(ck) + ".tmp").replace(ck)
            (a.out / "curve.json").write_text(json.dumps(curve, indent=1))
    else:
        log(f"epoch {epoch + 1}/{a.epochs}  train_mse={tr_mse:.5f} train_lpips={tr_lp:.4f}" + (f" train_dino={run_dn / max(run_n, 1):.4f}" if dino is not None else "") + f"  [{(time.time() - t0) / 3600:.2f} h]")
    if ddp:
        dist.barrier()
log("done")
if ddp:
    dist.destroy_process_group()
