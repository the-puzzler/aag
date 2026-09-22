"""Train the AAG inverse model E(x) -> z on the REAL assigned pairs only (see aag/inverse.py).

    torchrun --nproc_per_node=8 scripts/train_inverse256.py --assignment .../assign.pt --dataset imagenet256 \
        [--base-ckpt gen.pt]  --out .../inverse

E = frozen DINOv2 ViT-B/14 (CLS + patch tokens) -> InverseHead (attention pooling) -> target.
Target = full whitened z, or the generator's bottleneck code c = W0 z + b0 when --base-ckpt has a linear
bottleneck (then E predicts exactly what the generator reads). Targets are standardised per dimension by the
anchor statistics, so val MSE 1.0 = constant predictor; R^2 = 1 - MSE. Augmentation: random crop (85-100%) +
mild brightness/contrast jitter; NO flips (z encodes pose). Images are GPU-resident per rank like the trainer.
Held-out anchors (--val-frac) are never trained on -> honest R^2 for "how predictable is z from x".
"""
from __future__ import annotations

import argparse, json, os, time
import torch, torch.distributed as dist, torch.nn.functional as F
from aag.hf256 import load_uint8, manifest, rank_slice, to_float
from aag.inverse import InverseHead

ap = argparse.ArgumentParser()
ap.add_argument("--assignment", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--root", default=os.environ.get("AAG_HF_ROOT", "/data/aag_data/hf"))
ap.add_argument("--base-ckpt", default="", help="generator checkpoint; if it has a linear bottleneck the target is its code W0 z + b0")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--val-frac", type=float, default=0.02)
ap.add_argument("--encoder", choices=["dino", "conv"], default="conv", help="'conv': from-scratch mirror of the generator (default); 'dino': trained head on frozen DINOv2 tokens")
ap.add_argument("--enc-width", type=float, default=1.0)
ap.add_argument("--enc-res", type=int, default=2)
ap.add_argument("--enc-head", choices=["pool", "grid"], default="pool")
ap.add_argument("--enc-dropout", type=float, default=0.0)
ap.add_argument("--ema", type=float, default=0.0, help=">0: evaluate and save an EMA of the encoder weights (e.g. 0.999)")
ap.add_argument("--n-query", type=int, default=8)
ap.add_argument("--depth", type=int, default=2)
ap.add_argument("--aug", type=int, default=1)
ap.add_argument("--decode-workers", type=int, default=0)
ap.add_argument("--log-every", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
if ddp:
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
else:
    rank, world = 0, 1
local = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(local); dev = torch.device("cuda", local)
is_main = rank == 0
def log(*s):
    if is_main: print(*s, flush=True)
os.makedirs(a.out, exist_ok=True)

A = torch.load(a.assignment, map_location="cpu", weights_only=False)
z_all, lab_all = A["z"].float(), A["label"]; N, dim_z = z_all.shape
lo, hi = rank_slice(N, rank, world)
z = z_all[lo:hi].to(dev)
workers = a.decode_workers or max(1, (os.cpu_count() or 8) // world)
t = time.time(); x_u8, y_dec = load_uint8(a.root, a.dataset, lo, hi, workers=workers, chunk=2048); x_u8 = x_u8.to(dev)
if not torch.equal(y_dec, lab_all[lo:hi]):
    raise SystemExit(f"rank {rank}: label mismatch between parquet rows and assignment")
log(f"assignment {a.assignment}: N={N:,} dim_z={dim_z}; rank shard {hi - lo:,} images ({x_u8.numel() / 1e9:.1f} GB) decoded in {time.time() - t:.0f}s")

# target definition
W0 = b0 = None; kind = "z"
if a.base_ckpt:
    ck = torch.load(a.base_ckpt, map_location="cpu", weights_only=False); sd = ck.get("ema", ck.get("model"))
    if "bott.weight" in sd and sd["bott.weight"].dim() == 2:
        W0, b0 = sd["bott.weight"].float(), sd["bott.bias"].float(); kind = "code"
        log(f"target = generator bottleneck code: W0 {tuple(W0.shape)} from {a.base_ckpt}")
    else:
        log(f"base ckpt has no linear bottleneck -> target = full z")
def raw_target(zz):
    return zz @ W0.to(zz.device).t() + b0.to(zz.device) if kind == "code" else zz
with torch.no_grad():
    tgt_all = raw_target(z_all.to(dev)); mu, sd_ = tgt_all.mean(0), tgt_all.std(0) + 1e-6; del tgt_all
out_dim = mu.numel()
def target(zz): return (raw_target(zz) - mu) / sd_

# split (identical on every rank)
g = torch.Generator().manual_seed(a.seed)
val_mask_all = torch.zeros(N, dtype=torch.bool); val_mask_all[torch.randperm(N, generator=g)[: int(a.val_frac * N)]] = True
val_mask = val_mask_all[lo:hi].to(dev); tr_idx = (~val_mask).nonzero(as_tuple=True)[0]; va_idx = val_mask.nonzero(as_tuple=True)[0]

# encoder
head_kwargs = conv_kwargs = None
if a.encoder == "conv":
    from aag.inverse_conv import ConvInverse
    conv_kwargs = dict(out_dim=out_dim, width=a.enc_width, n_res=a.enc_res, head=a.enc_head, dropout=a.enc_dropout)
    head = ConvInverse(**conv_kwargs).to(dev)
    def predict(x): return head(x)
else:
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False, trust_repo=True, skip_validation=True).eval().to(dev)
    for p in dino.parameters(): p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    def feats(x):  # x in [-1,1], (B,3,H,W)
        x = F.interpolate((x + 1) / 2, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        o = dino.forward_features((x - mean) / std); return o["x_norm_clstoken"], o["x_norm_patchtokens"]
    head_kwargs = dict(dim_tok=768, n_tok=257, n_query=a.n_query, out_dim=out_dim, depth=a.depth)
    head = InverseHead(**head_kwargs).to(dev)
    def predict(x):
        with torch.no_grad():
            cls, pt = feats(x)
        return head(cls.float(), pt.float())
net = torch.nn.parallel.DistributedDataParallel(head, device_ids=[local]) if ddp else head
opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
steps_per_epoch = tr_idx.numel() // a.batch
if ddp:
    m = torch.tensor([steps_per_epoch], device=dev); dist.all_reduce(m, op=dist.ReduceOp.MIN); steps_per_epoch = int(m.item())
total_steps = steps_per_epoch * a.epochs; warm = min(500, total_steps // 10)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + torch.cos(torch.tensor(min(1.0, s / total_steps) * 3.14159265)).item()))
ema_net = None
if a.ema > 0:
    import copy
    ema_net = copy.deepcopy(head).eval()
    for p_ in ema_net.parameters(): p_.requires_grad_(False)
log(f"encoder={a.encoder} {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M params, target {kind} dim {out_dim}, "
    f"{steps_per_epoch} steps/epoch x {a.epochs}, batch {a.batch}x{world}, aug={bool(a.aug)}, ema={a.ema}")

def augment(x):
    if not a.aug: return x
    B, _, H, W = x.shape
    s = torch.empty(B, device=dev).uniform_(0.85, 1.0)
    theta = torch.zeros(B, 2, 3, device=dev); theta[:, 0, 0] = s; theta[:, 1, 1] = s
    theta[:, 0, 2] = (torch.rand(B, device=dev) * 2 - 1) * (1 - s); theta[:, 1, 2] = (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
    grid = F.affine_grid(theta, (B, 3, H, W), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    c = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.9, 1.1); br = torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.1, 0.1)
    return (x * c + br).clamp(-1, 1)

@torch.no_grad()
def evaluate():
    head.eval(); se = torch.zeros((), device=dev); n = torch.zeros((), device=dev); per_dim = torch.zeros(out_dim, device=dev)
    for i in range(0, va_idx.numel(), a.batch):
        idx = va_idx[i:i + a.batch]; x = to_float(x_u8[idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = (ema_net(x) if a.encoder == "conv" else predict(x)) if ema_net is not None else predict(x)
        d = (pred.float() - target(z[idx])).pow(2); se += d.sum(); n += d.numel(); per_dim += d.sum(0)
    if ddp:
        dist.all_reduce(se); dist.all_reduce(n); dist.all_reduce(per_dim)
    head.train(); mse = (se / n).item(); pd = per_dim / (n / out_dim)
    return mse, pd

gstep = 0; hist = []; best_mse = float("inf")
for ep in range(a.epochs):
    head.train(); perm = tr_idx[torch.randperm(tr_idx.numel(), device=dev)]; run = 0.0; rn = 0; t0 = time.time()
    for s in range(steps_per_epoch):
        idx = perm[s * a.batch:(s + 1) * a.batch]; x = augment(to_float(x_u8[idx]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if a.encoder == "conv":
                pred = net(x)
            else:
                with torch.no_grad():
                    cls, pt = feats(x)
                pred = net(cls.float(), pt.float())
        loss = F.mse_loss(pred.float(), target(z[idx]))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sched.step(); gstep += 1
        run += loss.item(); rn += 1
        if ema_net is not None:
            with torch.no_grad():
                for pe, pm in zip(ema_net.parameters(), head.parameters()): pe.lerp_(pm, 1 - a.ema)
                for be, bm in zip(ema_net.buffers(), head.buffers()): be.copy_(bm)
        if gstep % a.log_every == 0:
            log(f"  ep {ep + 1} step {s + 1}/{steps_per_epoch}  mse {run / rn:.4f}  lr {sched.get_last_lr()[0]:.2e}  {(s + 1) * a.batch * world / (time.time() - t0):.0f} img/s")
    mse, pd = evaluate()
    hist.append({"epoch": ep + 1, "train_mse": run / max(rn, 1), "val_mse": mse, "val_r2": 1 - mse})
    log(f"epoch {ep + 1}/{a.epochs}  train_mse={run / max(rn, 1):.4f}  val_mse={mse:.4f}  val_R2={1 - mse:.4f}  "
        f"per-dim R2 min/median/max {1 - pd.max().item():.3f}/{1 - pd.median().item():.3f}/{1 - pd.min().item():.3f}  [{(time.time() - t0) / 60:.1f} min]")
    if is_main and mse <= best_mse:                      # the FILE must hold the best weights, not the last epoch's
        best_mse = mse
        torch.save({"head": (ema_net if ema_net is not None else head).state_dict(), "head_kwargs": head_kwargs,
                    "conv_kwargs": conv_kwargs, "encoder": a.encoder, "kind": kind, "W": W0, "b": b0, "mu": mu.cpu(), "sd": sd_.cpu(),
                    "r2_val": 1 - mse, "per_dim_r2": (1 - pd).cpu(), "assignment": a.assignment, "base_ckpt": a.base_ckpt, "args": vars(a), "hist": hist},
                   os.path.join(a.out, "inverse.pt"))
        log(f"  saved (best so far, val_mse {mse:.4f}, R2 {1 - mse:.4f})")
    if is_main:
        json.dump(hist, open(os.path.join(a.out, "curve.json"), "w"))
if ddp: dist.barrier()
log("done")
