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
ap.add_argument("--z-bottleneck", type=int, default=0, help="flat z: hard rank-r linear bottleneck before the 8x8 stem (0 = off)")
ap.add_argument("--z-skip", choices=["none", "full", "lowrank"], default="none", help="z bypass into every up-stage: none | full (unrestricted, gate init 1) | lowrank (shared rank-k, gate init 0.1)")
ap.add_argument("--z-skip-rank", type=int, default=32)
ap.add_argument("--z-pre-depth", type=int, default=0, help="nonlinear compressor before the rank-r code: number of SiLU MLP layers (0 = linear)")
ap.add_argument("--z-pre-width", type=int, default=512)
ap.add_argument("--cond-dim", type=int, default=512)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=32, help="per GPU")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--warmup", type=int, default=2000)
ap.add_argument("--min-lr-frac", type=float, default=0.0, help="cosine floor as a fraction of --lr (1.0 = constant LR after warmup)")
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
ap.add_argument("--eval-assigned", type=int, default=0, help="1: also report the ON-ANCHOR FID (G at a fixed subset of assigned training z) at every eval")
ap.add_argument("--log-every", type=int, default=200)
ap.add_argument("--compile", action="store_true")
ap.add_argument("--no-amp", action="store_true")
ap.add_argument("--decode-workers", type=int, default=0, help="0 = cpu_count // world_size")
ap.add_argument("--resume", default=None, help="checkpoint path, or 'auto' = latest gen_ep*.pt under --out")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--vit-dim", type=int, default=768); ap.add_argument("--vit-depth", type=int, default=12); ap.add_argument("--vit-heads", type=int, default=12)
ap.add_argument("--vit-ztokens", type=int, default=8); ap.add_argument("--vit-patch", type=int, default=16)
ap.add_argument("--vit-mode", choices=["self", "cross"], default="self", help="--arch vit: z tokens prepended (self) or cross-attended in every block (cross)")
ap.add_argument("--arch", choices=["grid", "residual", "vit", "hybrid"], default="grid",
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
ap.add_argument("--ts-loss", choices=["none", "mmd", "swd"], default="none",
                help="NON-adversarial two-sample loss between G(z_fresh) and G(z_assigned) [detached] in frozen DINOv2 feature space "
                     "(user idea 2026-09-10): 'mmd' = RBF-mixture MMD^2, 'swd' = sliced Wasserstein-2. Only the fresh branch gets gradient.")
ap.add_argument("--ts-weight", type=float, default=1.0, help="two-sample loss weight (x adaptive unless --ts-fixed)")
ap.add_argument("--ts-fixed", action="store_true", help="use --ts-weight as a fixed multiplier")
ap.add_argument("--ts-floor", type=int, default=1, help="1: subtract the finite-sample floor = EMA of the same statistic between the current and the "
                                                        "previous step's assigned batches (independent), clamp at 0 -> optimise only until indistinguishable")
ap.add_argument("--ts-floor-mult", type=float, default=1.0, help="stop margin: clamp the loss at 0 once stat < mult x floor (fresh-vs-assigned can never reach the "
                                                                 "assigned-vs-assigned floor exactly: 28k discrete anchors vs a continuum)")
ap.add_argument("--ts-stop-steps", type=int, default=0, help=">0: FLOOR-STOP -- end training once the two-sample loss has been clamped at 0 (stat below the floor margin) on at least "
                                                              "--ts-stop-frac of the last N steps (the stat hovers AT the floor, so consecutive runs never happen); the checkpoint saved then is the unpicked endpoint")
ap.add_argument("--ts-stop-frac", type=float, default=0.8)
ap.add_argument("--fresh-local-k", type=int, default=0, help="LATENT-LOCAL matching (user idea 2026-09-25): >0 = for each fresh z use its k nearest ASSIGNED z "
                                                             "(whitened latent space; same class when conditional) as the anchor side of the MMD / critic instead of a random assigned batch")
ap.add_argument("--fresh-local-mode", choices=["nbhd", "persample", "nearest"], default="nbhd",
                help="'nbhd': per step --fresh-local-nb neighbourhood centres c~N(0,I); assigned side = k nearest assigned z to c, fresh side = k nearest points to c of a fresh "
                     "N(0,I) pool of the SAME size as the assignment (same class when conditional) -> both sides local to the same neighbourhood, statistic per neighbourhood "
                     "(MMD) or one neighbourhood per critic batch; floor = same statistic between two independent fresh pools. "
                     "'persample': each fresh z vs the kernel distribution of its own k nearest anchors (MMD) / its nearest anchor as the critic's real sample. "
                     "'nearest' (MMD, middle ground): batch-level MMD between G(z_fresh) and G(nearest anchor of each z_fresh) -- local pairing, pooled statistic; "
                     "floor = same statistic with the batch's anchors as queries vs their nearest OTHER anchor")
ap.add_argument("--fresh-critic-pair", type=int, default=0, help="PAIRWISE local critic (user idea 2026-09-28): critic input = channel-concat [image, G(nearest anchor of its z)]; "
                                                                  "fake pair = [G(z_fresh), G(nn(z_fresh))], real pair = [G(z_anchor), G(nearest OTHER anchor)]; requires --fresh-local-mode nearest --fresh-local-k 1; "
                                                                  "gradient only through the fresh image; same PatchGAN with 6 input channels")
ap.add_argument("--fresh-local-nb", type=int, default=4, help="neighbourhoods per step per rank in 'nbhd' mode (nb*k samples per side)")
ap.add_argument("--ts-fresh-mult", type=int, default=1, help="k: k fresh batches per step; the assigned side = the current + previous k-1 assigned batches "
                                                              "(feature history), the floor = stat between two disjoint k-batch histories -> larger n, lower floor, finer match")
ap.add_argument("--ts-features", choices=["cls", "cls+patch"], default="cls", help="DINO features: CLS token, or CLS ++ mean patch token")
ap.add_argument("--ts-slices", type=int, default=256, help="random directions for swd")
ap.add_argument("--ts-gather", type=int, default=1, help="1: all_gather features across ranks (32 -> 256 samples per side) before the statistic")
ap.add_argument("--fresh-gan-r1", type=float, default=0.0, help="R1 gradient penalty (gamma) on the fresh critic's 'real' side, lazy every --r1-every steps")
ap.add_argument("--r1-every", type=int, default=16)
ap.add_argument("--fresh-critic-source", choices=["fresh", "assigned", "gen_assigned"], default="fresh",
                help="what the fresh critic is TRAINED on as fakes: 'fresh' = G(z~N(0,I)) (default); 'assigned' = the supervised batch's "
                     "outputs G(z_i) only (user idea 2026-09-09: a generic realism boundary the generator is then pushed with at fresh z); "
                     "'gen_assigned' = NO real images: critic separates G(z_assigned) ['real'] from G(z_fresh) ['fake'] distributionally and only "
                     "the fresh branch is updated, so off-anchor outputs become indistinguishable from supervised ones (user idea 2026-09-09)")
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
if a.fresh_local_k > 0:
    # every rank needs ALL anchors for the neighbour search (the training shard is 1/world of them)
    z_knn = z_all.to(dev, torch.float16).contiguous(); y_knn = lab_all.to(dev)
    if n_classes > 0:
        _ord = torch.argsort(y_knn); _cls_cnt = torch.bincount(y_knn, minlength=int(y_knn.max()) + 1); _off = torch.cat([torch.zeros(1, device=dev, dtype=torch.long), _cls_cnt.cumsum(0)])
    else:
        _ord = _off = None
    def knn_assigned(zq, yq, k, exclude_self=None):
        """indices (B, k) of the k nearest assigned z to each query (same class when labels exist). exclude_self: (B,) row ids to drop."""
        out = torch.empty(zq.shape[0], k, device=dev, dtype=torch.long); zq16 = zq.to(torch.float16)
        for i in range(zq.shape[0]):
            if _ord is not None:
                c = int(yq[i]); rows = _ord[_off[c]:_off[c + 1]]; cand = z_knn[rows]
            else:
                rows = None; cand = z_knn
            d2 = torch.cdist(zq16[i:i + 1].float(), cand.float()).squeeze(0) if cand.shape[0] <= 65536 else torch.cat([torch.cdist(zq16[i:i + 1].float(), cand[j:j + 65536].float()).squeeze(0) for j in range(0, cand.shape[0], 65536)])
            if exclude_self is not None:
                own = exclude_self[i]
                if rows is not None:
                    hit = (rows == own).nonzero(as_tuple=True)[0]
                    if hit.numel(): d2[hit] = float("inf")
                else:
                    d2[own] = float("inf")
            nn_ = torch.topk(d2, k, largest=False).indices
            out[i] = rows[nn_] if rows is not None else nn_
        return out
    def local_fresh(c, yc, k, n_pools):
        """for each centre c_i: draw n_pools independent fresh N(0,I) pools of the same size as the (class-)assignment and return
        the k nearest pool points to c_i from each pool -> list of n_pools tensors (nb*k, dim_z). Same locality notion (kNN rank) as the anchors."""
        outs = [[] for _ in range(n_pools)]
        for i in range(c.shape[0]):
            P = int(_cls_cnt[int(yc[i])]) if _ord is not None else z_knn.shape[0]
            for j in range(n_pools):
                pool = torch.randn(P, c.shape[1], device=dev, generator=fresh_gen)
                d2 = torch.cdist(c[i:i + 1], pool).squeeze(0)
                outs[j].append(pool[torch.topk(d2, k, largest=False).indices])
        return [torch.cat(o, 0) for o in outs]
    log(f"latent-local matching on: mode={a.fresh_local_mode} k={a.fresh_local_k}" + (f" nb={a.fresh_local_nb}" if a.fresh_local_mode == "nbhd" else "") +
        f" (class-restricted={_ord is not None}); anchors = k nearest assigned z, fresh = k nearest of a fresh pool of the same size; critic architecture unchanged")
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
elif a.arch == "vit":
    from aag.vit_generator import ViTGenerator
    assert n_classes == 0, "--arch vit is unconditional"
    make = lambda: ViTGenerator(dim_z, image_size=DATASETS[a.dataset]["image_size"], patch=a.vit_patch, dim=a.vit_dim, depth=a.vit_depth,
                                heads=a.vit_heads, z_bottleneck=a.z_bottleneck, n_ztok=a.vit_ztokens, mode=a.vit_mode).to(dev)
elif a.arch == "hybrid":
    from aag.hybrid_generator import HybridGenerator
    # --vit-patch doubles as the token grid (16 -> 16x16 tokens); --vit-dim/depth/heads size the trunk; --width/--n-res size the conv upsampler;
    # class-conditional via one class token in the trunk
    make = lambda: HybridGenerator(dim_z, image_size=DATASETS[a.dataset]["image_size"], tok_grid=a.vit_patch, dim=a.vit_dim, depth=a.vit_depth,
                                   heads=a.vit_heads, z_bottleneck=a.z_bottleneck, n_ztok=a.vit_ztokens, width=a.width, n_res=a.n_res,
                                   n_classes=n_classes).to(dev)
else:
    make = lambda: Generator256(dim_z, grid=grid, image_size=DATASETS[a.dataset]["image_size"], n_classes=n_classes,
                                cond_dim=a.cond_dim, width=a.width, n_res=a.n_res,
                                z_bottleneck=a.z_bottleneck, z_skip=a.z_skip, z_skip_rank=a.z_skip_rank,
                                z_pre_depth=a.z_pre_depth, z_pre_width=a.z_pre_width).to(dev)
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
dino = DinoFeatures().to(dev) if (a.dino_weight > 0 or a.ts_loss != "none") else None

class TwoSample:
    """Two-sample statistic between fresh-z and assigned-z generations in frozen DINO feature space, with a finite-sample floor.
    Features are standardised by an EMA of the assigned-batch mean/std. mmd: biased (V-statistic) MMD^2 with an RBF mixture whose
    bandwidths follow the median pairwise distance of the assigned batch; swd: sliced W2^2 over --ts-slices random unit directions.
    floor: EMA (0.99) of the same statistic between the current assigned batch and the previous step's assigned batch, i.e. two
    independent samples of the SAME distribution at the same n -> what "indistinguishable" looks like at this batch size."""
    def __init__(self):
        self.mu = self.sd = None; self.prev = None; self.floor = None; self.calls = 0
        from collections import deque
        self.buf = deque(maxlen=2 * a.ts_fresh_mult)  # gathered, standardised-later assigned features (detached), newest last
        self.gen = torch.Generator(device=dev).manual_seed(24680)  # identical on every rank -> identical statistic on every rank
    def feats(self, x):
        cls, patch = dino(x)
        f = cls if a.ts_features == "cls" else torch.cat([cls, patch.mean(1)], 1)
        return f.float()
    def _std(self, f_assigned):
        with torch.no_grad():
            m, sd = f_assigned.mean(0), f_assigned.std(0) + 1e-6
            if self.mu is None: self.mu, self.sd = m, sd
            else: self.mu.lerp_(m, 0.01); self.sd.lerp_(sd, 0.01)
    def stat(self, f1, f2):
        if a.ts_loss == "mmd":
            with torch.no_grad():
                d2 = torch.cdist(f2, f2).pow(2); med = d2[d2 > 0].median().clamp_min(1e-6)
            def k(x, y):
                d = torch.cdist(x, y).pow(2)
                return sum(torch.exp(-d / (med * s_)) for s_ in (0.5, 1.0, 2.0)) / 3
            return k(f1, f1).mean() + k(f2, f2).mean() - 2 * k(f1, f2).mean()
        dirs = F.normalize(torch.randn(f1.shape[1], a.ts_slices, device=f1.device, generator=self.gen), dim=0)
        p1 = (f1 @ dirs).sort(0).values; p2 = (f2 @ dirs).sort(0).values
        return (p1 - p2).pow(2).mean()
    def _mmd_groups(self, x, y, k, med):
        """x, y: (nb, k, D) -> per-neighbourhood biased MMD^2 with the RBF mixture, mean over neighbourhoods"""
        def kern(p, q):
            d = torch.cdist(p, q).pow(2); return sum(torch.exp(-d / (med * s_)) for s_ in (0.5, 1.0, 2.0)) / 3
        return (kern(x, x).mean(dim=(1, 2)) + kern(y, y).mean(dim=(1, 2)) - 2 * kern(x, y).mean(dim=(1, 2))).mean()

    def neighbourhood(self, pred_f, pred_a, null_ab, k):
        """Per-neighbourhood two-sample statistic: fresh side (k nearest fresh-pool points to the centre) vs assigned side (k nearest anchors),
        MMD^2 computed inside each neighbourhood and averaged (no pooling across neighbourhoods). Floor = same statistic between two independent
        fresh pools of the same neighbourhoods (exact null: anchors distributed like the fresh prior)."""
        ff = self.feats(pred_f); nb = ff.shape[0] // k
        with torch.no_grad():
            fa = self.feats(pred_a); fn = [self.feats(p) for p in null_ab]
        with torch.autocast("cuda", enabled=False):
            ff = ff.float(); fa = fa.float(); self._std(fa)
            ff = ((ff - self.mu) / self.sd).view(nb, k, -1); fa = ((fa - self.mu) / self.sd).view(nb, k, -1)
            with torch.no_grad():
                d2 = torch.cdist(fa.reshape(nb * k, -1), fa.reshape(nb * k, -1)).pow(2); med = d2[d2 > 0].median().clamp_min(1e-6)
            raw = self._mmd_groups(ff, fa, k, med)
            fl = torch.zeros((), device=ff.device)
            if a.ts_floor and len(fn) == 2:
                with torch.no_grad():
                    f1 = ((fn[0].float() - self.mu) / self.sd).view(nb, k, -1); f2 = ((fn[1].float() - self.mu) / self.sd).view(nb, k, -1)
                    f0 = self._mmd_groups(f1, f2, k, med)
                self.floor = f0 if self.floor is None else self.floor.lerp(f0, 0.01)
                fl = self.floor * a.ts_floor_mult
        self.calls += 1
        return (raw - fl).clamp_min(0), raw.detach(), fl

    def persample(self, pred_f, pred_nn_all, k, pred_floor_q=None, pred_floor_nn=None):
        """Per-sample local MMD^2: each fresh output vs the kernel distribution of its own k neighbours' outputs
        (RBF mixture in standardised DINO space). Floor: the same statistic for anchors vs THEIR k nearest anchors (no grad)."""
        ff = self.feats(pred_f); B = ff.shape[0]
        with torch.no_grad():
            fa = self.feats(pred_nn_all).view(B, k, -1)
        with torch.autocast("cuda", enabled=False):
            ff = ff.float(); fa = fa.float(); self._std(fa.reshape(B * k, -1))
            ff = (ff - self.mu) / self.sd; fa = (fa - self.mu) / self.sd
            with torch.no_grad():
                d2 = torch.cdist(fa.reshape(B * k, -1), fa.reshape(B * k, -1)).pow(2); med = d2[d2 > 0].median().clamp_min(1e-6)
            def kern(x, y):  # x (B,1,D) vs y (B,k,D) -> (B,k) ; or (B,k,D) vs (B,k,D) -> (B,k,k)
                d = torch.cdist(x, y).pow(2); return sum(torch.exp(-d / (med * s_)) for s_ in (0.5, 1.0, 2.0)) / 3
            cross = kern(ff.unsqueeze(1), fa).mean(dim=(1, 2))                       # mean_j k(f, a_j)
            within = kern(fa, fa); within = (within.sum(dim=(1, 2)) - within.diagonal(dim1=1, dim2=2).sum(1)) / max(k * (k - 1), 1)
            raw = (1.0 - 2 * cross + within).mean()
            fl = torch.zeros((), device=ff.device)
            if a.ts_floor and pred_floor_q is not None:
                with torch.no_grad():
                    fq = (self.feats(pred_floor_q).float() - self.mu) / self.sd; fn = ((self.feats(pred_floor_nn).float() - self.mu) / self.sd).view(B, k, -1)
                    c0 = kern(fq.unsqueeze(1), fn).mean(dim=(1, 2)); w0 = kern(fn, fn); w0 = (w0.sum(dim=(1, 2)) - w0.diagonal(dim1=1, dim2=2).sum(1)) / max(k * (k - 1), 1)
                    f0 = (1.0 - 2 * c0 + w0).mean()
                self.floor = f0 if self.floor is None else self.floor.lerp(f0, 0.01)
                fl = self.floor * a.ts_floor_mult
        self.calls += 1
        return (raw - fl).clamp_min(0), raw.detach(), fl

    def __call__(self, pred_f, pred_assigned_detached, floor_pair=None):
        ff = self.feats(pred_f)
        with torch.no_grad():
            fa = self.feats(pred_assigned_detached)
            fp = [self.feats(x) for x in floor_pair] if floor_pair is not None else None
        if a.ts_gather and world > 1:
            # ranks can hold a PARTIAL last batch of different sizes (per-rank splits are not multiples of the batch);
            # all_gather needs equal shapes, so truncate every rank to the common minimum first (identical decision on all ranks)
            import torch.distributed.nn.functional as dnf
            n_min = torch.tensor([fa.shape[0]], device=ff.device); dist.all_reduce(n_min, op=dist.ReduceOp.MIN); n_min = int(n_min.item())
            ff, fa = ff[: n_min * a.ts_fresh_mult], fa[:n_min]  # fresh side holds k x the assigned rows
            ff = torch.cat(dnf.all_gather(ff), 0)
            with torch.no_grad():
                fa = torch.cat(dnf.all_gather(fa), 0)
                if fp is not None: fp = [torch.cat(dnf.all_gather(x[:n_min]), 0) for x in fp]
        with torch.autocast("cuda", enabled=False):
            ff = ff.float(); fa = fa.float()
            self._std(fa)
            k = a.ts_fresh_mult
            self.buf.append(fa.detach())
            hist = list(self.buf)
            fa_side = torch.cat(hist[-k:], 0) if len(hist) >= k else None
            fl = torch.zeros((), device=ff.device)
            if fa_side is None or fa_side.shape[0] != ff.shape[0]:
                # history not full yet (or ragged last batch): no two-sample term this step
                self.calls += 1
                return torch.zeros((), device=ff.device, requires_grad=False) + 0 * ff.sum(), torch.zeros((), device=ff.device), fl
            ff = (ff - self.mu) / self.sd; fa_side = (fa_side - self.mu) / self.sd
            raw = self.stat(ff, fa_side)
            if a.ts_floor and fp is not None:
                # matched null: the batch's anchors as queries vs their nearest OTHER anchor, same construction as fresh -> nearest anchor
                with torch.no_grad():
                    f0 = self.stat((fp[0].float() - self.mu) / self.sd, (fp[1].float() - self.mu) / self.sd)
                self.floor = f0 if self.floor is None else self.floor.lerp(f0, 0.01); fl = self.floor * a.ts_floor_mult
            elif a.ts_floor:
                if len(hist) == 2 * k and all(h.shape == hist[0].shape for h in hist):
                    with torch.no_grad():
                        A = (torch.cat(hist[:k], 0) - self.mu) / self.sd; B = (torch.cat(hist[k:], 0) - self.mu) / self.sd
                        f0 = self.stat(A, B)
                    self.floor = f0 if self.floor is None else self.floor.lerp(f0, 0.01)
                if self.floor is not None: fl = self.floor * a.ts_floor_mult
        self.calls += 1
        return (raw - fl).clamp_min(0), raw.detach(), fl
ts = TwoSample() if a.ts_loss != "none" else None

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=a.wd)
adv = a.gan_weight > 0
fadv = a.fresh_gan_weight > 0
disc = opt_d = fdisc = opt_fd = None
if adv or fadv or ts is not None:
    from aag.discriminator import NLayerDiscriminator, paired_batch, paired_d_loss, paired_g_loss, adaptive_weight, hinge_d_loss, g_loss_from
if ts is not None and not fadv:
    fresh_gen = torch.Generator(device=dev).manual_seed(8765 + rank)
if adv:
    disc = NLayerDiscriminator(6, a.gan_ndf, a.gan_layers).to(dev)
    opt_d = torch.optim.Adam(disc.parameters(), lr=a.gan_lr, betas=(0.5, 0.9))
    pair_gen = torch.Generator(device=dev).manual_seed(4321 + rank)
if fadv:
    fdisc = NLayerDiscriminator(6 if a.fresh_critic_pair else 3, a.fresh_gan_ndf, a.fresh_gan_layers).to(dev)
    if a.fresh_critic_pair: assert a.fresh_local_mode == "nearest" and a.fresh_local_k == 1, "--fresh-critic-pair needs --fresh-local-mode nearest --fresh-local-k 1"
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
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
start_epoch, gstep = 0, 0
curve = {"epoch": [], "train_mse": [], "train_lpips": [], "val_mse": [], "val_lpips": [], "fid": [], "fid_assigned": []}
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
            return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
    else:
        opt.load_state_dict(R["opt"])
    if adv and "disc" in R:
        disc.load_state_dict(R["disc"]); opt_d.load_state_dict(R["opt_d"])
    if fadv and "fdisc" in R:
        fdisc.load_state_dict(R["fdisc"]); opt_fd.load_state_dict(R["opt_fd"])
    start_epoch, gstep, curve = R["epoch"], R["gstep"], R["curve"]
    log(f"resumed {a.resume}: epoch {start_epoch}, step {gstep:,}, last val_mse {curve['val_mse'][-1] if curve['val_mse'] else None}"
        + (f"; fresh schedule: lr {a.lr}, warmup {a.warmup}, {total_steps:,} steps, min_lr_frac {a.min_lr_frac}" if a.reset_schedule else ""))
raw = model
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    if adv:
        disc = torch.nn.parallel.DistributedDataParallel(disc, device_ids=[local])
    if fadv:
        fdisc = torch.nn.parallel.DistributedDataParallel(fdisc, device_ids=[local])
if ts is not None:
    log(f"two-sample loss on: {a.ts_loss} in DINO {a.ts_features} space, weight={a.ts_weight} ({'fixed' if a.ts_fixed else 'x adaptive'}), "
        f"floor={f'{a.ts_floor_mult}x EMA of stat(assigned_t, assigned_t-1), clamp 0' if a.ts_floor else 'off'}, gather={bool(a.ts_gather)}, fresh_mult={a.ts_fresh_mult}; G(z_fresh) vs G(z_assigned).detach(), no real images, no critic")
if fadv:
    log(f"fresh-z adversary on: weight={a.fresh_gan_weight} ({'fixed' if a.fresh_gan_fixed else 'x adaptive'}), critic ndf={a.fresh_gan_ndf} "
        f"n_layers={a.fresh_gan_layers} ({sum(p.numel() for p in fdisc.parameters()) / 1e6:.1f}M), lr={a.gan_lr}; "
        f"critic trained on {'G(z_assigned) vs G(N(0,I)) -- no real images' if a.fresh_critic_source == 'gen_assigned' else 'real vs ' + ('G(z_assigned) [supervised batch]' if a.fresh_critic_source == 'assigned' else 'G(N(0,I))')}; generator pushed at fresh z only"
        + (f"; R1 gamma={a.fresh_gan_r1} every {a.r1_every} steps" if a.fresh_gan_r1 > 0 else ""))
if adv:
    log(f"pairwise adversary on: weight={a.gan_weight} ({'fixed' if a.gan_fixed else 'x adaptive'}), critic ndf={a.gan_ndf} "
        f"n_layers={a.gan_layers} ({sum(p.numel() for p in disc.parameters()) / 1e6:.1f}M), lr={a.gan_lr}")
fwd = torch.compile(model) if a.compile else model

log(f"generator: {n_params / 1e6:.1f}M params  arch={a.arch}{' ch=' + str(a.ch) if a.arch == 'residual' else ''}{f' vit dim={a.vit_dim} depth={a.vit_depth} heads={a.vit_heads} ztokens={a.vit_ztokens} patch={a.vit_patch} mode={a.vit_mode}' if a.arch in ('vit', 'hybrid') else ''} width={a.width} n_res={a.n_res}"
    f"{' z_bottleneck=' + str(a.z_bottleneck) if a.z_bottleneck else ''}{' z_pre=' + str(a.z_pre_depth) + 'x' + str(a.z_pre_width) if a.z_pre_depth else ''}{' z_skip=' + a.z_skip + ('/k' + str(a.z_skip_rank) if a.z_skip == 'lowrank' else '') if a.z_skip != 'none' else ''}  batch {a.batch}x{world}={a.batch * world}  "
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


@torch.no_grad()
def fid_assigned(net):
    """On-anchor FID: G(z_i) for a FIXED subset of assigned training rows (first `per` of each rank's shard), same stats."""
    from aag.fid import get_activations, fid_from_stats
    ref = np.load(a.fid_stats)
    per = math.ceil(a.fid_n / world)
    # FIXED RANDOM subset of this rank's training rows (rows can be class-ordered, e.g. ImageNet: the first rows would cover few classes)
    idx = tr_idx[torch.randperm(tr_idx.numel(), device=dev, generator=torch.Generator(device=dev).manual_seed(4242 + rank))[:per]]
    acts = []
    for i in range(0, idx.numel(), a.batch):
        b = idx[i:i + a.batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            img = net(z[b], y[b] if n_classes else None)
        acts.append(torch.from_numpy(get_activations((img.float().clamp(-1, 1) + 1) / 2, dev, batch=b.numel())))
    acts = torch.cat(acts).to(dev)
    if ddp:
        n_min = torch.tensor([acts.shape[0]], device=dev); dist.all_reduce(n_min, op=dist.ReduceOp.MIN); acts = acts[: int(n_min.item())]
        gathered = [torch.empty_like(acts) for _ in range(world)]
        dist.all_gather(gathered, acts); acts = torch.cat(gathered)
    return fid_from_stats(ref["mu"], ref["sigma"], acts[:a.fid_n].cpu().numpy()) if is_main else None


t0 = time.time()
gstep_run0 = gstep    # throughput/ETA count from here, whatever step the resume landed on
from collections import deque
ts_clamped_win = deque(maxlen=max(a.ts_stop_steps, 1)); floor_stop = False
for epoch in range(start_epoch, a.epochs):
    model.train()
    perm = tr_idx[torch.randperm(tr_idx.numel(), device=dev)]
    run_mse, run_lp, run_n = 0.0, 0.0, 0
    run_dn = 0.0
    run_g = run_d = run_w = 0.0
    run_gf = run_df = run_wf = 0.0
    run_ts = run_tf = run_tw = 0.0
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
        if fadv or ts is not None:
            # fresh-z term: samples from N(0,I) (labels drawn from the batch when conditional) judged by an
            # unpaired critic against the real batch -- supervision exactly where the pairs give none
            kf = a.ts_fresh_mult if ts is not None else 1
            anchor_side = pred.detach().clamp(-1, 1)
            if a.fresh_local_k > 0 and a.fresh_local_mode == "nbhd":
                # LATENT-LOCAL (nbhd): nb neighbourhoods; both sides = the k points nearest the centre from each distribution
                k, nb = a.fresh_local_k, a.fresh_local_nb
                with torch.no_grad():
                    yc = y[b][:nb] if n_classes else None
                    c = torch.randn(nb, dim_z, device=dev, generator=fresh_gen)
                    nn_idx = knn_assigned(c, yc, k)                                                                 # (nb, k)
                    pools = local_fresh(c, yc, k, 3 if (ts is not None and a.ts_floor) else 1)
                    yl = yc.repeat_interleave(k) if n_classes else None
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        anchor_side = fwd(z_knn[nn_idx.flatten()].float(), y_knn[nn_idx.flatten()] if n_classes else None).float().clamp(-1, 1)
                        null_ab = [fwd(pz, yl).float().clamp(-1, 1) for pz in pools[1:]]                           # two independent fresh pools -> floor
                zf = pools[0]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    pred_f = fwd(zf, yl).float().clamp(-1, 1)
            else:
                zf = torch.randn(b.numel() * kf, dim_z, device=dev, generator=fresh_gen)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    pred_f = fwd(zf, y[b].repeat(kf) if n_classes else None).float().clamp(-1, 1)
            if a.fresh_local_k > 0 and a.fresh_local_mode == "persample":
                # LATENT-LOCAL (persample, method 1): each fresh z vs its own k nearest anchors (points within a comparison are unrelated);
                # MMD: kernel distribution of the k neighbours, floor = anchors vs their k nearest other anchors; critic: real side = each fresh z's NEAREST anchor
                yf = y[b].repeat(kf) if n_classes else None
                with torch.no_grad():
                    nn_idx = knn_assigned(zf, yf, a.fresh_local_k)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        pred_nn_all = fwd(z_knn[nn_idx.flatten()].float(), y_knn[nn_idx.flatten()] if n_classes else None).float().clamp(-1, 1)
                    if fadv:
                        anchor_side = pred_nn_all.view(zf.shape[0], a.fresh_local_k, *pred_nn_all.shape[1:])[:, 0]
                if ts is not None:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        q_idx = b[: zf.shape[0]] if b.numel() >= zf.shape[0] else b.repeat(2)[: zf.shape[0]]
                        nnq = knn_assigned(z[q_idx], y[q_idx] if n_classes else None, a.fresh_local_k, exclude_self=q_idx + lo)
                        pred_fq = fwd(z[q_idx], y[q_idx] if n_classes else None).float().clamp(-1, 1)
                        pred_fnn = fwd(z_knn[nnq.flatten()].float(), y_knn[nnq.flatten()] if n_classes else None).float().clamp(-1, 1)
            pair_ref_f = None
            if a.fresh_local_k > 0 and a.fresh_local_mode == "nearest":
                # LATENT-LOCAL (nearest, middle ground): assigned side = G(nearest anchor of each fresh z), pooled batch statistic;
                # with --fresh-critic-pair the critic sees [image || G(nearest anchor)] pairs (relational test, gradient only via the fresh image)
                assert kf == 1, "nearest mode needs --ts-fresh-mult 1"
                yf = y[b] if n_classes else None
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    nn1 = knn_assigned(zf, yf, 1)[:, 0]
                    anchor_side = fwd(z_knn[nn1].float(), y_knn[nn1] if n_classes else None).float().clamp(-1, 1)
                    nnq = knn_assigned(z[b], yf, 1, exclude_self=b + lo)[:, 0]
                    floor_pair = (pred.detach().clamp(-1, 1), fwd(z_knn[nnq].float(), y_knn[nnq] if n_classes else None).float().clamp(-1, 1))
                if a.fresh_critic_pair:
                    pair_ref_f = anchor_side                                                    # G(nn(z_fresh)), no grad
                    anchor_side = torch.cat([floor_pair[0], floor_pair[1]], 1)                  # real pair: [G(z_a) || G(nearest other anchor)]
        if ts is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                if a.fresh_local_k > 0 and a.fresh_local_mode == "nearest":
                    ts_loss, ts_raw, ts_fl = ts(pred_f, anchor_side, floor_pair)
                elif a.fresh_local_k > 0 and a.fresh_local_mode == "nbhd":
                    ts_loss, ts_raw, ts_fl = ts.neighbourhood(pred_f, anchor_side, null_ab, a.fresh_local_k)
                elif a.fresh_local_k > 0 and a.fresh_local_mode == "persample":
                    ts_loss, ts_raw, ts_fl = ts.persample(pred_f, pred_nn_all, a.fresh_local_k, pred_fq, pred_fnn)
                else:
                    ts_loss, ts_raw, ts_fl = ts(pred_f, anchor_side)
            ts_loss = ts_loss.float()
            wt = torch.tensor(a.ts_weight, device=dev) if a.ts_fixed else (adaptive_weight(loss, ts_loss, raw.out.weight) * a.ts_weight if ts_loss.item() > 0 else torch.zeros((), device=dev))
            total = total + wt * ts_loss
            if a.ts_stop_steps > 0:   # global mode: identical on every rank (gathered statistic); local modes: per-rank statistic -> synchronise the vote
                clamped = 1.0 if (ts_loss.item() == 0 and ts_fl.item() > 0) else 0.0
                if a.fresh_local_k > 0 and ddp:
                    cv = torch.tensor(clamped, device=dev); dist.all_reduce(cv, op=dist.ReduceOp.SUM); clamped = 1.0 if cv.item() >= 0.5 * world else 0.0
                ts_clamped_win.append(clamped)
                if len(ts_clamped_win) == a.ts_stop_steps and sum(ts_clamped_win) >= a.ts_stop_frac * a.ts_stop_steps:
                    floor_stop = True
        if fadv:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                # score fakes inside the same real||fake batch the critic is trained on: the critic has
                # BatchNorm, so a fake-only batch would be normalised with different statistics and the
                # generator would receive a critic signal unrelated to the one the critic was trained with
                real_side = anchor_side if a.fresh_critic_source == "gen_assigned" else tgt
                n_real = real_side.shape[0]; pred_f = pred_f[:n_real]
                crit_fake = torch.cat([pred_f, pair_ref_f[:n_real]], 1) if pair_ref_f is not None else pred_f
                g_adv_f = g_loss_from(fdisc(torch.cat([real_side, crit_fake], 0)).float()[n_real:])
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
            fake_d = pred.detach().clamp(-1, 1) if a.fresh_critic_source == "assigned" else crit_fake.detach()
            real_d = anchor_side if a.fresh_critic_source == "gen_assigned" else tgt
            do_r1 = a.fresh_gan_r1 > 0 and gstep % a.r1_every == 0
            if do_r1:
                # R1 on the 'real' side, inside the same real||fake forward (BatchNorm statistics unchanged), fp32, double backward
                real_d = real_d.detach().requires_grad_(True)
                lg = fdisc(torch.cat([real_d, fake_d], 0)).float()
                g_r1, = torch.autograd.grad(lg[: tgt.shape[0]].sum(), real_d, create_graph=True)
                r1 = g_r1.pow(2).sum(dim=[1, 2, 3]).mean() * (a.fresh_gan_r1 / 2) * a.r1_every
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    lg = fdisc(torch.cat([real_d, fake_d], 0)).float()
                r1 = None
            d_loss_f = hinge_d_loss(lg[: tgt.shape[0]], lg[tgt.shape[0]:])
            opt_fd.zero_grad(set_to_none=True)
            (d_loss_f + r1 if r1 is not None else d_loss_f).backward()
            opt_fd.step()
            run_gf += g_adv_f.item() * b.numel(); run_df += d_loss_f.item() * b.numel(); run_wf += float(wf) * b.numel()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), raw.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        gstep += 1
        if floor_stop:
            log(f"FLOOR-STOP: two-sample loss clamped on >= {a.ts_stop_frac:.0%} of the last {a.ts_stop_steps} steps at step {gstep} (epoch {epoch + 1}) -> evaluating and saving, then stopping")
            break
        run_mse += mse.item() * b.numel(); run_lp += perc.item() * b.numel(); run_n += b.numel(); run_dn += dn.item() * b.numel()
        if ts is not None:
            run_ts += float(ts_raw) * b.numel(); run_tf += float(ts_fl) * b.numel(); run_tw += float(wt) * b.numel()
        if is_main and gstep % a.log_every == 0:
            el = time.time() - t0
            print(f"  ep {epoch + 1} step {gstep - sched0:,}/{total_steps:,}  mse {run_mse / run_n:.5f}  lpips {run_lp / run_n:.4f}  "
                  + (f"g {run_g / run_n:.3f} d {run_d / run_n:.3f} w {run_w / run_n:.3g}  " if adv else "")
                  + (f"gf {run_gf / run_n:.3f} df {run_df / run_n:.3f} wf {run_wf / run_n:.3g}  " if fadv else "")
                  + (f"ts {run_ts / run_n:.4g} fl {run_tf / run_n:.4g} wt {run_tw / run_n:.3g}  " if ts is not None else "") +
                  f"lr {lr_at(gstep):.2e}  {(gstep - gstep_run0) * a.batch * world / el:,.0f} img/s  "
                  f"eta {(total_steps - (gstep - sched0)) * el / max(gstep - gstep_run0, 1) / 3600:.1f} h",
                  flush=True)
    tr_mse, tr_lp = all_reduce_mean(run_mse / max(run_n, 1), run_n), all_reduce_mean(run_lp / max(run_n, 1), run_n)
    curve["epoch"].append(epoch + 1); curve["train_mse"].append(tr_mse); curve["train_lpips"].append(tr_lp)
    if (a.eval_every > 0 and (epoch + 1) % a.eval_every == 0) or epoch + 1 == a.epochs or floor_stop:
        vm, vl = evaluate(ema)
        fid = fid_fresh(ema) if a.fid_stats else None
        fid_a = fid_assigned(ema) if (a.fid_stats and a.eval_assigned) else None
        curve["val_mse"].append(vm); curve["val_lpips"].append(vl); curve["fid"].append(fid); curve.setdefault("fid_assigned", []).append(fid_a)
        gates = f"  gates={[round(float(g), 3) for g in raw.skip_gate]}" if getattr(raw, "z_skip", "none") != "none" else ""
        log(f"epoch {epoch + 1}/{a.epochs}  train_mse={tr_mse:.5f} train_lpips={tr_lp:.4f}" + (f" train_dino={run_dn / max(run_n, 1):.4f}" if dino is not None else "") + gates + "  "
            f"heldout_mse={vm:.5f} heldout_lpips={vl:.4f}" + (f"  fid@{a.fid_n}={fid:.2f}" if fid is not None else "")
            + (f"  fid_assigned@{a.fid_n}={fid_a:.2f}" if fid_a is not None else "")
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
                        "cond_dim": a.cond_dim, "assignment": a.assignment, "arch": a.arch, "ch": a.ch,
                        "z_bottleneck": a.z_bottleneck, "z_skip": a.z_skip, "z_skip_rank": a.z_skip_rank,
                        "z_pre_depth": a.z_pre_depth, "z_pre_width": a.z_pre_width,
                        "vit_dim": a.vit_dim, "vit_depth": a.vit_depth, "vit_heads": a.vit_heads, "vit_ztokens": a.vit_ztokens,
                        "vit_patch": a.vit_patch, "vit_mode": a.vit_mode}, str(ck) + ".tmp")
            Path(str(ck) + ".tmp").replace(ck)
            (a.out / "curve.json").write_text(json.dumps(curve, indent=1))
            if floor_stop:
                (a.out / "FLOOR_STOP").write_text(f"epoch {epoch + 1} step {gstep} checkpoint {ck.name}\n")
    else:
        log(f"epoch {epoch + 1}/{a.epochs}  train_mse={tr_mse:.5f} train_lpips={tr_lp:.4f}" + (f" train_dino={run_dn / max(run_n, 1):.4f}" if dino is not None else "") + f"  [{(time.time() - t0) / 3600:.2f} h]")
    if ddp:
        dist.barrier()
    if floor_stop:
        if ddp: dist.barrier()
        log('training stopped by FLOOR-STOP'); break

log("done")
if ddp:
    dist.destroy_process_group()
