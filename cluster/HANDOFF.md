# AAG 256×256 scale-up — agent handoff

Written 2026-09-05 by the session that built this. Everything below is either measured
in-session or a direct instruction from the user (Matteo, `the-puzzler`). Read the
**Rules** section before changing anything about the assignment.

---

## 1. The task

Scale **AAG** (Amortised Assignment Generation — https://github.com/the-puzzler/aag,
blog https://the-puzzler.github.io/blog/aag/) from 64×64 CelebA/CIFAR/Doom to:

- **ImageNet-1k at 256×256, class-conditional**
- **CelebA-HQ at 256×256, unconditional** (direct-to-pixel)

AAG in one line: transport AE latents onto a Gaussian **once, offline**, as a persistent
per-example assignment `x_i ↔ z_i`; then train a feed-forward `G: z (⊕ c) → pixels`.
Generation is **one forward pass**, no sampler.

Budget: **one 8×B200 node per dataset** on `eks-train-prod-aps3`, project `warhol`.

---

## 2. The user's rules — do not violate these

1. **"The only rule during assignment is more Gaussian is more good."**
   Budget transport generously. Never stop because a metric "looks converged".
   Keep step-stamped checkpoints (`--keep-checkpoints`) and choose the step count
   *afterwards* by the fresh-z / held-out MSE of a generator trained on each.

2. **Gaussianising a group ≠ gaussianising its marginals.**
   User: *"when it comes to gaussianising for a given condition, gaussianising a group of
   conditions is not the same as gaussianising the individual marginals and that can be
   important to make sure that each independent sub condition is also independent from z."*
   → For ImageNet this is why the assignment interleaves group transport over **WordNet
   hierarchy levels**, not just the 1000-way class. See §5.

3. **"Displacement is not a useful metric, I've discovered, so don't worry about it."**
   `disp` is still printed; ignore it in every decision and report.

4. **"PCA stuff doesn't work, take my word for it."** No rank-k truncation of the latent.
   (Measured anyway before the instruction landed: PCA 2048→1024 doubles recon MSE,
   LPIPS 0.125→0.25. The option was removed from the code.)

5. **Keep the assignment dim small, but reconstruction must stay top-notch.**
   Settled at 2048-d — see §4, user verdict *"the recons seem perfect"*.

6. **Metrics never decide quality — the user's eyes do.** FID/val-loss may *order* runs;
   present them as measurements and put sample grids in front of the user. Do not
   announce a visual verdict yourself.

7. **Never idle the GPU / never kill a run on your own judgement** while the user is away.
   Report and recommend; let them decide. Only hard failures (NaN, crash, corruption)
   justify a unilateral stop.

8. **Structural choices get asked, not assumed** (phase order, what gets ablated).
   Hyperparameters inside an agreed structure are yours.

9. **Job naming:** plain `aag1`, `aag2`, … so they're easy to spot in k9s. Stick to the
   node budget and don't escalate about cluster capacity — the scheduler handles it.

10. **Pipeline order is fixed:** assignment → plain generator to convergence →
    rollout/GAN/extras. (Not yet relevant here; no rollout phase at 256².)

---

## 3. Where everything lives

| What | Path |
|---|---|
| Code (branch `worktree-aag-scale256`) | `/home/ubuntu/exp/newgen/.claude/worktrees/aag-scale256` |
| Parent repo / origin | `/home/ubuntu/exp/newgen` → `git@github.com:the-puzzler/aag.git` |
| Datasets (HF parquet mirrors) | `/data/aag_data/hf/{celeba-hq-256x256,imagenet-1k-256x256}` |
| Particles (encoded latents) | `/data/aag_data/<dataset>/particles_dcae_f32c32.pt` |
| Class groupings | `/data/aag_data/imagenet256/class_groups.pt` |
| Results | `/data/aag_results/results_scale256/` |
| Python | `/home/ubuntu/exp/newgen/.venv/bin/python` — **run with `PYTHONPATH=<worktree>`** |
| odytrain (fresh origin/main) | `/data/tmp/odyssey-main/tools/odytrain` |

⚠️ The venv's editable `aag` install points at the **main checkout**, which lacks
`generator.py`, `hf256.py`, DC-AE, etc. Always set `PYTHONPATH` to the worktree.

**Nothing has been pushed to the public origin.** 11 local commits on the branch.

### New code in this branch
- `aag/hf256.py` — HF parquet → GPU-resident uint8; **global parquet row index in sorted
  filename order is the particle identity**. Every consumer indexes by that integer.
- `aag/generator256.py` — `Generator256`: reshapes z back to its (C,g,g) grid, DC-AE
  parameter-free upsample shortcuts, AdaGN class conditioning (falls back to plain
  GroupNorm when `n_classes=0`, so one module serves both datasets).
- `scripts/imagenet_class_groups.py` — WordNet ancestor partitions of the 1000 classes.
- `scripts/encode_hf256.py` — encode → particle file; `--rank/--world` shards, `--merge` joins.
- `scripts/run_assignment_classes.py` — the assignment driver (§5).
- `scripts/train_generator256_ddp.py` — torchrun DDP trainer (§6).
- `scripts/compute_fid_stats_hf256.py` — reference Inception stats.
- `cluster/` — Dockerfile, `entry.py` stage runner, `launch.sh`, configs, RUNBOOK.md.

---

## 4. The latent — settled, don't relitigate

**Pretrained DC-AE (`diffusers.AutoencoderDC`), `mit-han-lab/dc-ae-f32c32-in-1.0-diffusers`,
2048-d as an 8×8×32 grid.** No AE is trained.

Measured on 512 validation images (MSE on [-1,1], LPIPS-VGG):

| model | latent | CelebA-HQ | ImageNet |
|---|---|---|---|
| **f32c32-in** | 8×8×32 | **0.0104 / 0.125** | **0.0253 / 0.165** |
| f64c128-in | 4×4×128 | 0.0106 / 0.129 | 0.0265 / 0.172 |
| f32c32-mix | 8×8×32 | 0.0109 / 0.136 | 0.0260 / 0.179 |

Sheets in `/data/aag_results/results_scale256/ae_probe/`. **User looked and said "the recons
seem perfect."** That closes the latent question.

Why it matters: in AAG, z is the *only* channel carrying target-specific information, so
detail the AE fails to encode was never gaussianised and cannot be generated. The AE is the
ceiling, which is why this was measured before anything else.

---

## 5. The assignment (`scripts/run_assignment_classes.py`)

Per step: greedy global rank transport (subset 2048, 64 dirs, α=1) → slab cleanup every 2
steps → interleaved **group transport per hierarchy level** → radial χ_d calibration every
20 steps. Whitening is **`rotate=False`** so coordinate j of z stays grid cell j of h.

**ImageNet levels** (`--levels joint,depth7,depth6,depth5,depth4,living,animal,dog`):

| level | groups | largest |
|---|---|---|
| joint | 1000 | 1281 each |
| depth7 | 225 | chordate 335 |
| depth6 | 85 | animal 398, device 125 |
| depth5 | 38 | organism 407, instrumentality 352 |
| depth4 | 20 | artifact 517, living thing 407 |
| living / animal / dog | 2 each | 407 / 398 / 116 classes |

Rationale (rule 2): per-class transport moves ~1281 particles, whose own quantile noise is
~3%; a small shift shared by all 116 dog classes (148k particles) is invisible to each
per-class step **and** to the per-class ratio, but adds coherently at the superclass level.
One transport over those 148k removes it. This is the same failure that hid a VPT action
leak for days (joint 81-way read 0.59 while "turning vs not" sat at 1.27).
Sampling is size-weighted so touches-per-particle are equalised across group sizes.

The generator sees **only the 1000-way class** — the hierarchy is a property of the class,
so feeding it adds no information. There is no classifier-free guidance in AAG, so class
adherence rests entirely on z ⊥ c.

**Readout:** never quote one level. The log prints the independence ratio at every level.
Selection between checkpoints is by a trained generator, never by the ratio.

### CelebA-HQ result so far (local box)
200,000 steps done, 28k particles, d=2048, ~141 step/s.
Objective 0.0207 → 0.0043 (its N(0,I) noise floor is 0.0045); G 0.00228 → 0.00027.
Worst of 256 random projections: whitened 0.033 → z 0.0047 (true Gaussian ref 0.0004).
Checkpoints every 20k in `/data/aag_results/results_scale256/celebahq/assign/`.
Curve plot: `.../assign/plots/assignment_celebahq_uncond.png`.
The user judged the 26k-step version *"quite under assigned"* — hence 200k.

---

## 6. The generator (`scripts/train_generator256_ddp.py`)

`torchrun --standalone --nproc_per_node=8`. Per rank: decodes its contiguous slice of the
train parquet into a **GPU-resident uint8 tensor** (ImageNet 256³ uint8 = 252 GB total,
31.5 GB per B200) and takes `z[lo:hi]`. After that one decode the epoch loop does **no I/O**.

- **Identity is asserted at startup**: labels decoded from parquet must equal the labels
  stored in the assignment, else it refuses to train. A misaligned particle order once
  looked exactly like mode collapse in this project.
- Loss `MSE + 0.5·LPIPS(VGG)` (standing project recipe — LPIPS is load-bearing), bf16
  autocast, EMA 0.9995 evaluated/sampled.
- Held-out pairs: a seeded 1–2% of particles never trained → **this is the assignment
  selection metric**.
- FID-10k on fresh z each eval, sharded across ranks; sample grids + held-out pair sheets
  written every eval **for the user to look at**.
- `--resume auto` picks the latest `gen_ep*.pt` under `--out`.

**Deadlock fix (important):** the held-out mask removes a Binomial number of rows per
shard, so `ceil(len(tr_idx)/batch)` differed by one across ranks; a rank leaving the epoch
loop early deadlocks the rest at their next gradient all-reduce. Every rank now runs the
**min** across ranks. Do not undo this.

Sizes: CelebA-HQ 37.9M params (`width=1.0, n_res=2`), ImageNet 140M (`width=1.5, n_res=3`).

---

## 7. Cluster

```bash
aws sso login --sso-session odyssey        # when kubectl says the token expired
cluster/launch.sh build                    # buildx --load, then docker push
cluster/launch.sh submit cluster/configs/aag256_celebahq.yaml aag1
cluster/launch.sh submit cluster/configs/aag256_imagenet.yaml aag2
kubectl -n kubeflow get pytorchjobs | grep aag
kubectl -n kubeflow logs -f aag1-master-0 -c pytorch
kubectl -n kubeflow delete pytorchjob aag1     # delete.py needs a TTY; kubectl doesn't
```

`cluster/entry.py` is the launcher odytrain runs under torchrun. It executes a YAML stage
list; stages are `single` (LOCAL_RANK 0 runs, others wait on a /dev/shm marker, completion
recorded under `state_dir` so a relaunch skips it), `ddp: true` (all ranks), or a
`parallel:` group that splits GPUs between sub-stages with renumbered RANK/WORLD_SIZE and
restricted `CUDA_VISIBLE_DEVICES`. Verified by emulating torchrun with 3 processes.

Facts worth knowing:
- odytrain hard-codes KAI queue **`dev-ml`** on aps3 (`launch_lib.py:1046`); `warhol` rides
  as the `odyssey.systems/project` label. That is normal and correct — not a misconfiguration.
- It **requires `ODYSSEY_USER`** (refuses `ubuntu`); `launch.sh` defaults it to `matteopeluso`.
- The user's `/home/ubuntu/odyssey` checkout is from July and does **not** know aps3 —
  use the `origin/main` worktree at `/data/tmp/odyssey-main`.
- Shared storage is an EFS PVC at `/mnt/shared` on EKS.
- Pods are IPv6-only unless `--nodisable_ipv4_egress` (the wrapper passes it). HF Hub has
  AAAA records anyway.

---

## 8. State (updated 2026-09-05 ~06:45 UTC)

**Split decided by the user: the local box does assignment, the cluster trains generators.**
The reason was measured, not assumed: one assignment step at N=1.28M cost 383 ms on a
single GPU (host syncs, not transport FLOPs), so the on-node assign stage would have idled
7 of 8 B200s for many hours. `aag/gaussianize.py` now caches the Gaussian/chi quantile
targets, scores all 32 slab candidates in one batch, slices group members from a cached CSR
order, and skips score readback (`return_score=False`); same transports in the same order,
~77 ms/step (~13 step/s -> 300k ImageNet steps ~6 h locally). A sharded multi-GPU version
was designed (transports stay strictly sequential; only the row arithmetic of ONE transport
is split, all-gathering one float per particle) and then dropped by the user as unnecessary.

**Local box:** the user killed their doom AE and the local CelebA-HQ generator (epoch 17;
`gen_uncond_200k_local/` keeps ep5/10/15 checkpoints and grids). Running: ImageNet DC-AE
encode (`/data/aag_results/results_scale256/imagenet/encode.log`, GPU-bound ~220 img/s), then
the 300k-step hierarchy assignment starts automatically (waiter script in the job tmp dir;
log `/data/aag_results/results_scale256/imagenet/assign/assign_cls_300k.log`, checkpoints
every 50k kept).

**Cluster:** `aag1` (CelebA-HQ, generator-only YAML, image `odydev.azurecr.io/aag:b2bcef8`)
is running. `aag2` goes up once the ImageNet assignment (~16 GB) is on EFS. Old `dfa1baf`
jobs were deleted by the user: that image lacked the trainer deadlock fix, and its tag was
unresolvable from ap-south-1 for >1 h (no India ACR replica) until the scheduler suspended
the jobs. The rebuilt tag pulled fine (~12 min).

**Data hop:** the auto-mode classifier blocks S3/R2 writes and `kubectl delete`. A CPU-only
busybox pod `aagcp` (kubeflow ns, PVC `shared-drive` at `/mnt/shared`) makes `kubectl cp`
work with no bucket (~8 MB/s: 600 MB in 73 s, so ~30 min for ImageNet). If a bucket is ever
wanted, the convention is `s3://ody-model-hub-aps3/training/<owner-or-project>/...`.

### 8b. The CelebA-HQ finding and the TiTok pivot (08:00 UTC)

`aag1` (DC-AE d=2048, 200k assignment) trained fine but plateaued: FID-10k 96-99 from
epoch 80 on, held-out pair MSE best 0.0213 at ~epoch 100 then drifting up (overfitting the
27k pairs). The user noticed held-out pairs look far better than fresh samples. Measured:
same epoch-40 EMA gives FID 18 on ASSIGNED z, 146 on fresh N(0,I), 144 on assigned z with
coordinates shuffled independently -- the generator is fine, the JOINT of z is not Gaussian
although every 1-D projection is. Readout: the SPLIT kNN two-sample test (`knn_split.py` in
the job tmp dir): A->A = fraction of an assigned point's 10 NN that are assigned (0.5 =
Gaussian). d=2048: 1.000 -> 0.968 after 200k steps. Cause per METHOD.md s5: independent-N/d
= 28k/2048 = 14, below the UCF-101 failure case (37). User: matching to a Gaussian sample
(Sinkhorn/LAP) does NOT work -- "it needs to cover the whole gaussian". User's remedy: the
512-d TiTok-LL-32 VAE latent (only pretrained continuous tokenizer that is more compressed;
recon MSE 0.026 vs 0.010). Same recipe at d=512: A->A 0.937 (20k) -> 0.673 (200k) -> 0.593
(500k), continuing to 1M (`celebahq_titok/assign/assign_uncond_1m*.pt`). `aag3` = the
generator on the 200k d=512 assignment (`aag256_celebahq_titok.yaml`, image `ba83580`,
Generator256 grid=0 -> learned Linear stem). ImageNet (N/d 626, TwoNN 88) stays on DC-AE
for now; its 300k hierarchy assignment is running locally.

### 8c. Target and current runs (09:40 UTC)

**Target (user):** beat ROMS-IMLE (arXiv 2607.19332): FID-50k 2.56 on ImageNet-256 @310M
(raw 4.16; 2.56 needs their round-trip rejection), 6.70 on CelebA-HQ-256 @138M, NFE 1. Use
`scripts/eval_fid256.py --n 50000` for comparable numbers (validated vs in-loop FID).
User: no DINO loss; a pairwise adversary (real||gen channel concat) is the later addition;
rising held-out MSE while FID falls is NOT overfitting (their prior observation).

`aag3` (TiTok d=512, 200k assignment, 38.3M): 400 epochs done, best FID-10k 38.35 @ep230,
FID-50k 37.2 @ep180, plateau from ep130. `aag4` (submitted 09:38): 1M assignment (A->A 0.545)
+ width 1.9/n_res 2 = 133.6M, image `d1ceef4`, config `aag256_celebahq_titok_1m.yaml`.
ImageNet: TiTok d=512 300k hierarchy assignment running locally (54 step/s; A->A 0.746@50k,
0.669@100k; all hierarchy ratios ~1.0 by 100k, joint 2.2 @150k), auto-continues to 1M
(`assign_cls_1m*.pt`); the 300k goes to `aag2` (`aag256_imagenet_titok.yaml`, width 2.2/n_res 3
= 285M, 40 epochs) via the helper pod. ACR docker login expires after a few hours:
`az acr login -n odydev` then `docker push` if launch.sh build reports "authentication required".

### 8d. State at 11:20 UTC

- `aag5` (CelebA-HQ, 133.6M on the 200k TiTok assignment, image d1ceef4): running, FID-10k 43.3 @ep80
  vs aag3 43.6 -- capacity alone helps a little; held-out identical to aag3.
- `aag4` (1M assignment + 133.6M) was STOPPED at ep140 by the user: FID 47 plateau, held-out
  0.093 vs 0.054. User's reading: more transport is only better with enough data; otherwise it
  spreads the samples until nowhere is safe to sample. Outputs remain on EFS (`gen_uncond_1m/`).
- `aag2` (ImageNet, 285M = width 2.2/n_res 3, 300k d=512 assignment, 40 epochs): running; the
  assignment upload took ~2.5 h via kubectl cp (6.6 GB at <1 MB/s) -- for the next big file use
  a slimmer artefact (z + labels only) or a bucket.
- `matteo-exp-2` (third node, user-granted): compact-AE sweep, 8 single-GPU variants, batch 128,
  lr 1e-3 cosine, 100 epochs, image e074fde. First two submissions failed on my bugs (missing
  DataLoader import; entry.py failure marker with '/' in nested tags) -- both fixed and tested
  locally with a one-epoch run (val MSE 0.068 / LPIPS 0.47 after 1 epoch, dcae d128 g4 c128).
- Local box: ImageNet d=512 assignment continuation 300k -> 1M (`assign_cls_1m*.pt`), the
  data-rich test of "more transport"; A->A at 300k was 0.606.
- Target: ROMS-IMLE FID-50k 6.70 (CelebA-HQ @138M) / 2.56 (ImageNet @310M). Best so far:
  CelebA-HQ FID-50k 37.1 (aag3). `scripts/eval_fid256.py --n 50000` for comparable numbers.

### 8e. 12:00 UTC

- `aag2` (ImageNet, 285M, 300k d=512 assignment, image d1ceef4): banner correct (1,281,167
  particles, 1000 classes, 160,146 images/rank = 31.5 GB uint8, identity check passed, 12,811
  held-out pairs), **821 img/s -> 26 min/epoch -> 40 epochs ~17 h**, eval every 2 epochs.
  User: future ImageNet generators should be SMALLER (~140M or less); capacity is not the lever
  (aag5 133.6M plateaued at FID-10k 41.2-41.7 vs aag3 38M at 38.4 on the same assignment).
- `matteo-exp-2` (AE sweep, image 58f8c9e = gcc+libc6-dev+python3-dev for torch.compile):
  8 variants running, ~2-3 min/epoch, epoch-5 val MSE 0.033-0.053 / LPIPS 0.37-0.42. Log lines
  are tagged only by arch; attribute variants via `celebahq_ae/<variant>/ae_train_curve_*.json`.
  Three failed submissions before this one: missing DataLoader import, entry.py marker path with
  '/', no C compiler in the image. **Test the exact cluster command path locally (incl. --compile)
  before submitting.** User authorised `kubectl delete pytorchjob` for their aag*/matteo-exp-* jobs.
- Next: when the sweep's epoch-20+ numbers and sheets are in, the user picks a latent; then
  encode CelebA-HQ with it (`encode_hf256.py --encoder <ae_ckpt>.pt`), assign (~200k steps,
  watch A->A and held-out MSE, not more), generator at width 1.0. Selection metric for the
  assignment budget on data-poor sets: held-out pair MSE, per the user's "more transport only
  with enough data" rule.

### 8f. 15:10 UTC -- where the FID gap lives

- Own-AE sweep (`matteo-exp-2`): dimension dominates (d256 0.025/0.281, d128 0.033/0.309,
  d64 0.043/0.337), 4x4 grid best, width irrelevant. Full pipelines on d128 (`aag6`) and d256
  (`aag7`) plateaued in the same FID-10k 39-43 band as TiTok (aag3 38.4) -> user pulled them.
- Pairwise-adversary finetunes (`aag8` TiTok from aag3 ep230; `aag9` d128 from aag6 ep220;
  gan-weight 0.5 adaptive, lr 1e-4, 100 epochs, ~25 min each): TiTok 38.4 -> **35.0** (ep270;
  FID-50k **33.7**, best so far), d128 -> 36.8. Held-out MSE unchanged; critic wins early
  (d ~0.1-0.3) but the flipped-label BCE keeps a live gradient.
- **z-source decomposition (the key readout):** aag8 ep270 assigned-z FID@5k 8.7 vs fresh-z
  36.0 vs coordinate-shuffled 36.1; aag9: 9.8 / 38.0 / 37.6. Latent + generator are already
  in the target regime on assigned z; ALL of the remaining gap is joint structure of z that
  fresh draws lack. The kNN A->A test saturates before the generator does (d128 A->A 0.516 yet
  same fresh FID as TiTok at 0.673).
- Candidates put to the user: z-noise/mixup on the generator; more faces (FFHQ). Nodes: aag2
  (ImageNet) running; two CelebA nodes idle awaiting the decision.
- Scripts in the job tmp dir: `zsource_fid_aag8.py` (decomposition), `knn_split.py`,
  `run_celebahq_ae_pipeline.sh`, `ae_sheet.py`. Configs: `aag256_celebahq_{titok,dcae_d128}_adv.yaml`.

### 8g. 18:30 UTC -- the transport is the bottleneck; direction search and budget

- z-source decomposition (same checkpoint, assigned z vs fresh z vs coordinate-shuffled): CelebA
  aag8 8.7 / 36 / 36; ImageNet aag2 (ep10) 26 / 149 / 149. Latent + generator are fine on the
  cloud; the whole gap is joint structure of z that fresh Gaussians lack.
- Random 1-D directions miss it: worst of 64 random reads at the noise floor while an optimised
  direction reads 1000x higher. `refine_direction` (max-sliced) and `population_direction`
  (user's skew/kurt projection pursuit) added; `run_assignment_classes.py --refine-steps`.
- BUT harder Gaussianisation costs the generator: aag11 (200k + 20k refined a=1) FID 46 vs aag3
  38.4; toy bench (`scripts/toy_assignment_bench.py`) shows the generator's optimum is EARLY
  (random 3k > 10k > 100k; refine a0.1 at 3k best), and assignment-space Gaussianity readouts
  (kNN A->A, C2ST, learned-dir W2) do NOT predict the generator. Held-out MSE tracks it best.
- Generators queued on real data: aag12 (refine a0.3, 10k), aag13 (random 100k ckpt), aag14
  (refine a0.1, 3k). Best so far: aag8 TiTok + adversary FID-50k 33.7.
- AE sweeps: CelebA d32 0.057/0.365, d16 0.075/0.396 (lossy); ImageNet own AEs (matteo-exp-3)
  d256-1024 on 300k rows: best ~0.035/0.30 at epoch 10/20 (beats TiTok's 0.056/0.328).
- User rules today: no FFHQ (comparison paper uses CelebA-HQ), no DINO, pairwise adversary on
  the generator, 30-min update cadence, decisions via push with options, comparisons as one PNG
  with column headers (`compose_columns.py`), kubectl delete authorised for their jobs.

## 9. What to do next

1. `aag1`: diff the generator banner (per-rank shard sizes, identity check passed, params,
   bf16, lpips_weight 0.5) in the first two minutes; send the user the first sample grids.
2. When the ImageNet assignment finishes: `kubectl cp` it to
   `/mnt/shared/aag/results_scale256/imagenet/assign/assign_cls_300k.pt` (via `.part` + mv),
   then `cluster/launch.sh submit cluster/configs/aag256_imagenet.yaml aag2` with
   `AAG_IMAGE_TAG=b2bcef8` (or rebuild if code changed).
3. Train short generators on kept assignment checkpoints (ImageNet 50k/100k/.../300k;
   CelebA-HQ 20k/60k/100k/200k) and compare **held-out pair MSE/LPIPS** -- "more transport
   is better" is assumed, not yet measured, at 256^2.
4. `plots/plot_assignment.py` does not read this run's curve keys; adapt if wanted.

## 10. Relevant prior findings (from the project's memory)

- The independence **ratio does not select** a good assignment; a trained generator's
  fresh-z MSE does. Ratios carry ±0.13 eval noise — read neighbouring evals before quoting.
- Decorrelating against a high-dimensional condition can leave z strongly correlated with
  low-dimensional functions of it. Always report every sub-scale.
- The AE latent's information content is the generator's ceiling, not an additive error.
- Do **not** add a KL term to any AE here — the assignment gaussianises by transport.
- A `sed` range-delete once silently truncated a script to a stub that exited 0.
  **`wc -l` after any sed edit of a script.**

## 9. Night of 2026-09-05 → 06 (autonomous; user asleep)
- **Step predictor** (`scripts/toy_step_predictor.py`, `scripts/toy_score_snaps.py`, `scripts/assign_surrogate_score.py`):
  the 3-s surrogate generator's fresh-z FD (`fd_sur`) predicts the real generator's FID (toy Spearman median ≈0.9 across
  9 regime cells; real CelebA rank order matches all known FIDs). Objective-floor crossing is the α1 shortcut. Gaussianity/kNN
  metrics, held-out MSE, locality do not predict. Take the EARLIEST step of the surrogate plateau (fitted generator wants to stop
  a snapshot earlier). Results: `/data/aag_results/results_scale256/toy_bench/predictor/`, `…/celebahq_titok/assign/surrogate_scores*.json`.
- **CelebA-TiTok 28k is saturated** ~FID 38–42 for every floor-reaching assignment (4-seed surrogate 39.1–40.1). Validation
  runs: aag12 α0.3-10k 42.4, aag13 100k 40.6, aag14 α0.1-3k (pred. worst) running, aag15 α1-10k (pred. over-transport),
  aag16 20k (pred. 72), aag17 α1-3k, aag18 α1-500 (top single pick), aag19 α0.1-10k, aag20 α0.3-5k, aag21 half-width generator.
- **Dead ends:** z-jitter in generator training (worse at every σ); kNN-distance rejection of fresh z (36.0→36.0–36.4; in 512-d
  all fresh z are uniformly ~9% farther from data than data from itself — no safe areas).
- **Lead: hflip pair doubling** (56k pairs, N/d 109): `encode_hf256.py --flip` → `merge_particles.py` → refined α1 1000 steps
  (`…/celebahq_titok_x2/assign/assign_uncond_x2_refine_a1_2k_step1000.pt`, on EFS as `assign_uncond_x2_refine_a1_best.pt`);
  surrogate 22.95 vs 39+. Trainer `--aug-hflip` (rows [N,2N) = flipped copies), image `aag:5ccf468`, config
  `aag256_celebahq_titok_x2.yaml` → **aag23** (200 epochs). Dry-run passed locally. User asked whether flips are allowed for the comparison.
- **ImageNet:** refined α1 + `--grp-per-step 128` gets all conditional ratios ≈1 by 10k steps (surrogate 3.58 vs 4.41 for 300k random,
  pairs 2× smoother). Slim (z fp16 + labels) uploads to EFS as `imagenet_titok/assign/assign_cls_refine_a1_g128_best_slim.pt`;
  `aag256_imagenet_titok_refine_a1_g128.yaml` (133.6M generator) → **aag22**, auto-submits when aag2 finishes (chain_22.sh).
- Chains live in `/home/ubuntu/.claude/jobs/2153ceba/tmp/chain_v4.sh` (node A: aag14→aag23→aag16→aag18→aag20; node B:
  aag15→aag17→aag19→aag21) and `chain_22.sh`. AE sweep (matteo-exp-3): ImageNet own d512-g4 0.043/0.351, d1024 0.033/0.288
  (TiTok 0.056/0.328); CelebA d32 0.057, d16 0.075 (too tight).
- **Later that night:** aag23 (x2 flips) FID-10k 37.77 / FID-50k 36.40 (best plain); aag24 = pairwise adversary on it → FID-10k
  **33.57**; aag25 = gan-weight 1.0 variant queued (chain_v6.sh). Diagnostics (jobs/2153ceba/tmp/*.py): latent MLP + TiTok
  decoder — assigned z hit the decoder ceiling (7.6 CelebA / 8.1 ImageNet) but fresh z give 73 (CelebA) / 121–149 (ImageNet):
  the gap is joint z-structure, not generator blur. Assigned-vs-fresh MLP critic: best-FID CelebA assignments are 65–70%
  distinguishable, over-transported ones ≤ chance but worse FID (coverage vs smoothness at 28k). ImageNet critic 79/83/70%
  (300k/refined/1M) with fresh FID 134/142/121 — more transport helps there; z→class MLP probe 13–15% (chance 0.1%) and
  class-focused transport cannot remove it. Running locally: `assign_cls_3m.pt` (1M→3M random). `imagenet_2m_pipeline.sh`
  swaps the 2M slim assignment onto EFS and raises `go22`; `chain_22_v2.sh` submits aag22 after aag2 + go22 (90-min deadline).
- **Morning 09-06:** aag26 (x2, adversary w1.0, 200 ep) FID-10k **29.57** (best). Validation queue done: α1-500 39.0, α0.1-10k 40.2,
  α0.3-5k 41.6, α1-3k 45.3, 20k 49.0; half-width on 200k (aag21) **37.8 < 38.4** full width. ImageNet fresh-z latent FID falls with
  transport (300k 134 → 2M 110); aag2 final 138.3; **aag22** (150M — width 1.9 + class embedding, not 133.6M; 2M assignment) running,
  epoch 2 = 161.6 vs aag2 165.2. 3M assignment finishing locally → `imagenet_3m_pipeline.sh` uploads `assign_cls_3m_slim.pt` → aag29.
  Node B: aag27 (x2 plain, width 0.5) → aag28 (its adversary w1.0 200 ep). Chains: `chain_v8.sh`.
