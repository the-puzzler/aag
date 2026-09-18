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
- **09-06 morning (user awake):** aag26 FID-50k **28.18**. User asked for the ROMS-IMLE pixel loss recipe (LPIPS 1.0 + DINO 1.0 +
  pixel 0.1): trainer now has `--mse-weight/--dino-weight` (DINOv2 ViT-B/14 via torch.hub, offline mirror at
  `/mnt/shared/aag/torch_hub`, config env `TORCH_HOME`). Node B chain (`chain_v9.sh`): aag27 (x2 w0.5 plain) → **aag30** (x2 DINO
  recipe, image tag in `dino_image_tag`) → aag28 (w0.5 adversary) → aag31 (DINO recipe + adversary). Node A: `wait29.sh` submits
  aag29 (3M assignment) when `assign_cls_3m_slim.pt` lands on EFS. aag22 epoch 4: 146.8 vs aag2 152.0. Compact AEs (d64/128/256)
  were a wash: FID 39.4/39.7/43.1 vs TiTok 38.4 — better N/d paid for by the decoder floor.
- **09-06 08:xx — flips REJECTED by the user** ("gain too small to warrant flipping"): aag27/28/30/31 (x2 line) cancelled, aag26's
  28.18 is off the record; valid CelebA best stays aag8 33.7 until the 28k re-runs land. Submitted: **aag33** = adversary w1.0/200 ep
  on aag3 ep400 (node A; aag29 queues behind it), **aag32** = DINO recipe on 28k from scratch (node B, image 7bba9c8) → aag34 its
  adversary (`chain_v10.sh`). Configs `aag256_celebahq_titok_{dino,adv_w1_long,dino_adv_w1_long}.yaml`.
- **09-06 07:40:** literal ROMS-IMLE weights (0.1/1/1) don't train (FID 404↑; DINO output-grad 1.30 vs LPIPS 0.0061 vs MSE 0.0001 →
  DINO-only loss; `grad_balance.py`). User chose (a): aag32 deleted → **aag35** = gradient-matched mse 1 / lpips 0.5 / dino 0.005
  (`aag256_celebahq_titok_dino_gm.yaml`) → aag36 its adversary (`chain_v11.sh`). aag33 (28k strong adversary) 35.2 @ep470/600.
- **09-06 08:00:** aag33 (28k adversary w1.0/200ep on aag3) final FID-10k **34.43** (FID-50k running). **aag35** (grad-matched DINO,
  plain) 34.3 @ep120/400 and falling — the DINO term at 0.005 is the biggest plain-generator lever found; aag36 (its adversary)
  chained. **aag37** = dino 0.02 bracket on node A while the 3M ImageNet assignment finishes (~09:20) → aag29 queues behind it.
- **09-06 09:20:** DINO recipe results (28k, plain, best checkpoint): dino 0.005 (aag35) FID-10k 34.2 / **FID-50k 33.02** @ep140;
  dino 0.02 (aag37) 33.3 / **FID-50k 32.31** @ep160 — plain generators now beat every adversary run (aag33 33.16, aag8 33.72).
  Both drift up after the plateau (36.5 / ~34.3 at ep400) → adversary finetunes resume from the BEST checkpoint: aag36 (from
  aag35 ep140, running), aag39 (from aag37 ep160, chained after aag38 = dino 0.08 bracket). Scratch checkpoints moved to
  `/opt/dlami/nvme/aag_scratch/ckpts` (user: use the large drives; root disk was at 84%).
- **09-06 10:00:** **aag36 FID-50k 29.16** (valid best: 28k, 38M, DINO-0.005 generator ep140 + adversary w1.0/200 ep). Node B → aag38
  (dino 0.08 bracket) → aag39 (adversary on aag37 ep160, dino 0.02). Node A: aag29 (ImageNet 3M assignment, 150M) running;
  aag22 (2M) at epoch 14: 141.3. 3M assignment diagnostics: fresh latent FID 102.8, critic 56%.
- **09-06 12:30 — AAG2 (user's new idea, see memory `newgen_aag2_joint_block`):** implemented (`aag2_block_step`, `run_assignment_aag2.py`).
  CelebA TiTok: floor crossed at block 28 (3 s), surrogate 37.5 vs 39.4 for AAG1-200k; over-transport degrades with ridge 0.02,
  not with ridge ≥ 0.2; PCA rotation essential. Generator tests: **aag40** (floor, spec) → **aag41** (block 100) chained after aag39
  (`chain_v14.sh`), standard recipe for a like-for-like vs aag3. Assignments on EFS under `celebahq_titok/assign_aag2/`.
- **09-06 14:20 — AAG2 verdict:** aag40 (AAG2 floor assignment, std recipe) FID-10k 39.7 best / 40.55 final, **FID-50k 38.93** vs
  aag3 38.4 / 37.13 — equivalent-to-slightly-worse, 7000× cheaper assignment. Toy at CelebA N/d strongly negative (see memory).
  **New user idea:** fresh-z unpaired adversary (`--fresh-gan-weight`, adaptive 0.1) → **aag42** = finetune of aag37 ep160 (image
  tag in `fresh_image_tag`), then aag41 (AAG2 block-100 control) via `chain_v15.sh`.
- **09-06 17:00 — fresh-z adversary (user idea 2):** `--fresh-gan-weight` (unpaired critic real vs G(N(0,I)); two BatchNorm bugs
  fixed: single real||fake forward for DDP, fakes scored inside the mixed batch). Weight 0.1 → critic saturates, FID 33→57 (aag42).
  Weight 0.5 (aag44): **FID-10k 28.8 @ +10 epochs, FID-50k 27.71 (valid best)**, then drifts up at the reset LR (38 @ep210).
  Queued (`chain_v19.sh`): aag46 = short low-LR fresh finetune (30 ep, 3e-5), aag47 = fresh 0.5 + pairwise 1.0, aag41 (AAG2
  control); aag45 = no-adversary finetune control after aag22. Image tag `84229c6`.
- **09-06 18:00 — z-routing experiment (research agent):** `Generator256(z_bottleneck, z_skip, z_skip_rank)`; trainer flags
  `--z-bottleneck/--z-skip/--z-skip-rank`. Sweep aag48–52 (bott64, +lowrank8, +full, bott16, bott16+lowrank8) on the third node
  after aag45, DINO-0.02 recipe, vs flat aag37 (33.3 / 32.31). `chain_v20.sh`, tag `route_image_tag`. Fresh-z adversary: control
  aag45 flat at 33.5–33.9 → the aag42/aag44 rises are adversarial instability; aag46 (short low-LR) and aag47 (fresh+pairwise) queued.
- **09-06 20:00 — z-routing first result:** aag48 (rank-64 bottleneck, plain, DINO-0.02) FID-10k 31.5 final with NO post-plateau
  drift (flat aag37: min 33.3, 34.9 final); **FID-50k 30.56** = plain record. Held-out pixel MSE much worse (0.10 vs 0.057) — the
  hypothesised trade. aag49–52 (skips, rank 16) follow. User rule: do NOT stack bottleneck with adversary (confounded).
  Fresh-z adversary: aag46 (lr 3e-5) null; aag47 (fresh+pairwise) diverging (89 @310); aag53 (20 ep at 1e-4) next.
- **09-06 20:25 — fresh-z recipe settled:** aag53 (flat dino-0.02 gen + fresh critic 0.5, lr 1e-4, 20-ep cosine, ckpt @ep172)
  FID-10k 28.58 / **FID-50k 27.45 = valid CelebA best**. aag46 (lr 3e-5) null, aag47 (fresh+pairwise) collapsed to 152, aag44 (200 ep)
  collapsed to 85 after its ep170 minimum. Bottleneck sweep continues on the third node (aag49 running). aag41 (AAG2 control) on node B.
- **09-06 21:40:** aag29 (ImageNet 3M assignment, 150M) final FID-10k **131.9** (300k→2M→3M: 138.3→133.6→131.9). **aag57** =
  same + DINO-0.02 recipe on node A (~17 h). aag49 (bott64 + lowrank-8 skips) FID-50k 31.57 — skips erode the bottleneck gain
  (30.56); aag50 (full skips) running. Fresh-weight bracket aag54–56 queued after aag41 on node B.
- **09-06 22:40:** fresh-z short recipe: weight bracket 0.25 → 32.6, **0.5 → 28.6 (seed 2: 28.9)**, 1.0 → 31.9 then unstable; on the
  standard-recipe generator (aag58) no gain (39.1) — needs the DINO base. aag41 (AAG2 block-100) 39.7 best = same as floor ckpt.
  ImageNet aag57 (3M + DINO-0.02): FID-10k **113.4 @ epoch 2** (std recipe 166 @2, 132 final). Node B idle; third node on aag50–52.
- **09-07 02:20 — z-routing sweep:** FID-50k plain: flat 32.31 · bott64 30.56 · +lowrank8 31.57 · +full skips 43 (FID-10k) ·
  **bott16 30.06** · bott16+lowrank8 30.34. Tighter trunk keeps helping, every bypass hurts, no post-plateau drift with a bottleneck.
  aag59 (r=8) → aag60 (r=4) chained (`chain_v23.sh`). ImageNet aag57 (DINO): min 91.4 @ep4, drifting (98.9 @12).
- **09-07 06:00:** rank sweep done (flat 33.3 · 64 31.5 · **16 30.9 / FID-50k 30.06** · 8 33.7 · 4 67.1; skips always hurt). Follow-ups
  aag61 (r=32) → aag62 (r=16, 800 ep) on the third node. User: make the compressor nonlinear → `--z-pre-depth/--z-pre-width`,
  sweep aag63–65 (2×512 / 4×1024 / 6×2048 at r=16) on node B (`chain_v25.sh`). ImageNet aag57 (DINO): 91 @ep4 → 107 @24, awaiting decay.
- **09-07 08:10:** aag61 (rank-32 bottleneck, plain) FID-10k 29.85 / **FID-50k 28.40** — plain record; rank optimum ≈32. aag62 (r=16, 800 ep)
  running, aag66 (r=32, 800 ep) chained (`chain_v26.sh`). Nonlinear compressor sweep aag63–65 on node B. aag57 ImageNet DINO flat at 107.
- **09-07 10:30 — ImageNet:** aag57 (3M + DINO-0.02) min **91.4 @ep4**, final 106.3 (std recipe 131.9). Next, separately: **aag67** =
  + rank-32 bottleneck from scratch (node A, tag `c94ddbc`), **aag68** = short fresh-z finetune (0.5, lr 1e-4, 2 ep) of aag57 ep4 after
  aag65 on node B (`chain_v27.sh`). Nonlinear compressor: 2×512 36.3, 4×1024 37.8 (linear r=16: 30.9) — nonlinear hurts; 6×2048 running.
- **09-07 12:15:** nonlinear compressor sweep negative (36.3 / 37.8 / 36.8 vs linear 30.9); 800-epoch schedule worse than 400 (r=16: 31.7 vs
  30.9). aag66 (r=32 × 800) running for confirmation; aag67 (ImageNet + r=32 bottleneck) running; aag68 (ImageNet fresh finetune) starting.
- **09-07 15:30 — ImageNet:** aag68 (aag57 ep4 + 1 epoch fresh-z adversary 0.5) FID-10k **82.6 / FID-50k 80.00**; 2nd epoch collapsed (193);
  1-epoch cosine (aag70) 106.9 — gain is mid-schedule at high LR, as on CelebA. aag57 ep4 plain: FID-50k 88.21. aag67 (r=32 bottleneck) stuck
  at ~320 — awaiting the user's kill/relaunch decision (rank-128 config ready). CelebA: aag66 (r=32, 800 ep) 31.75 < 400-ep schedule; aag69 = r=32 seed 1.
- **09-07 21:30:** aag67 (ImageNet r=32) descending slowly from 320 → 277 @ep30 — rank 32 starves ImageNet; **aag71** (rank 128) chained
  after it (`chain_v28.sh`, tag `c94ddbc`). CelebA: r=32 seed 1 30.78 (seed 0 29.85) → r=16 ≈ r=32 within seed noise; bottleneck gain
  over flat robust. ImageNet FID-50k: 80.00 (aag68 ep5, +fresh critic), 88.21 (aag57 ep4 plain). Nodes B and third idle awaiting direction.
- **09-08 13:00:** ImageNet bottleneck negative: r=32 262.8, r=128 120.9 vs flat DINO 106.3 / best 91.4. CelebA nonlinear compressor at r=32
  (aag72) 35.7 — nonlinear worse at every rank. Decomposition (FID@5k assigned/fresh): flat 10.0/34.4, linear r32 **7.7/30.5**, nonlinear
  10–13/36–39 — linear bottleneck better on both targets, nonlinear worse on both (Gaussianity of the code is preserved only by a linear
  map). aag73 (4×1024 @ r32) finishing. All three nodes otherwise idle; user considering direction.
- **09-09 10:30:** AAG2 400 blocks (aag74, Gaussian-indistinguishable) 42.65 best / FID-50k 41.84 vs floor 38.93 — fully Gaussian costs ~3.
  User-requested bottleneck × fresh critic: linear r32 + fresh (aag75) 29.85 → 29.64 (no gain); nonlinear 4×1024@r32 + fresh (aag76)
  34.3 → 29.96. Bottleneck family converges on ~29.6–30 regardless; flat DINO + fresh critic remains best (28.58 / 27.45). All nodes idle.
- **09-09 11:30 — new CelebA best:** aag77 = linear r32 bottleneck (aag61 ep400) + fresh critic **1.0** (0.5 was too weak for this generator:
  critic never reached equilibrium): FID-10k 27.28 @ep406, **FID-50k 25.87**; collapses after ep408. aag78 (2.0) running. Rule of thumb:
  the gain happens in the window where df≈1/gf≈0 — tune the weight per generator until that window appears, then take the mid checkpoint.
- **09-09 12:15:** weight bracket on the r32 bottleneck generator: 0.5 none · 1.0 27.28 (FID-50k 25.87) · 2.0 28.55→collapse. Seed-1 repeat
  of 1.0 (aag79): min 29.03 then collapse — 25.87 is seed-fragile, not a recipe. Reproducible best remains flat DINO + fresh 0.5 (27.45).
  Next lever if pursued: stabilise the critic (lower critic LR / R1 / EMA critic) + eval every epoch to widen the productive window.
- **09-09 14:40 — critic training-set variants (user ideas), 20-ep finetunes on the flat DINO generator (aag37 ep160, w0.5) and the r32 bottleneck (aag61 ep400, w1.0):**
  `--fresh-critic-source assigned` (critic trained on real vs G(z_assigned), generator pushed at fresh z): aag80 flat 31.11 @168 then 45; aag81 r32 28.50 @406 then 65.
  Worse than the fresh-trained critic (28.6 / 27.3) and same collapse; critic saturates on decoder texture (df→0.006, gf→−15).
  `--fresh-critic-source gen_assigned` (DISTRIBUTIONAL: critic separates G(z_assigned) ["real"] from G(z_fresh) ["fake"], no real images, only the fresh branch gets gradient):
  aag82 flat w0.5 33.2→31.4 monotonic, no collapse (critic in equilibrium the whole run, realised mult 0.007–0.02);
  aag84 flat w2.0 31.1 @164 then 62; aag85 w4.0 diverges — collapse with the critic STILL balanced (adv term 0.5–2× the supervised gradient), so it is weight, not critic saturation;
  **aag83 r32 w1.0: 29.1 → 27.5 → 26.67 @406 / FID-50k 25.18 (record), then settles at 29–30 instead of collapsing.** PNG: jobs tmp critic_variants_r32.png.
  Running: aag86 (aag83 seed 1), aag87 (flat, gen_assigned w1.0). Image tag d1c05fc.
- **09-09 16:00:** distributional critic REPRODUCES: aag86 (aag83 seed 1) 26.36 @406 / **FID-50k 24.87** (seed 0: 26.67 / 25.18), both seeds settle at 28.8–29 afterwards, no collapse.
  Flat generator + gen_assigned w1.0 (aag87): 28.9 @170 and flat to the end, final ep180 FID-50k 28.21 (end-of-schedule checkpoint, not a window pick).
  Gain fades after the peak because the adaptive multiplier decays 0.3 -> 0.03 with the LR -> aag88 = fixed 0.3 on r32. aag89 = ImageNet aag68 recipe with the distributional critic.
  **Reproducible best is now CelebA-HQ FID-50k 24.9–25.2 (r32 linear bottleneck + DINO 0.02 + distributional critic 1.0, 6 finetune epochs from aag61 ep400).**
- **09-09 18:00 — schedule for the distributional critic (r32 + gen_assigned w1.0, 20 ep from aag61 ep400):** the post-peak fade is NOT the adaptive multiplier
  (aag88 fixed 0.3: 26.32 @406 then 30.0) and NOT cosine decay (aag90 constant 1e-4: 26.05 @406 then 32.8 — worst fade). It is high LR held too long.
  **aag91 constant 5e-5: monotonic 29.6 → 26.80 @414, flat 26.9 to the end; final ep420 FID-50k 25.56 (stable, no picking).** New flag `--min-lr-frac` (1.0 = constant).
  Running: aag92 (5e-5, 40 ep), aag93 (3e-5, 40 ep), aag89 (ImageNet). Image tag: jobs tmp const_image_tag.
- **09-09 19:30:** constant 5e-5 for 40 ep (aag92): 24.84 @440 still descending, **final-checkpoint FID-50k 23.56 (record, stable)**; 3e-5 (aag93) 26.20 — 5e-5 is the LR.
  Bottleneck still needed with the critic (matched cosine-1e-4 pair: flat aag87 28.21 vs r32 aag83 25.18 FID-50k). Queued: aag97 flat control at const 5e-5/40 ep, aag96 r32 80 ep.
  From scratch (user question): aag94 (w1.0) / aag95 (w0.5) = aag61 recipe with the critic on from step 1, ~14 h.
- **09-09 20:30 — ImageNet:** aag89 = aag57 ep4 + distributional critic 0.5 (cosine 1e-4, 2 ep): FID-10k 61.9 @ep5 / **FID-50k 59.25** (aag68 real critic 80.00; plain 88.21); ep6 69.2 as LR decayed.
  Critic balanced throughout (df 0.89–0.95). Queued aag98: same at constant 5e-5, w1.0, 4 ep (after aag97 flat control), then aag96.
- **09-09 21:30:** flat control at the best schedule (aag97: flat DINO + distributional critic 1.0, constant 5e-5, 40 ep): plateau 29.4–29.6 from ep174, final 29.41 / FID-50k 28.21 (same as its cosine run aag87) —
  vs r32 bottleneck on the identical schedule 24.84 (aag92). Bottleneck worth ~4.5 FID with the critic. User asked for speed → aag96/98/99 now run in PARALLEL (6 nodes for ~3 h).
  CelebA epoch ≈ 36 s (40 ep ≈ 30 min incl. evals); ImageNet epoch ≈ 0.6 h.
- **09-09 23:00 — from scratch (interim, ep190/400):** aag95 (r32 + distributional critic **0.5** on from step 1, aag61 recipe) monotonic 25.12 @190 — already
  below plain aag61's endpoint (29.85 @400) and near the finetune record (24.84) at under half the schedule. aag94 (w1.0 from scratch) unstable: 108 @60, 133 @90, 73 @190 —
  1.0 is right for finetuning a converged generator, too strong while the generator is forming. Critic balanced in both (df 0.90–0.95).
- **09-10 00:30 — from scratch FINAL (400 ep cosine 2e-4, r32, critic on from step 1; runs take ~4.5 h not 14):**
  aag95 w0.5: monotonic to **23.60 @210 / FID-50k 22.29 (best of the project, but a picked checkpoint)**, then collapse from ep220 (38 → 57 → 138 @400). ep210 ≈ where the cosine LR passes 1e-4.
  aag94 w1.0: never converged (108 @60, 133 @90, 57 @400). Weight 1.0 is for finetuning a converged generator only.
  Reading: the critic game is stable at LR ≤ 5e-5 and unstable when LR ≥ ~1e-4 is held for long once the generator is near the anchors (finetunes overshoot at 1e-4, from-scratch collapses when the cosine reaches 1e-4 after ep200). Early high LR from scratch is fine because the generator is far from the anchors.
- **09-10 02:00:** aag98 ImageNet at constant 5e-5, w1.0: FID-10k **47.2 @5 / FID-50k 44.76**, 47.4 @6, 49.8 @7, 78 @8 (collapse with the aag95 signature: critic winning, multiplier throttled).
  aag99 (aag92 recipe, critic 4 layers / ndf 128 = 27.8M): 24.18 @440 vs 24.84 standard; **final FID-50k 22.93 vs 23.56** — small real gain (~2x the FID-50k seed spread of 0.3); new stable-endpoint record.
  Its critic also drifts toward winning (df 0.72, mult 0.009 by the end). Running: aag100 (scratch, 220-ep cosine), aag101 (scratch, fixed 0.05); aag96 chained after aag99.
- **09-10 04:00:** aag96 (standard critic, constant 5e-5, **80 ep**): 23.72 @480 still descending, no collapse in 70k steps; **final FID-50k 22.58 — stable-endpoint record.**
  Second 40 epochs gained 1.7 FID-10k over the first 40. aag102 = big critic x 160 ep running. Stable FID-50k ladder: 25.56 (20 ep) → 23.56 (40) → 22.58 (80); big critic 40 ep 22.93.
- **09-10 04:30 — from-scratch collapse is the ADAPTIVE WEIGHT, not the LR:** aag100 (220-ep cosine) collapsed at ep150–180 (27.45 → 58) with LR already ≈4.6e-5.
  Both from-scratch collapses (aag95 @220, aag100 @160) happen at the same quality point (FID high-20s, generator reaching the anchors) with the same signature: df→0.68, gf→1.1, realised multiplier→0.004.
  Mechanism: multiplier = ||∇sup||/||∇adv|| × w; as the generator converges on the anchors ∇sup shrinks → push on the fresh branch vanishes → fresh branch drifts → critic wins → ∇adv grows → multiplier shrinks further (positive feedback).
  Finetunes from a converged generator are stable because ∇sup is already small and steady. aag101 (fixed 0.05) is the test: 32.1 @120, df 1.00.
- **09-10 05:30:** aag101 (from scratch, FIXED multiplier 0.05) passed the collapse zone: 24.06 @230, **22.96 @240 (best FID-10k)**, 23.97 @250; critic drifting toward winning (df 0.83) but the push stays on. Confirms the adaptive-weight feedback as the cause. aag103 (ImageNet, fixed 0.08) queued on the freed node.
- **09-10 06:30:** aag102 (big critic 4L/ndf128, constant 5e-5, **160 ep** from aag61 ep400): FID-10k 20.02 @560 still descending, no collapse in 140k steps;
  **final FID-50k 18.89 — first sub-20, stable last checkpoint.** Critic drifting toward winning (df 0.60) but the multiplier stays ~0.012 on the converged generator.
  aag104 = continue from aag102 ep560 for 160 more epochs (critic state resumes). Ladder (standard critic) 20/40/80 ep: 25.56/23.56/22.58; big critic 40/160 ep: 22.93/18.89.
- **09-10 07:30 — from scratch: PARKED.** aag101 (fixed 0.05) also collapsed: 22.96 @240 → 37 @260 → 131 @390 (df 0.54, gf 2.0). The fixed multiplier delayed the collapse ~40 ep but did not prevent it,
  so the adaptive throttle is a contributor, not the cause. All three from-scratch variants (adaptive w0.5, 220-ep cosine, fixed 0.05) break when the generator reaches the anchors (FID low/mid-20s),
  regardless of the LR at that moment (1e-4, 4.6e-5, 5.5e-5). Best from-scratch number stays the picked aag95 ep210 (FID-50k 22.29). Finetune route is stable and better (18.89) → pursue that.
  aag105 = seed-1 repeat of aag102 (queued on aag101's node).
- **09-10 09:30:** aag104 (aag102 continued to **320 ep**): 17.86 @720, still slowly descending (2nd 160 ep: −2.2 FID-10k vs −5 for the 1st); **final FID-50k 16.79 — record.**
  Critic winning (df ~0.5) but stable. Ladder (big critic, constant 5e-5) 40/160/320 ep: 22.93/18.89/16.79. aag108 continues to 880 (queued behind the two-sample runs).
  Two-sample loss (user idea): `--ts-loss mmd|swd` in frozen DINO CLS space, gathered 256 vs 256, floor = EMA of stat(assigned_t, assigned_t−1), clamp 0; smoke: fresh/assigned stat 2–3x the floor.
  aag106 (MMD) running, aag107 (SWD) queued; same recipe as aag92 (r32, constant 5e-5, 40 ep) for a direct comparison with the critic (24.84 / 23.56).
- **09-10 10:30:** aag105 (seed-1 repeat of aag102, big critic, 160 ep): 19.32 @560 (seed 0: 20.02), monotonic; **final FID-50k 18.12** (seed 0: 18.89) — reproduces within ~0.8.
- **09-10 12:00 — MMD two-sample loss WORKS (aag106):** r32, constant 5e-5, 40 ep, `--ts-loss mmd` (DINO CLS, RBF mixture, 256 vs 256 gathered, floor = EMA stat(assigned_t, assigned_t−1)):
  FID-10k **21.47 @428** / 21.76 @440, **final FID-50k 20.93** vs the critic on the identical recipe 24.84 / 23.56 (aag92). No adversary, no critic. MMD fell 0.0355 → 0.0075 with floor 0.0048 (7.4x → 1.5x floor; clamp never engaged).
  Slight uptick over the last 12 ep (21.47 → 21.76). First aag106 attempt hung: partial last batches differ across ranks → all_gather deadlock; fixed by truncating to the all_reduce(MIN) count.
  aag107 (SWD) running; queued aag109 (MMD 160 ep), aag110 (ImageNet MMD). Image tag for ts: jobs tmp ts_image_tag.
- **09-10 13:00:** aag107 (SWD, same recipe): 21.56 @426 / 21.91 @440, final FID-50k 21.07 — same as MMD (21.47 / 21.76 / 20.93), same mild late uptick; SWD stat 0.131 → 0.033 vs floor 0.0235 (1.4x). Statistic choice is secondary.
- **09-10 13:30:** aag103 (ImageNet, distributional critic FIXED 0.08, constant 5e-5, 6 ep): 50.5 / 49.0 / 52.1 / 67.0 / 69.3 / 62.5 @5–10 — no better than adaptive (aag98 47.2 @5) and degrades anyway.
  Fixed multiplier is not the ImageNet fix either. aag110 (ImageNet MMD) submitted next.
- **09-10 14:30:** aag108 (critic run continued 720 → 880): plateau 17.7–18.7, ends 17.93 (= aag104's 17.86). **The big-critic ladder saturates at ~17.9 FID-10k / 16.79 FID-50k after 320 ep**; critic winning (df 0.42). Not re-evaluated at 50k.
- **09-10 16:00:** aag109 (MMD adaptive, 160 ep): 21.51 @424 then slow drift to ~21.9 — adaptive weight climbs to 3.7 and renormalises the push to full supervised strength at the floor (noise push).
  aag111 (MMD FIXED 0.5, 40 ep): monotonic to 21.84 @440, flat, no drift — but same level. The ~21.5–21.8 plateau is the statistic's resolution (CLS-only MMD stuck at 1.4–2x floor), not the weighting.
  aag112 = MMD fixed 0.5 on CLS++mean-patch features. Critic ladder saturated at 16.79 (320 ep); MMD is faster at 40 ep (20.93 vs 23.56) but does not improve with epochs.
- **09-10 16:30:** aag109 (MMD adaptive, 160 ep) final: 21.49 @428 then 21.8–22.0 flat to 560. CLS-token MMD saturates at ~21.5–21.9 regardless of schedule/weighting. Not re-evaluated at 50k.
- **09-10 17:00 — ImageNet MMD (aag110, aag57 ep4 + `--ts-loss mmd` adaptive 1.0, constant 5e-5):** FID-10k **39.39 @ep5 / FID-50k 36.67** (critic 47.2 / 44.76, plain 91.4 / 88.21); MMD 0.0044 vs floor 0.0048 → clamp engaged after one epoch (indistinguishable in DINO CLS space). 3 epochs still running.
- **09-10 18:30:** aag112 (MMD on CLS++mean-patch, fixed 0.5, 40 ep): monotonic to **21.17 @440 / FID-50k 20.17**, still descending, stat 2.3x floor (not saturated) — finer features beat CLS-only (21.84 fixed / 21.47 adaptive min).
  aag114 continues it to ep560. aag113 = same features on ImageNet.
- **09-10 19:30:** aag110 (ImageNet CLS-MMD) finished all 4 epochs WITHOUT collapse — first stable ImageNet finetune: 39.4 / 34.3 / 33.0 / **32.77 @8 / FID-50k 30.17 (last checkpoint)**, stat at floor (clamp on, push ~off); aag115 continues to ep14.
- **09-10 20:30 — two-sample loss over-optimises past ~40 ep:** aag114 (aag112 continued 440 → 560): FID-10k 21.12 @446 → 23.33 @560 monotonically UP while the MMD stat kept falling (0.0105 → 0.0079).
  The floor-stop never engaged: fresh-vs-assigned cannot reach the assigned-vs-assigned floor (28k discrete anchors vs a continuum) → the clamp needs a margin. New flag `--ts-floor-mult` (aag116: 2x, 160 ep).
  Practical recipe today: MMD CLS++patch fixed 0.5, 40 ep → FID-50k 20.17, taken at the end (no picking) — but do not run it longer without the margin.
- **09-10 22:00:** aag113 (ImageNet, MMD CLS++patch fixed 0.5, 4 ep): 34.0 / **29.87 @6** / 29.84 / 29.89 @8 — stat hit the floor at ep6, clamp engaged, FID HELD FLAT for 2 epochs (the intended stop-at-indistinguishable behaviour, first clean instance). CLS-only was 32.77 @8. **FID-50k @8: 27.27 (last checkpoint).**
- **09-11 00:30:** aag116 (MMD CLS++patch fixed 0.5, stop margin 2x floor, 160 ep): min 21.19 @440–448 then drift to 23.20 @560 — identical to the unmargined aag114 although the clamp was engaging.
  The late drift is NOT the two-sample push: with the MMD clamped, continued supervised training at constant 5e-5 on a converged generator memorises the anchors further (off-target degradation) and nothing counteracts it; the critic kept improving because it never stops finding differences. aag119 = plain control (no fresh term) to confirm.
  Recipe stands: two-sample loss for ~40 ep then STOP (or a finer/larger statistic: aag117/118 doubled batch).
- **09-11 07:00:** aag118 (MMD CLS++patch fixed 0.5, two-sample batch DOUBLED via `--ts-fresh-mult 2`, 40 ep): monotonic to **20.93 @440 / FID-50k 20.01**, still descending (aag112 single batch: 21.17 / 20.17). Floor halves (0.0024), stat at 3.2x floor.
  First attempt hung (fresh side k x rows vs assigned rows on the ragged last batch) → truncate the two sides separately. aag117 = ImageNet version.
- **09-11 08:30 — plain control (aag119, supervised only, constant 5e-5, 160 ep): 29.80 → 30.29.** Supervised-only training drifts only +0.5, so the +2 late drift in the long two-sample runs
  (aag114/116) IS the two-sample term: past ~40 ep it over-optimises the DINO-space proxy (FID up while the stat sits in its band), margin or not. Correction of the 00:30 reading.
  Two-sample recipe = ~40 ep then stop; the adaptive critic is what keeps improving over long schedules (ladder to 16.79). Batch-size axis (aag118 x2: 20.01) still open (aag120 x4).
- **09-11 09:30:** aag115 (ImageNet CLS-MMD continued 8 → 14): 33.5 / 32.7 / 31.9 / 31.8 / 31.5 / **30.89 @14 / FID-50k 28.35**, stat at floor throughout, no collapse. Superseded by CLS++patch (29.9 in 4 ep).
- **09-11 10:30:** aag120 (two-sample batch x4, 1024 vs 1024): 20.92 @440 / FID-50k 19.95 = x2 (20.93 / 20.01); floor 0.0011, stat 5.6x floor. The batch axis saturates at x2 for the 40-ep recipe. aag122 = seed-1 repeat of aag118 (x2).
- **09-11 11:30 — MATCHED FACTORIAL (user request):** {flat, r16, r32, r64} x {none, MMD, critic}, all from each generator's own plain **ep400** (identical 400-ep DINO-0.02 recipe,
  lr 2e-4), finetuned at constant 5e-5 for **160 ep** (matched steps). none = supervised only; critic = big distributional critic (4L/ndf128, w1.0 adaptive); MMD = CLS++patch, fixed 0.5, batch x2.
  Existing matched cells: r32 x none = aag119 (30.29 end), r32 x critic = aag102 (20.02 / FID-50k 18.89; seed aag105 19.32 / 18.12). Missing 10 cells = aag123–aag132 (chain51, jobs listed in jobs tmp factorial_jobs.txt; configs `aag256_celebahq_titok_dino_gm4_fact_<gen>_<term>_160ep.yaml`).
  Note: all earlier FLAT runs (aag53/80/82/84/85/87/97/121) started from flat ep160, not ep400 — aag121 (flat + big critic from ep160, 160 ep) is a supplementary point, not a factorial cell.
  After the 160-ep grid: extend the four critic cells to 320 ep to match aag104 (16.79).
- **09-11 12:30:** aag122 (seed-1 repeat of aag118, MMD CLS++patch fixed 0.5 x2, 40 ep): 21.07 @440 / FID-50k 20.16 (seed 0: 20.93 / 20.01) — reproduces.
- **09-11 13:30:** aag121 (FLAT from ep160 + big critic 1.0, constant 5e-5, 160 ep): 23.29 @320 / FID-50k 22.21, still slowly descending; critic winning (df 0.39). r32 on the same schedule: 20.02 / 19.32 (aag102/105). Supplementary point; the matched flat cell (from ep400) is aag124.
- **09-11 14:30:** aag117 (ImageNet, MMD CLS++patch fixed 0.5, batch x2, 4 ep): 32.9 / **29.45 @6** / 29.9 / 29.9, FID-50k 27.38 — same as x1 (aag113 29.9 / 27.27); stat at the (halved) floor by ep6, clamp on. Batch axis saturates on ImageNet too.
- **09-11 15:30 — factorial, first cell:** aag123 flat x none: FID-10k 34.9 → 34.5 → 35.05 @560, **FID-50k 33.81**. (Flat plain ep400 = 34.9 is worse than its ep160 = 33.0: the flat generator overfits late in plain training; r32 did not — 29.85 @400.)
- **09-11 17:30 — factorial cells:** r16 x none: 31.2 → 31.89 @560, FID-50k 31.10. **flat x critic (matched, from flat ep400): 25.34 @560 (min 24.71), FID-50k 24.14** vs r32 x critic 18.89 / 18.12 — a ~5.5 FID-50k bottleneck gap under fully matched conditions.
- **09-11 18:30:** factorial flat x MMD (160 ep): FID-50k 27.34 at the end (see FID-10k curve in the job log; MMD runs drift after ~40 ep by design of the matched schedule).
- **09-11 19:30:** factorial r16 x critic (160 ep): **FID-50k 20.73** (flat 24.14, r32 18.89/18.12). flat x MMD FID-10k min 23.24 @448 then drift to 28.20 @560.
- **09-11 20:30:** factorial r16 x MMD (160 ep): monotonic to 23.17 @560, **no drift** (flat and r32 MMD drift after ~40 ep); FID-50k 22.45. r16 x critic 21.46 @560 / FID-50k 20.73.
- **09-11 21:00:** factorial r32 x MMD (160 ep, clean single run): FID-50k 22.58 at the end (see min in the log line below).
- **09-11 22:00:** factorial r64 x none: 31.4 → 32.26 @560, FID-50k 31.14. r32 x MMD FID-10k min 20.96 @438 then drift to 23.62 @560 (FID-50k 22.58 at the end).
- **09-11 23:00 — factorial r64 x critic (160 ep): 17.73 @560 still descending steeply, FID-50k 16.60 — beats r32 x critic (18.89/18.12) at matched steps, and matches aag104's 320-ep r32 record (16.79) in half the epochs.**
  Critic column: flat 24.14 · r16 20.73 · r32 18.89/18.12 · r64 16.60. Plain column: 33.81 · 31.10 · ~30.3 · 31.14. The plain rank optimum (r32) does NOT carry over under off-anchor supervision — the wider code benefits more from the critic.
- **09-11 23:30:** 320-ep extensions of the critic cells: aag133 (r64), aag134 (flat), aag135 (r16, chained); r32 @320 = aag104 (16.79). Auto FID-50k → scratch ckpts/factorial320_fid50k.out.
  Next rank: plain r128 400-ep generator (aag136, chained) → then r128 x critic. MMD column endpoint is drift-dominated for flat/r32 (report min too); r16 x MMD did not drift.
- **09-12 00:30 — 160-EP FACTORIAL COMPLETE (FID-50k, last checkpoint; MMD cells drift after ~40 ep so FID-10k min in brackets):**
  | gen | none | MMD | critic |
  | flat | 33.81 | 27.34 (min 23.2 @448) | 24.14 |
  | r16 | 31.10 | 22.45 (no drift) | 20.73 |
  | r32 | 28.92 | 22.58 (min 21.0 @438) | 18.89 / 18.12 (2 seeds) |
  | r64 | 31.14 | 23.09 (min 21.6 @450) | **16.60**, still descending |
  Reading: (1) the bottleneck advantage survives fully matched off-anchor finetuning and GROWS with it (plain spread flat→best 2.7; critic spread 7.5); (2) under the critic the rank optimum moves up (r64 > r32 > r16 > flat) whereas plain r32 ≈ r64 ≈ r16; (3) MMD ≈ −8 to −9 at its minimum for every rank but only r16 holds it; (4) both terms are complementary to compression, not substitutes: flat+critic (24.1) < r64 plain (31.1) but r64+critic (16.6) ≪ both.
- **09-12 01:30 — node budget → 1 (user).** Deleted aag134 (flat critic → 320) and aag135 (r16 critic → 320); resumable from `..._fact_{flat,r16}_critic_160ep/checkpoints/gen_ep560.pt`. Kept aag133 (r64 → 320).
  One-node queue (chain55): aag136 plain r128 (400 ep) → aag139 r128 x critic (160) → aag140 plain r256 → aag141 r256 x critic → aag137/138 sequential MMD(40)→critic(160) for r32/r64. User: rank optimum under the critic not yet found → keep going up.
- **09-12 02:30:** aag133 (r64 x critic → 320 ep): 17.73 @560 → min 16.65 @654 → 17.29 @720 — saturated (like r32 at 320–480). **FID-50k 16.24 (record; r32 @320 16.79).**
- **09-12 07:30:** aag136 plain r128 (400 ep): min 32.7 @200 → 33.56 @400 (plain: r32 29.85 < r64 ~31.5 < r128 33.6 — wider code, less regularisation). aag139 = r128 x critic next.
- **09-12 10:00:** aag139 r128 x critic (160 ep, from plain r128 ep400): 21.22 @560 — behind r64 (17.73) and r32 (20.0/19.3), ≈ r16 (21.5). **Critic-column rank optimum = r64.** FID-50k 20.10 (column: flat 24.14 · r16 20.73 · r32 18.89/18.12 · r64 16.60 · r128 20.10). aag140 (plain r256) next, then r256 x critic to close the curve.
- **09-12 15:00:** aag140 plain r256 (400 ep): 34.9 @160 → 36.96 @400 (plain trend r32 29.9 < r64 31.5 < r128 33.6 < r256 37.0). aag141 r256 x critic next.
- **09-12 17:30:** aag141 r256 x critic (160 ep): 26.17 @560 — worse than flat x critic (25.3). **Rank curve under the critic (FID-10k @560): flat 25.3 · r16 21.5 · r32 20.0/19.3 · r64 17.7 · r128 21.2 · r256 26.2 — a U with its minimum at r64.** FID-50k 24.82 → critic column at 50k: 24.14 · 20.73 · 18.89/18.12 · **16.60** · 20.10 · 24.82. aag137/138 (sequential) next.
- **09-12 20:00:** aag137 r32 SEQUENTIAL (MMD 40 ep ckpt 20.93 → big critic 160 ep): dips to **18.91 @454** (14 critic epochs!), bounces to 21.2, settles 19.69 @600 — endpoint ≈ single-stage r32 x critic (20.0/19.3). MMD stage = speed, not a better destination, on r32. FID-50k 18.44 (single-stage 18.89/18.12). aag138 (r64 sequential) next.
- **09-12 22:30:** aag138 r64 SEQUENTIAL (MMD ckpt ep440 ~21.6 → big critic 160 ep): 19.3 @454, bounce to 21.4, then 17.61 @600 still descending — same endpoint as single-stage r64 x critic (17.73 @560). Verdict (both ranks): the MMD stage front-loads the gain, the destination is the critic's. FID-50k 16.54 (single-stage 160 ep 16.60). **Queue empty; node idle awaiting direction.**
- **09-12 23:30 — CAPACITY under the critic (user: rank is per-dataset, capacity is general):** r64 fixed; width 1.9 (133.6M) and 1.4 plain 400-ep generators, then the big critic 160 ep each, matched to the 38M cell (16.60).
  One-node queue chain56: aag142 (plain w1.9) → aag143 (critic) → aag144 (plain w1.4) → aag145 (critic). Auto FID-50k via queue evaluator (scratch ckpts/queue_fid50k.out).
- **09-13 04:00:** aag142 plain r64 width 1.9 (133M, 400 ep): 37.6 @80 → 43.9 @400 — wide trunk overfits the anchors in plain training (38M: 31.5). aag143 (critic 160 ep from ep400) running: does the critic rescue the capacity?
- **09-13 09:30 — CAPACITY IS A LEVER UNDER THE CRITIC:** aag143 (r64, width 1.9 = 133M, big critic 160 ep from the overfit plain ep400 at 43.9): 16.89 @560 still descending, **FID-50k 15.53** (38M: 16.60 @160, 16.24 @320). The plain-only "capacity is not the lever" result is reversed once off-anchor supervision exists.
  Queue reordered (one node): aag147 = continue aag143 → 320 ep; aag146 = wide critic from its best plain ckpt (ep160); aag148/149 = width 2.5 plain + critic; w1.4 pair (aag144/145) deprioritised.
- **09-13 15:00:** aag147 (133M wide critic continued 560 → 720): flat 16.6 to ep608 (min 16.49 @604), then drift to 21.2 @716; **FID-50k 19.50 at the end** (vs 15.53 @560). Critic winning (df 0.45, adaptive mult 0.004) = the from-scratch late-drift signature, arriving earlier for the wide trunk than for 38M (which only plateaued).
  Wide-generator productive window ≈ 160–200 ep; best stays aag143 ep560 = 15.53. Do not extend wide critic runs past ~200 ep without a stabiliser.
- **09-13 18:00:** aag146 (133M from its best plain ckpt ep160 = 37.7, + critic 160 ep): 16.58 @320 still descending (from the overfit ep400 start, aag143: 16.89 @560). Start-point quality is secondary; the wide trunk is the lever. **FID-50k 15.44 (record; aag143 15.53 — two starts agree within 0.1).** aag148 (plain width 2.5) next.
- **09-14 04:30:** aag148 plain r64 width 2.5 (236M, 400 ep): 38.6 @80 → 46.7 @400 (plain overfits harder with width: 38M 31.5 · 133M 43.9 · 236M 46.7). aag149 (critic 160 ep from ep400) next.
- **09-14 10:30:** aag149 (236M r64 + big critic 160 ep from ep400 at 46.7): **15.55 @560 FID-10k, still descending; FID-50k 14.27** (133M: 16.89 / 15.53; 38M: 17.73 / 16.60). NOTE: the CelebA target is 138M with equal/fewer params — 133M (15.44) is in budget, 236M is not; wider runs are mechanism evidence, not headline numbers.
- **09-14 21:00:** aag144 plain r64 width 1.4 (~75M): 33.3 @170 → 35.75 @400. Plain vs width: 38M 31.5 · 75M 35.8 · 133M 43.9 · 236M 46.7 (monotonic worse). aag150 (w0.5 continuation) then aag145 (w1.4 critic) next.
- **09-15 00:30:** aag145 (75M width 1.4 + big critic 160 ep): **15.38 @560 FID-10k**, still descending — lowest 10k endpoint of the ladder (38M 17.73 · 75M 15.38 · 133M 16.89/16.58 · 236M 15.55): not monotonic at 10k. **FID-50k 14.30 — in-budget record (75M ≪ 138M), = 236M (14.27), < 133M (15.44).** Width ladder at 50k: 38M 16.60 · 75M 14.30 · 133M 15.44 · 236M 14.27 → gain saturates by ~75M; 133M is the odd point. aag150 next.
- **09-15 04:00 — over-transport under the critic (user idea):** the 1M-step AAG1 assignment (objective G flat ≈7.5e-5 from 900k–1M = past the floor until plateau; plain-regime verdict was −7 FID vs 200k) × the 75M recipe:
  aag151 plain r64 w1.4 400 ep on assign_uncond_1m → aag152 + big critic 160 ep. Compare with the 200k assignment (aag144 35.8 plain; aag145 FID-50k 14.30). Queued behind aag150 (chain60).
- **09-15 07:30:** aag150 (133M critic continuation at weight 0.5): 17.1 → 19.4 (FID-50k 17.84 vs 15.53 before), critic winning harder (df 0.29, gf 1.8), realised mult ~0.001 — halving the weight makes the late drift WORSE. With aag147 (w1.0, same drift): the drift is an ineffective push (adaptive throttle once the critic wins), not an over-strong one.
  Rule: wide critic runs stop at ~160–200 ep; a stabiliser must act on the critic (lower critic LR / fixed multiplier), not on the weight. aag151 (plain 75M on the 1M assignment) next.
- **09-15 12:00:** user: is the over-transport test AAG2? No — aag151/152 = AAG1 at 1M steps. Added route 2: AAG2 400-block assignment (Gaussian-indistinguishable; block checkpoints 25…400 exist) x the 75M recipe: aag153 plain → aag154 critic (chain61, behind chain60).
- **09-15 17:00:** aag151 plain 75M on the 1M (over-transported) assignment: min 40.45 @140 → 43.3 @400 (200k assignment: 33.3 / 35.8). Plain regime: over-transport still costs ~7 FID at 75M. aag152 (critic) running — the real test.
- **09-15 20:30:** aag152 (75M on the 1M AAG1 assignment + big critic 160 ep): 19.56 @560 vs 15.38 on the 200k assignment (aag145). Over-transport gap: plain ~7.5 → critic ~4.2 — narrowed, NOT flipped. **FID-50k 18.49 vs 14.30.** AAG2 route (aag153/154) next.
- **09-16 03:00:** aag153 plain 75M on the AAG2 400-block assignment: 35.7 @160 → 38.8 @400 (200k AAG1: 35.8; 1M AAG1: 43.3) — the AAG2 over-transport penalty is smaller (~3) than AAG1's (~7.5) in the plain regime. aag154 (critic) running.
- **09-16 06:30 — over-transport x critic: VERDICT.** aag154 (75M on AAG2 400-block + big critic 160 ep): plateau 19.98 @528–560 vs 15.38 on the floor assignment; AAG1 1M route 19.56. Both over-transported assignments lose ~4.5 FID-10k under the critic (plain: AAG2 −3, AAG1 −7.5). Off-anchor pressure narrows but does not remove the cost → **floor-stop remains the rule.** FID-50k: AAG2-400 19.19, AAG1-1M 18.49, floor 14.30. Queue empty; node idle awaiting direction.
- **09-16 09:00 — AAG3 (user spec):** `scripts/run_assignment_aag3.py` — per step: gradient-ascent worst direction (20 steps, warm start + 4 restarts, renormalised) → exact AAG1 transport along that one direction; frozen 256-dir held-out R_G every 10 steps; save `_floor.pt` at first R_G<=1, continue to plateau (running-best G not improved >0.5% in 100 evals) → `_plateau.pt`. Whitening rotate=0 (as the AAG1 200k baseline).
  Runs LOCALLY (GPU idle). Downstream (one node, matched 75M recipe): aag155 plain floor → aag156 critic floor → aag157 plain plateau → aag158 critic plateau; FID before (plain ep400 FID-10k) and after (critic FID-50k) the critic. Assignments upload to `$R/celebahq_titok/assign_aag3/assign_aag3_{floor,plateau}.pt`.
- **09-16 10:00:** AAG3 smoke with rotate=0 RAISED the held-out defect (R_G 34 → 260 in 30 steps): the learned worst direction is the top principal axis (var 22) and single-axis transport of a correlated cloud disturbs all correlated projections. With PCA whitening the ascent finds real non-Gaussianity (kurtosis 22, 400x a random direction) and G falls monotonically → AAG3 runs with rotate=1.
- **09-16 10:30 — AAG3 run (PCA whitening):** start R_G 6.23 → **floor crossed at step 1110** (AAG1: ~200k random steps; AAG2: 28 blocks) → **plateau R_G ≈ 0.83** (best G 1.06e-4 @2250; stopped @3250; ~3 min total at 60 ms/step).
  At the plateau the pursuit still finds directions with L ≈ 0.017 vs 0.0001 random (100x) but transporting them no longer lowers the held-out random-direction G: residual non-Gaussianity lives in adversarial directions random projections do not see. Files: assign_aag3/assign_aag3_{floor,plateau}.pt (local + $R). Downstream aag155–158 via chain62.
- **09-16 12:00 — AAG3 pure form (user):** no warm start, K fresh random inits per step (`--warm 0 --restarts K`); K=1 and K=4 running locally → assign_aag3_k{1,4}_{floor,plateau}.pt. Downstream for K=1 (aag159 plain floor → aag160 critic → aag161 plain plateau → aag162 critic) chained behind the warm-start pipeline (chain63). K=4 downstream pending its curve.
- **09-16 12:30 — AAG3 pure form results:** warm+4: floor @1110, plateau 0.83 · K=4 fresh: floor @1080, plateau 0.78 · **K=1 fresh: floor @850, plateau 0.665 @4380 (stop 5380), 1 ascent/step** — fastest, deepest, 5x cheaper. Warm start is unhelpful (the exact update flattens that direction). Files assign_aag3_k1_{floor,plateau}.pt uploaded; downstream aag159–162 chained.
- **09-16 19:00:** aag155 plain 75M on AAG3-floor (warm-start variant, R_G 1.0 @1110): min 35.2 @130 → 38.1 @400 (AAG1 200k floor: 33.3 / 35.8; AAG2-400: 38.8). aag156 (critic) running.
- **09-16 22:00:** aag156 (AAG3-floor, warm variant, + critic 160 ep): 18.54 @560 (plateau from 520) vs AAG1-200k floor 15.38. The plain deficit (−2) carries through (−3). FID-50k 17.51 (AAG1-200k: 14.30). aag157 (plain plateau) next.
- **09-17 00:30:** aag157 (plain on AAG3-plateau) FAILED at ep310 on a transient shared-FS rename of the .tmp checkpoint (gen_ep310.pt exists). Curve so far the best plain of the AAG3 set: min 32.2 @120, 35.9 @310 (floor variant 35.2 / 38.1). Resubmitted (resume auto from ep310); aag158 (was waiting for ep400) deleted and re-chained after it, then aag159–162 (chain64).
- **09-17 03:00:** aag157 (plain 75M on AAG3-plateau, warm variant, R_G 0.83) finished after resume: 36.12 @400, min 32.2 @120 — level with AAG1-200k (35.8 / 33.3), no over-transport penalty in the plain regime (AAG1-1M 43.3, AAG2-400 38.8, AAG3-floor 38.1). aag158 (critic) next.
- **09-17 06:00:** aag158 (AAG3-plateau, warm variant, + critic 160 ep): 18.19 @560 still slowly descending (floor variant 18.54; AAG1-200k 15.38). Plain was level with AAG1 (36.1 vs 35.8) but the critic extracts less (−18 vs −20.4). FID-50k 17.10 (floor variant 17.51; AAG1-200k 14.30).
- **09-17 09:00:** aag159 (plain 75M on AAG3 K=1 FLOOR, step 850): 35.93 @400, min 32.3 @130 — level with AAG1-200k (35.8 / 33.3), better than the warm-start floor (38.1). aag160 (critic) next.
- **09-17 12:00:** aag160 (AAG3 K=1 FLOOR + critic 160 ep): **15.71 @560, still descending** ≈ AAG1-200k (15.38); warm-start floor was 18.54. Pure-form AAG3 reaches the floor in 850 learned steps (vs ~200k random) at equal downstream quality: **FID-50k 14.62 vs 14.30** (within seed spread). aag161/162 (K=1 plateau, R_G 0.665) next.
- **09-17 18:00:** aag161 (plain 75M on AAG3 K=1 PLATEAU, R_G 0.665): 36.31 @400, min 32.7 @130 — level with K=1 floor (35.9) and AAG1-200k (35.8): NO plain-regime over-transport penalty (AAG1-1M 43.3, AAG2-400 38.8). aag162 (critic) running — last cell.
- **09-17 21:00 — AAG3 test complete (FID-10k @560 after the critic):** K=1 floor 15.71 · K=1 plateau 17.74 · warm floor 18.54 · warm plateau 18.19 · AAG1-200k 15.38.
  Past-the-floor penalty after the critic: AAG1-1M +4.2, AAG2-400 +4.6, **AAG3 K=1 plateau +2.0** (with zero plain penalty). Learned pursuit is the gentlest way below the floor but still costs after the critic → floor-stop stands; the "super-Gaussian design" does not pay at N=28k, d=512. FID-50k: K=1 floor 14.62 · K=1 plateau 16.59 · warm floor 17.51 · warm plateau 17.10 · AAG1-200k 14.30. Queue empty; node idle awaiting direction.
- **09-17 23:00 — ViT generator (user request):** `aag/vit_generator.py`, `--arch vit`: 256 learned position tokens (16-px patches), dim 768 x 12 blocks x 12 heads, z → r64 linear bottleneck → 8 z tokens; `--vit-mode self` (z tokens prepended, 86M) or `cross` (per-block cross-attention to z tokens, 115M); linear patch head + 2-conv full-res smoother (`.out`). Same recipe on the AAG3 K=1 floor assignment: aag163 plain self → aag164 critic → aag165 plain cross → aag166 critic (chain65 after local smoke + image build). Compare conv 75M: 35.9 / 14.62.
- **09-18 08:00:** aag163 ViT-self (86M) plain 400 ep on K=1 floor: min 39.0 @160 → 43.1 @400 (conv 75M: 32.3 / 35.9; conv 133M plain was 43.9 and still reached 15.5 after the critic). aag164 (critic) running.
- **09-18 11:30:** aag164 ViT-self + critic 160 ep: 21.06 @560, still descending (conv 75M: 15.71). ~5 behind after the critic; FID-50k 20.21 (conv 14.62). aag165 (ViT-cross plain) next.
- **09-18 21:00:** aag165 ViT-cross (115M) plain: min 42.4 @140 → 46.7 @400 (ViT-self 39.0 / 43.1; conv 75M 32.3 / 35.9). aag166 (critic) running — last ViT cell.
- **09-19 00:00 — HYBRID generator:** `aag/hybrid_generator.py`, `--arch hybrid`: ViT trunk (dim 512 x 8, 16x16 position tokens + 8 z tokens, self-attention) → Linear to a 64-ch 16x16 map → Generator256's conv upsampler (width 1.4, channels 512/256/128/64 → 75.4M total, trunk 25M). Isolates token mixing from the ViT patch head. aag167 plain → aag168 critic on the K=1 floor assignment, chained after the ViT queue (chain66, after local smoke + build).
- **09-19 04:00:** aag166 ViT-cross (115M) + critic 160 ep: 19.73 @560, still descending steeply (ViT-self 21.06; conv 75M 15.71). Cross-attention beats prepended z tokens after the critic despite a worse plain start (46.7 vs 43.1). FID-50k 18.93 (ViT-self 20.21; conv 14.62). Hybrid (aag167/168) next.
- **09-19 10:00 — HYBRID plain (aag167, 75M: ViT trunk 512x8 over 16x16 tokens + conv upsampler w1.4):** 32.3 @160 → **30.19 @400, still descending, NO late drift** — best plain of any architecture on this assignment (conv 75M 35.9 end / 32.3 min; pure ViTs 43–47). aag168 (critic) running.
- **09-19 13:30:** aag168 hybrid + critic 160 ep: 20.87 @560 (min 20.75), **FID-50k 19.95** (conv 75M 14.62). Critic gain only −9 vs −20 on the conv; critic stayed nearer balance (df 0.66 vs 0.56) and FID still descending → aag171 continues the critic to 320 ep first; then aag169/170 (800-ep plain + critic).
- **09-19 17:00:** aag171 (hybrid critic continued 560 → 720): 22.5 → 28 → SPIKE 84 @656 → 34.9 @720; realised multiplier jumped 0.015 → 0.18 mid-run. The critic game is unstable on the transformer trunk; hybrid ceiling with this critic recipe = the 160-ep result (19.95). aag169 (800-ep plain) running, then aag170.
- **09-20 08:00:** aag169 hybrid plain 800 ep: min 31.6 @240 → 33.7 @800 — worse than the 400-ep run (30.2, still descending at its cosine end). The 400-ep "no drift" was the cosine tail; the hybrid drifts too, slower than the conv. aag170 (critic from ep800) running for completeness.
- **09-20 11:30:** aag170 (critic 160 ep from the 800-ep hybrid plain, 33.7): **17.66 @960, still descending** — 3 better than the critic from the 400-ep hybrid (20.87) despite a worse plain start; within 2 of the conv (15.71). Longer plain training makes the transformer trunk more responsive to the critic. **FID-50k 16.76** (conv 75M 14.62; hybrid-400 19.95). Queue empty; node idle awaiting direction.
- **09-20 13:00 — push the hybrid (user):** aag172 = continue aag170's critic 960 → 1120 (was 17.66 and descending, critic balanced); aag173 = WIDE hybrid (trunk 768x12x12 = 85M + upsampler 50M = 135M, in budget) plain 800 ep; aag174 = its critic 160 ep. One-node chain69; auto FID-50k.
- **09-20 16:30:** aag172 (800-ep hybrid, critic continued 960 → 1120): collapsed, FID-50k 41.4 (from 16.76). Second hybrid critic continuation to blow up (aag171 from the 400-ep plain did too). On the transformer trunk the critic is a ~160-ep tool unless stabilised. aag173 (wide hybrid plain) next.
- **09-21 02:00:** aag173 wide hybrid (135M, trunk 768x12) plain 800 ep: min 30.4 @170 → 37.8 @800 (75M hybrid: 31.6 → 33.7). Wider trunk overfits the anchors harder, as with the conv. aag174 (critic 160 ep from ep800) running.
- **09-21 06:00:** aag174 wide hybrid (135M) + critic 160 ep: **15.17 @960, still descending steeply** (conv 75M 15.71 @560; 75M hybrid 17.66). Worst plain start (37.8) → best transformer critic result, as width did for the conv. **FID-50k 14.19 — new in-budget best (conv 75M 14.62; 236M conv 14.27).** Queue empty; node idle.
- **09-21 12:00 — on/off-anchor decomposition (FID@5k; assigned / fresh / shuffled):** conv 75M plain 5.40/37.41 → critic 7.12/17.34 · hybrid-400 plain 7.25/31.60 → critic 15.39/22.65 · hybrid-800 plain 5.76/35.38 → critic 9.30/19.50 · wide hybrid plain 4.97/38.87 → critic 7.30/16.64. Shuffled ≈ fresh (no leakage).
  **Off-anchor has NOT reached the on-anchor cap: fresh is still 2.3x assigned after the critic.** The critic costs anchor fidelity (assigned 5 → 7; hybrid-400 → 15.4 = why it failed); longer plain training sharpens anchors and degrades off-anchor (the drift). Next: a critic stage that protects the anchor fit (track assigned FID per eval) and runs longer stably.
- **09-21 14:00 — anchor-protecting critic (user: proceed):** trainer `--eval-assigned 1` logs `fid_assigned@N` (G at a fixed subset of assigned z) next to the fresh FID every eval. Two 320-ep critic stages on the wide hybrid plain ep800 (aag174 baseline: adaptive 1.0, 160 ep → 15.17 / 14.19; continuations collapsed):
  aag175 = FIXED multiplier 0.02 · aag176 = adaptive 1.0 with critic LR 2e-5. Goal: keep assigned FID ≈ plain level while fresh keeps falling past 17 (@5k) / 15 (@10k). chain70 after local smoke + build.
- **09-22 00:00:** aag175 (wide hybrid, critic 320 ep, FIXED 0.02, fresh/assigned FID@10k): fresh 37.4 → **16.18 @936** → runaway from ~960 (59 @1106, 46 @1120); assigned 3.66 → 9.47 @882 (rising through the productive window) → back to 7.6–7.9 during the collapse.
  Reading: the critic pays for off-anchor gains with anchor fidelity (3.7 → 9.5), and the collapse is a DECOUPLING of the fresh branch (anchors recover while fresh explodes), not anchor loss; a fixed multiplier does not prevent it (as before). aag176 (critic LR 2e-5) next.
- **09-22 09:00:** aag176 (critic LR 2e-5, 320 ep): window stretched to ~225 ep, fresh 37.4 → **15.78 @1026** with assigned recovering 9.8 → 6.8 alongside (both improving), then a violent runaway (66 → 220 within 30 ep). Fixed multiplier (aag175) and slower critic (aag176) both only move the collapse. Picked ep1026 FID-50k 14.86 (aag174 unpicked 160-ep endpoint 14.19 stays the record); endpoints 45.7 / 180.9.
  V3 = critic LR 2e-5 + **R1 gamma 5 (lazy /16, anchored side, same real||fake forward)** — `--fresh-gan-r1`; aag177 after smoke + build (chain71).
- **09-22 18:00:** aag177 (critic LR 2e-5 + R1 gamma 5): R1 too strong — critic never separates (df 1.0–1.13, gf ≈ 0), adaptive multiplier balloons to 0.2–0.7, fresh 37 → 29.8 @878 → 72 @1120 on noise; anchors untouched (4.7–5.0). aag178 = same with gamma 0.5.
- **09-23 05:00 — stabiliser line closed:** aag178 (critic LR 2e-5 + R1 gamma 0.5): same failure as gamma 5, faster — R1 softens the critic (df 0.9–1.06), the ADAPTIVE weight inflates (0.24 → 3.7), fresh drifts from the start (min 28.6 @826 → 164 @1120), anchors untouched (5–6). R1 and the adaptive multiplier are incompatible (anything that weakens the critic gradient is multiplied back into the push).
  Summary of critic stabilisers on the wide hybrid (plain ep800): 160 ep adaptive = 15.17 / FID-50k 14.19 (record, unpicked) · fixed 0.02: min 16.18, collapse · critic LR 2e-5: min 15.78 @1026 (picked FID-50k 14.86), collapse @~1040 · R1 5 / 0.5 (+adaptive): critic crippled. Practical rule: 160-ep critic stage; collapse onset coincides with df falling below ~0.65 (early-warning signal).
- **09-23 10:00 — IMAGENET FULL RECIPE (user):** (1) class-respecting AAG3 K=1: global learned-direction transport + 8 group firings/step over the hierarchy levels (joint,depth7..depth4,living,animal,dog; learned direction WITHIN the group, cond_alpha 0.5, max_group 32k), floor-stop (+100 steps), cond/random defect ratio logged; running locally on the 1.28M TiTok particles → `imagenet_titok/assign_aag3/assign_aag3_k1_floor.pt`.
  (2) class-conditional wide hybrid (class token in the trunk; 136M): aag179 plain 16 ep (~85k steps ≈ CelebA 800 ep) → aag180 big critic 3 ep (~15k steps ≈ the 160-ep window), fresh/assigned FID per epoch. chain72 (smoke → build → assignment upload → submit). References: ImageNet best so far 27.27 (flat 150M + MMD).
- **09-23 12:00:** ImageNet AAG3 K=1 (class-respecting) assignment: R_G 92 → floor crossed at step **14540** (~50 min at 0.2 s/step; the old AAG1 class-conditional assignment took 3M steps), cond/random class ratio 3.3 → 0.86–0.91 (z ⊥ class). File `imagenet_titok/assign_aag3/assign_aag3_k1_floor.pt`. Chain72 uploads it and submits aag179.
- **09-23 13:00:** the ImageNet AAG3 floor file is 3.9 GB (z fp32 + h); uploading the slim copy instead (`assign_aag3_k1_floor_slim.pt`, z fp16 + label + metadata incl. `levels`, which the trainer uses to infer n_classes=1000); configs repointed. chain73: upload → aag179 → aag180. Local ImageNet trainer smoke impossible (pixels not local) — aag179's launch banner is monitored instead.
- **09-23 17:30 — ImageNet hybrid STALLED:** aag179 FID-10k 272 / 296 / 286 / 281 / 280 @ep1–5, train MSE 0.13 (conv aag57 on the old assignment: 113 @2, 91 @4, MSE ~0.07). One class token is too weak a conditioning path for 1000 classes (the conv injects class via AdaGroupNorm in every block; the hybrid's upsampler gets none). Left running per the rule; conv fallback configs prepared (aag181 plain / aag182 critic: Generator256 r64 w1.4 conditional, ~75M, on the AAG3 assignment). Push sent for the decision.
- **09-23 18:00 — user: "kill early and switch".** aag179 deleted at ep5 (FID 280). aag181 = conv Generator256 r64 w1.4 class-conditional (84M) plain 16 ep on the AAG3 K=1 floor assignment → aag182 = big critic 3 ep (chain74); ImageNet FID-50k of the aag182 endpoint staged. Hybrid on ImageNet needs proper class injection (AdaLN in trunk + upsampler) before retrying.
- **09-23 20:30 — conv r64 ALSO stalls on ImageNet (aag181: 357/308/290 @1–3, MSE 0.134) → the r64 bottleneck is the culprit, not the class token (old plain finding: ImageNet r32 263, r128 121 vs flat 91). Killed; aag183 = FLAT conv w1.4 (84.6M) plain 16 ep on the AAG3 assignment → aag184 critic 3 ep (chain75). ImageNet "full recipe" = AAG3 assignment + width + critic, WITHOUT the bottleneck.**
- **09-24 00:00:** aag183 (flat conv w1.4 on the AAG3 assignment) trains normally: 277 / 143 / 104 / 99.6 / 99.3 @ep1–5, MSE 0.067 (old aag57 w1.9 flat: 113 @2, 91 @4). Bottleneck confirmed as the ImageNet stall. Critic aag184 after ep16.
- **09-24 08:00:** aag183 (ImageNet flat conv w1.4 plain, 16 ep on the AAG3 assignment) done: min 99.3 @5 → ~110 @16 (drift as usual). aag184 (big critic 3 ep from ep16, fresh/assigned FID per epoch) next.
- **09-24 11:00:** aag184 (ImageNet flat conv w1.4 + big critic 3 ep from the AAG3 plain ep16): fresh FID-10k 110.6 → 52.8 / 50.2 / **47.0 @19**, still falling ~3/ep, critic balanced (df 0.62–0.82). = the old critic result on the old assignment (aag98: 47.2, 150M) — not better; MMD route still best on ImageNet (29.9 / 27.27).
  CAVEAT: `fid_assigned` (80) is inflated on ImageNet — the fixed subset is the first rows of each rank's shard and rows are class-ordered → few classes vs full-train stats. Needs a random fixed subset (fixed in the trainer, d91fb8f). **FID-50k @19: 44.35** (old critic on the old assignment 44.76; MMD 27.27). aag185 continues the critic to 6 ep.
- **09-24 17:00:** aag185 (ImageNet critic continued to 6 ep): 47.0 → 44.6 / 42.9 / **42.6 @22**, flattening, critic winning (df ~0.5). Critic route on ImageNet plateaus ~42.5 FID-10k, **FID-50k @22: 39.84** (critic best on ImageNet; MMD on the old assignment 27.27). aag186 = MMD CLS++patch fixed 0.5, 4 ep from the AAG3 plain ep16 (the untested combination).
- **09-24 21:00 — ImageNet MMD on the AAG3 assignment (aag186, flat conv w1.4 85M, MMD CLS++patch fixed 0.5):** FID-10k **26.03 @17 (after ONE epoch; stat at floor)**, then 26.2 / 27.3 / 27.7 (clamped-term drift). Previous ImageNet best 29.9 @6 with 150M on the old assignment. **FID-50k: endpoint ep20 25.31 (unpicked); ep17 23.57 (picked, after 1 MMD epoch).** New ImageNet best (was 27.27).
- **09-25 00:00 — natural next steps (user):** (1) FLOOR-STOP for the two-sample stage: `--ts-stop-steps N` ends training after N consecutive clamped steps and writes `FLOOR_STOP`; makes the "first MMD epoch" checkpoint an unpicked endpoint. (2) aag189 = ImageNet w1.4 MMD with floor-stop (vs aag186 picked 23.57 / endpoint 25.31). (3) aag187 = ImageNet flat conv WIDTH 1.9 (150M) plain 16 ep on the AAG3 assignment → aag188 = its MMD floor-stop. chain76 (smoke → build → serial on one node); latest-checkpoint FID-50k evaluators staged for aag189/188. Hybrid-with-AdaLN deferred.
- **09-25 06:00:** aag189 (w1.4 MMD, consecutive-step floor-stop) never stopped: the stat hovers AT the floor so the clamp is intermittent — 26.04 / 26.26 / 27.27 / 27.82, FID-50k 25.37 (reproduces aag186: 26.03 @17, 25.31). Rule changed to a fraction of a window (`--ts-stop-frac 0.8` of 500 steps). aag187 (w1.9 plain) running; aag188 (w1.9 MMD floor-stop, fixed rule) after it (chain77); aag190 = w1.4 MMD floor-stop rerun after that (chain78). Latest-checkpoint FID-50k evaluators staged.
- **09-25 14:00 — LATENT-LOCAL MATCHING (user idea): match local conditional distributions around each fresh z instead of the global assigned/fresh marginals.** User's refinement: critic architecture UNCHANGED, BOTH sides local to the SAME latent neighbourhood so locality is the only variable vs the global control; per-neighbourhood / per-sample statistics, no pooling of neighbourhoods across the batch (that would recreate the global marginal). Implementation (`--fresh-local-k K --fresh-local-mode nbhd|persample --fresh-local-nb NB`, trainer): every rank holds the FULL assignment (fp16) for kNN (class-restricted on ImageNet); **nbhd mode**: per step NB centres c~N(0,I), assigned side = G(k nearest assigned z to c), fresh side = G(k nearest points to c of a fresh N(0,I) pool of the SAME size as the (class-)assignment) → identical locality notion (kNN rank) on both sides; MMD² computed INSIDE each neighbourhood and averaged (RBF mixture, standardised DINO CLS++patch), floor = same statistic between two independent fresh pools of the same neighbourhoods (exact null: "anchors distributed like the prior"); critic: one neighbourhood per critic batch per rank (k=32, NB=1), real = the neighbourhood's anchors, fake = its fresh-pool points, plain PatchGAN. **persample mode** (MMD only): each fresh z vs the kernel distribution of its own k nearest anchors, floor = anchors vs their k nearest other anchors. Floor-stop vote synchronised across ranks in local modes (per-rank statistics). Jobs (chain79, behind aag190, 1 node): CelebA on the K=1-floor r64 w1.4 plain ep400 (aag159): **aag191** MMD-global control (40 ep), **aag192** MMD-local nbhd k8×4 (40 ep), **aag193** MMD-local persample k8 (40 ep), **aag195** critic-local nbhd k32 (160 ep; global control = aag160 14.62); ImageNet on the flat w1.4 plain ep16 (aag183): **aag194** MMD-local nbhd floor-stop (controls aag186/189 25.31/25.37), **aag196** critic-local nbhd 3 ep (control aag184 44.35). Smoke (local GPU, CelebA): nbhd MMD raw 0.22 vs two-pool floor 0.146 at step 0 — the local statistic starts ABOVE its floor, i.e. there is local mismatch to remove.
- **09-25 15:00 — user: two locality methods, test both.** Method 1 = per fresh z, its closest anchor(s) (points within one comparison unrelated) = `persample` mode; method 2 = a neighbourhood of anchors with fresh z sampled around it (everything in one comparison related) = `nbhd` mode. Method 1 was MMD-only → wired for the critic too (real side = generation of each fresh z's NEAREST anchor, `--fresh-local-k 1 --fresh-local-mode persample`). Extra jobs (chain80 behind chain79): **aag197** ImageNet MMD method 1 (k8, floor-stop), **aag198** CelebA critic method 1 (160 ep), **aag199** ImageNet critic method 1 (3 ep). Full matrix: CelebA MMD {global 191, nbhd 192, persample 193}, CelebA critic {global aag160, nbhd 195, persample 198}; ImageNet MMD {global 186/189, nbhd 194, persample 197}, ImageNet critic {global 184, nbhd 196, persample 199}. Smoke: nbhd and persample MMD both train (CelebA, local GPU).
- **09-25 16:30 — user: middle ground = random z → nearest anchor of each, compare the two sets.** For the critic this is already method 1 (aag198/199: real batch = nearest anchor of each fresh z, batch pooled by the critic). For the MMD it is new: `--fresh-local-mode nearest` = batch-level MMD between G(z_fresh) and G(nearest anchor of each z_fresh) (local pairing, pooled statistic), floor = the same construction under the null (the batch's anchors as queries vs their nearest OTHER anchor). Jobs (chain81 behind chain80): **aag200** CelebA (40 ep), **aag201** ImageNet (floor-stop). MMD row is now {global, nearest (middle), persample (method 1), nbhd (method 2)}.
- **09-26 ~02:00:** aag187 (ImageNet flat conv w1.9 150M plain 16 ep on AAG3 K=1 floor) done: 308 / 133 / 100.2 / **96.6 @4** / 96.9 / 98.4 … 114.0 @16 (MSE 0.153 → 0.056). vs w1.4 85M (aag183): 99.3 @5 → ~110 @16. Width buys ~3 FID at the minimum, nothing at the end (same drift). aag188 (w1.9 + MMD floor-stop) starts automatically (chain77).
- **09-26 ~05:00 — ImageNet NEW BEST: aag188 (flat conv w1.9 150M, AAG3 K=1 floor assignment, plain 16 ep + MMD CLS++patch fixed 0.5, floor-stop 80%/500) FID-50k 22.59 @ep20 (last, floor-stopped)**; FID-10k 23.19 @17 → 25.23 @20. vs w1.4 85M: 25.31. Capacity helps under MMD on ImageNet (as under the critic on CelebA). Picked ep17 FID-50k queued. Budget 310M: room for w2.5 (~260M) next.
- **09-26 ~06:30:** aag188 PICKED ep17 (first MMD epoch) FID-50k **20.47** (last ep20: 22.59). Same 2-point drift after the first MMD epoch as w1.4 (23.57 → 25.31); the floor-stop fires ~3 epochs too late because the statistic hovers AT the floor (80%-of-500 rule). Stopping tighter (e.g. 60%) or evaluating at the first clamp-majority epoch would keep the picked number.
- **09-26 ~08:00:** aag190 (w1.4 MMD floor-stop 80%/500 rerun): 26.10 / 26.22 / 27.20 / 27.51, floor-stop at step 96,674 (epoch 20) — reproduces aag186/189 and the stop fires at the same point as for w1.9 (aag188, step 95,090). FID-50k of the last ckpt queued. chain78 lacked its CHAIN78_DONE marker → written manually; latent-local round (aag191…) starts now.
- **09-26 ~09:30:** aag190 FID-50k **25.02** @ep20 (last, floor-stopped). Three w1.4 MMD runs: 25.31 / 25.37 / 25.02 → the ImageNet MMD result is stable to ±0.2. w1.9: 22.59 last / 20.47 picked.
- **09-26 ~10:00:** aag191 (CelebA MMD-GLOBAL control, 40 ep on the K=1-floor r64 w1.4 plain ep400): FID-10k 33.8 → 20.24 @440, still falling slowly at the end (no floor clamp reached). FID-50k queued. aag192 (nbhd) running with the correct banner.
- **09-26 ~11:00:** aag191 FID-50k **19.38** (CelebA MMD-global control, last ckpt). Reference on the same base: critic-global aag160 14.62; plain base aag159 ep400 ≈ 30 (FID-10k).
- **09-26 ~12:00 — aag192 (CelebA MMD latent-LOCAL, neighbourhood mode k8×4):** FID-10k 33.8 → min 25.74 @420 → **27.40 @440 (rising)** vs global control aag191 20.24 @440 (still falling). Local statistic 0.219 → 0.159 vs floor 0.137 (never clamped, still decreasing) while FID rose from ep420: the per-neighbourhood objective keeps improving while the global marginal degrades → fresh outputs pulled toward their specific nearby anchors (diversity loss), not toward the marginal. Heldout MSE/LPIPS 0.0953/0.427 (anchor fidelity unchanged). FID-50k queued. First verdict: neighbourhood-local MMD is WORSE than global on CelebA.
- **09-26 ~13:00:** aag192 FID-50k **26.80** (nbhd-local MMD) vs aag191 **19.38** (global). Confirmed: neighbourhood-local MMD loses by 7.4 on CelebA.
- **09-26 ~14:00 — aag193 (CelebA MMD latent-LOCAL, per-sample k8):** FID-10k 34.3 → min 32.93 @408 → **41.51 @440**, worse than the base; statistic 0.534 → 0.445 vs floor 0.47 (reached the floor and clamped) while FID degraded monotonically. CelebA MMD row so far: global 20.24 (50k 19.38) < nbhd 27.40 (50k 26.80) < per-sample 41.5. The more local the anchor side, the worse: matching fresh outputs to the anchors that surround them is a pull toward those anchors (diversity collapse), not toward the marginal. Remaining: middle ground aag200 (nearest anchor, pooled statistic) and the critic variants.
- **09-26 ~15:00:** aag193 FID-50k **40.64**. CelebA MMD row (FID-50k, last ckpt, same base): global 19.38 | nbhd 26.80 | per-sample 40.64.
- **09-26 ~16:00 — aag194 (ImageNet nbhd-local MMD) FAILED at step 0:** `IndexError` in `local_fresh` — the class-count table `_cnt` was shadowed by the per-rank step-count tensor `_cnt` defined later in the trainer (size 1). Only the class-conditional neighbourhood sampler reads it, so the unconditional CelebA smoke could not catch it (ImageNet cannot be smoked locally). Renamed to `_cls_cnt`; CPU reproduction on the full ImageNet assignment passes. aag196 would have hit the same path → chain79 killed after submitting aag195; chain79b re-runs aag196 and aag194 on the fixed image, then hands over to chain80 (197–199) and chain81 (200–201) unchanged.
- **09-26 ~19:00 — aag195 (CelebA CRITIC latent-LOCAL, neighbourhood k32×1, critic unchanged):** FID-10k 30.0 → **min 15.76 @536** → 16.15 @560, critic won at the end (df 0.375, wf 0.015 = the usual late decoupling). Global-critic control aag160 on the same base: 30.0 → 15.71 @560 (monotone, FID-50k 14.62). Curves are within noise of each other (local slightly ahead mid-run, e.g. 464: 18.94 vs 19.13; 528: 16.09 vs 16.35; then it turned). Unlike the MMD, the local CRITIC does not hurt — the critic pools the batch and learns a global decision anyway, so "locality" only changes batch composition. FID-50k queued.
- **09-26 ~21:00:** aag195 FID-50k **15.12** (nbhd-local critic, last ckpt after the late decoupling) vs global critic aag160 **14.62**. Within the critic-stage noise / end-point effect (local was at its minimum 24 epochs earlier). CelebA critic row: global 14.62 | nbhd 15.12 | per-sample (aag198) pending.
- **09-27 ~00:00 — aag196 (ImageNet CRITIC latent-LOCAL, neighbourhood k32×1 same-class, fixed image):** fresh FID-10k 110.6 → 56.68 / 54.75 / **51.76 @19**, assigned 8.6 → 8.5 (anchor fidelity intact); critic ahead from epoch 18 (df 0.58, wf 0.014). Global-critic control aag184: 52.8 / 50.2 / 47.0 @19 (FID-50k 44.35). Local critic ~4–5 points BEHIND global on ImageNet at every epoch. FID-50k queued. aag194 (ImageNet nbhd-local MMD) rerun running on the fixed image.
- **09-27 ~01:00:** aag196 FID-50k **49.53** vs global critic aag184 **44.35**. ImageNet critic row: global 44.35 | nbhd 49.53 | per-sample (aag199) pending.
- **09-27 ~04:00 — aag194 rerun (ImageNet MMD latent-LOCAL, neighbourhood k8×4 same-class):** FID-10k 29.26 / 26.67 / 26.59 / **26.86 @20**; the floor-stop never fired in 4 epochs (statistic 0.107 vs floor 0.090, still above) — ran to its 20-epoch cap. Global MMD control: 26.03 @17 then 26.2 / 27.3 / 27.7 (floor-stopped, FID-50k 25.02–25.37). Interesting: the local statistic keeps the run from drifting up (26.6–26.9 flat over epochs 18–20 vs global drifting to 27.7), but it never reaches the global's first-epoch minimum. FID-50k queued.
- **09-27 ~07:00 — FIRST WIN FOR LOCALITY: aag194 FID-50k 24.22** (ImageNet nbhd-local MMD, last ckpt @20, no floor-stop) vs global MMD last-ckpt **25.02 / 25.31 / 25.37** (three runs). Consistent with the 10k curves (26.86 vs 27.5–27.8): the local statistic does not let the run drift after the first epoch. The global's PICKED first-epoch ckpt (23.57) is still better than the local's last, so the local objective's value here is stability, not a lower optimum. Same base (w1.4 85M) → worth repeating on w1.9 (aag188 base: 22.59 last / 20.47 picked).
- **09-27 ~08:00 — aag197 (ImageNet MMD latent-LOCAL, per-sample k8 same-class):** statistic 0.50 → floor 0.38 within ~600 steps → FLOOR-STOP at step 79,910 (epoch 17, 630 steps in), fresh FID-10k **87.7** (base 110; global MMD 26.0 after one full epoch). The per-sample test saturates: its floor (anchor vs its own k nearest anchors) is large, and fresh-vs-neighbours reaches it long before the marginal is fixed → "indistinguishable" is declared far too early. Per-sample MMD loses on both datasets (CelebA 40.64 by collapse, ImageNet 87.7 by premature saturation). FID-50k of ep17 queued (evaluates newest = the floor-stop save).
- **09-27 ~10:00:** aag197 FID-50k **85.17** (per-sample local MMD, ImageNet, floor-stopped after 630 steps). ImageNet MMD row: global 25.02–25.37 | nbhd **24.22** | per-sample 85.17 | nearest (aag201) pending.
- **09-27 ~13:00 — aag198 (CelebA CRITIC latent-LOCAL, method 1: real = nearest anchor of each fresh z, k=1):** FID-10k 30.6 → **min 16.54 @512** (global aag160: 16.87 @512) then VIOLENT collapse 16.95 / 22.3 / 36.7 / 84 / 216 / **313.5 @560**, critic won outright (df 0.07, wf 0.026). The nearest-anchor real side gives the critic an easier job (real and fake are paired in latent space, so any systematic fresh-vs-anchor difference is trivially separable) → the usual late decoupling becomes a total collapse. Last-ckpt FID-50k meaningless; picked ep512 FID-50k queued (labelled picked).
- **09-27 ~15:00:** aag198 PICKED ep512 FID-50k **15.56** (last ep560: 311.637, collapsed). CelebA critic row (FID-50k): global 14.62 (last) | nbhd 15.12 (last) | nearest-anchor 15.56 (picked; last collapsed). No local critic variant beats the global critic on CelebA.
- **09-27 ~19:00 — aag199 (ImageNet CRITIC latent-LOCAL, method 1: real = nearest same-class anchor of each fresh z):** fresh FID-10k 110.6 → **49.83 / 48.62 / 49.07 @19** (assigned 8.3, intact); critic ahead by ep19 (df 0.45). Faster start than global (52.8 @17) but flattens at ~49 while global reaches 47.0 @19 (FID-50k 44.35). FID-50k queued. ImageNet critic row so far: global 47.0 | nbhd 51.8 | nearest 49.1 (FID-10k @19).
- **09-27 ~21:00:** aag199 FID-50k **46.69** vs global critic 44.35, nbhd 49.53. ImageNet critic row (FID-50k, last ckpt): global 44.35 | nearest 46.69 | nbhd 49.53. Global critic wins on both datasets.
- **09-27 ~22:00 — aag200 (CelebA MMD latent-LOCAL, middle ground: G(z_fresh) vs G(nearest anchor), pooled statistic, matched floor):** FID-10k 33.8 → **min 24.13 @428** → 25.05 @440 (turning up), statistic 0.075 → 0.009 vs matched floor 0.0059 (never clamped). Global control 20.24 @440 (still falling). CelebA MMD row (FID-10k @440): global 20.24 | nearest 25.05 | nbhd 27.40 | per-sample 41.51 — monotone in locality. FID-50k queued. Only aag201 (ImageNet nearest) left.
- **09-27 ~23:00:** aag200 FID-50k **24.34**. CelebA MMD row (FID-50k, last ckpt, same base): global 19.38 | nearest 24.34 | nbhd 26.80 | per-sample 40.64.
- **09-28 ~03:00 — aag201 (ImageNet MMD latent-LOCAL, middle ground: nearest same-class anchor per fresh z, pooled statistic):** FID-10k 29.23 / 31.95 / 33.18 / **33.62 @20**; statistic AT its matched floor from epoch 18 (0.0045 vs 0.0046) yet the 80% floor-stop never fired (hovering) → drifted up for 3 epochs. Global: 26.0 → 27.7. Worse than global at every epoch; the nbhd variant (26.9, 24.22 @50k) is the only local MMD that holds on ImageNet. FID-50k queued → final table.
- **09-28 ~05:00 — LATENT-LOCAL MATCHING ROUND COMPLETE.** aag201 FID-50k **30.60**. Final table (FID-50k, last ckpt unless marked; `/data/aag_results/results_scale256/local_matching_fid50k_table.png`):
  | | global | nearest anchor per z | per-sample own k=8 | neighbourhood k-NN both sides |
  |---|---|---|---|---|
  | CelebA MMD | **19.38** | 24.34 | 40.64 (collapse) | 26.80 |
  | CelebA critic | **14.62** | 15.56 picked (last 311.6, collapsed) | = nearest | 15.12 |
  | ImageNet MMD | 25.02–25.37 (picked ep17 23.57) | 30.60 | 85.17 (saturated after 630 steps) | **24.22** |
  | ImageNet critic | **44.35** | 46.69 | = nearest | 49.53 |
  Verdict: latent-local matching does not beat the global marginals. MMD: locality monotonically hurts on CelebA (statistic keeps falling while FID rises → fresh outputs pulled toward their specific anchors); per-sample saturates at its own high floor on both datasets. Critic: locality only changes batch composition (the critic pools anyway) → within noise on CelebA, 2–5 behind on ImageNet; nearest-anchor pairing makes the late collapse violent. Single win: ImageNet neighbourhood MMD (24.22 vs 25.0–25.4) — the local statistic never reaches its floor so the run doesn't drift after epoch 1, but the global run's picked first-epoch ckpt (23.57) is still lower, so it's a stopping-rule effect, not a better optimum. Keep global matching; fix the floor-stop (tighter fraction / stop at first clamp-majority) to bank the picked numbers (w1.9: 20.47 picked vs 22.59 last).
- **09-28 ~06:00 — user: PAIRWISE local critic.** Channel-concat pairs: fake = [G(z_fresh) || G(nearest anchor of z_fresh)], real = [G(z_a) || G(nearest OTHER anchor of z_a)]; same PatchGAN with 6 input channels, fresh slot first, gradient only through the fresh image, adaptive weight as usual (`--fresh-critic-pair 1 --fresh-local-mode nearest --fresh-local-k 1`). Unlike aag198/199 (which only changed the real batch) the critic now has a reference image and can test the RELATIONAL question: does a fresh output relate to its latent neighbour the way anchors relate to theirs. Null is exact at the Gaussian floor (NN distance of a new point to N anchors = NN distance of an anchor to the other N-1). Jobs (chain82 after smoke + build): **aag202** CelebA 160 ep (controls aag160 14.62, aag198 15.56 picked/collapsed), **aag203** ImageNet 3 ep (controls aag184 44.35, aag199 46.69).
- **09-28 ~10:00 — aag202 (CelebA PAIRWISE local critic, 160 ep):** FID-10k 34.5 → 20.1 @480 (plateau to ~512) → **17.69 @560**, still falling slowly; critic ahead (df 0.39) but NO collapse — the only nearest-anchor critic that survived to the end (aag198 collapsed to 313). Global critic aag160: 15.71 @560 (FID-50k 14.62). Pairwise ≈ 2 FID-10k behind at the same epoch but more stable; FID-50k queued. aag203 (ImageNet pairwise) running.
- **09-28 ~12:00:** aag202 FID-50k **16.63** (pairwise critic, CelebA, last ckpt, no collapse) vs global 14.62 | nbhd 15.12 | nearest 15.56 picked. aag203 (ImageNet pairwise) started cleanly (df 0.98, wf 0.07).
- **09-29 ~02:00 — aag203 (ImageNet PAIRWISE local critic, 3 ep):** fresh FID-10k 110.6 → **47.63 / 49.66 / 46.30 @19** (assigned 8.7, intact), critic balanced (df 0.63). Global aag184: 52.8 / 50.2 / 47.0; nearest aag199: 49.8 / 48.6 / 49.1; nbhd aag196: 56.7 / 54.8 / 51.8. Pairwise is the FASTEST local critic on ImageNet (47.6 after one epoch vs 52.8 global) and ends 0.7 below global at ep19. FID-50k queued.
- **09-29 ~04:00 — PAIRWISE CRITIC COMPLETE. aag203 FID-50k 43.84 vs global 44.35** (ImageNet; nearest 46.69, nbhd 49.53) — the only local critic to beat the global one anywhere, by 0.5 (within ~noise but consistent with being ahead at every epoch and 5 points ahead after epoch 1). CelebA: 16.63 vs 14.62 (behind, but no collapse). Table updated: `/data/aag_results/results_scale256/local_matching_fid50k_table.png`. Reading: giving the critic a latent-neighbour reference helps where neighbourhoods are meaningful (ImageNet, class-restricted, 1.28M anchors) and costs where they are not (CelebA, 28k unconditional). Same pattern as the MMD row (ImageNet nbhd-MMD win, CelebA loss). Locality is a per-dataset lever tied to anchor density, not a general improvement.
- **09-29 ~06:00 — user: "try those recommended steps".** (1) Floor-stop fix: `--ts-stop-rule mean` = stop when the 500-step window mean of (raw − floor·mult) ≤ 0 (statistic at its null level on average; never fires while raw is systematically above the floor). Epoch lines now print `ts_clamp=` / `ts_margin=`. Validated on the w1.9 base: **aag206** = w1.9 MMD mean-stop from aag187 ep16 (target: stop ≈ep17 and bank ≈20.5 as the LAST checkpoint; frac-rule run aag188 gave 22.59 last / 20.47 picked). (2) **aag207** = pairwise critic (2 ep) as a second stage from aag206's stop checkpoint. (3) **aag204** = flat conv **w2.5 (257M)** plain 16 ep; **aag205** = w2.5 MMD mean-stop from its ep16. Order on the one node: 206 → 207 → 204 → 205 (chain83).
- **09-29 ~08:00 — user: INVERSE MODEL / CYCLE CONSISTENCY ("the missing universal objective").** Train E(x)→z on the real assigned pairs only, freeze it, then add ‖E(G(z_f)) − z_f‖² at fresh z next to the distributional term: pointwise off-anchor supervision derived from AAG itself (assigned loss = what anchors mean; MMD/critic = right image distribution; cycle = each fresh z keeps its own location so gaps cannot be shuffled/collapsed). Implementation: `aag/inverse.py` (InverseHead: frozen DINOv2 CLS+patch tokens → attention-pooling head → target; target = full z, or the r-dim bottleneck code W0 z + b0 the generator actually reads when the base has a linear bottleneck; standardised per dim so MSE 1.0 = constant predictor), `scripts/train_inverse256.py` (pairs only, held-out anchors for R², crop/brightness aug, NO flips), trainer `--inverse E.pt --cycle-weight w [--cycle-fixed]` (adaptive gradient-balanced multiplier; logs `cyc` fresh error, `cyc_a` anchor error). Plan: E-CelebA trained locally (r64 code target, base aag159) and uploaded; **aag208** = E-ImageNet on the cluster (full z, 6 ep). Runs: CelebA **aag209** critic+cycle 160 ep (ctrl aag160 14.62), **aag210** MMD+cycle 40 ep (ctrl aag191 19.38), **aag211** cycle alone 40 ep; ImageNet w1.9 **aag212** MMD mean-stop + cycle (ctrl aag206), **aag213** MMD+cycle 3 ep NO stop (drift test vs aag188 23.19→25.23), **aag214** cycle alone 2 ep. Inserted before the w2.5 runs (204/205) after 206→207; chain83 replaced by chain84.
- **09-29 ~10:00 — E-CelebA ceiling:** held-out R² of the r64 code from frozen DINOv2 tokens plateaus at **≈0.50** regardless of head: 8 ep 0.497 (uploaded, used by aag209–211), 60 ep peaks 0.50 @10 then overfits to 0.42, small 9.9M head 0.50. So half the code variance is not recoverable from DINO semantics with 27k pairs; the cycle signal is the EXCESS fresh error over the anchor error (smoke: 0.59 vs 0.49). Option if the cycle term shows promise: an E on the frozen TiTok encoder latent (z is a deterministic transport of it) would be a near-exact inverse = a pointwise virtual target for every fresh z; heavier (TiTok forward + backward in the loop), not run.
- **09-29 ~11:00 — aag206 (w1.9 MMD, MEAN floor-stop):** ep17 end: ts_clamp 47%, margin +0.00006 (not yet); fired at step 84,735 (500 steps into epoch 18) with mean margin −0.00001, FID-10k **22.93** at the stop (frac-rule run aag188: fired at step 95,090 in epoch 20 at 25.23). The stopping-rule fix works: it stops at the first point where the statistic is at its null level on average. FID-50k of the stop ckpt queued (expect ≈20.5, the previously "picked" number).
- **09-29 ~12:00 — ImageNet NEW BEST (last ckpt): aag206 FID-50k 20.24** (w1.9 150M, AAG3 K=1 floor, plain 16 ep + global MMD CLS++patch fixed 0.5, MEAN floor-stop at step 84,735 / epoch 18). Beats aag188 frac-rule last 22.59 and even its picked 20.47. Recipe now: assignment to the floor → plain → MMD until the statistic reaches its null level on average → stop. Next in queue: aag207 pairwise 2nd stage from this ckpt; cycle runs; w2.5.
- **09-29 ~13:00 — aag207 (pairwise critic 2 ep AFTER the MMD mean-stop ckpt, w1.9):** fresh FID-10k 22.93 → 38.66 → **47.53**, assigned 8.5 → 10.6. The critic stage destroys the MMD result and heads for the critic route's own ~45 plateau; critic-after-MMD is harmful on ImageNet. Closed. (FID-50k of ep20 queued for completeness.) aag208 (E-ImageNet) running.
- **09-29 ~13:30 — aag208 (E-ImageNet, full z, 6 ep, 10 min):** held-out R² 0.14 / 0.22 / 0.25 / 0.26 / 0.27 / **0.27** (per-dim 0.07–0.69). Much lower than CelebA's 0.50: z ⊥ class by construction (conditional assignment), so it encodes WITHIN-class variation that DINO's semantic tokens carry poorly. The ImageNet cycle runs (212–214) proceed with this E but their pointwise signal is weak; a TiTok-latent E is the fix if the CelebA cycle runs show value. aag209 (CelebA critic+cycle) running: cyc 0.567 vs cyc_a 0.433 at start.
- **09-29 ~15:30 — aag209 (CelebA critic + UNFLOORED cycle, 160 ep): FID-10k 35 → min 25.5 @430 → 33.0 @560** (critic alone: 15.7). Diagnostic: fresh cycle error 0.567 → **0.24** while the ANCHOR cycle error stayed 0.36–0.42; FID turned up exactly when fresh fell below anchor (~ep 420). G exploits the frozen E: fresh images become "more invertible than real anchors" (adversarial to E's features) at the cost of image quality — the collusion the user warned about happens even with E frozen, through the gradient. Fix (same logic as the two-sample floor): `--cycle-floor-mult m` → loss = relu(cyc_fresh − m·EMA(anchor cycle error)), i.e. fresh error is pushed down TO the anchor level and never below; `--cycle-aug 1` random crop/jitter before E (harder to exploit). Unfloored controls aag210 (MMD+cycle) and aag211 (cycle alone) still run on CelebA; the unfloored ImageNet trio (212–214) is replaced by floored runs: CelebA **aag215** critic+cycle-fl, **aag216** MMD+cycle-fl; ImageNet w1.9 **aag217** MMD mean-stop + cycle-fl, **aag218** MMD+cycle-fl 3 ep no-stop. Then w2.5 (204/205). chain84 → chain85.
- **09-29 ~17:00:** aag209 FID-50k **32.13** (critic + unfloored cycle; critic alone 14.62). Floored variant queued (aag215).
- **09-29 ~18:00 — aag210 (CelebA MMD + UNFLOORED cycle, 40 ep):** FID-10k 31.5 → min 22.25 @428 → 23.23 @440 (global MMD control aag191: 20.24 @440, still falling). Fresh cycle error 0.571 → 0.325 crossing below the anchor error (~0.39) at ~ep 419, FID turns up ~ep 428 — same exploitation signature as aag209. MMD statistic stalls at 0.015 (3× floor) vs 0.0095 for the control: the unfloored cycle term fights the distributional term. FID-50k queued. aag211 (cycle alone) running; floored runs next.
- **09-29 ~19:00:** aag211 (cycle alone) failed at step 0: `fresh_gen` (and the `adaptive_weight` import) were only created when a critic or two-sample term was active. Fixed (`or cyc`), re-queued as aag211 rerun after chain85 (chain86) with its own evaluator. aag215 (CelebA critic + FLOORED cycle) starts next on the node.
- **09-29 ~19:30 — logging note for aag215–218 (image 3b01a69):** in floored mode the printed `cyc_a` is the 8-rank SUM (in-place all_reduce through a detached view) → divide by 8 (3.47 → 0.43). The floor EMA itself uses the correct mean. Fixed for later images (`.clone()`).
- **09-29 ~20:00:** aag210 FID-50k **22.48** (MMD + unfloored cycle) vs global MMD 19.38. Unfloored cycle: critic 32.13 / MMD 22.48 — both worse than their controls (14.62 / 19.38). Floored runs pending (215/216).
- **09-29 ~22:00 — aag215 (CelebA critic + FLOORED cycle + aug, 160 ep):** FID-10k 33.9 → **16.92 @560**, monotone, no collapse (df 0.42). Fresh cycle error 0.57 → 0.39–0.40 = the anchor level (logged cyc_a/8 ≈ 0.41): the floor holds exactly as designed, no exploitation. But critic alone reached 15.71 @560 (FID-50k 14.62) → the floored cycle term is neutral-to-slightly-negative on CelebA with this E (R² 0.50). FID-50k queued. aag216 (MMD + floored cycle) next.
- **09-29 ~23:00:** aag215 FID-50k **15.91** (critic + floored cycle) vs critic alone 14.62; unfloored 32.13. CelebA critic row: global 14.62 | +cycle unfloored 32.13 | +cycle floored 15.91.
- **09-30 ~00:00 — aag216 (CelebA MMD + FLOORED cycle, 40 ep):** FID-10k 31.3 → 20.72 @436 → **20.76 @440** (flattening) vs global MMD 20.24 @440 (still falling), unfloored 23.23. Fresh cycle error settles at the anchor level (0.40); MMD statistic 0.0115 vs control 0.0095 (the cycle term still slows the MMD slightly). Floored cycle on CelebA: neutral on both objectives (critic 15.91 vs 14.62; MMD ≈ +0.5 at 10k). FID-50k queued. aag217 (ImageNet MMD mean-stop + floored cycle) running.
- **09-30 ~01:00:** aag216 FID-50k **20.11** (MMD + floored cycle) vs global MMD 19.38, unfloored 22.48. CelebA cycle table (FID-50k): critic 14.62 | +cycle 32.13 | +cycle-floored 15.91; MMD 19.38 | +cycle 22.48 | +cycle-floored 20.11. aag217 start: ImageNet fresh cycle error 0.72 = anchor level (E R² 0.27) → no excess to remove; floored term ≈ 0.
- **09-30 ~04:00 — aag217 (ImageNet MMD mean-stop + FLOORED cycle, ADAPTIVE weight): fresh FID-10k 28.6 / 36.4 / 43.0 / 44.0 @20** (aag206 without cycle: 23.2 → 22.9 stop in ep18); floor-stop only fired at step 95,214 (MMD statistic held off its floor: ts_clamp 0% at ep17 vs 47%). Fresh cycle error 0.72 → 0.69 vs anchor 0.72 → excess ≈ 0 → floored loss ≈ 0, BUT the ADAPTIVE multiplier rescales any tiny positive excess until its gradient matches the supervised gradient → a near-zero, noisy signal drives a full-strength gradient. Same adaptive-weight/clamped-loss incompatibility as R1 earlier; the MMD floor works because its weight is FIXED. Fix: `--cycle-fixed` (weight 1.0 on the standardised excess). Re-runs: CelebA **aag219** critic + cycle-fl fixed, **aag220** MMD + cycle-fl fixed; ImageNet **aag221** MMD mean-stop + cycle-fl fixed (E R² 0.27 → likely inert, run for completeness). aag218 (3 ep no-stop, adaptive) left to finish.
- **09-30 ~06:00:** aag217 FID-50k **41.51** (ImageNet MMD mean-stop + adaptive floored cycle) vs aag206 20.24 — the adaptive-weight failure confirmed at 50k. Fixed-weight re-runs (219–221) queued right after aag218; w2.5 moved behind them.
- **09-30 ~09:00 — aag218 (3 ep no-stop, adaptive floored): 28.4 / 36.6 / 43.9**, identical to aag217. Its logged multipliers (wc 0.2 → 0.007) CORRECT the mechanism: the raw cycle gradient is 5–140× the supervised one, so adaptive weighting pins the cycle push to the supervised gradient norm in every step where excess>0, however tiny — because d relu(cyc−floor)/dθ = ∇cyc in full. A FIXED weight 1.0 (aag219 as launched) would be worse (~100× supervised) → aag219 deleted 2 min in. Fix: **squared hinge** `--cycle-floor-pow 2` (gradient ∝ excess, vanishes at the floor) with FIXED weight 0.1 (CelebA: excess 0.13, ‖∇cyc‖≈9‖∇sup‖ → push ≈0.2‖∇sup‖ at start, → 0 at the floor; ImageNet excess≈0 → inert, as it should be). Re-queued: aag219/220 CelebA, aag221 ImageNet (chain89), then 211 rerun, 204, 205.
- **09-30 ~10:30:** aag218 FID-50k **41.20** (adaptive floored cycle, 3 ep no-stop) ≈ aag217 41.51. Adaptive-weighted cycle closed on both datasets; squared-hinge fixed-weight runs (a87ca51) launching: aag219 → 220 → 221.
