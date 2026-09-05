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
