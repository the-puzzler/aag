# AAG @ 256² — agent handoff (2026-09-06, research paused by the user)

Read this first; then `cluster/HANDOFF.md` (chronological log), `cluster/RUNBOOK.md` (mechanics), and the memory files
under `~/.claude/projects/-home-ubuntu/memory/` whose names start with `newgen_` / `feedback_`.

## 1. What we are working on

**AAG** (Amortised Assignment Generation): transport the autoencoder latents of a dataset onto a Gaussian *once* (the
"assignment", producing pairs (z_i, x_i) with z ~ N(0,I) jointly), then train a feed-forward generator z → pixels on the
pairs. Sampling = one forward pass from fresh z ~ N(0,I).

**Goal:** beat ROMS-IMLE (arXiv 2607.19332, https://serchirag.github.io/roms-imle/) at NFE=1 with equal or fewer params:
- CelebA-HQ 256², unconditional: FID-50k **6.70 @ 138M**. Our valid best: **29.16** (aag36, 38M).
- ImageNet 256², class-conditional: FID **2.56 @ 310M** (raw 4.16). Our best: **138.3** (aag2, 285M); aag22/aag29 running.

Owner: Matteo (matteo.peluso@odyssey.systems). Worktree: `/home/ubuntu/exp/newgen/.claude/worktrees/aag-scale256`,
branch `worktree-aag-scale256` (many commits, NOT pushed to origin). Run everything with
`PYTHONPATH=<worktree> /home/ubuntu/exp/newgen/.venv/bin/python`.

## 2. The user's standing rules (verbatim where possible)

- "the only rule during assignment is more gaussian is more good" — refined: "more transport is better provided there is
  enough data to support [it]", otherwise you space samples apart and "nowhere is safe to sample from anymore".
- "displacement is not a useful metric"; "PCA stuff doesn't work"; "I don't really trust locality as a useful measure";
  "val mse rising is not necessarily overfit".
- Metrics never decide quality — the user's eyes do. Present comparisons as ONE PNG with column headers (recons: model
  names as column names). Don't claim quality from FID alone.
- Never kill a healthy run on your own judgement. Structural choices get asked. "Ping me when I need to make a decision
  with a push notification and present the options." Updates every ~30 min, not per log line.
- Job names: plain `aag<N>` in project `warhol`, namespace `kubeflow`. Next free number: **aag40**. AE jobs were `matteo-exp-<N>`.
- You may run `kubectl -n kubeflow delete pytorchjob <name>` on the user's aag*/matteo-exp-* jobs.
- Node budget: two 8×B200 nodes for AAG (a third was granted temporarily for AE sweeps). "Capacity is the scheduler's problem."
- Comparison must be CelebA-HQ ONLY (no FFHQ). **Horizontal-flip pair doubling was REJECTED** ("gain too small to warrant
  flipping") — all `celebahq_titok_x2` results (aag23–28, 30, 31) are off the record.
- The user reopened DINO: "have the loss be mse, lpips and dino literally the same recipe as that paper". The literal
  weights don't train (see §6); the user then chose the gradient-matched version.
- "use the large storage drives attached": transient checkpoint copies go to `/opt/dlami/nvme/aag_scratch/ckpts`
  (1.7 TB ephemeral NVMe), durable outputs to `/data/aag_results`. Never park big files in `~/.claude/jobs/*/tmp` (root disk).
- The published AAG recipe is the baseline (MSE + 0.5·LPIPS, bf16, EMA 0.9995); pipeline order is assignment → plain
  generator → finetunes (adversary etc.).

## 3. Infrastructure and how to run things

**Cluster** `eks-train-prod-aps3`, kubeflow PyTorchJobs, 8×B200 per node. Image repo `odydev.azurecr.io/aag:<git-sha>`
(`az acr login -n odydev` when pushes say "authentication required"; ACR replication to ap-south-1 lags a few minutes
after a push — ImagePullBackOff right after pushing is normal, wait). Tags in use:
- `da81e12` — base trainer (assignment/gen/adversary), no hflip, no DINO.
- `5ccf468` — adds `--aug-hflip` (unused now) and the eval_every guard.
- `7bba9c8` — adds `--mse-weight/--dino-weight` (DINOv2 loss). **Use this for anything new.**

```
cluster/launch.sh build                                  # docker build + push, prints tag (~10 min)
AAG_IMAGE_TAG=7bba9c8 cluster/launch.sh submit cluster/configs/<cfg>.yaml aag40
kubectl -n kubeflow get pytorchjob aag40 -o jsonpath='{.status.conditions[-1].type}'
kubectl -n kubeflow logs -f aag40-master-0 | grep -E "fid@|Traceback"
kubectl -n kubeflow delete pytorchjob aag40             # wait for the pod to vanish before reusing a name
```
Configs are YAML stage lists run by `cluster/entry.py` (stages: download HF mirror → `wait_for_file.py $ASSIGN` →
FID stats → generator on 8 GPUs). The YAML is mounted at submit time (config edits need no rebuild; code edits do).
Test the exact command locally on 1 GPU before submitting (`torchrun --nproc_per_node 1 scripts/train_generator256_ddp.py …`).

**Shared storage (EFS)** `/mnt/shared/aag/` inside pods: `hf/` (parquet mirror), `data/` (FID stats), `results_scale256/<ds>/…`,
`torch_hub/` (DINOv2 mirror; configs set `TORCH_HOME` to it). Reach it from the box via the CPU helper pod **`aagcp`**
(busybox + PVC `shared-drive`, 30-day sleep; recreate with the spec in `cluster/RUNBOOK.md`/HANDOFF if it shows Succeeded):
`kubectl -n kubeflow cp <local> kubeflow/aagcp:<remote>.part && kubectl exec aagcp -- mv …` — slow (0.3–8 MB/s), verify
sizes with `stat -c %s` on both ends (truncation happens). Pull checkpoints the same way to `/opt/dlami/nvme/aag_scratch/ckpts`.

**Local box** (this machine, 1 GPU 98 GB, 124 GB RAM): does the assignments (sequential algorithm, fast here) and FID-50k.
Data: `/data/aag_data/{celebahq256,imagenet256}/particles_*.pt`, `/data/aag_data/hf` (parquet), FID stats
`/data/aag_data/celebahq256/fid_stats_train_28000.npz`, `/data/aag_data/imagenet256/fid_stats_train_full.npz`.
Results: `/data/aag_results/results_scale256/<ds>/assign/*.pt` (+ `.log`, `.curve.json`), `…/fid50k.jsonl`.
`/data` is ~90 % full (ImageNet assignments are 6.5 GB each; slim them with `scripts/slim_assignment.py` before upload —
the trainer only needs z+label). Background shell jobs get killed by a memory heuristic when page cache is high: run heavy
Python in the foreground or with `nohup`, load big tensors with `mmap=True`. `pkill -f <pattern>` kills your own shell if
the pattern appears in the command line — kill by PID.

**Pipeline commands**
```
# encode (TiTok-LL-32 VAE, 512-d flat latent; posterior mean; images in [0,1])
python scripts/encode_hf256.py --dataset celebahq256 --encoder titok:yucornetto/tokenizer_titok_ll32_vae_c16_imagenet --amp --out /data/aag_data/celebahq256/particles_titok_ll32.pt
# assignment (CelebA: unconditional; ImageNet: --groups class_groups.pt --levels joint,depth7,…,dog)
python scripts/run_assignment_classes.py --particles <particles.pt> --steps 200000 --alpha 1.0 [--refine-steps 30 --search-subset 8192] [--grp-per-step 128] --save-every 20000 --keep-checkpoints --out <assign.pt>
# score assignments without a generator (3 s each; ranker, not calibrated)
python scripts/assign_surrogate_score.py --seeds 4 <assign1.pt> <assign2.pt> …
# generator: see any cluster/configs/aag256_celebahq_titok*.yaml (400 epochs ≈ 1.3 h on 8 GPUs for 28k pairs; ImageNet 40 epochs ≈ 17 h)
# FID-50k on a checkpoint (the reporting protocol; in-loop fid@10000 reads ~1 higher)
python scripts/eval_fid256.py <ckpt.pt> --fid-stats /data/aag_data/celebahq256/fid_stats_train_28000.npz --n 50000 --out /data/aag_results/results_scale256/celebahq_titok/fid50k.jsonl
```
Adversary finetune = same trainer with `--resume <ckpt> --reset-schedule --gan-weight 1.0 --gan-layers 3 --lr 1e-4
--warmup 500` and `--epochs <base+200>`; resume from the generator's BEST checkpoint, not its final one (they drift).

## 4. Results (CelebA-HQ 28k pairs unless marked; 38.3M generator = width 1.0; FID vs 28k train stats)

| run | what | FID-10k (in-loop) | FID-50k |
|---|---|---|---|
| aag3 | TiTok-512, 200k random assignment, MSE+0.5 LPIPS (baseline) | 38.4 | 37.13 |
| aag5 | same, 133.6M generator | 41.5 | — |
| aag21 | same, half-width (~10M) generator | 37.8 | — |
| aag12/13/14/15/16/17/18/19/20 | assignment-budget sweep (α0.3-10k / 100k / α0.1-3k / α1-10k / 20k / α1-3k / α1-500 / α0.1-10k / α0.3-5k) | 42.4 / 40.6 / 46.2 / 50.3 / 49.0 / 45.3 / 39.0 / 40.2 / 41.6 | — |
| own AEs d64 / d128 / d256 (200k) | compact latents | 39.4 / 39.7 / 43.1 | — |
| aag8 | aag3 ep230 + adversary w0.5 / 50 ep | ~36 | 33.72 |
| aag33 | aag3 ep400 + adversary **w1.0 / 200 ep** | 34.43 | 33.16 |
| aag35 | **DINO recipe** mse 1 / lpips 0.5 / dino 0.005, plain, best ep140 | 34.2 (drifts to 36.5 @400) | 33.02 |
| aag37 | dino 0.02, plain, best ep160 | 33.3 (34.9 @400) | **32.31** (plain best) |
| aag38 | dino 0.08, plain | 36.58 final (worse than 0.02) | — |
| aag36 | aag35 ep140 + adversary w1.0 / 200 ep | 30.29 | **29.16 (valid best)** |
| aag39 | aag37 ep160 + adversary (queued after aag38) | — | — |
| off-record (x2 flips): aag23 plain 37.77/36.40; aag24–26 adversary 33.6/31.6/29.57 (FID-50k 28.18) | | | |

ImageNet (TiTok-512, class-conditional, hierarchy-level transport, FID-10k vs full-train stats):
aag2 (285M, 300k steps) final **138.3**; aag22 (150M, 2M steps) 139.9 @ep18/40 (≈4–5 ahead of aag2 at equal epoch);
aag29 (150M, 3M steps) 166.3 @ep2 (running). Latent-space fresh-z FID by assignment length: 300k 134 → 1M 121 → 2M 110 → 3M 103.

## 5. Key scripts (all in the worktree)
`aag/gaussianize.py` (transport primitives incl. `refine_direction`, `population_direction`), `scripts/run_assignment_classes.py`,
`scripts/train_generator256_ddp.py` (flags: `--width --n-res --arch residual --mse-weight --lpips-weight --dino-weight
--gan-weight --gan-layers --reset-schedule --resume --aug-hflip`), `scripts/eval_fid256.py`, `scripts/encode_hf256.py`
(`--flip` exists, don't use), `scripts/merge_particles.py`, `scripts/slim_assignment.py`, `scripts/assign_surrogate_score.py`,
`scripts/toy_step_predictor.py`, `scripts/toy_score_snaps.py`, `scripts/toy_assignment_bench.py`, `scripts/compare_direction_search.py`.
Diagnostics (in `~/.claude/jobs/2153ceba/tmp/`, copy them somewhere durable if needed): `zsource_fid_*.py` (assigned-z vs
fresh-z vs shuffled-z FID on one checkpoint), `latent_gen_fid*.py` (MLP z→latent + TiTok decode), `assigned_vs_fresh_probe.py`
/ `imagenet_critic.py` (MLP critic assigned vs N(0,I)), `z_class_probe.py` (does z predict the class), `nn_ratio.py`,
`grad_balance.py` (per-loss-term output gradient norms), `rejection_fid.py`, `compose_columns.py` (comparison PNGs).

## 6. What we discovered

1. **The fresh-z gap is joint z-structure, not generator blur.** Same checkpoint: assigned z → FID 8.7, fresh N(0,I) z → 36,
   per-coordinate-shuffled assigned z → 36 (CelebA); ImageNet 26 / 149 / 149. A latent-space MLP hits the decoder ceiling at
   assigned z (7.6) but 73 at fresh z. Per-coordinate/sliced Gaussianity readouts are blind to it; use the z-source FID
   decomposition, the split kNN test or an MLP critic (assigned vs fresh; 50 % = indistinguishable).
2. **At 28k points in 512-d, coverage and smoothness trade off.** Every assignment that reaches the sliced-W2 noise floor
   without overshooting gives the same generator (FID 38–42); under-transport is worse (20k: 49) and so is over-transport
   (1M: 47, refined-α1 10k: 50) even though the latter is *more* Gaussian by every readout (critic ≤ chance, NN-ratio → 1).
   The user's intuition ("more transport spaces samples apart") measured. Random vs refined (max-sliced) directions: refined
   reaches the floor in ~500–1k steps instead of 100k+, same endpoint quality.
3. **A 3-second surrogate predicts the ranking of assignments** (`assign_surrogate_score.py`: tiny MLP z→latent, Fréchet
   distance at fresh z). Spearman ≈0.9 with real generator FID across toy regimes; correct order on 9 real runs; but effect
   sizes are inflated (it said flips would give 23 vs 39; reality 37.8 vs 38.4). Take the EARLIEST step of its plateau.
   Nothing else predicts (C2ST, coverage, kNN regression, held-out MSE, locality).
4. **Generator capacity is not the lever** (133M vs 38M: nothing; half-width slightly better). A less-fitted generator gives
   better fresh-z samples (toy + real). Input jitter on z: worse at every σ. kNN rejection of fresh z at sampling: no effect
   (in 512-d all fresh points are uniformly ~9 % farther from the data than data from itself; no "safe areas").
5. **Latent choice**: DC-AE 2048-d cannot be Gaussianised (N/d 14). TiTok-LL-32 (512-d, recon 0.026/0.26) is the CelebA
   latent. Own compact AEs (d64–d256) buy N/d but lose the same on the decoder floor — a wash. d32/d16 too lossy.
6. **What actually moves CelebA: what the generator is asked to match beyond pixels.** DINOv2 feature term −5 FID plain;
   pairwise adversary (real‖gen channel concat, PatchGAN, adaptive weight × gan-weight) −4; additive: 37.1 → 29.2 FID-50k.
   The literal ROMS-IMLE weights (pixel 0.1 / LPIPS 1 / DINO 1) do NOT train (FID 404↑): at a trained generator the output
   gradients are DINO 1.30 vs LPIPS 0.0061 vs MSE 0.0001, so the loss is DINO-only. Gradient-matched weights work:
   mse 1 / lpips 0.5 / dino 0.005–0.02 (0.02 best, 0.08 worse). Adversary: gan-weight 1.0 for 200 epochs ≫ 0.5 for 50;
   FID first jumps up, then falls with LR decay; still improving at the end of 100 epochs, saturated by 200.
7. **DINO-recipe generators drift after their plateau** (34.2 @ep140 → 36.5 @ep400) more than MSE+LPIPS ones; keep the
   best checkpoint (`--eval-every 10` saves them) and finetune from it.
8. **ImageNet is the opposite regime (N/d 2500):** more transport keeps helping (fresh latent FID 134 → 103 from 300k → 3M
   steps; critic 79 % → 56 %) while the pair fit degrades only mildly. Refined directions reach the sliced floor in 1k steps
   but do NOT buy joint Gaussianity there — step count does. Assigned z still leaks the class (MLP probe 13–15 % top-1 vs
   0.1 % chance) even when every hierarchy-level ratio ≈1; class-focused transport (512 firings/step) doesn't remove it.
   DINO and the adversary have NOT been tried on ImageNet yet — obvious next lever. Own ImageNet AEs exist
   (`/mnt/shared/aag/results_scale256/imagenet_ae/dcae_d{256,512,1024}_g{4,8}_c128`; d512-g4 recon 0.043/0.35 beats TiTok's 0.056/0.33) but were not used.
9. **Toy benchmarks** (`toy_assignment_bench.py`, `toy_step_predictor.py`, results in
   `/data/aag_results/results_scale256/toy_bench/`) reproduce all of the above on synthetic manifolds; use them before
   spending a node.

## 7. Live state at handoff (2026-09-06 ~11:30 UTC)
- Node A: **aag29** ImageNet 3M assignment (150M), ~16 h left. Node B: **aag38** dino 0.08 (~40 min left) → **aag39**
  (adversary on aag37 ep160) auto-submits via `~/.claude/jobs/2153ceba/tmp/chain_v13.sh`. **aag22** (ImageNet 2M) is on
  the third node, ~5 h left. Watch with `kubectl -n kubeflow get pytorchjob`.
- Nothing else queued. Chain/monitor scripts live in `~/.claude/jobs/2153ceba/tmp/` (chain_v*.sh, wait29.sh); they die
  with that session — nothing depends on them except the aag39 submission.
- Decision still open for the user: none. Unanswered idea from the user: pairwise adversary on ImageNet; DINO on ImageNet.

## 8. If research resumes — the obvious next steps
1. ImageNet: apply the gradient-matched DINO term and the strong adversary to the best assignment (aag22/aag29 outcome);
   re-measure the DINO/LPIPS/MSE gradient balance there (`grad_balance.py`) before picking weights.
2. CelebA: aag39's number; then whether dino ∈ (0.02, 0.08) or a DINO term inside the adversary phase helps; consider
   DINO-L/14 or multi-layer features; consider training the adversary phase longer than 200 epochs with the DINO base.
3. The remaining CelebA gap (29 vs 6.7) is the fresh-z joint structure at 28k points; nothing on the assignment side moved
   it. The user is rethinking AAG fundamentally — don't spend nodes on assignment variants without a new idea.
