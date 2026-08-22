#!/bin/bash
# Doom world model: predict frame t from the 3 preceding frames + the action.
# Frame FID 60.18 (fresh z, 6k samples vs real held-out frames), from 84.0.
#
# The whole gain came from the ASSIGNMENT, not the generator. The previous
# version appeared to ignore the action: with its own assigned z it identified
# the true action 91% of the time, but with a fresh z -- which is what
# generation does -- only 12.7%, against 5.6% chance.
#
# Decomposing the independence ratio into its two factors, each against the same
# random-subset floor at k=4096, located the fault. The action factor was nearly
# converged while the frames factor was two orders out:
#
#   conditional steps per global step   action   frames   joint   frame FID
#     2 (the old recipe)                  4.18    83.83   14.19        84.0
#     8                                   1.85    70.66    7.08           -
#    32                                   1.23    33.83    1.92           -
#   128                                   1.14    16.64    0.97           -
#   112 context-only + 16 action          1.07     1.10    0.90       59.56
#   the same, action sampled uniformly    1.01     1.12    1.12       60.18
#
# The last two rows differ only in how the action step picks its query. Sampling
# a particle uniformly gives each action a share of the conditional budget equal
# to its frequency, and the actions are skewed: Forward/Turn variants hold
# 11-12% of particles each, Attack 2.8%. Per-action independence came out at
# 1.05 for Forward but 1.67 for Attack, and Attack-ness leaking into a fresh z
# put muzzle flashes into Forward rollouts. Sampling the ACTION uniformly first
# gives every action 1/18 of the budget: Attack 1.67 -> 1.22, per-action mean
# 1.17 -> 1.04, and the false-Attack rate falls to 5.4% against a 5.6% chance
# baseline. This costs 0.6 FID and is the shipped model -- FID scores frame
# quality, not whether the requested action is respected.
#
# Action control remains the open weakness. Measured on the shipped checkpoint
# over 64 contexts, with a fresh z the generated frame matches the requested
# action only 15.0% of the time against 5.6% chance, i.e. z overrides the action
# in most samples. Both variants behave this way, so the cause is the
# conditioning architecture -- 18 of 274 input dims, concatenated flat -- rather
# than the assignment. AdaLN conditioning, as used by the video generator, is
# the obvious next thing to try.
#
# So z was not independent of the visual CONTEXT, which put a fresh z
# off-manifold and produced a frame 3-4x further from the truth whatever action
# was supplied. run_assignment_doom.py spends every conditional step on an
# action-filtered k-NN neighbourhood, i.e. half its effort on a constraint that
# had already converged; run_assignment_doom_ctx.py spends most of the budget on
# context-only neighbourhoods instead and reaches ratio ~1 on both factors.
#
# Note fresh-z reconstruction MSE is NOT a useful target here: it stayed flat at
# ~0.0103 across every assignment and all 112 epochs while frame FID improved
# 24 points. With z genuinely independent of c, a fresh z encodes a different
# residual and so produces a different valid continuation -- per-sample MSE
# punishes that, FID does not.
set -euo pipefail
cd "$(dirname "$0")/.."
V=.venv/bin/python
R=results_doom
C=/data/doom/cache_train_p9

$V scripts/preprocess_doom.py --src /data/doom/p-doom/train --out $C --frames 16 --size 64 --per-record 9

# per-frame 2D autoencoder: temporal modelling belongs in the dynamics, not the encoder
$V scripts/train_ae.py --dataset doom_frames --data $C --arch hybrid --dim 64 --ch 64 \
  --image-size 64 --lpips-weight 0.5 --topk-frac 0.2 --epochs 14 --eval-every 7 --batch 128 \
  --out $R/ae_frames_p9

AE=$(ls -t $R/ae_frames_p9/checkpoints/*.pt | head -1)
$V scripts/prepare_doom_particles.py --cache $C --checkpoint "$AE" --context 3 \
  --per-episode 9 --out $R/particles_p9          # 630k particles, ~35k per action

$V scripts/run_assignment_doom_bal.py --particles $R/particles_p9/particles.pt \
  --steps 16000 --ctx-per-step 112 --act-per-step 16 --k 4096 --eval-k 4096 \
  --eval-every 800 --out $R/assign_wm_bal
# run_assignment_doom_ctx.py is the same without the balanced action sampler.

$V scripts/train_generator_doom.py --assignment $R/assign_wm_bal/assignment.pt \
  --cache $C --epochs 112 --eval-every 8 --out $R/gen_wm_bal
# FID by epoch: 62.0 @16, 60.6 @32, 60.6 @48, 61.2 @64, 60.0 @80, 60.6 @96,
# 60.2 @112 -- flat within +-0.6 from ep32, so this is converged.
