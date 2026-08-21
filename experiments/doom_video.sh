#!/bin/bash
# Doom video, conditioned on a first frame.  Held-out FID 56.89.
#
# What moved this from 58.81 to 56.89 was data volume plus training length, not
# any change to the transport. Each ~160-frame episode yields up to 9
# non-overlapping 16-frame segments and the earlier recipe took only 3, so the
# same download supports 550k particles instead of 190k -- from the SAME 70,000
# episodes, so it is more segments rather than more episodes. Both AEs are
# retrained on the larger cache at matched gradient-step budgets (epoch counts
# cut ~3x), which helps the 2D frame AE clearly (test LPIPS 0.0735 vs 0.0911)
# and the 3D video AE not at all (0.1781 vs 0.1784).
#
# Things that did NOT help, so they are deliberately absent:
#   * driving the assignment further: 16k -> 160k steps took G from 2.55 to 0.72
#     and the independence ratio from 3.13 to 0.82, with no FID change.
#   * generator LPIPS weight 0.5 -> 8: catastrophic on video (~100 vs 63),
#     the opposite of its effect on CIFAR.
#   * k = 32768 instead of 4096: 5-6 FID worse. The independence ratio predicted
#     this correctly, but only with eval_k pinned -- at eval_k = k it says the
#     opposite, because a wider neighbourhood is trivially more independent.
#   * a sustained lr of 1e-3: no better than the single restart at 5e-4.
set -euo pipefail
cd "$(dirname "$0")/.."
V=.venv/bin/python
R=results_doom
C=/data/doom/cache_train_p9

# 9 segments per episode instead of 3 -- ~550k particles, ~124 GB of cache
$V scripts/preprocess_doom.py --src /data/doom/p-doom/train --out $C --frames 16 --size 64 --per-record 9
$V scripts/preprocess_doom.py --src /data/doom/p-doom/val  --out /data/doom/cache_val \
  --frames 16 --size 64 --per-record 3

# both AEs on the larger cache, epoch counts cut ~3x so the step budget matches
$V scripts/train_ae.py --dataset doom_frames --data $C --arch hybrid --dim 64 --ch 64 \
  --image-size 64 --lpips-weight 0.5 --topk-frac 0.2 --epochs 14 --eval-every 7 --batch 128 \
  --out $R/ae_frames_p9

# 3D video AE: spatial grid latent + DC-AE style skips, d=64 (see docs/METHOD.md §5 on why not 256)
$V scripts/train_ae.py --dataset doom --data $C --arch hybrid --dim 64 --ch 64 --image-size 64 \
  --lpips-weight 0.5 --topk-frac 0.2 --epochs 6 --eval-every 2 --batch 32 --t-out 4 --lr 5e-4 \
  --out $R/ae_video_p9

VAE=$(ls -t $R/ae_video_p9/checkpoints/*.pt | head -1)
FAE=$(ls -t $R/ae_frames_p9/checkpoints/*.pt | head -1)
$V scripts/prepare_doom_video_particles.py --video-ae "$VAE" --frame-ae "$FAE" \
  --cache $C --out $R/video_particles_p9

$V scripts/run_assignment_doom_video.py --particles $R/video_particles_p9/particles.pt \
  --steps 16000 --cond-per-step 4 --cond-alpha 0.25 --k 4096 --eval-k 4096 \
  --cond-metric cosine --out $R/assign_p9

# Training length is the lever. The cosine anneals lr to ~0 by the end of each
# cycle and FID stalls there while train loss keeps falling, so each restart is
# what buys the next drop -- not the peak lr itself.
# 64.69 -> 58.84 over ep2-10, then 57.77 at ep12, then a ~56.8 plateau by ep16.
$V scripts/train_generator_doom_video.py --assignment $R/assign_p9/assignment.pt \
  --cache $C --cond-mode adaln --epochs 10 --eval-every 2 --batch 32 --out $R/gen_p9

$V scripts/train_generator_doom_video.py --assignment $R/assign_p9/assignment.pt \
  --cache $C --cond-mode adaln --epochs 20 --start-epoch 10 --eval-every 2 --batch 32 \
  --lr 5e-4 --resume $R/gen_p9/generator_ep10.pt --out $R/gen_p9

$V scripts/train_generator_doom_video.py --assignment $R/assign_p9/assignment.pt \
  --cache $C --cond-mode adaln --epochs 200 --start-epoch 12 --eval-every 2 --batch 32 \
  --lr 1e-3 --resume $R/gen_p9/generator_ep12.pt --out $R/gen_p9_lr1e3
# (--epochs 200 only stretches the cosine so lr stays near 1e-3; stop at ep24.)

$V scripts/eval_doom_video_fid.py --frame-ae "$FAE" \
  --generators $R/gen_p9_lr1e3/generator_ep24.pt
$V scripts/make_doom_mp4.py --generator $R/gen_p9_lr1e3/generator_ep24.pt --frame-ae "$FAE" \
  --rows 3 --cols 6 --real-strip --out $R/doom_video.mp4
