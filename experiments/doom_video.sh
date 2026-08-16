#!/bin/bash
# Doom video, conditioned on a first frame.  Best result: held-out FID 60.10
set -euo pipefail
cd "$(dirname "$0")/.."
V=.venv/bin/python
R=results_doom
C=/data/doom/cache_train

$V scripts/preprocess_doom.py --src /data/doom/p-doom/train --out $C --frames 16 --size 64 --per-record 3
$V scripts/preprocess_doom.py --src /data/doom/p-doom/val  --out /data/doom/cache_val \
  --frames 16 --size 64 --per-record 3

# 3D video AE: spatial grid latent + DC-AE style skips, d=64 (see docs/METHOD.md §5 on why not 256)
$V scripts/train_ae.py --dataset doom --data $C --arch hybrid --dim 64 --ch 64 --image-size 64 \
  --lpips-weight 0.5 --topk-frac 0.2 --epochs 15 --batch 32 --t-out 4 --lr 5e-4 \
  --out $R/ae_video_dim64

VAE=$(ls -t $R/ae_video_dim64/checkpoints/*.pt | head -1)
FAE=$(ls -t $R/ae_frames_dim64/checkpoints/*.pt | head -1)   # from doom_worldmodel.sh
$V scripts/prepare_doom_video_particles.py --video-ae "$VAE" --frame-ae "$FAE" \
  --out $R/video_particles_dim64

$V scripts/run_assignment_doom_video.py --particles $R/video_particles_dim64/particles.pt \
  --steps 16000 --cond-per-step 4 --cond-alpha 0.25 --k 4096 --eval-k 4096 \
  --cond-metric cosine --out $R/assignment_video

$V scripts/train_generator_doom_video.py --assignment $R/assignment_video/assignment.pt \
  --cond-mode adaln --epochs 16 --eval-every 4 --batch 32 --out $R/generator_video

$V scripts/eval_doom_video_fid.py --frame-ae "$FAE" \
  --generators $R/generator_video/generator_ep16.pt
$V scripts/make_doom_mp4.py --generator $R/generator_video/generator_ep16.pt --frame-ae "$FAE" \
  --rows 3 --cols 6 --real-strip --out $R/doom_video.mp4
