#!/bin/bash
# Doom world model: predict frame t from the 3 preceding frames + the action.
set -euo pipefail
cd "$(dirname "$0")/.."
V=.venv/bin/python
R=results_doom
C=/data/doom/cache_train

$V scripts/preprocess_doom.py --src /data/doom/p-doom/train --out $C --frames 16 --size 64 --per-record 3

# per-frame 2D autoencoder: temporal modelling belongs in the dynamics, not the encoder
$V scripts/train_ae.py --dataset doom_frames --data $C --arch hybrid --dim 64 --ch 64 \
  --image-size 64 --lpips-weight 0.5 --topk-frac 0.2 --epochs 40 --batch 128 \
  --out $R/ae_frames_dim64

AE=$(ls -t $R/ae_frames_dim64/checkpoints/*.pt | head -1)
$V scripts/prepare_doom_particles.py --checkpoint "$AE" --context 3 --per-episode 3 --out $R/particles

$V scripts/run_assignment_doom.py --particles $R/particles/particles.pt \
  --steps 4000 --cond-per-step 2 --cond-alpha 0.25 --k 4096 --out $R/assignment_cond

$V scripts/train_generator_doom.py --assignment $R/assignment_cond/assignment.pt \
  --epochs 500 --eval-every 25 --out $R/generator_cond
