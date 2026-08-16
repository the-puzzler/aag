#!/bin/bash
# CelebA, unconditional.  Reported: FID 19.36
set -euo pipefail
cd "$(dirname "$0")/.."
V=.venv/bin/python
R=results_celeba

$V scripts/train_ae.py --dataset celeba --data /data/hf_cache --arch residual \
  --dim 64 --ch 64 --image-size 64 --lpips-weight 0.5 --epochs 40 --out $R/ae

AE=$(ls -t $R/ae/checkpoints/*.pt | head -1)
$V scripts/run_assignment.py --dataset celeba --data /data/hf_cache --checkpoint "$AE" \
  --N 162770 --assign-steps 4000 --out $R/assignment

$V scripts/train_generator.py --dataset celeba --data /data/hf_cache \
  --assignment $R/assignment/assignment.pt --epochs 2000 --out $R/generator_uncond
