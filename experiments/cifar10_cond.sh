#!/bin/bash
# CIFAR-10, conditional on the 10 classes (exact, disjoint groups).
set -euo pipefail
cd "$(dirname "$0")/.."
V=.venv/bin/python
R=results_cifar

$V scripts/train_ae.py --dataset cifar10 --data data --arch residual \
  --dim 64 --ch 64 --image-size 32 --lpips-weight 0.5 --epochs 40 --out $R/ae

AE=$(ls -t $R/ae/checkpoints/*.pt | head -1)
$V scripts/run_assignment.py --dataset cifar10 --data data --checkpoint "$AE" \
  --N 50000 --assign-steps 4000 --out $R/assignment

$V scripts/run_assignment_conditional.py --dataset cifar10 --data-root data \
  --h-source $R/assignment/assignment.pt --N 50000 --steps 4000 \
  --cond-per-step 1 --cond-alpha 0.5 --k 4096 --out $R/assignment_cond

$V scripts/train_generator_conditional.py --dataset cifar10 --data data \
  --z-source $R/assignment_cond/assignment.pt --image-size 32 \
  --epochs 500 --eval-every 25 --out $R/generator_cond
