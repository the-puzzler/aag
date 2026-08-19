#!/bin/bash
# CIFAR-10, unconditional.  FID 45.91 (10k samples vs the CIFAR-10 test set).
#
# Three things matter here and none of them are the assignment budget:
#   - the generator must train on the UNCONDITIONAL assignment. Feeding it the
#     class-conditional one costs ~7 FID: conditional transport reshapes z within
#     each class, and a generator that never sees the class just pays for it.
#   - --lpips-weight 32 rather than the usual 0.5, worth ~4.6 FID. Weights 8/16/32
#     all land within noise of each other; 0.5 is clearly too low.
#   - the AE is trained past its best test-LPIPS epoch. Test LPIPS bottoms around
#     epoch 50 while reconstruction FID keeps improving to ~150, because FID
#     rewards the sharper reconstructions a mildly overfit decoder produces.
set -euo pipefail
cd "$(dirname "$0")/.."
V=.venv/bin/python
R=results_cifar/uncond

$V scripts/train_ae.py --dataset cifar10 --data data --arch residual \
  --dim 64 --ch 64 --image-size 32 --lpips-weight 0.5 --epochs 150 \
  --eval-every 50 --out $R/ae

AE=$(ls -t $R/ae/checkpoints/*.pt | head -1)
$V scripts/run_assignment.py --dataset cifar10 --data data --checkpoint "$AE" \
  --N 50000 --assign-steps 64000 --out $R/assignment

$V scripts/train_generator.py --dataset cifar10 --data data \
  --assignment $R/assignment/assignment.pt --image-size 32 --ch 64 \
  --lpips-weight 32 --epochs 40 --eval-every 10 --particle-order \
  --out $R/generator
