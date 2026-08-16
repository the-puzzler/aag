#!/bin/bash
# CelebA, conditional on 40 binary attributes.  Reported: FID 20.83
set -euo pipefail
cd "$(dirname "$0")/.."
V=.venv/bin/python
R=results_celeba

$V scripts/extract_celeba_attrs.py                      # -> results_celeba/attrs.pt

AE=$(ls -t $R/ae/checkpoints/*.pt | head -1)    # reuse the unconditional AE
$V scripts/run_assignment_conditional.py --dataset celeba --data-root /data/hf_cache \
  --h-source $R/assignment/assignment.pt --N 162770 --steps 4000 \
  --cond-alpha 0.25 --k 4096 --out results_celeba_conditional/assign_4k_alpha0.25

$V scripts/train_generator_conditional.py --dataset celeba \
  --z-source results_celeba_conditional/assign_4k_alpha0.25/assignment.pt \
  --epochs 500 --eval-every 25 --out results_celeba_conditional/generator_cond

$V scripts/run_fid_conditional.py results_celeba_conditional/generator_cond/checkpoints/generator_ep500.pt
