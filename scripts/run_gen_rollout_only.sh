#!/usr/bin/env bash
# Rollout-only finetune: the SAME finetune as gen_rollgan_lag1 with the
# adversary switched off, so the two together give full attribution.
#
#   gen_pix_lag1 ep40  ->  rollout only          isolates ROLLOUT
#   rollout only       ->  rollout + adversary   isolates the ADVERSARY
#
# One flag apart from gen_rollgan_lag1 (--gan-weight absent, so 0.0). Everything
# else identical: same resumed checkpoint, same 10 epochs, same seq_prob 0.5 and
# L=8, same lr schedule, same batch, same seed. That is the point -- the pair is
# only interpretable if nothing else moves.
#
# This replaces the queued 4x generator, which was dequeued. Worth noting the
# tension: the failures actually observed in the ep42 rollouts are SEMANTIC
# substitutions -- a plant read as grass terrain, sand becoming wood, then
# propagated faithfully by the rollout -- and that looks more like a
# capacity/resolution limit than something rollout or an adversary fixes. At
# 64x64 a plant is a handful of pixels and genuinely resembles grass. So the 4x
# run remains the more direct test of THAT failure and is worth requeueing after
# this; attribution comes first by choice.
#
# Expect this to be FASTER than the rollout+GAN run's 38 min/epoch: no critic
# forward per step, and more importantly no adaptive_weight, which costs two
# extra autograd.grad calls with retain_graph per rollout step -- 16 partial
# backward passes per sequence batch.
set -uo pipefail
WT=/home/ubuntu/exp/newgen/.claude/worktrees/vpt-cache-hardening
PY=/home/ubuntu/exp/newgen/.venv/bin/python
ASG=/data/aag_results/results_vpt/assign_12d_lag1/assignment.pt
CK=/data/aag_results/results_vpt/gen_pix_lag1/checkpoints/gen_seq_ep40.pt
cd "$WT" || exit 1

echo "=== waiting for the rollout+GAN finetune $(date -u +%H:%M:%S) ==="
while pgrep -f "[t]rain_generator_vpt_seq\.py" > /dev/null; do sleep 120; done
echo "=== GPU free $(date -u +%H:%M:%S) ==="
sleep 30

echo "=== rollout-only finetune from gen_pix_lag1 ep40 $(date -u +%H:%M:%S) ==="
PYTHONPATH=$WT "$PY" scripts/train_generator_vpt_seq.py \
  --assignment "$ASG" --cache /opt/dlami/nvme/vpt_full \
  --pixel-context --pix-ch 96 --pix-depth 2 \
  --resume "$CK" --start-epoch 40 --epochs 50 \
  --seq-len 8 --seq-prob 0.5 --seq-warmup 0 \
  --ch 192 --batch 128 --lr 3e-4 --amp \
  --out /data/aag_results/results_vpt/gen_rollonly_lag1 > /data/vpt/gen_rollonly_lag1.log 2>&1
echo "=== generator exited rc=$? $(date -u +%H:%M:%S) ==="
tail -5 /data/vpt/gen_rollonly_lag1.log
