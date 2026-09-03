#!/usr/bin/env bash
# Step 2 of the pipeline: a PURE generator, trained conventionally.
#
#   assignment -> generator (this) -> rollout finetune / GAN / extras
#
# No rollout supervision at all: --seq-prob 0 means every batch is the ordinary
# single-step problem, and --seq-warmup is set past the horizon so the log says
# "(warmup)" throughout rather than printing zeros that could be mistaken for a
# per-step curve. Rollout training belongs afterwards, as a small finetune on a
# STRONG generator -- applying it to a weak model is a different intervention
# and confounds the result, which is what happened to gen_seq_lag1.
#
# The context is RAW PIXELS through a learned encoder trained end-to-end with
# the transformer and decoder, so the pipeline is
#
#   3 context frames (pixels) -> encoder -> transformer -> decoder -> pixels
#
# The frozen AE is now used ONLY by the assignment, to define the context metric
# and the h_target that z was assigned against. It never touches the generator.
# Measured reason: one AE encode-decode destroys 6.55 mean |pixel| against a
# real consecutive-frame step of 7.16 -- 91% -- and velocity lives in the
# DIFFERENCE between consecutive context vectors, so it was a poor channel for
# reading speed. That matches the observed failure, which is not that the model
# is static but that its motion barely varies: own-motion sat at 0.78-1.11
# across every action at ep28 while the two test scenes really moved 1.40 and
# 7.94.
#
# Encoder size: CAPACITY-MATCHED to the AE's own encoder, which is 17,384,000
# parameters, so what this run tests is the encoder being LEARNED on this task
# rather than the encoder being bigger. ch=96 depth=2 gives 17,702,944, +1.8%.
# ch=120 depth=1 is marginally closer at +1.0% but shallower -- 8 stages against
# 12 -- and at 64x64 the match is better spent on depth, since the job is to
# expose motion across three frames, not to hold a wide basis.
#
# Batch back to 256 (the sequence runs used 128 because a sequence batch held L
# generator graphs at once); with no rollout there is one encoder forward per
# batch instead of eight, so this should run far closer to the plain baseline's
# 4.3 min/epoch than to the sequence run's 19.
set -uo pipefail
WT=/home/ubuntu/exp/newgen/.claude/worktrees/vpt-cache-hardening
PY=/home/ubuntu/exp/newgen/.venv/bin/python
ASG=/data/aag_results/results_vpt/assign_12d_lag1/assignment.pt
cd "$WT" || exit 1

echo "=== pure generator, pixel context, no rollout $(date -u +%H:%M:%S) ==="
PYTHONPATH=$WT "$PY" scripts/train_generator_vpt_seq.py \
  --assignment "$ASG" --cache /opt/dlami/nvme/vpt_full \
  --pixel-context --pix-ch 96 --pix-depth 2 \
  --seq-len 8 --seq-prob 0.0 --seq-warmup 9999 \
  --ch 192 --epochs 40 --batch 256 --lr 3e-4 --amp \
  --out /data/aag_results/results_vpt/gen_pix_lag1 > /data/vpt/gen_pix_lag1.log 2>&1
echo "=== generator exited rc=$? $(date -u +%H:%M:%S) ==="
tail -4 /data/vpt/gen_pix_lag1.log
