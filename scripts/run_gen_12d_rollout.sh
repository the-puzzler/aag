#!/usr/bin/env bash
# Rollout fine-tune of the 12-d generator, continuing from the plain ep40.
#
# Why: ep40 is sharp single-step but falls apart under autoregressive refeed --
# within ~10-25 frames, washing out with a fresh z per frame and wandering off
# the starting scene with a held z. It was trained with --rollout-k 0, so it had
# never once seen its own output as context; fed AE-encoded self-predictions at
# inference it is off-distribution from the first step and compounds.
#
# --rollout-k 3, not the previous lineage's 2, and the reason is CTX:
#
#   the context window is 3 frames. At inference, from the fourth generated
#   frame onward EVERY frame in that window is the model's own output. With
#   k=2 at most two of the three are synthetic, so a fully synthetic window --
#   exactly the input distribution the whole rollout lives in -- never appears
#   in training. k=3 is the smallest value that covers it. n_roll is drawn
#   uniformly from 1..k, so the shorter mixed cases are still trained too.
#
# Costs 3 forward-target buffers instead of 2 (~19 GB pinned vs ~12.6 GB); there
# is ~117 GB free, so this is not the constraint.
#
# --rollout-prob 0.5 keeps half the batches single-step, which is what stops the
# fine-tune from trading away the sharpness ep40 already has.
#
# Still NO adversary: --gan-weight stays at its 0.0 default, per the standing
# instruction to keep this plain MSE + LPIPS. Rollout is the one thing being
# added, so if the multi-step behaviour changes it is attributable.
#
# --start-epoch 40 --epochs 60 gives ONE continuous cosine over the new
# 60-epoch horizon, fast-forwarded to epoch 40 (lr ~7.5e-5), rather than a warm
# restart at the 3e-4 peak -- and the Adam moments come back with the weights.
#
# Writes to a NEW directory so the plain ep40 baseline stays intact for
# comparison; the whole point is to be able to tell the two apart.
set -uo pipefail
WT=/home/ubuntu/exp/newgen/.claude/worktrees/vpt-cache-hardening
PY=/home/ubuntu/exp/newgen/.venv/bin/python
ASG=/data/aag_results/results_vpt/assign_12d_scratch/assignment.pt
CK=/data/aag_results/results_vpt/gen_12d_scratch/checkpoints/gen_vpt_ep40.pt
GEN=/data/aag_results/results_vpt/gen_12d_rollout
cd "$WT" || exit 1

echo "=== rollout fine-tune from ep40 $(date -u +%H:%M:%S) ==="
PYTHONPATH=$WT "$PY" scripts/train_generator_vpt.py \
  --assignment "$ASG" --cache /opt/dlami/nvme/vpt_full \
  --arch transformer \
  --ch 192 --batch 256 --lr 3e-4 --amp --eval-every 1 \
  --resume "$CK" --start-epoch 40 --epochs 60 \
  --rollout-k 3 --rollout-prob 0.5 \
  --out "$GEN" > /data/vpt/gen_12d_rollout.log 2>&1
echo "=== generator exited rc=$? $(date -u +%H:%M:%S) ==="
tail -4 /data/vpt/gen_12d_rollout.log
