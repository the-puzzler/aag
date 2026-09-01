#!/usr/bin/env bash
# Wait for the from-scratch 12-d assignment, then train the transformer
# generator on it. Plain reconstruction: MSE + LPIPS, no adversary.
#
# Transformer because at every matched epoch it beat the conv version on
# reconstruction with half the parameters (ep12 lpips 0.166 vs 0.182, 53.4M vs
# 107.6M) and had better true-z temporal coherence (1.51x vs 1.70x). LR 3e-4,
# not the conv version's 2e-3: measured, 2e-3 gave lpips 0.373 at epoch 3
# against 0.160 at 3e-4, so reusing the conv LR would test the optimiser
# setting rather than the architecture.
#
# What is deliberately OFF, versus the previous lineage's generator runs:
#
#   --gan-weight   default 0.0, so no paired adversary. Was 0.5 from epoch 3.
#   --rollout-k    default 0, so single-step only. Was 2 at prob 0.5.
#
# Both off makes this a clean baseline that isolates ONE change -- the 12-d
# action representation and the assignment built around it. Adding either back
# is a one-flag edit, and rollout is a planned pipeline addition rather than
# something dropped.
#
# --act-vec and --n-actions are not passed: the first is now a no-op (the action
# vector is always the conditioning) and the second is only read under
# --action-onehot, which is the legacy path. Generator input is
# z 256 + cond 768 + act 12 = 1036, against 1114 for the old 81-way + 9-d
# condition.
#
# --ctx-frames is not passed either: the assignment's cond is already the 3-frame
# slice, so the generator derives CTX = 768/256 = 3 from it. Passing a number
# here could only introduce a disagreement.
#
# --cache is the staged copy on the ephemeral NVMe, which is what the particles
# were built from and where the generator reads its PIXEL targets. It does not
# need the patched clicks: those are already baked into the particle file's
# action_raw/action_vec. Verified that the two caches share segment ordering --
# the 81-way index recomputed from cache_train's keys/mouse agrees with the
# action field the particles recorded from vpt_full at build time on
# 512,000/512,000 particles.
set -uo pipefail
WT=/home/ubuntu/exp/newgen/.claude/worktrees/vpt-cache-hardening
PY=/home/ubuntu/exp/newgen/.venv/bin/python
ASG=/data/aag_results/results_vpt/assign_12d_scratch/assignment.pt
ALOG=/data/vpt/assign_12d_scratch.log
GEN=/data/aag_results/results_vpt/gen_12d_scratch
cd "$WT" || exit 1

echo "=== waiting for the 12-d assignment $(date -u +%H:%M:%S) ==="
while pgrep -f "scripts/run_assignment_vpt.py" > /dev/null; do sleep 120; done
echo "=== assignment process gone $(date -u +%H:%M:%S) ==="

# "saved ->" only appears on the FINAL save, so this distinguishes completion
# from a kill or a crash. A step-stamped checkpoint existing is not enough.
if ! grep -q "saved ->" "$ALOG"; then
  echo "ABORT: assignment did not reach its final save. Last lines:"
  tail -8 "$ALOG"
  echo "Step-stamped checkpoints that DO exist (any can be trained on by"
  echo "pointing --assignment at one):"
  ls -1 "$(dirname "$ASG")" 2>/dev/null | tail -5
  exit 1
fi
grep -E "^step 384000|disp=" "$ALOG" | tail -3

echo "=== transformer generator, MSE + LPIPS, no adversary $(date -u +%H:%M:%S) ==="
PYTHONPATH=$WT "$PY" scripts/train_generator_vpt.py \
  --assignment "$ASG" --cache /opt/dlami/nvme/vpt_full \
  --arch transformer \
  --ch 192 --epochs 40 --eval-every 1 --batch 256 --lr 3e-4 --amp \
  --out "$GEN" > /data/vpt/gen_12d.log 2>&1
echo "=== generator exited rc=$? $(date -u +%H:%M:%S) ==="
tail -5 /data/vpt/gen_12d.log
