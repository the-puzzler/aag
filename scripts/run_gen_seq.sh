#!/usr/bin/env bash
# Sequence-rollout generator on the lag-corrected assignment.
#
# Waits for assign_12d_lag1 to finish, then trains ONE run -- no plain phase
# followed by a fine-tune. Two reasons: the new assignment has new z, so the old
# ep40 weights are mismatched and there is nothing to resume from; and step 0 of
# a sequence batch IS the ordinary single-step problem, so single-step signal is
# present throughout rather than needing its own phase. --seq-warmup 3 keeps the
# first three epochs pure single-step, because rolling a barely-trained model on
# its own output supervises against noise.
#
# L=8 covers 2.7x the 3-frame context window, which is the regime where drift
# first appeared (~frame 10-25). Batch 128 rather than 256: a sequence batch
# holds L generator graphs at once, so this keeps peak activations near the old
# single-step footprint.
set -uo pipefail
WT=/home/ubuntu/exp/newgen/.claude/worktrees/vpt-cache-hardening
PY=/home/ubuntu/exp/newgen/.venv/bin/python
ASG=/data/aag_results/results_vpt/assign_12d_lag1/assignment.pt
ALOG=/data/vpt/assign_12d_lag1.log
cd "$WT" || exit 1

echo "=== waiting for the lag-corrected assignment $(date -u +%H:%M:%S) ==="
while pgrep -f "[r]un_assignment_vpt\.py" > /dev/null; do sleep 120; done
echo "=== assignment process gone $(date -u +%H:%M:%S) ==="
if ! grep -q "saved ->" "$ALOG"; then
  echo "ABORT: assignment did not reach its final save"; tail -8 "$ALOG"
  ls -1 "$(dirname "$ASG")" 2>/dev/null | tail -5
  exit 1
fi
grep -E "^step 384000|disp=" "$ALOG" | tail -2

echo "=== sequence-rollout generator $(date -u +%H:%M:%S) ==="
PYTHONPATH=$WT "$PY" scripts/train_generator_vpt_seq.py \
  --assignment "$ASG" --cache /opt/dlami/nvme/vpt_full \
  --seq-len 8 --seq-prob 0.5 --seq-warmup 3 \
  --ch 192 --epochs 40 --batch 128 --lr 3e-4 --amp \
  --out /data/aag_results/results_vpt/gen_seq_lag1 > /data/vpt/gen_seq_lag1.log 2>&1
echo "=== generator exited rc=$? $(date -u +%H:%M:%S) ==="
tail -4 /data/vpt/gen_seq_lag1.log
