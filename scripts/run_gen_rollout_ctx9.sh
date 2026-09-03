#!/usr/bin/env bash
# Rollout-only finetune at 9 context frames instead of 3.
#
# Queued THIRD, after the 3-frame rollout-only run, because that run is the
# reference for two separate comparisons and neither works without it:
#
#   gen_pix_lag1 ep40   ->  rollout only @3      isolates ROLLOUT
#   rollout only @3     ->  rollout + GAN @3     isolates the ADVERSARY
#   rollout only @3     ->  rollout only @9      isolates CONTEXT LENGTH
#
# Run 9 frames before the 3-frame one and the third comparison has no baseline,
# so the order is not arbitrary even though the 9-frame run is the one expected
# to help most.
#
# WHY 9 FRAMES IS FREE. The particles were built with 24 frames of context and
# the target frame t is drawn from [24, 80), so frames t-9..t-1 always exist --
# no rebuild, no re-encode. And with --pixel-context the generator reads FRAMES,
# so the assignment's 3-frame cond is used for nothing but deriving the default
# width; --ctx-frames overrides it.
#
# WHAT IT COSTS. The positional embedding is sized ctx_frames+2, so resuming a
# 3-frame generator at 9 requires widening it. pos layout is
# [z, ctx_0..ctx_(C-1), action] with oldest context first, so the last 3 learned
# context embeddings are copied onto the last 3 new slots -- the NEWEST are
# preserved exactly, since the single-frame independence result says those carry
# the signal -- and the 6 extra older slots are seeded from the oldest learned
# one. Interpolating would have perturbed the newest slot instead.
#
# Adam's moments for `pos` no longer fit after that, so they are dropped for
# that parameter alone and every other parameter keeps its own. Without this the
# fused Adam update raises "size of tensor a (5) must match tensor b (11)".
#
# ON INDEPENDENCE. The assignment decorrelated z against the 3-frame context,
# not a 9-frame one. That is expected to hold rather than assumed: the ctx3
# lineage transported against scales 1 and 3 only and came out independent at
# 5/12/24 frames (0.793/0.862/0.907), which is the collider argument working. It
# is still worth re-probing on the result.
#
# Expect ~2-3x the 3-frame epoch time: the 17.7M encoder now runs on 9 frames per
# step instead of 3, and that is the dominant added cost, not the 6 extra
# transformer tokens.
set -uo pipefail
WT=/home/ubuntu/exp/newgen/.claude/worktrees/vpt-cache-hardening
PY=/home/ubuntu/exp/newgen/.venv/bin/python
ASG=/data/aag_results/results_vpt/assign_12d_lag1/assignment.pt
CK=/data/aag_results/results_vpt/gen_pix_lag1/checkpoints/gen_seq_ep40.pt
PREV=/data/vpt/gen_rollonly_lag1.log
cd "$WT" || exit 1

# wait for the 3-frame rollout-only run to COMPLETE, not merely for the GPU to
# free -- otherwise this races the other queued waiter when the current run ends
echo "=== waiting for the 3-frame rollout-only run to finish $(date -u +%H:%M:%S) ==="
while true; do
  if [ -f "$PREV" ] && grep -q "^epoch 50/50" "$PREV"; then break; fi
  sleep 180
done
echo "=== 3-frame run complete $(date -u +%H:%M:%S) ==="
while pgrep -f "[t]rain_generator_vpt_seq\.py" > /dev/null; do sleep 60; done
sleep 30

echo "=== rollout-only @ 9 context frames $(date -u +%H:%M:%S) ==="
PYTHONPATH=$WT "$PY" scripts/train_generator_vpt_seq.py \
  --assignment "$ASG" --cache /opt/dlami/nvme/vpt_full \
  --pixel-context --pix-ch 96 --pix-depth 2 --ctx-frames 9 \
  --resume "$CK" --start-epoch 40 --epochs 50 \
  --seq-len 8 --seq-prob 0.5 --seq-warmup 0 \
  --ch 192 --batch 128 --lr 3e-4 --amp \
  --out /data/aag_results/results_vpt/gen_rollonly_ctx9 > /data/vpt/gen_rollonly_ctx9.log 2>&1
echo "=== generator exited rc=$? $(date -u +%H:%M:%S) ==="
tail -5 /data/vpt/gen_rollonly_ctx9.log
