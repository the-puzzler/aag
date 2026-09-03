#!/usr/bin/env bash
# Step 3 of the pipeline: rollout + adversary, as a SMALL finetune on the strong
# generator.
#
#   assignment -> generator -> rollout finetune / GAN   <- this
#
# Resumes gen_pix_lag1 ep40 (single-step train 0.00809 / 0.09888, learned 17.7M
# pixel-context encoder, lag-corrected assignment). 10 epochs, not 40: the point
# of the phase ordering is that this is an adaptation of a converged model, not a
# training run in its own right.
#
# Rollout and adversary run TOGETHER, by choice. That means an improvement
# cannot be attributed to one or the other -- the tradeoff was accepted for a
# single ~6h run instead of two sequential ones. If the result is good and the
# attribution matters later, the ablation is one flag either way.
#
# Two things worth knowing about the adversary here:
#
#   * --gan-start-epoch 1, i.e. on from the first epoch. The usual reason to
#     delay it is that a randomly-initialised generator gives the critic nothing
#     to learn from; that does not apply to a finetune off a converged model.
#   * the critic sees the real frame and the generated one for the SAME z,
#     context and action, stacked on the channel axis in random order, and is
#     asked which is the true continuation. Better posed than judging realism in
#     isolation, and it needs no conditioning input because the pairing carries
#     it. In a sequence batch EVERY step has its exact real counterpart, so it
#     applies at all 8 rollout steps rather than only the first, with the
#     adaptive weight recomputed per step so the balance holds even though later
#     steps are harder.
#
# --start-epoch 40 --epochs 50 makes ONE continuous cosine over the new horizon,
# fast-forwarded to 2.86e-5, roughly a tenth of the 3e-4 base -- a finetune LR
# rather than a warm restart at peak. The zero-LR trap is guarded in the trainer:
# a cosine that ran its full horizon ends at exactly 0.0, load_state_dict
# restores that, and CosineAnnealingLR.step() is recursive in group["lr"], so
# fast-forwarding from 0 would stay at 0 and train nothing while logging
# plausible losses.
set -uo pipefail
WT=/home/ubuntu/exp/newgen/.claude/worktrees/vpt-cache-hardening
PY=/home/ubuntu/exp/newgen/.venv/bin/python
ASG=/data/aag_results/results_vpt/assign_12d_lag1/assignment.pt
CK=/data/aag_results/results_vpt/gen_pix_lag1/checkpoints/gen_seq_ep40.pt
cd "$WT" || exit 1

echo "=== rollout + GAN finetune from gen_pix_lag1 ep40 $(date -u +%H:%M:%S) ==="
PYTHONPATH=$WT "$PY" scripts/train_generator_vpt_seq.py \
  --assignment "$ASG" --cache /opt/dlami/nvme/vpt_full \
  --pixel-context --pix-ch 96 --pix-depth 2 \
  --resume "$CK" --start-epoch 40 --epochs 50 \
  --seq-len 8 --seq-prob 0.5 --seq-warmup 0 \
  --gan-weight 0.5 --gan-start-epoch 1 \
  --ch 192 --batch 128 --lr 3e-4 --amp \
  --out /data/aag_results/results_vpt/gen_rollgan_lag1 > /data/vpt/gen_rollgan_lag1.log 2>&1
echo "=== generator exited rc=$? $(date -u +%H:%M:%S) ==="
tail -5 /data/vpt/gen_rollgan_lag1.log
