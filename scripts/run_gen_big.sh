#!/usr/bin/env bash
# A 4x generator, otherwise identical to gen_pix_lag1, to see whether capacity
# moves the loss.
#
# Waits for the rollout+GAN finetune to finish, then trains from scratch. PLAIN:
# no rollout, no adversary. That is deliberate -- the question is whether
# capacity moves the loss, and the only comparable reference is gen_pix_lag1's
# plain 40-epoch run (0.00809 / 0.09888). Adding capacity and two new objectives
# at once would answer nothing.
#
# WHERE THE 4x GOES, which is a real choice and not a detail. The 53.38M
# baseline is 18.91M transformer (35%) and 34.19M conv decoder (64%). Scaling
# d_model 512->1024 and depth 6->12 with ch 192->256 gives 218.1M, i.e. 4.09x,
# split 151.2M transformer (69%) and 66.4M decoder (30%). So it shifts the
# balance toward the transformer.
#
# That is intended. The failures this project is chasing are DYNAMICS -- motion
# magnitude that does not modulate, scene identity that does not persist -- and
# the transformer is what reasons over the context and the action. Per-frame
# image quality was already a dead heat between the AE-latent and learned-pixel
# encoders (0.00812 vs 0.00809), so decoder capacity is not the thing in short
# supply. Preserving the original 35/64 split would have poured most of the 4x
# into the decoder.
#
# Everything else is held fixed against gen_pix_lag1 so the comparison is
# capacity alone: same lag-corrected assignment, same 17.7M learned pixel
# context encoder (ch 96 depth 2), same 40 epochs, same lr, same batch.
#
# Batch stays 256 to keep the comparison honest, but this may not fit -- 53M at
# batch 256 used 21 GB, so 218M could want 50-70 GB against 97 GB total. If it
# OOMs, halve the batch and say so, because the LR then no longer matches.
set -uo pipefail
WT=/home/ubuntu/exp/newgen/.claude/worktrees/vpt-cache-hardening
PY=/home/ubuntu/exp/newgen/.venv/bin/python
ASG=/data/aag_results/results_vpt/assign_12d_lag1/assignment.pt
cd "$WT" || exit 1

echo "=== waiting for the rollout+GAN finetune $(date -u +%H:%M:%S) ==="
while pgrep -f "[t]rain_generator_vpt_seq\.py" > /dev/null; do sleep 120; done
echo "=== GPU free $(date -u +%H:%M:%S) ==="
sleep 30

echo "=== 4x generator, plain, pixel context $(date -u +%H:%M:%S) ==="
PYTHONPATH=$WT "$PY" scripts/train_generator_vpt_seq.py \
  --assignment "$ASG" --cache /opt/dlami/nvme/vpt_full \
  --pixel-context --pix-ch 96 --pix-depth 2 \
  --d-model 1024 --depth 12 --heads 16 --ch 256 \
  --seq-len 8 --seq-prob 0.0 --seq-warmup 9999 \
  --epochs 40 --batch 256 --lr 3e-4 --amp \
  --out /data/aag_results/results_vpt/gen_big_lag1 > /data/vpt/gen_big_lag1.log 2>&1
echo "=== generator exited rc=$? $(date -u +%H:%M:%S) ==="
tail -5 /data/vpt/gen_big_lag1.log
