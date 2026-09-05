#!/usr/bin/env bash
# Resume the dim-256 AE where it left off and add the adversary.
#
# WHY THIS IS THE ONLY LEVER LEFT AT 256 DIMS. Measured on this project's own
# history: training longer has flatlined (test_mse 0.00737 at ep5 to 0.00716 at
# ep9, non-monotone, while LPIPS creeps ~1%/epoch and decelerates), and more
# capacity at the same latent dim was already tried and LOST
# (ae_dcae_grid8_ch368_dim256, 50.1M params, mse 0.00822 / lpips 0.13218 at ep3
# against grid4 ch192's 0.00776 / 0.11035 at the same epoch). gan_weight is 0.0
# in every AE checkpoint here, so the adversary is untried -- and it is the only
# option that changes the OBJECTIVE rather than optimising the same one harder,
# which is what a flat curve requires.
#
# The mechanism, which is the point rather than cosmetics: pressure on the
# decoder to produce sharp detail is pressure on the ENCODER to retain the
# information needed to produce it. In AAG that matters structurally, because z
# is the only channel carrying target-specific information and z is a transported
# copy of h_target = AE_enc(frame). Detail the encoder discards has no
# representation in z, so the assignment never had a handle on it and it was
# never gaussianised -- which is why MSE resolves it to a conditional mean, i.e.
# blur on fine texture. The generator is already at 1.13x this AE's MSE floor and
# BEATS it on LPIPS, so nothing on the generator side can recover that detail.
#
# NOT using --gan-head-modules. That is DC-AE's phase 3, which freezes everything
# but the decoder head and therefore keeps the encoder -- and every existing
# latent -- unchanged. It would preserve the particle set and the assignment, but
# it cannot put missing detail INTO the latent, which is the entire objective
# here. So the encoder trains, the latents change, and the downstream cost is
# real: particles must be re-encoded, the 13h assignment re-run, and the
# generator retrained. That is the price of the diagnosis being right.
#
# --lr 1e-4, not the original 2e-3. --resume gives a FRESH optimiser and a fresh
# cosine that starts at --lr, so peaking at the original rate would wreck weights
# already converged to 0.00716. 1e-4 is a refinement rate that still anneals to
# zero over the run.
#
# --gan-start-epoch 1, i.e. on immediately. The flag exists so a
# randomly-initialised decoder is sane before being critiqued; that does not
# apply to a resume from nine epochs of reconstruction training.
#
# The critic here is the trainer's built-in unpaired PatchGAN with hinge loss --
# the literal SD-VAE / VQGAN recipe, at their weight of 0.5, and n_layers 2 for a
# 34px receptive field rather than the usual 3's 70px which would exceed a 64px
# frame. A PAIRED critic (real and reconstruction channel-stacked) would be
# better posed still, since an AE's pair is perfectly registered, and the
# implementation exists in aag/discriminator -- worth an A/B later, but this run
# uses the proven recipe rather than introducing a second variable.
# --amp AND --loader-workers ARE NOT OPTIONAL HERE, and the first attempt at this
# run proved it by omitting both. It ran 4h without finishing one epoch. The
# original nine epochs of this same AE took 99 min each (checkpoint mtimes,
# ae_dcae_ch192_dim256: ep1 13:15, ep2 14:54, ep3 16:33, ...) and both original
# logs print "torch.compile enabled" and "bf16 autocast enabled". Dropping amp
# put the run in fp32, and 2.4x+ of GPU time went to nothing recoverable -- no
# checkpoint is written until an epoch ends, so the whole 4h was discarded.
#
# --compile is correctly absent: train_ae.py refuses it under --gan-weight,
# because the alternating D/G steps and adaptive_weight's two autograd.grad
# probes graph-break repeatedly. So this run is permanently eager, and part of
# the gap against the original 99 min cannot be closed. cudnn.benchmark (added
# to train_ae.py) recovers some of it by autotuning the conv algorithms once,
# which is worth more in eager mode than under compile.
#
# --log-every 500 is the other half of the fix. train_ae.py previously printed
# NOTHING between epoch lines, so "is this run healthy" was unanswerable for
# hours -- which is exactly how a missing --amp survived a night. The line now
# carries the GAN health terms too (g, d, and the adaptive weight w), so a
# critic that has won (d -> 0) or a blown-up gradient probe (w at the 1e4 cap)
# shows up in minutes rather than at the next epoch boundary.
set -uo pipefail
WT=/home/ubuntu/exp/newgen/.claude/worktrees/vpt-cache-hardening
PY=/home/ubuntu/exp/newgen/.venv/bin/python
CK=/data/aag_results/results_vpt/ae_dcae_ch192_dim256_gan/checkpoints/ae_doom_frames_dcae_lpips_ch192_dim256_ep1.pt
cd "$WT" || exit 1

echo "=== AE + adversary, resumed from 9 epochs $(date -u +%H:%M:%S) ==="
PYTHONPATH=$WT "$PY" scripts/train_ae.py \
  --dataset doom_frames --data /opt/dlami/nvme/vpt_full \
  --arch dcae --ch 192 --dim 256 --grid 4 --image-size 64 \
  --lpips-weight 0.5 --lpips-upsample 1 \
  --resume "$CK" \
  --gan-weight 0.5 --gan-start-epoch 1 --gan-layers 2 --gan-ndf 64 \
  --gan-lr 4.5e-5 \
  --amp --loader-workers 12 --log-every 500 \
  --epochs 5 --batch 128 --lr 1e-4 --eval-every 1 \
  --out /data/aag_results/results_vpt/ae_dcae_ch192_dim256_gan \
  > /data/vpt/ae_gan2.log 2>&1
echo "=== AE exited rc=$? $(date -u +%H:%M:%S) ==="
tail -5 /data/vpt/ae_gan2.log
