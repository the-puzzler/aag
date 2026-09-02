#!/usr/bin/env bash
# The 12-d action representation, assigned FROM SCRATCH, on lag-corrected actions.
#
# THE ACTION LAG. Earlier runs conditioned target frame t on the action recorded
# at tick t. That is one tick late: VPT stores (observation, action) in the
# standard RL convention, so the action at tick t is chosen after seeing frame t
# and produces frame t+1. The action that produces frame t is a_(t-1). Measured
# two ways on this cache:
#
#   corr(|mouse|, |frame_t - frame_(t-1)|)  a_(t-2) 0.383  a_(t-1) 0.481  a_t 0.389
#   signed horizontal image shift into t    dx_(t-1) -0.588     dx_t -0.477
#   sign agreement of that shift            dx_(t-1) 69.4%      dx_t 61.7%
#
# The lag profile peaks at t-1 with a symmetric falloff, and only the SIGNED test
# separates the two properly -- mouse motion is autocorrelated enough that the
# wrong lag still scores 0.39, which is why this survived so long. It degraded
# rather than broke.
#
# Why it forces a rerun rather than just a retrain: the action-side transport
# built its k-NN neighbourhoods and its sixteen marginal groups from the WRONG
# action per particle, so z was decorrelated from something other than the action
# the generator will be handed. That is a mechanism for the one leak that never
# closed -- dxsign sat at +2.46 after 384,000 steps against an explicit dxsign
# grouping, because the correct turn signal had no reason to leave z.
#
# An earlier attempt resumed z from assign_ctx3_384k (step 352,000) on the
# reasoning that context transport is representation-independent, so it could be
# reused. That was wrong, for two compounding reasons:
#
#   * Path dependence. Greedy rank transport is order dependent, so 352,000
#     steps had already arranged z to satisfy constraints that never included
#     attack or use. Measured, that arrangement was NOT click-independent: the
#     null-referenced leakage probe read anyclick +3.08 and dxsign +4.22 on that
#     z. Fixing those marginals from there means paying further displacement to
#     undo an arrangement built without them.
#
#   * Displacement, which is the decisive one. Mean ||z - z_whitened|| grows
#     monotonically while the transport objective saturates and stops being able
#     to see it. On CelebA the 4k assignment moved particles 15% of their own
#     radius and the 60k one 43%, and the 60k z fit the generator 1.6-2.2x WORSE
#     at matched epochs -- particles moved that far no longer have neighbouring z
#     mapping to similar images, so the z->image map loses locality. The doom
#     world model that actually worked used 16,000 steps. Resuming at 352,000 and
#     adding more pushes further along an axis already shown to hurt.
#
# So: fresh whitening, the full context budget restored (from scratch the context
# side has real work to do again, unlike in the resumed run), and all sixteen
# marginals of the 12-d vector interleaved from step 1 -- the arrangement is
# built with clicks accounted for rather than patched afterwards.
#
#   112 ctx + 24 act + 96 grp = 232 firings/step, 16 groupings -> 6 firings each
#
# 384,000 steps, matching the budget scale this lineage has actually used. Do NOT
# shorten this on the strength of the CelebA 4k-vs-60k displacement result: that
# comparison had NO per-marginal transport, so its extra steps bought
# displacement without buying marginal independence, and it is confounded for
# this purpose. On VPT more transport has consistently been better (192k -> 384k
# improved I_ctx; 16k -> 48k lifted the generator's fresh-z HF ratio 0.633 ->
# 0.921), and this is far bigger and more complex data.
#
# disp= is logged as a free diagnostic, not as a stopping rule. Measured here it
# is a fast-SATURATING curve -- 53% of ||z|| by step 1,000 and 97.2% by step
# 352,000 -- so it mostly reports how far the cloud has rearranged from whitened,
# and CelebA's 15%/43% figures sit in the early, under-transported part of it.
# It is a progress meter, not a damage meter.
#
# --keep-checkpoints every 16,000 steps (~62 GB total) so the run can be stopped
# anywhere and the step count chosen afterwards by fresh-z MSE, which is the only
# thing that has ever selected an assignment here.
set -euo pipefail
WT=/home/ubuntu/exp/newgen/.claude/worktrees/vpt-cache-hardening
cd "$WT"
PYTHONPATH=$WT /home/ubuntu/exp/newgen/.venv/bin/python scripts/run_assignment_vpt.py \
  --particles /data/vpt/particles_dim256_512k_12d_lag1.pt \
  --ctx-frames 3 --ctx-scales 1,3 --ctx-metric cosine \
  --act-groups w,a,s,d,space,shift,ctrl,e,attack,use,dxsign,dysign,dxmag,dymag,anyclick,moving \
  --steps 384000 --eval-every 2000 --save-every 16000 --keep-checkpoints \
  --k 3329 --k-act 13316 \
  --ctx-per-step 112 --act-per-step 24 --grp-per-step 96 \
  --max-group 8192 --cond-alpha 0.25 \
  --out /data/aag_results/results_vpt/assign_12d_lag1/assignment.pt
