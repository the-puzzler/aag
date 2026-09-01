#!/usr/bin/env bash
# Continue the ctx3 512k lineage onto the 12-d action representation.
#
# Resumes z from assign_ctx3_384k (step 352,000). That work is not wasted by the
# representation change: the context transport and the global gaussianisation are
# independent of what the action vector contains, and the old action groupings
# (marginals of the 81-way index) are marginals of a SUBSET of the new vector, so
# their transport is still valid -- just incomplete. Resuming is strictly
# additive.
#
# Budget is reallocated rather than grown. At step 352,000 the context side is
# well transported (I_ctx 0.61-0.74) and the global objective G sits at 0.00001
# against a 0.0035 floor -- deep inside its own noise band -- so extra context
# firings buy little. The new work is the sixteen action marginals, ten of which
# (E, attack, use, per-key W/A/S/D, mouse magnitude) have never been transported
# against at all.
#
#   was:  112 ctx + 24 act + 48 grp = 184 firings/step,  6 groupings -> 8 each
#   now:   56 ctx + 24 act + 96 grp = 176 firings/step, 16 groupings -> 6 each
#
# Same step cost, six firings for every marginal the generator can now key on.
set -euo pipefail
WT=/home/ubuntu/exp/newgen/.claude/worktrees/vpt-cache-hardening
cd "$WT"
PYTHONPATH=$WT /home/ubuntu/exp/newgen/.venv/bin/python scripts/run_assignment_vpt.py \
  --particles /data/vpt/particles_dim256_512k_12d.pt \
  --resume-z /data/aag_results/results_vpt/assign_ctx3_384k/assignment.pt \
  --ctx-frames 3 --ctx-scales 1,3 --ctx-metric cosine \
  --act-groups w,a,s,d,space,shift,ctrl,e,attack,use,dxsign,dysign,dxmag,dymag,anyclick,moving \
  --steps 96000 --eval-every 2000 --save-every 8000 \
  --k 3329 --k-act 13316 \
  --ctx-per-step 56 --act-per-step 24 --grp-per-step 96 \
  --max-group 8192 --cond-alpha 0.25 \
  --out /data/aag_results/results_vpt/assign_ctx3_12d/assignment.pt
