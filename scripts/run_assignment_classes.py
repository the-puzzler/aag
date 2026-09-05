#!/usr/bin/env python
"""Gaussian assignment for a CATEGORICAL condition with a hierarchy, or none.

Used for ImageNet-256 (class-conditional) and CelebA-HQ-256 (unconditional).
Two of the user's rules shape it:

  1. "More Gaussian is more good." The transport budget is generous, the run
     saves step-stamped checkpoints (--keep-checkpoints) and the step count is
     chosen AFTERWARDS by the generator trained on each -- never by the
     transport objective, which sits inside its own noise floor after a few
     thousand steps and cannot see further progress.

  2. Gaussianising a group is not gaussianising its marginals. Independence
     from the 1000-way class does not buy independence from "is a dog": each
     per-class transport moves ~1281 particles and cannot resolve a shift
     shared by all 116 dog classes, while one transport over those 148k
     particles removes it. So group transport is INTERLEAVED across every
     hierarchy level (--levels), size-weighted so touches per particle are
     equalised across groups of very different sizes, and the readout reports
     the ratio at EVERY level -- one number against the joint class is exactly
     the reading that hid the VPT action leak for days.

Global steps are the published recipe (greedy rank transport + slab cleanup +
radial chi calibration). Whitening is rotate=False so the spatial grid of the
AE latent survives into z.
"""
from __future__ import annotations

import argparse, gc, json, time
from pathlib import Path

import torch

from aag.diagnostics import group_w2, random_subset_w2, transport_objective_floor
from aag.gaussianize import (greedy_rank_transport_step, group_rank_transport_step,
                             offset_slab_cleanup_step, radial_chi_calibration, whiten)

ap = argparse.ArgumentParser()
ap.add_argument("--particles", required=True, help="output of encode_hf256.py")
ap.add_argument("--groups", default=None, help="class_groups.pt from imagenet_class_groups.py")
ap.add_argument("--levels", default="",
                help="comma list of hierarchy levels to transport against, e.g. "
                     "'joint,depth7,depth6,depth5,depth4,living,animal,dog'. Empty = unconditional.")
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--search-subset", type=int, default=2048)
ap.add_argument("--n-dirs", type=int, default=64)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--grp-per-step", type=int, default=8,
                help="group-transport firings per global step, split evenly over levels")
ap.add_argument("--cond-alpha", type=float, default=0.5)
ap.add_argument("--max-group", type=int, default=32768,
                help="particles per group firing; a coarse group like 'artifact' has "
                     "660k members and 32k already puts quantile noise at 0.6%%")
ap.add_argument("--cleanup-every", type=int, default=2)
ap.add_argument("--chi-every", type=int, default=20)
