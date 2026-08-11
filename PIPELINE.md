# AAG — Amortised Assignment Generation

The main thread. CelebA 64x64.

Each example keeps a *persistent* particle: the identity x_i <-> z_i is fixed for
the whole run while z_i is transported toward N(0,I) by cheap rank operations.
That assignment is expensive and non-parametric, computed once offline — then a
single feedforward network **amortises** it, so generation is one forward pass
from fresh z ~ N(0,I). No iterative sampling.

(Formerly "PGGA / Persistent Global Gaussian Assignment" — same method.)

    images ──► AE (dim=64, MSE+LPIPS) ──► assignment ──► generator ──► pixels
                                          (+conditional)   (direct-to-pixel,
                                                            MSE+LPIPS)

Everything below is the canonical path. Anything not listed lives in `archive/`.

## The four stages

**1. Autoencoder** — `train_ae.py`
dim=64, residual arch, ch=64, loss = MSE + 0.5·LPIPS(vgg).
Only the *encoder* is used downstream. LPIPS matters: MSE alone gives misleading
checkpoint selection.
→ `results_celeba/ae/residual_lpips/`

**2. Assignment** — `run_assignment.py` (global) / `run_assignment_conditional.py` (+conditional)
Whiten the AE latents, then iteratively rank-transport the worst random 1D
projection toward Gaussian quantiles, with slab cleanup and radial chi
calibration.

*Stop when the transport objective enters its N(0,I) noise floor* (~4k steps for
N=162770, d=64; floor ≈ 0.0043). Past that the objective cannot see progress
while transport displacement keeps accumulating, and displacement costs the
generator — see "Open questions".

The conditional variant additionally rank-transports k-NN-by-Hamming
neighbourhoods in the 40-dim binary attribute space. This is a *correctness
condition*, not a refinement: sampling z~N(0,I) and pairing it with a chosen
condition c is only valid if z ⊥ c. Track
`independence_ratio = conditional_group_w2 / random_subset_w2 → 1.0`.

**3. Generator** — `train_generator.py` (uncond) / `train_generator_conditional.py` (cond)
Direct-to-pixel: z (or z⊕cond) → image, loss = MSE + 0.5·LPIPS, grad-clip 1.0,
200 epochs. **Two-stage generation through a frozen AE decoder is abandoned** —
the decoder is brittle to off-manifold h and produces garish artifact tiles that
direct-to-pixel does not.

Pass `--particle-order` whenever the assignment came from AE latents: those use
`celeba_loaders`' permuted particle order, while raw-embedding pipelines use
natural dataset order. Getting this wrong silently misaligns (z, image) pairs —
it once looked exactly like mode collapse.

**4. Attributes** — `extract_celeba_attrs.py` → `results_celeba/attrs.pt`
40 binary attributes per image, aligned to the same particle order.

## Standard plots — `plots/`

    plots/plot_assignment.py   <assignment.pt>...  --out fig.png
    plots/plot_generations.py  <run_dir>...        --out fig.png
    plots/plot_conditional.py  <run_dir>...        --out fig.png
    plots/plot_ae.py           <ae_ckpt.pt>... [--curve curve.json] --out fig.png

All accept multiple inputs and stack them as rows for comparison. Samples use
seed=0 everywhere so grids are comparable across runs.

## Evaluation

`compute_real_fid_stats.py` (once) → `run_fid_sweep.py` → `results_fid/`.
FID is a **measurement, not a verdict** — it has contradicted visual judgement in
this project (LeJEPA). Always look at the samples.
`demo_rare_conditions.py` renders rare/contradictory attribute combinations,
which is where residual z–condition dependence actually shows up.

## The flagship

The best-established result is this pipeline *minus* the conditional step:
AE(dim64, LPIPS) → 4k global assignment → direct-to-pixel generator.
→ `results_celeba/pixel_generator_lpips_flagshipz/`

## Open questions

- **Gaussianity vs learnability.** More transport buys Gaussianity but moves
  particles further (displacement 1.17 @4k → 2.91 @4k-dense → 3.41 @60k), which
  degrades z→image locality and makes the generator fit ~1.5–2x worse. Whether
  that hurts *generation* is being tested now.
- **Independence has a displacement cost.** Concentrating conditional budget into
  4k global steps reached ratio 1.21 but at displacement 2.91, not the flagship's
  1.17. Untested fixes: weaker conditional steps (alpha<1) or a lower
  conditional:global ratio.
- **No conditional generator has ever used a low-displacement assignment** — the
  cheapest conditional z so far is displacement 2.53.
