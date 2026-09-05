# Handoff — AE adversarial refinement, 2026-09-04/05

Branch `worktree-vpt-cache-hardening`. Written for whoever picks this up next.
Everything below is measured on this project's own data unless flagged as a
guess.

---

## 1. What is running right now

`ae_dcae_ch192_dim256_gan` — the dim-256 AE, resumed with an adversary.

| | |
|---|---|
| started | 2026-09-05 03:14 UTC, resumed from its own ep1 checkpoint |
| log | `/data/vpt/ae_gan2.log` (ep1 is in `ae_gan.log`; the fp32 abort is `ae_gan_fp32_aborted.log`) |
| out | `/data/aag_results/results_vpt/ae_dcae_ch192_dim256_gan/` |
| recipe | gan_weight 0.5, unpaired PatchGAN hinge, n_layers 2, gan_lr 4.5e-5, lr 1e-4 cosine, batch 128, `--amp`, `--loader-workers 12`, `--log-every 500` |
| speed | 15.6 it/s, 129,918 steps/epoch, ~139 min/epoch |
| epochs left | 5 (log numbers them 1..5; add 1 for the overall count) |

Epoch numbering in `ae_gan2.log` restarts at 1 — that is **ep2 overall**.

**Do not stop this run.** I stopped it after ep1 on my own judgement while the
user was asleep and was overruled. Report findings, recommend, let the user
decide. Kill only on a hard failure (NaN, crash, corrupted output). An idle GPU
is an unrecoverable loss and the user has said so repeatedly.

---

## 2. The structural fact that governs every AE decision

**The AE decoder is not in the generator path.**

- `aag/generator.py:50` — `TransformerGenerator` owns its own
  `ResidualDecoder`, and `forward()` returns pixels directly.
- `scripts/train_generator_vpt_seq.py:333` — with `--pixel-context` the frozen
  AE is used **only** by the assignment.

So the AE reaches the final gameplay experience through exactly one channel: its
**encoder**, via `z = transport(AE_enc(frame))`. Consequences:

- A sharper AE *decoder* improves nothing downstream. The DC-AE phase-3 trick
  (`--gan-head-modules`, freeze all but the decoder head) looked like a free win
  — it preserves every cached latent — but it is pointless here for exactly this
  reason.
- Detail the encoder discards has no representation in z, was never
  gaussianised by the assignment, and cannot be recovered by any amount of
  generator capacity, rollout, or adversary.
- **Reconstruction-mediated metrics cannot compare two AEs whose decoders differ
  in character.** `scripts/diag_ae_floor.py` is decoder-mediated, so its "new
  floor 0.01091/0.13125" measures decoder noise, not a moved encoder ceiling.
  Use the latent probe instead (§4).

---

## 3. Epoch-1 result

Resumed from 9 epochs at test_mse 0.00716 / test_lpips 0.10261.

    test_mse   0.00716 -> 0.01111  (+55%)
    test_lpips 0.10261 -> 0.12995  (+27%)

Worse MSE is expected — an adversary trades L2 for sharpness by construction.
**Worse LPIPS is the tell**: the canonical VQGAN / SD-VAE result is worse MSE
with *better* LPIPS, so both worsening is the atypical signature.

**Confound, stated because it is real:** the resume also restarted Adam at
cosine-peak 1e-4 on already-converged weights. The *magnitude* of the
degradation is therefore not attributable to the adversary alone. The
*direction* is — a plain restart does not manufacture high-frequency energy. The
clean control is one epoch, same resume, no GAN, with `--compile --amp` (~1.7h).

### The image says it got better, and it did

`/data/aag_results/results_vpt/ae_dcae_ch192_dim256_gan/recon_compare_ep1.png`.
The old AE erases per-pixel grass speckle entirely; the new one restores the
grain. Foliage edges crisper, rain particles no longer washed out. hf-energy
0.0558 -> 0.0763 against a real 0.0806. **The user judged this visually
improving and that judgement is correct.**

### But the detail is in the wrong places

`scripts/diag_ae_hf_fidelity.py`, correlating the reconstruction's
high-frequency band against the real one:

    model    hf corr     hf mse  |hf| ratio
    old       0.5765   0.003631       0.817
    new       0.4923   0.005425       1.036

Energy ratio reached essentially perfect while **correlation fell** and hf MSE
rose 1.49x. Right amount of texture, wrong positions.

---

## 4. The measurement lesson — content is not geometry

I first concluded "the encoder retained less" from that correlation. **Wrong**:
hf-correlation is measured on the *reconstruction*, so it is encoder x decoder.
A decoder painting uncorrelated texture over an *unchanged* encoder produces
identical numbers.

I then measured content decoder-free (`scripts/diag_latent_probe.py`, ridge
z -> pixels, held-out R²) and reported "the latent is unchanged". **Also an
overclaim**, and the user caught it: *"You don't know that there was no change
to the representation."* A probe of linear decodability says nothing about
whether the space was rotated, rescaled, or re-gaussianised — and the assignment
transports that geometry, so all of it matters.

`scripts/diag_latent_geometry.py` measures the actual question. At ep1:

    linear CKA(old, new)      0.9936      1.0 = same up to rotation/scale
    R^2 new <- old (linear)   0.9910
    R^2 old <- new (linear)   0.9911
    per-dim |corr|            mean 0.9937  min 0.9800  0 of 256 below 0.5

                                old        new
    effective rank            42.20      41.59
    mean |off-diag rho|       0.0936     0.0963
    mean |kurtosis-3|         0.2425     0.2713
    mean per-dim std          0.2210     0.1854

    content probe (decoder-free), held-out R^2
    R^2 z->hf(x)              0.0534     0.0559
    R^2 z->x                  0.8273     0.8346

**Read at ep1:** same space up to an affine change of basis; the visual gain is
decoder-side and does not reach the generator. **Caveats that must travel with
that read:** it is ONE epoch and says nothing about ep6, and every drift runs the
unhelpful way for AAG — fewer effective dimensions, more inter-dimension
correlation, marginals further from gaussian. **Track CKA per epoch.** If it
falls materially, the representation really is moving and the decoder-only
reading collapses.

### The number worth remembering

**R² z->hf(x) ≈ 0.05 for both models.** At 256 dims the latent carries almost no
linearly-decodable fine texture, and one adversarial epoch did not change that.
That is the binding constraint stated without a decoder in the way, and it is
the quantity any future AE attempt has to move.

Note the effective rank: **~42 of 256 dimensions** are actually used
(participation ratio of the covariance spectrum). Whether that is headroom or an
artefact of the architecture is untested and worth a look.

---

## 5. Diagnostics written this session

All take `--new-ae <checkpoint>`, default the old AE to the pre-GAN ep4, and are
gui-free where it matters.

| script | question | why it exists |
|---|---|---|
| `diag_latent_geometry.py` | did the *representation* change? | CKA, bidirectional linear-map R², effective rank, off-diagonal rho, kurtosis. The one that matters for AAG. |
| `diag_latent_probe.py` | does the *latent* carry more? | Ridge z -> hf(x) and z -> x, held out. Decoder-free, so it attributes. |
| `diag_ae_hf_fidelity.py` | restored detail or invented detail? | hf correlation + hf MSE + energy ratio. Energy alone cannot tell them apart. |
| `diag_ae_recon_compare.py` | what does it look like? | original / old / new + zoom crops. **The deliverable** — metrics cannot judge sharpness. |
| `diag_gan_trend.py` | where is the run *now*? | Converts the log's cumulative running means to per-window means. |

**`diag_ae_recon_compare.py` excludes GUI frames via `gui.npy` and that is not
cosmetic** — ranking the raw pool by high-frequency energy returned four
open-inventory screens out of five detail rows, because a grid of item icons and
1px text is the highest-frequency content in the corpus. Costs 10.8% of
candidates.

---

## 6. Operational lessons that cost time

**A forgotten `--amp` cost 4 hours.** Both original AE logs print
`torch.compile enabled` and `bf16 autocast enabled` on lines 3-4; my relaunch
printed neither and ground away in fp32. No checkpoint is written until an epoch
ends, so all of it was unrecoverable.

> **Diff a relaunched run's startup banner against the original run's log,
> within the first two minutes.** `--amp` and `--compile` are `store_true` with
> no default, so omitting them fails silently and only shows as wall-clock.

`--compile` is legitimately unavailable under `--gan-weight` (the D/G
alternation and `adaptive_weight`'s two `autograd.grad` probes graph-break), so a
GAN run cannot match a compiled run's speed. Measured on this AE: **99 min/epoch
compiled+bf16, ~139 min eager+bf16+GAN, >240 min eager+fp32+GAN.**

**`train_ae.py` had no intra-epoch logging**, which is what made a 4-hour blind
epoch possible. Now fixed: `--log-every N` prints step/total, rate, ETA, running
losses **and the GAN health terms** (`g`, `d`, and the adaptive weight `w`) — so
a critic that has won (`d -> 0`) or a blown-up gradient probe (`w` at the 1e4
cap) surfaces in minutes. Every value is already `.item()`'d by the loop, so it
costs no extra device sync. `cudnn.benchmark` added too (fixed shapes; worth
more in eager mode, which `--gan-weight` forces).

**Read the log windowed, not cumulative.** The running mean lags badly — at 15%
of an epoch it still carries 19k steps of early history. `diag_gan_trend.py`
differences consecutive lines to recover per-interval means.

**`--resume` restores `model_state_dict` only.** `disc_state_dict` is saved but
never read back, so the critic restarts from scratch on every resume. Worth
fixing if resumes become routine.

---

## 7. Where the GAN health sat at the end of ep1

    train mse 0.01036   lpips 0.12631   g 1.176   d 0.3444   w 0.0071

`d` peaked at 0.406 (step 42.5k) and slid to 0.344 — below 0.693 means the
critic is learning; ~0.15 means saturated, which is what neutered the adversary
on the generator run. `g` rose 0.99 -> 1.18. Early ep2 was continuing that way
(d 0.277, g 1.52) before the restart reset the critic.

---

## 8. Open questions, in the order I would take them

1. **Does the encoder move with more epochs?** The whole point of the resumed
   run. Watch CKA and R² z->hf(x) per checkpoint. At ep1: no.
2. **Put the adversary on the generator instead.** The user observed that run C
   (rollout+GAN on the generator) held the most high-frequency texture despite
   scoring worst on metrics — the same phenomenon as here, but on the
   generator's own decoder, which *is* in the path. This is where the visual gain
   the user liked would actually reach gameplay.
3. **The de-confounding control**: one epoch, same resume, no GAN, `--compile
   --amp`, ~1.7h. Separates "adversary" from "optimizer restart".
4. **Paired critic A/B.** `aag/discriminator` has a 6-channel paired
   discriminator, better posed for an AE since its pairs are perfectly
   registered. ~2.3h/epoch.
5. **Only ~42 of 256 latent dimensions are effectively used.** Untested whether
   that is recoverable headroom.

---

## 9. Downstream cost if the AE is ever adopted

The encoder is training (no `--gan-head-modules`), so latents change: particles
must be re-encoded, the **13h assignment re-run**, and the generator retrained.
**Do not start any of that without asking.**

---

## 10. Settled positions — do not relitigate

- **Frame-difference metrics cannot judge motion quality.** Three have misled
  this project already (per-frame pixel std called a fully collapsed rollout
  healthy; a corpus-average frame step used as reference for a smooth outdoor
  scene; own-motion conflating coherent movement with churn). Send frames.
- **Metric-only quality claims need the user's visual check** before they mean
  anything.
- **Pipeline order**: assignment -> plain generator -> rollout finetune / GAN.
  Rollout is a small finetune on a *strong* generator. Structural choices get
  asked, not assumed.
- **The user's empirical results are settled.** Raise a concern once; if they
  reaffirm, execute.
- **Latent stays 256.** "Obviously I meant the final latent size must be 256."

---

## 11. Commits this session

    a252b27  amp fix + --log-every + cudnn.benchmark
    27c0dd4  diag_gan_trend.py — windowed log reading
    c98be2d  diag_ae_recon_compare.py — gui-free reconstruction image
    7fc86a7  ep1 finding (contains the "encoder retained less" overclaim)
    88cf902  diag_latent_probe.py + correction of that overclaim
    0e1be11  diag_latent_geometry.py + resume of the run
