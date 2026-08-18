# AAG — Method, Theory, Metrics, Scaling

A working reference: what the method is, why it is valid, what we measure, and
the empirical scaling behaviour. Every claim here is backed by a run in this
repo; `assets/curves/` holds the per-experiment assignment and training curves.

---

## 1. The method in one page

Three stages, all offline; generation afterwards is a single forward pass.

1. **Encode.** Train an autoencoder (MSE + LPIPS). Keep the encoder; latents
   `h_i = E(x_i)`, `i = 1..N`, dimension `d`.
2. **Assign.** Transport the whitened latent cloud onto `N(0, I_d)` by iterated
   1-D rank transports (below). The output is a *persistent assignment*: a fixed
   coordinate `z_i` paired with every training example `x_i`.
3. **Amortise.** Train a generator `G : z (⊕ c) → x` on the frozen pairs,
   direct to pixels, MSE + LPIPS. (Decoding through the frozen AE decoder is
   measurably worse — it is brittle off-manifold.)

Sampling: `z ~ N(0, I)` → `G(z)` (optionally with a chosen condition `c`).
No denoising loop, no autoregression.

---

## 2. The assignment step

```text
input   h[1..N, d]                     # AE latents
        cond[1..N] (optional)          # condition per particle
params  S        global steps          # a budget, not part of the method — see
        #   the stopping note below the loop
        α_g      global step size      # 1.0 in all successful runs
        c        conditional steps per global step
        α_c      conditional step size # 0.25 everywhere that worked
        k        conditional neighbourhood size   # < median condition-group size
        n_dirs   directions searched per step     # small is fine (see §5)

z ← PCA-whiten(h)                      # rotation is load-bearing, see §5
q_N ← Φ⁻¹((rank + ½) / N)              # Gaussian order statistics, fixed

repeat S times:
    # ---- global step: make the whole cloud Gaussian --------------------
    draw n_dirs random unit vectors; on a random subset of z,
        score(a) = mean_j ( sort(z·a)_j − q_j )²      # empirical 1-D W2² to N(0,1)
    a* ← argmax score                                  # worst direction found
    rank-transport ALL of z along a*:
        target[rank order of z·a*] = q_N
        z += α_g · (target − z·a*) · a*

    # ---- conditional steps: make z ⊥ c ---------------------------------
    if conditional:
        repeat c times:
            i ← random particle
            S_i ← particles with a similar condition to i:
                    categorical c :  exact group { j : cond_j = cond_i }
                    continuous  c :  k nearest under a metric (cosine, L2)
                    hybrid       :  exact tag filter, then k-NN inside it
            a* ← worst direction for z[S_i] (same score, within the subset)
            rank-transport z[S_i] only, along a*, scaled by α_c

# stopping: not part of the method — S and c are budgets, chosen by watching
# the diagnostics. Stop the global budget when G (measured against a matched
# iid N(0,I) reference) flattens, and the conditional budget when I flattens.
# Caution: the transport objective's own noise floor is NOT a reliable stop —
# it is computed on a subset along single directions and saturates while G is
# still genuinely improving (observed: floor reached by ~3k steps at d=256,
# G still falling at 160k; the best d=64 run used 4x its floor budget).
```

Optional refinements, interleaved on the CelebA runs (the video runs did
without): an offset-slab cleanup that Gaussianises a tangent coordinate inside a
thin slab `|n·z − b| < ε` (catching localized tail spikes single projections
miss), and a radial calibration that rank-corrects `‖z‖` toward the exact χ_d
law (the high-d shell error survives every 1-D test).

The inner operation is exact: sorting projections and matching them to Gaussian
order statistics **is** optimal transport in one dimension, and `score` is the
1-D Wasserstein-2² being minimised. Everything around it — the greedy direction
search, the subset sampling, the interleave ratio — is heuristic, and §5 records
which of those heuristics matter.

---

## 3. Why it is valid

**Unconditional.** The generator is trained on pairs `(z_i, x_i)`; at sampling
time it receives fresh `z ~ N(0, I)`. The pushforward `G#N(0,I)` matches the data
distribution to the extent that `{z_i}` is indistinguishable from an iid Gaussian
sample. Cramér–Wold gives the direction-wise route: Gaussian along every 1-D
projection ⇒ jointly Gaussian, which is exactly what the iterated sliced
transport drives toward. Two honest caveats:

- The correct target is a *finite Gaussian sample*, not the analytic density. A
  genuine `N`-draw at this `(N, d)` has sampling fluctuation; rank transport
  suppresses it (the transported cloud scores far *below* the iid reference on
  sliced W2). This is a mild train/inference mismatch — fresh `z` has the
  fluctuation the training coordinates lacked.
- Matching the marginal `{z_i}` says nothing about *which* particle holds which
  coordinate. The coupling is what the generator must interpolate; see R in §4.

**Conditional.** Generation draws `z` independently and pairs it with a chosen
`c`. That is only licensed by

    p(z | c) = p(z)        i.e.  z ⊥ c .

Without it, some `(z, c)` combinations never occurred in training and the
generator is queried off-distribution exactly where conditioning is most
interesting. The conditional transport steps enforce it by Gaussianising every
condition-neighbourhood *on its own terms*: if `z` restricted to each
neighbourhood is `N(0, I)`, the conditional and marginal laws agree.

The residual the condition leaves for `z` to model is

    f_z = E_c[ tr Cov(h | c) ] / tr Cov(h)   ∈ [0, 1] ,

and how hard the independence constraint is to satisfy scales with how much of
`h` the condition already explains (`1 − f_z`): weak labels (CIFAR classes,
R² ≈ 0.05) reach `I ≈ 1` almost for free; a 3-frame+action context (R² ≈ 0.88)
never got below `I ≈ 14`.

---

## 4. The metrics: G, I, R

| | measures | target | verdict from ground truth |
|---|---|---|---|
| **G** | is the cloud Gaussian? | ≈ 1 (iid floor) | predicted the FID winner every time |
| **I** | is z independent of c? | ≈ 1 | predicted the FID winner every time |
| **R** | is the coupling locally decodable? | low | mis-ranked both FID comparisons |

**G — Gaussian defect** (`proj_over_gauss`). Mean sliced W2² over random
directions, normalised by the same statistic for a genuine iid `N(0,I)` sample of
matched `(N, d)`. `G ≈ 1` means "as Gaussian as a real sample"; our transported
clouds routinely sit below the floor on the raw statistic (over-regular).

**I — independence ratio.**

    I = W2( z within a condition-neighbourhood ) / W2( z within a random subset of equal size )

evaluated at matched subset size. `I = 1` ⇔ conditional slices look like random
slices ⇔ `z ⊥ c` at the resolution measured. It can overshoot below 1 harmlessly
(toy: 0.09 with no loss). **Not comparable across conditioning setups** — it
tracks how much dependence existed to remove; normalise against the ratio at
zero conditional steps if two setups must be compared.

**R — prior-local decodability.** Probes come from the *prior*, not the data:
sample `u ~ N(0, I)`, take the `k` assigned `z` nearest to `u`, inspect their
targets `h`:

    R_rel      = local affine prediction error / local target variance      (≈ 1 − R² of a local fit)
    R_rel|c    = same, but neighbourhoods restricted to similar c, and the
                 denominator is Var(h − E[h|c])  — the variance z should explain
    R_impact   = f_z · R_rel|c   = local error as a fraction of TOTAL Var(h)

**Empirical standing.** A controlled scramble —
permuting the pairing while leaving the cloud untouched — degrades generation
674× with G *exactly constant*, and R tracks it with Spearman 1.0. So R detects
a real failure G cannot. But monotone rank transport cannot produce that
failure, and on real assignments R has mis-ranked both comparisons where
held-out FID exists, including an assignment at `R_impact = 1.57` (nominally a
dead latent) that produced the best video model. **Rank by G first, then I; read
R as description, not prediction; never rank by generator training loss** — it
rewards whichever assignment transported least.

---

## 5. Scaling behaviour

All empirical, from the runs in this repo.

**Samples per dimension is the binding constraint on the AE.**
Count *independent* units, not augmented segments:

| dataset | independent N | d | N/d | outcome |
|---|---|---|---|---|
| CelebA | 162,770 | 64 | 2,543 | works |
| CIFAR-10 | 50,000 | 64 | 781 | works, but underperforms a flow-matching baseline |
| UCF-101 | 9,537 clips | 256 | **37** | fails — thin structure unrecoverable |
| Doom video | 70,000 episodes | 256 | 273 | marginal: 20k+ steps needed |
| Doom video | 70,000 episodes | 64 | 1,094 | best result (FID 60.10) |

Rule of thumb: keep independent-N/d in the ~10³ regime; reducing `d` toward a
few × the intrinsic dimension (TwoNN) is the cheapest lever and improved FID
outright.

**More samples keep paying on some datasets and not others, and the intrinsic
dimension does not predict which.** Holding the AE, the assignment budget
(64k steps) and the generator recipe fixed and varying only N:

| dataset | N | TwoNN | N / TwoNN | G | FID (40 ep) |
|---|---|---|---|---|---|
| CIFAR-10 | 6,250 | 29.8 | 210 | 0.60 | 67.24 |
| CIFAR-10 | 12,500 | 30.3 | 413 | 0.55 | 56.49 |
| CIFAR-10 | 25,000 | 29.6 | 845 | 0.55 | 51.11 |
| CIFAR-10 | 50,000 | 29.0 | 1,690 | 0.60 | 46.14 |
| CelebA | 50,000 | 27.0 | 1,852 | 0.57 | 21.73 |
| CelebA | 162,770 | 27.2 | 5,985 | 0.76 | 22.54 |

(Probe runs at a fixed 40 epochs for comparability — not the showcase numbers.)

CIFAR gains ~5 FID per doubling of N with no sign of flattening at 50k, so it
is genuinely sample-hungry. But the obvious explanation — too few samples per
effective dimension — does not survive the control: **at essentially matched
samples-per-intrinsic-dim (1,690 vs 1,852) CelebA reaches 21.7 where CIFAR
reaches 46.1**, and CelebA is already saturated at N = 50,000, gaining nothing
from 3.25× more data. TwoNN barely separates the two datasets (29 vs 27), so
whatever makes CIFAR hard for this method is not captured by it. The working
hypothesis is that the deterministic one-pass z→pixel map suits a smooth,
aligned, low-diversity manifold (faces) and struggles on a fragmented one (ten
disjoint semantic classes, unaligned) — untested.

Two confounds, pulling opposite ways. In the full-N arm's favour: 40 epochs
over 162,770 samples is 3.25× more gradient steps. Against it: at a fixed 64k
budget the larger assignment is less converged (G 0.76 vs 0.57). The second is
bounded by the CIFAR sweep in §5 — moving G from 1.07 to 0.56 there was worth
only ~1.7 FID, so a 0.76 → 0.57 difference can account for well under 1 FID,
far short of the gain 3.25× more data should have produced.

**Steps scale hard with d.** `d = 64` reaches its G floor in ~4k steps;
`d = 256` was still improving at 160k. The greedy search is nearly blind in high
d — 64 random directions achieve best |cos| ≈ 0.16 against an arbitrary
direction at `d = 256` (0.32 at `d = 64`).

**Two budgets, not one.** The global and conditional budgets need separate
stopping decisions: `I` keeps improving long after the global objective
saturates, so run conditional steps until `I` flattens. Do not stop the global
budget on the transport objective's own noise floor — it saturates while true G
is still improving (see the stopping note in §2). Conditional steps are also the cheaper way to buy `I` — doubling
`c` halved `I` at fixed global steps — and they *contribute* to G as well
(each one Gaussianises a k-subset; removing them made G worse at equal steps).

**Knobs that don't pay.**
- `n_dirs`: a 64× stronger direction search left G unchanged and added 57%
  displacement. Keep the search weak.
- Removing the whitening rotation (to preserve latent grid topology): 8.7×
  displacement; the transport spends its entire budget undoing correlations
  that whitening removes analytically.
- `k` beyond the median condition-group size: the continuous k-NN silently
  degenerates into a plain categorical step (U-shaped ratio in k).
- Matching to pre-drawn Gaussian samples (exact OT): degenerate in high d —
  the *nearest* available draw sits at ~⟨x−y⟩ for independent Gaussians.
  Continuous movement is the essential feature of the method, not a shortcut.

---

## 6. Practical notes

- **Particle order is identity.** The pairing (x_i, z_i) must survive every
  stage. Loaders that permute (CelebA's seeded particle subset) differ from
  natural dataset order; misaligning them silently trains the generator on
  shuffled pairs and looks exactly like mode collapse. When wiring a new
  dataset, verify alignment before anything else.
- **Keep LPIPS in the AE loss.** MSE-only training gives misleading checkpoint
  selection; LPIPS is also load-bearing during sparse-supervision experiments
  (removing it collapsed a top-k-MSE fine-tune outright).
- **Feed an unconditional generator an unconditional assignment.** Conditional
  transport steps reshape z to be Gaussian *within each condition group*; a
  generator that never sees the condition cannot exploit that and simply pays
  for the extra scrambling. Training CIFAR's unconditional generator on the
  class-conditional assignment cost ~7 FID at a matched step budget — the
  single largest error found in that pipeline. (That 7 is an upper bound on the
  bug alone: the comparison also carries an AE-ceiling improvement of 33.4 →
  31.0, so ~2.4 of it is representation, not wiring.)
- **The generator's LPIPS weight matters far more than its size.** On CIFAR,
  raising `--lpips-weight` from 0.5 to 8 was worth ~4.6 FID, while a 4×
  parameter increase (2.1M → 7.9M) bought under 2. Tune the loss before the
  architecture. An L1 pixel term instead of L2, and an added focal-frequency
  term, were both washes.
- **Standard figures**: `plots/plot_assignment.py <assignment.pt> --out f.png`
  (six-panel view used throughout), `plots/plot_ae.py`, `plots/plot_generations.py`.

## 7. Open questions

- Why does a locally-unpredictable coupling still decode well? The best video
  model sits at `R_impact ≈ 1.6` — nominally past the point where z explains
  less than the local mean — yet generates coherently. Plausibly the decoder is
  not restricted to locally-affine behaviour, so a locally rough but globally
  consistent map stays learnable; untested.
- Rank transport over-regularises: the transported cloud is *more* uniform than
  a genuine Gaussian sample, while fresh z at inference has full sampling
  fluctuation. Whether injecting that fluctuation into the assignment helps has
  not been tried.
- The reconstruction/generation gap (a particle's own z reconstructs far better
  than fresh z generates) is the clearest signature that the coupling, not the
  marginal, is what limits quality — but no measured coupling property has yet
  predicted it.
