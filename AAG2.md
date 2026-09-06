# AAG2: Joint Assignment Generation

## Summary

AAG2 modifies only the offline assignment stage of Amortised Assignment Generation. Generator training and inference remain unchanged: every training example receives one persistent Gaussian coordinate, and a feed-forward generator learns the fixed mapping in one step.

AAG2 introduces two rules:

1. **Stop at the finite-sample Gaussian noise floor.** Do not continue optimizing until the empirical particle cloud looks “more Gaussian” than a matched finite Gaussian sample.
2. **Transport a full block of `K=d` nonorthogonal projections jointly.** Compute all rank corrections from the same pre-update cloud, solve for one coupled `d`-dimensional particle movement, and apply it once.

The working hypothesis is that joint blocks extract coherent multidimensional assignment from the representation before the floor, while the floor rule prevents fitting finite-sample noise afterward.

## What changed from AAG1

The original assignment repeatedly:

1. samples candidate directions;
2. chooses the worst-looking projection;
3. exactly rank-transports that one projection;
4. repeats.

This is greedy coordinate descent over changing, generally nonorthogonal directions. Every update is individually simple, but later directions can react to or undo geometry created by earlier directions. Once the Gaussian floor has been reached, choosing the worst of many candidates preferentially selects finite-sample fluctuations.

AAG2 instead samples one full block of directions and solves their requested movements simultaneously.

## AAG2 assignment

Let the encoder representations be

\[
H=\{h_i\}_{i=1}^N,\qquad h_i\in\mathbb R^d.
\]

### 1. Whiten

Initialize the persistent particles with PCA whitening:

\[
Z\leftarrow \operatorname{PCAWhiten}(H).
\]

The identity of every pair \((x_i,z_i)\) remains fixed throughout assignment.

### 2. Draw a square nonorthogonal direction block

Draw `K=d` random directions and normalize each row:

\[
A=
\begin{bmatrix}
a_1^\top\\
\vdots\\
a_d^\top
\end{bmatrix}
\in\mathbb R^{d\times d},
\qquad \|a_j\|_2=1.
\]

Do not orthogonalize the directions. Overlapping projections carry information about cross-coordinate dependence that an orthogonal coordinate basis does not expose within a single block.

### 3. Compute all rank targets before moving particles

Project the current particles along every direction:

\[
P=ZA^\top\in\mathbb R^{N\times d}.
\]

For each column \(j\), sort \(P_{:j}\) and match its ranks to the corresponding standard-Gaussian order statistics:

\[
Q_{ij}=\Phi^{-1}\!\left(\frac{\operatorname{rank}(P_{ij})+1/2}{N}\right).
\]

The requested projection corrections are

\[
D=Q-P.
\]

All columns of \(D\) are computed from the same pre-update particle cloud.

### 4. Solve one joint movement

Find the particle movement \(\Delta Z\) that best realizes all requested projection corrections:

\[
\Delta Z
=
\arg\min_U
\left\|UA^\top-D\right\|_F^2
+\lambda\|U\|_F^2.
\]

The ridge solution is

\[
\boxed{
\Delta Z
=DA(A^\top A+\lambda I)^{-1}
}
\]

and the block update is

\[
Z\leftarrow Z+\Delta Z.
\]

The synthetic experiments used `λ=0.02`. Because a random square direction matrix can be poorly conditioned, the ridge is part of the method rather than an optional numerical detail.

### 5. Repeat with a fresh block

Draw a new `d×d` direction matrix, recompute ranks, solve the new joint movement, and update the particles.

### 6. Stop at the matched finite-Gaussian floor

Use a frozen set of evaluation directions that is separate from the directions used to construct updates. Measure the projection defect

\[
G(Z)=
\mathbb E_a\left[
W_2^2\!\left(a^\top Z,\mathcal N(0,1)\right)
\right].
\]

Estimate the finite-sample floor by applying exactly the same diagnostic to several iid Gaussian clouds with the same \(N\) and \(d\):

\[
G_{\mathrm{floor}}(N,d)
=
\mathbb E_{Z_g\sim\mathcal N(0,I_d)^N}[G(Z_g)].
\]

Stop at the first block for which

\[
\boxed{
G(Z)/G_{\mathrm{floor}}(N,d)\leq 1
}.
\]

The threshold should not be tuned by looking for the minimum possible empirical defect. Below the floor, the remaining rank errors are not distinguishable from finite-sample Gaussian fluctuations.

### 7. Train the normal AAG generator

After assignment, train the same one-pass supervised generator:

\[
G_\theta(z_i\oplus c_i)\rightarrow x_i.
\]

At inference:

\[
z\sim\mathcal N(0,I_d),\qquad x=G_\theta(z\oplus c).
\]

No change to the generator architecture or inference procedure is required.

## Experimental extension: police off-anchor gaps

The core AAG2 assignment still gives one target only at each persistent anchor:

\[
G_\theta(z_i\oplus c_i)\rightarrow x_i.
\]

In sparse latent regimes, this can leave most fresh Gaussian inputs far from any supervised anchor. An optional extension draws a separate batch `z_f ~ N(0,I)` during generator training and applies a distribution-only adversarial loss to `G(z_f)`, without assigning any individual `z_f` to a training example.

Keep this loss subordinate by controlling its generator-gradient norm rather than using a fixed loss coefficient:

\[
L_G=L_{\mathrm{pair}}+\lambda_tL_{\mathrm{adv}},
\qquad
\lambda_t=\rho
\frac{\|\nabla_\theta L_{\mathrm{pair}}\|_2}
{\|\nabla_\theta L_{\mathrm{adv}}\|_2+\epsilon}.
\]

The current experimental range is `ρ ∈ {0.05, 0.10, 0.20}`, starting at `ρ=0.10`. Recompute the coefficient every update and do not impose a positive minimum: if the paired gradient vanishes, the adversarial gradient must not become the primary signal by accident.

Across two synthetic targets and six `N,d` settings, the best tested ratio in this range beat paired-only AAG2 in all three matched seeds per setting. Improvements ranged from 7% to 92% in held-out sliced-\(W_2^2\). On the sparse 64-D tests, the eight-mode target improved 9% and the connected banana improved 28% at `ρ=0.20`. GAN-only training was highly unstable on both sparse 64-D targets. A 50% gradient target catastrophically failed on one banana seed.

This supports the mechanism—pairing specifies correspondence while a weak distributional loss constrains the extension between anchors—but not a universal sample-count rule. The auxiliary loss also helped in 8-D and sometimes with 512 or 2,048 examples. Its value depends on off-anchor coverage, target geometry, assignment complexity and model capacity, not dimension or dataset size alone.

For now this is an **experimental option, not part of the default AAG2 recipe**. It changes generator training, adds discriminator compute, and has only synthetic evidence. Default inclusion requires fixed-budget FID/KID tests on real encoder particles and comparison with non-adversarial alternatives such as MMD or sliced-Wasserstein regularization. Inference remains a single generator pass.

## Reference pseudocode

```python
z = pca_whiten(encoder(x))
floor = matched_gaussian_floor(N=len(z), d=z.shape[1], eval_dirs=frozen_dirs)

while True:
    d = z.shape[1]

    # K=d unit directions; deliberately nonorthogonal
    A = randn(d, d)
    A = A / row_norm(A)

    # Every target is calculated from the same pre-update cloud
    P = z @ A.T
    Q = gaussian_quantiles_at_column_ranks(P)
    D = Q - P

    # One regularized joint movement
    delta = D @ A @ inverse(A.T @ A + ridge * eye(d))
    z = z + delta

    # Frozen held-out projections, not the update directions
    if gaussian_defect(z, frozen_dirs) / floor <= 1.0:
        break

train_generator(z, x)  # unchanged from AAG1
```

For numerical stability, an implementation should use `solve` rather than explicitly form the inverse.

## Why stop at the noise floor?

For a population Gaussian, every projected rank correction is zero. A finite Gaussian particle set still has nonzero order-statistic errors in every projection. Searching many candidate directions and selecting the largest error creates an extreme-value or winner's-curse effect: even a correct finite Gaussian cloud always has a “worst” direction.

Continuing transport therefore fits the finite particle realization rather than the target distribution. It keeps changing the persistent correspondence without adding population-level Gaussianity.

In the 2-D banana experiment, generator architecture and 1,200 training steps were held fixed:

| assignment checkpoint | Gaussian defect / floor | generated sliced-W2 ↓ |
|---:|---:|---:|
| 10 steps | 0.707 | **0.095** |
| 5,000 steps | 0.064 | 0.152 |

Driving the empirical defect far below its matched floor made the final generator approximately 61% worse relative to the early checkpoint's score, or equivalently early stopping reduced the later score by about 38%.

## Why use `K=d`?

The joint movement for one particle has \(d\) unknown components.

- **`K<d`: underdetermined.** One block constrains only part of the possible movement.
- **`K=d`: square and potentially full-rank.** One block supplies enough overlapping constraints to determine a coupled movement in every latent direction.
- **`K>d`: overdetermined.** Independently rank-corrected projections are generally not jointly realizable by one \(d\)-vector movement. Least squares averages incompatible requests, suppressing the transport signal.

The `K` sweep at fixed `d=8` showed a sharp optimum:

| joint directions `K` | 2 | 4 | **8** | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|---:|
| generated sliced-W2 ↓ | 0.311 | 0.282 | **0.240** | 0.340 | 0.380 | 0.380 | 0.409 |

The dimension sweep then tested `K/d ∈ {¼, ½, 1, 2, 4}` over 60 matched generator runs:

| latent `d` | `K=d/4` | `K=d/2` | **`K=d`** | `K=2d` | `K=4d` |
|---:|---:|---:|---:|---:|---:|
| 4 | **0.138** | 0.156 | 0.140 | 0.147 | 0.180 |
| 8 | 0.278 | 0.301 | **0.224** | 0.315 | 0.369 |
| 16 | 0.470 | 0.480 | **0.251** | 0.514 | 0.599 |
| 32 | 0.722 | 0.677 | **0.208** | 0.759 | 0.773 |

`K=d` was the clear winner for `d=8,16,32`. At `d=4`, `K=1` and `K=4` were statistically tied within three-seed variance.

## What the ablations ruled out

The improvement is not explained by simply averaging many directions:

- averaging directions reduced finite-sample movement but also canceled useful transport signal;
- applying eight nonorthogonal directions sequentially scored 0.294, versus 0.240 when their requests were solved jointly;
- a full orthogonal block scored 0.318, so orthogonality itself was not the source of the gain;
- random single directions slightly outperformed greedy worst-direction selection, suggesting that extreme-direction search introduces finite-sample bias.

The winning joint method also had more total movement and worse generator training MSE than greedy assignment. Neither displacement, neighborhood preservation, nor supervised training loss should therefore be used as the final assignment-selection metric.

## Selection rule

The only final selection criterion is downstream generator quality at a fixed budget:

- same generator parameter count;
- same initialization protocol;
- same optimizer and batch construction;
- same number of generator steps;
- same evaluation sample count and score implementation;
- multiple seeds.

Assignment diagnostics have restricted roles:

- the matched Gaussian defect determines when to stop;
- conditioning and numerical diagnostics catch invalid assignments;
- they do not override a worse fixed-budget generator score.

## Conditional AAG2

The experiments above test unconditional assignment. A direct conditional extension would apply the same joint-block primitive inside exact condition groups or condition-local neighborhoods, while retaining global AAG2 blocks:

1. global `K=d` joint block;
2. one or more condition-local `K=d` joint blocks;
3. global Gaussian-floor test;
4. conditional independence test against its matched random-subset floor;
5. stop when both have reached their respective floors.

This extension has not yet been validated. In particular, small conditional groups may not contain enough particles to estimate a stable square system or a meaningful Gaussian floor.

## Known limitations and open tests

1. **The dimension sweep changed ambient representation dimension, not target intrinsic dimension.** The target remained a 2-D distribution embedded through nonlinear features. These results establish an assignment block-size signal; they do not show that AAG2 defeats genuinely high-dimensional sampling sparsity.
2. **Square random direction matrices can be ill-conditioned.** `K=d` produced the largest transport paths. A ridge sweep is needed to determine how much of the gain comes from full-rank joint constraints versus near-singular amplification.
3. **The evidence is synthetic.** The next decisive test is AAG1 versus AAG2 on real encoder particles, using identical generator models and training budgets and evaluating FID/KID over several seeds.
4. **The floor estimate must match the diagnostic exactly.** It must use the same `N`, `d`, number of evaluation directions, projection statistic, and candidate-selection protocol where applicable.
5. **AAG2 does not remove sparse-prior coverage.** It improves how the finite assignment is constructed. If fresh Gaussian samples remain far from all particles relative to the target's variation scale, additional data, lower effective dimension, stronger conditioning, or off-particle teacher supervision is still required.
6. **The gap-policing adversary is not yet a default component.** It improved every tested synthetic scenario at a selected 5–20% gradient ratio, but the ratio was selected on the same seeds used for evaluation and adversarial training adds instability and compute. Real-data confirmation is still required.

## Current recommended configuration

```text
initialization:       PCA-whitened encoder representations
directions/block:     K = latent dimension d
direction geometry:   random, unit-normalized, nonorthogonal
block solve:          ridge least squares
ridge:                0.02 in the current synthetic experiments
floor evaluation:     frozen held-out directions
stopping:             first defect/floor crossing at or below 1
generator selection:  fixed-budget downstream score over multiple seeds
optional experiment:   fresh-z adversary at 5–20% of paired gradient norm
```
