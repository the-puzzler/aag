# AAG — independence, marginals, and what "fully gaussianised" actually requires

The method core, consolidated for whoever picks this up. Several entries below
are **corrections of earlier conclusions that were wrong**; those are kept with
their history rather than tidied away, because the wrong version is intuitive
and will be re-derived by anyone who does not know it was tested.

---

## 1. Why conditional independence is a correctness condition, not a refinement

At generation you draw `z ~ N(0,I)` and pair it with a chosen condition `c`.
That is a valid sample of `p(x|c)` **only if `p(z|c) = p(z) = N(0,I)` for every
`c`** — i.e. z is independent of c.

A global assignment only guarantees the *marginal* `p(z)` is gaussian. If some
attribute occupies a lobe of z-space, the generator only ever saw
(z-from-that-lobe, c) pairs; sampling z from the whole sphere and pairing it
with c produces a combination absent from training. User's framing (2026-08-11):
*"the condition needs to be independent to the z — independence of information."*

Observed failure mode, CelebA: one fixed z rendered as red/white garbage under
every condition except `Wearing_Hat`, which gave a clean face. That z was
off-manifold for every other condition.

**The sharp test is rare/contradictory combinations** (Male+Heavy_Makeup,
Bald+Wearing_Lipstick), where `p(z|c)` deviates most from `p(z)` — not average
sample quality.

---

## 2. THE central lesson: every marginal must be independent, not just the joint

**Decorrelating against a high-dimensional condition can hide dependence on any
low-dimensional function of it.** This is the single most expensive thing this
project has learned, and it cost days twice — once on the context side, once on
the action side.

**Context side** (VPT 24-frame cosine assignment, n_eval=200, ±0.03):

    weighted, as assigned   0.954
    unweighted              0.897
    newest-5 unweighted     3.030
    newest-1                8.253

z was independent of the 24-frame trajectory and **8× dependent on where the
camera actually was**. Two particles can agree on the newest frame yet sit far
apart in the full 6144-d space, so they never share a k-NN neighbourhood and
transport never decorrelates z across them.

Downstream symptoms this explained, all of which had resisted other
explanations: resampling z moved the generated frame 254% of a real
consecutive-frame step against the action's 63%; fresh-z retrieval scored 53%
rank-1 where copying the previous frame scored 94%; fresh-z MSE sat at 5.0× the
AE floor across epochs 17/21/30/36/38 while train LPIPS improved 7%.

**Action side**, same night, the 81-way index (`action = move*9 + turn*3 +
tilt`):

    joint 81-way        mean 0.593
    move only (9)            0.822
    turn only (3)            1.043
    moving vs still (2)      1.003
    turning vs not (2)       1.272   <- worst

The joint class looks well decorrelated while z still carries *whether the
player is turning*, so at inference a fresh z argues with the turn command.

Note the two have **opposite shape**: on context the FINER probe was worse; on
action the COARSER marginal is. So there is no safe direction to probe in —
enumerate them all.

> **Rule: never quote a single independence number.** Report it at every
> sub-scale the downstream model could key on — context prefix lengths
> 1/3/5/12/24, and for actions the joint class plus every marginal. A single
> number against the full condition is the reading that hid this for days.

**Marginals are enumerated up front, never discovered later.** `--act-groups`
carries one grouping per binary control, dx/dy direction (3 each, cache
deadzones), and magnitude terciles — 16 groupings for the 12-d vector.
`--ctx-scales` interleaves transport across context sub-scales within every
step.

---

## 3. The causal rule — you only ever need single-frame transitions

The user's rule (2026-09-01), in their words: *"if every single frame transition
is uncorrelated, then every long range one must also be uncorrelated. Which
isn't necessarily true, except in the case in which you have causality."*

`z_t` is a parent of `X_t` and of nothing else, and history reaches `X_t` only
along `X_{t-2} -> X_{t-1} -> X_t`. The only path from `z_t` to `X_{t-2}` is

    z_t -> X_t <- X_{t-1} <- X_{t-2}

on which `X_t` is a **collider**, so the path is blocked and `z_t` is
d-separated from `X_{t-2}`. Independence at every long range follows from
independence at range one.

**Do not call this "the Markov property"** — that framing was mine and the user
corrected it. The mechanism is d-separation by a collider; saying Markov
obscures both the reason and the exception.

**What it buys:** transport cost stops scaling with context length. `cond_dim`
stays small forever. Measured: 48 minutes vs 2 hours per 48k steps at 512k
particles.

**Why 3 frames rather than 1**, stated correctly: not "longer is safer".
Velocity is not observable in a single image, so the causal parent of `X_t` is
the last two or three frames. Picking 3 is a claim about what constitutes the
parent — a far more precise justification than context length.

**Empirical support** (VPT 512k, ctx3 lineage transported against scales 1 and 3
only, then measured at scales it was never transported against):

    newest-1   0.924   <- transported against
    newest-3   0.834   <- transported against
    newest-5   0.793
    newest-12  0.862
    newest-24  0.907

**Short implies long; long does not imply short.** The 24-frame assignment read
0.897 on its own metric and 8.253 against newest-1.

**The exception, and it is real:** a SKIP EDGE — information reaching `X_t` from
`X_{t-2}` without passing through `X_{t-1}`. Occlusion, off-screen state,
inventory, anything the dynamics depend on that the current frame does not
render. There the collider argument does not apply. Re-check the rule on any
task with real memory.

**Not yet run:** a `--ctx-frames 1` assignment, which the rule predicts is the
minimal sufficient configuration. ~50 min at 512k particles.

---

## 4. Measurement traps — three instruments that lied

### 4a. The leakage probe is uninterpretable without a noise null

Never read `cond+z` minus `cond` as a leak. Reference it to `cond + N(0,I)` of
identical shape — the correct null, because a valid assignment's z *is*
marginally N(0,I), so the only difference is the information z carries.

    marginal   chance    cond   cond+z   z adds   noise   z-noise
    dxsign     68.55%  68.51%   68.04%    -0.46   -4.69    +4.22
    anyclick   69.21%  84.51%   84.01%    -0.50   -3.58    +3.08
    moving     59.46%  75.26%   73.81%    -1.45   -3.55    +2.11
    attack     74.80%  89.68%   89.60%    -0.09   -1.73    +1.64

Appending 256 uninformative dims to a 768-dim probe input costs accuracy, and
**the cost is not a constant offset** — it ranges from −4.69 (dxsign, where cond
carries no signal so the probe is fragile) to −0.03 (A/S/D, already at chance).
The bias is largest exactly where sensitivity is highest. Uncontrolled, the
table reported "worst leak +0.06, nothing anywhere" over a real **+4.22 turn
leak**. Reading it uncontrolled reverses the conclusion.

`scripts/diag_action_leakage.py --control`.

### 4b. Per-marginal W2 is structurally blind to turning

**A turn has no single-frame signature.** An MLP probe extracts +16.1 points of
"moving vs still" from `h` (the AE latent of the true target frame) but **+0.1**
of "turning vs not" from the same `h`. A turn is only visible as a DIFFERENCE
between consecutive frames, so no statistic computed on one latent can see it.
Same for clicks.

This nearly hid a real fix: W2 for turning-vs-not moved only 1.272 → 1.121 and I
reported the action side as unfixed, while the conditional probe showed the leak
had actually closed:

    marginal          z adds before   z adds after
    joint 81-way          +2.34           +1.02
    move only             +1.77           +0.30
    turn only             +2.41           +0.36
    turning vs not        +2.34           -2.12
    moving vs still       +2.81           -0.38

> **Use the conditional probe for independence; use per-marginal W2 only for
> continuity with old runs.** The conditional form is also the honest statement
> of what conditional independence *means*.

### 4c. Sensitivity structure — a data fact, not an artefact

`cond` beats chance by ~+15 on attack/anyclick/moving but by ~0.0 on
dxsign/dxmag/dysign. Attack is 95.6% lag-1 autocorrelated so recent context
nearly determines it; a turn is not predictable from past frames at all.

**Consequence: context transport incidentally decorrelates z from anything the
context determines, while mouse marginals require explicit action-side
transport.** Do not expect the two sides to need equal effort.

---

## 5. What gaussianisation cannot reach — the chain that ends in blur

This is the link between the assignment and the autoencoder, and it is the
reason the AE keeps coming up.

    detail the ENCODER discards
      -> has no representation in h_target = AE_enc(frame)
      -> so it is absent from z = transport(h_target)
      -> so the assignment never had a handle on it
      -> so it was never gaussianised
      -> so at generation it is unspecified
      -> and MSE + LPIPS resolve unspecified detail to its conditional mean
      -> which is blur, concentrated exactly on fine texture.

**z is the only channel carrying target-specific information, and z can only
carry what the encoder kept.** So the AE encoder is a hard ceiling on what the
assignment can gaussianise, and therefore on what the generator can ever render.

Quantified (2026-09-05, `scripts/diag_latent_probe.py`, ridge z → pixels,
held-out R²): **R² z→hf(x) ≈ 0.05.** At 256 dims the latent carries almost no
linearly-decodable fine texture. Effective rank of the latent covariance is
**~42 of 256** dimensions actually used.

Note the corollary for judging any AE change: **a reconstruction-mediated metric
cannot attribute**, because reconstruction is encoder × decoder. Use the
decoder-free probe. See `docs/HANDOFF_ae_adversary.md`.

---

## 6. Stopping criteria — the corrected story

This one went through three revisions. The end state:

**Do NOT stop on the independence ratio.** It was claimed that I must be driven
"decently below 1.0, target ~0.8". Auditing the raw logs killed that: the cited
supporting values were single noisy evals, not levels. The doom assignment that
actually won logged `joint=0.86` and `joint=1.12` **on the same step**. On a
±0.13 instrument, the working and failing regimes are not separable by this
number.

**But lower is still the right direction.** User, 2026-09-01: *"the working band
is kinda nonsense, lower always seems to be better."* These are not in conflict:
the audit is about *resolution between neighbours* (a noisy instrument cannot
rank adjacent runs), the user's claim is about *direction* (pushing I lower has
consistently produced better generators). Operationally both say: keep
transporting, do not stop at a threshold.

**The only selection rule that has ever worked here: fresh-z MSE of a generator
trained on the assignment.** Doom selected `assign_wm_ctx (fresh_mse=0.010313)`
over `assign_wm_c32 (0.012583)` this way.

**Displacement is a progress meter, not a damage meter.** `||z - z_whitened||`
on VPT 512k:

    step      1     disp 0.37    2% of ||z||
    step   1000     disp 8.55   53%
    step   2000     disp 9.61   60%
    step 352000     disp 15.46  97.2%   (typical ||z|| = 15.90)

53% is reached in the first 1000 steps — it is mostly the whitened-to-gaussian
rearrangement, not gradual drift.

**Do NOT transfer the CelebA displacement result.** User, 2026-09-01: *"dont
draw on informationm from those tests… more has alwys been better. its bigger
data and more complex and the celeba stuff is full of confoudsnds, like no
maringals being taken care of etc"*. The confound is decisive and specific: the
CelebA 4k-vs-60k comparison had **no per-marginal transport**, so its extra
steps bought displacement without buying marginal independence. On VPT more
transport has consistently been better.

**Watch the metric changing meaning across a resume.** The same z re-evaluated
against the new 12-d action vector read I_act 1.054 against 0.584 on the 9-d
vector. Not a regression — z had never been decorrelated against attack/use/E,
so the wider condition exposes dependence the narrower one could not see. Label
such discontinuities in any curve spanning a representation change.

---

## 7. The representation, and why it is shaped this way

12 dimensions, not 90:

    0..3  W A S D      4 space   5 shift   6 ctrl   7 E
    8     attack (LMB) 9 use (RMB)
    10,11 dx, dy

**Why the 81-way index had to go.** User: *"i really didnt liek the 81way thing.
we should just have continuous and discrete as descirbed reducing dimensionality
massivley."* Three measured costs: no code point for a mouse button (so a model
conditioned on it can never respond to a click, and E is absent too); it folds a
5px nudge and a 500px flick into one class past a deadzone; and as an integer it
puts opposites at adjacent code points, which is what made a turn command read
as a tilt. It is also a deterministic function of a subset of the 12-d vector,
so it carries zero extra information — feeding both invites the network to key
on the coarse discrete signal and underuse magnitude.

**Two scalings, and conflating them is the trap.**
- As a **k-NN distance**: binaries stay bounded 0/1 and are **NOT standardised**.
  E at 0.455% standardises to a ±15.6 swing that would decide every
  neighbourhood it appears in.
- Mouse: signed `log1p`, standardised, then scaled so both mouse columns
  together hold **8×** the binaries' total variance (8 = measured pixel-variance
  ratio, mouse 16.4% vs 2.1% for all keys).
- As **generator input** the scaling is an irrelevant linear map.

**Press rates** (543k segments / 240k frames): W 0.335, **attack 0.246**, ctrl
0.110, space 0.090, shift 0.075, D 0.072, A 0.059, use 0.055, S 0.036, E 0.0046.
~30% of frames have a mouse button down, and that entire third of "what explains
screen changes" was missing from the condition before this. attack and use
co-occur on 0.1% of frames. dwheel is nonzero on 0.58% — not worth conditioning
on.

**`act_norm` is an inference contract, not bookkeeping.** It travels with the
particles and into the generator checkpoint, because a live controller must be
encoded with the same constants the generator trained on. `encode_live()` is
that path, and it reproduces stored vectors bit-identically.

**ACTION_LAG = 1**: the action producing frame t is recorded at tick t−1 (VPT
stores (observation, action) in the RL convention). But **the lag is not
uniform** — mouse/space/use peak at t−1, all four movement keys peak at t−2
(acceleration: mouse-look rotates on the tick applied, WASD goes through
velocity). The key peaks are broad and flat, so do not chase a per-control
scalar lag; the flatness is the finding. **A single action tick cannot express
hold duration, and under acceleration hold duration is speed** — the likely
cause of unmodulated motion. Proposed and not yet run: condition on a short
action history (t−1, t−2, t−3), 36 dims instead of 12.

---

## 8. Practical checklist

1. Transport against the **causal parent** (~3 frames), never a long window.
2. **Enumerate the marginals up front** — one grouping per binary control, sign
   and magnitude groupings for continuous ones. Interleave transport across them
   and across context sub-scales within every step.
3. Judge independence with the **conditional probe against an N(0,I) null**,
   never a bare `cond+z − cond`, and never per-marginal W2 for anything that is
   a between-frame difference (turns, clicks).
4. **Report every sub-scale**, never one number.
5. Do not stop on the ratio. Keep transporting; select on **fresh-z MSE of a
   trained generator**.
6. Log displacement as progress, not damage.
7. Before attributing generator error to conditioning, **decompose against the
   AE floor** — most of VPT's absolute deficit was the autoencoder.
8. Never let a metric adjudicate anything visual. Frame-difference statistics
   have misled this project three times; send frames.

---

## 9. Provenance

Consolidated 2026-09-05 from the project memory files, which carry the dated
per-finding detail:

    newgen-conditional-independence-rationale
    newgen-high-dim-conditioning-hides-correlations
    newgen-single-transition-decorrelation-rule
    newgen-leakage-probe-needs-null-control
    newgen-assignment-stopping-criterion
    newgen-vpt-action-representation
    newgen-action-lag-and-history
    newgen-ae-is-the-binding-constraint
