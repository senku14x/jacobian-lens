# 007 — Which operator transports an intermediate representation to the output?

**Date** 2026-08-12 · **Model** Qwen/Qwen3.6-27B (64 layers, d_model 5120) · **Status** Phase A + B
complete; Phase C routed and closed. **Revised 2026-08-12 after external review — see Corrections.** Numbers reproducible from `research/experiments/` at the
commit recorded in each `research/outputs/*/qwen3.6-27b.json`.

---

## Corrections (2026-08-12, after external review)

Three claims in the first version of this report were wrong or overstated. They are corrected in
place below; this block exists so the corrections are not discoverable only by diffing.

**C1 — the smoothing-mechanism conclusion is RETRACTED.** The first version claimed the SmoothGrad
advantage is "denoising, not path-averaging," on the strength of orthogonal-only smoothing
recovering 99–109% of the isotropic gain in 9 of 9 conditions. That comparison does not
discriminate the hypotheses. For isotropic `u` in d=5120, `E[(u·δ̂)²] = 1/d ≈ 1.95e-4`, so only
0.02% of its squared norm is parallel to δ; projecting that out leaves a direction with
**cos(u_iso, u_orth) = 0.99990** (measured, 20k samples). The two arms are the same experiment to
0.01%, and the result was guaranteed by high-dimensional geometry regardless of mechanism. The
parallel arm was independently mis-specified: it evaluated `J(h ± σδ̂)` at σ=0.2·median‖h‖ while the
path at ε=0.05 extends only to 0.05·median‖h‖, i.e. it sampled points 4× beyond the path, so its
poor score does not refute a path explanation. **Supported statement: spherical averaging around
the operating point substantially improves finite-effect prediction. The mechanism is open.**
Note also that `T_IG`, the genuine path operator, is the best method in the table — which if
anything favours path effects mattering.

**C2 — the split leaked, and the in-family secant result was largely template memorisation.**
Calibrate/held-out was split by base prompt. A base is keyed by its own token ids, so the pair
(a→b) and the pair (b→a) live on different bases and can fall on opposite sides — with deltas that
are *exact negatives* of each other. Two arguments in one template leak the same way. Re-run under
nested splits (§ "Split leakage"), `T_sec` falls from 0.43–0.70 to **0.01–0.15** once templates
cannot cross the split, while `J̄` and `R̄` — never fitted on this data — stay flat. G-SECANT's
verdict is unchanged but its *reason* is now much simpler and stronger than the learning-curve
argument the first version used.

**C3 — `J_loc` is not "the exact local Jacobian."** It is a central secant at
`ε_ref = 0.01·median‖h‖` through a bf16 upper stack. The wording is corrected throughout. The
SmoothGrad advantage could therefore be partly numerical — finite-difference truncation, bf16
quantisation, or noise acting as dither — and that is not yet excluded. An `ε_ref` convergence
sweep and an fp32 replication are queued.

Smaller items also corrected: cosine is supplemented with relative residual error, which reveals
that `J̄`'s transport scores relerr ≈ 0.96–1.00, i.e. **no better in magnitude than predicting zero
effect**; the primary target is switched to the native one-sided patch, with the normalised
antithetic effect demoted to a diagnostic; and "context-averaging is the dominant failure" is
weakened to what is actually controlled (see §"What this does not establish").

Credit: this review came from outside the project and materially changed three conclusions.

---

## Question

The J-lens reads an intermediate representation by transporting it to the final layer with a single
averaged matrix, `J̄_ℓ = E[∂h_final/∂h_ℓ]`, then unembedding. The R-lens replaces the backward pass
with LRP-style stop-gradient rules. Both are *one linear operator standing in for a
context-dependent map*. This study treats them, plus four alternatives, as **competing hypotheses
about the same object**, and ranks them against a directly measured causal quantity rather than
against readability:

> for a perturbation δ applied to `h_ℓ` at one position of one prompt, which operator best predicts
> the actual downstream change `Δ = [F(h+δ) − F(h−δ)]/2` in the final residual stream?

Operators compared: `I` (logit lens), `J̄`, `R̄`, `J̄+λI`, the input-specific local transport
`J_loc(x)` (a bf16 central secant at ε_ref=0.01·median‖h‖, not an exact Jacobian), a fitted secant `T_sec = C_Δδ(C_δδ+ηI)⁻¹`, its R-anchored variant
`T_RC = (C_Δδ+λR)(C_δδ+λI)⁻¹`, SmoothGrad-J, and the path-integrated `T_IG`.

**North Star.** A vocabulary-readable instrument for causally active intermediates. Cosine against
the true finite effect is the **proxy**; the validity argument is that a lens whose transport does
not predict what a perturbation actually does cannot be read causally. The proxy's known divergence
risk — a good cosine that never changes behaviour — is why the judge-free behavioural ablation (M2)
is carried as a separate, span-immune gate rather than folded into the same number.

---

## Method in one page

**Measurement is autograd-free.** `forward_from(h, ℓ)` replays blocks ℓ+1…62 on a supplied hidden
state using kwargs captured from a clean forward. Those kwargs (`position_embeddings`,
`position_ids`, `attention_mask=None`) are hidden-state-independent, so the replay is exact —
verified at `0.000e+00` max error against a full forward, re-checked at the start of every
model-side run. Every Jacobian-vector product is a central difference
`[F(h+εδ)−F(h−εδ)]/2ε`, which *is* the JVP up to O(ε²). This keeps `torch.func` and
double-backward — both untested on this GatedDeltaNet hybrid (48 of 64 blocks are
`linear_attention`) — off the critical path entirely.

**Perturbation bank.** Token-aligned minimal pairs (one template, two arguments differing in one
contiguous block) embedded in a 96-token `pile-10k` prefix, so every perturbed position sits ~104
tokens deep and stays on-distribution for operators fitted with `skip_first=4` on 128-token pile
text. Four direction families: **D4** natural activation deltas (primary), **D1** isotropic, **D2**
lens-token directions `v_t = J̄ᵀ(γ⊙u_t)`, D3 not built. Each direction measured antithetically at
ε ∈ {0.01, 0.05, 0.2, 1.0}×median‖h‖ plus native magnitude, with summed-over-current-and-future
(primary — matches the J̄/R̄ estimator) and self reductions.

| | original bank | **broadened bank (B1)** |
|---|---|---|
| base prompts | 63 | **240** |
| categories × templates | 4 × 16 | **10 × 40** |
| minimal pairs | 186 | **1,200** |
| sites (prompt × position) | 169 | **684** |
| deltas **per layer** | 7,222 | **30,096** |
| D4 / D1 / D2 | 2,490 / 4,056 / 676 | 17,100 / 10,944 / 2,052 |
| calibrate / held-out base prompts | 31 / 32 | **120 / 120** |
| D4 effective rank @ε=0.2 (λ>1e-6·λmax) | 117 | **555** |
| D4 participation ratio | 57.6 | 169.2 |
| D4+D1 fit-set effective rank | 615 | **1,911** |

Layers 16 / 31 / 46 = 26% / 50% / 74% of the target row (62). Across 3 layers the broadened bank is
**90,288 measured deltas**.

**Statistics.** Paired bootstrap, 2,000 resamples, resampling **base prompts, not deltas** — deltas
within a prompt are not independent, and resampling them would inflate n by ~250×. "CI-clear" means
the paired difference CI excludes zero. Splits are by base prompt with a fixed seed, so no template
instance appears on both sides.

---

## The identifiability confound, and why the whole study was re-run

`T_sec` is only determined on `span(C_δδ)`; ridge sends it toward zero elsewhere. With the original
bank the natural-delta family spanned **117 of 5,120 dimensions**, so a held-out direction from a
different family sat mostly *outside* the identified subspace and was mispredicted **by
construction** — a fact about the fitting set, not about transport.

This was not a hypothetical. On the original bank the study concluded the secant was dead. On the
broadened bank, with effective rank 4.7× higher:

| held-out D4, ε=0.2 | original | broadened |
|---|---|---|
| `T_sec − J̄` @ L16 | +0.075 | **+0.294** |
| `T_sec − J̄` @ L31 | +0.050 | **+0.180** |
| `T_sec − J̄` @ L46 | +0.050 | **+0.142** |

with CIs that went from overlapping J̄ to clear at every layer. **One conclusion of this study is
methodological: any transport-transfer result reported without an effective-rank and
energy-captured diagnostic is uninterpretable.** Two of the three gates would have been called
wrong on the narrow bank.

The corollary is that the gates had to be placed on quantities the confound cannot touch. Three
were: the Stein/SmoothGrad control (compares two operators on the *same* directions), in-span
transfer (evaluates only the component inside `span(C_δδ)`, with true effects freshly measured),
and behavioural ablation (never asks any operator to predict a held-out δ).

![Operator ladder](figures/fig1_operator_ladder.png)

---

## Result 1 — the secant is not a transport operator (G-SECANT: FAIL)

It improves in-family with more data and does not improve on any span-immune axis.

**Stein / SmoothGrad control**, held-out D4 @ε=0.2, broadened bank:

| | L16 | L31 |
|---|---|---|
| SmoothGrad-J | **0.714** | **0.862** |
| J_loc | 0.485 | 0.590 |
| T_RC | 0.418 | 0.418 |
| T_sec | 0.408 | 0.393 |
| R̄ | 0.124 | 0.192 |
| J̄ | 0.111 | 0.188 |
| `T_sec − SmoothGrad-J` | **−0.306** [−0.367,−0.243] | **−0.469** [−0.520,−0.416] |
| `cos(T_sec·δ, SG-J·δ)` | 0.254 | 0.320 |

**In-span transfer** (held-out D1/D2 projected onto the top-r eigenvectors of `C_δδ`,
renormalised, true effects re-measured — the only transfer number the confound admits), L16, r=555:

| family | J̄ | R̄ | T_sec | `T_sec − J̄` |
|---|---|---|---|---|
| D1 | 0.161 | 0.133 | 0.042 | **−0.119** [−0.132,−0.107] |
| D2 | 0.311 | 0.250 | 0.112 | **−0.199** [−0.216,−0.184] |

Inside the subspace where it is fully identified, the fitted secant is **worse than the released
J-lens**, CI-clear, at every rank and both families — and the margin *widened* when the bank grew.

**Learning curves** settle what "worse" means. Fitting on 15 → 120 calibrate prompts:

![Learning curves](figures/fig2_learning_curves.png)

| L31 | 15 | 30 | 60 | 120 | gain over last doubling |
|---|---|---|---|---|---|
| in-family D4 cos | .117 | .231 | .313 | .384 | +22.9% |
| cross-family D2 cos | .021 | .047 | .060 | .092 | +53.0% |
| D2 energy in span | .316 | .422 | .528 | .654 | +24.0% |
| effective rank | 325 | 636 | 1115 | 1911 | +71.4% |

**Nothing has plateaued.** The honest statement is not "the secant cannot work" but: *at 240 base
prompts it reaches 0.092 cross-family where J̄ reaches 0.311, and it is closing at ~+0.03 per
doubling of data.* Linear extrapolation in log-data puts parity with J̄ around 6 more doublings —
~64× the corpus — and that extrapolation assumes a trend that must saturate. The negative is
**a bounded, quantified negative at this data scale**, not a proof of impossibility.

**Behavioural M2** (judge-free Δ log-prob of the correct answer under ablation of each operator's
probe direction, 40 probe-swap items, identical band {16,31,46} for every operator, matched-norm
random controls):

Run twice — once with the original-bank operators, once with the **broadened-bank refits**:

| operator | Δ logp (orig ops) | Δ logp (**broad ops**) | 95% CI (broad) | top-1 kept on pile |
|---|---|---|---|---|
| **J̄** | +0.856 | **+0.856** | [+0.437, +1.354] | .977 |
| R̄ | +0.587 | +0.587 | [+0.290, +0.975] | .980 |
| logit lens | +0.022 | +0.022 | [+0.007, +0.039] | .980 |
| T_RC | +0.006 | +0.016 | [+0.001, +0.035] | .984 |
| T_sec | −0.002 | −0.006 | [−0.011, −0.000] | .988 |
| T_sec^D4 | −0.029 | −0.009 | [−0.017, −0.002] | .973 |
| matched-norm random | −0.005 | −0.005 | [−0.008, −0.001] | — |

`T_sec − J̄` = **−0.861** [−1.358, −0.445] CI-clear on the refit operators — the broadening changed
this by 0.003. Top-1 retention .97–.99 throughout, so nobody wins by breaking the model.
**This is the axis with no identifiability escape**, and the secant sits at the random-direction
noise floor on it: the only operators that identify a causal direction at all are `J̄`, `R̄`, and
(barely, +0.020 over random) `T_RC`.

*Correction to the earlier read:* on the original-bank operators `T_sec^D4 − random` was −0.025
CI-clear, i.e. worse than a random direction. On the refits that gap closes to −0.005 and is no
longer CI-clear against random. The "worse than random" finding was specific to the rank-starved
D4-only fit and does not survive; the "indistinguishable from random" finding does.

> **Verdict — G-SECANT: FAIL.** Three independent span-immune checks, all against, all CI-clear,
> and the margins did not shrink when the confound was removed. The in-family gain that survives is
> consistent with `T_sec` matching the *distribution* of natural deltas rather than transporting
> arbitrary ones.

---

## Result 2 — the fixed lens loses 2–4× to input-specific transport (G-CONTEXT: PASS 2/3)

`J_loc(x)` — the same estimator conditioned on the actual prompt and position — against the
released `J̄`, held-out D4 @ε=0.2:

![Context ratio](figures/fig3_context_ratio.png)

| layer | J̄ | J_loc | ratio | 95% CI |
|---|---|---|---|---|
| L16 (26%) | .110 | .479 | **4.34** | [4.14, 4.53] |
| L31 (50%) | .188 | .597 | **3.17** | [3.05, 3.29] |
| L46 (74%) | .380 | .742 | **1.95** | [1.90, 2.00] |

Point estimates barely moved from the original bank while CIs tightened ~4×, so this is the
best-determined effect in the study. Averaging the Jacobian over contexts costs a factor of **2–4.3
in effect prediction**, and the cost is **monotone in depth**: largest early, decaying to ~2× by
three-quarters depth. L46 crossed from CI-clear *above* 2× on the narrow bank to CI-clear *below*
it on the broadened one — the gate reads 2 of 3, with the failing layer the deepest.

**This gap is not isolated to context-averaging.** `J_loc(x)` is compared against the *released*
`J̄`, fitted on 25 different pile passages, so the ratio bundles context averaging with
source-position mismatch, fitting-corpus mismatch, finite-sample estimation noise, and possible
convention differences. Isolating it needs a convention-matched
`J_own = E_{our calibration contexts}[J_loc(x)]`, which was not computed. The supported wording is
therefore **"the released fixed J-lens loses 2–4× to the input-specific estimator under this
evaluation"**, not "context averaging is responsible for the gap."

Interpretation, at *supported claim* level: the single averaged matrix is a much better
approximation late than early. That is the direction the workspace picture predicts if
representations become more context-independent as they approach the readout.

---

## Result 3 — the deviation is real but not low-rank-shared (G-CONDVIABLE: FAIL)

If `J_loc(x) − J̄` were low-rank and shared across inputs, a lens could carry a handful of
correction modes. Sketching `Y_i = J_loc(x_i)·S` on the top-192 eigenvectors of pooled `C_δδ` over
56 sites, 29 held out:

![Conditional model](figures/fig5_conditional.png)

| layer | top-1 | top-4 | top-16 | top-32 | rank-16 **oracle** recovery of the J̄→J_loc gap |
|---|---|---|---|---|---|
| L16 | .148 | .305 | .606 | .839 | **23.2%** |
| L31 | .061 | .192 | .525 | .799 | **40.0%** |
| L46 | .054 | .172 | .495 | .787 | **85.2%** |

The pre-registered criterion (top-16 > 50% variance) reads 2/3 — but 0.525 and 0.495 straddle the
threshold, so the proxy is doing no work, and the direct measurement it stood in for fails.
Coefficients here are **oracle** — fitted on the held-out site's own deviation, an upper bound no
deployable model can reach. Even so, recovery is **anti-correlated with need**: it works at L46
where the gap is 1.95×, and recovers under half at L16/L31 where the gap is 3–4.3×. Rank-32 adds
+2.4 and +0.4 points over rank-16. Predicted-coefficient variants were not run: a failing oracle
bounds them.

---

## Result 4 — smoothing helps a lot; why is open

SmoothGrad-J beats the local finite-difference transport `J_loc` by +0.17…+0.27, CI-clear at every
layer and every scale ε ≤ 0.2. That is a large, replicated, and (within this model, precision, and
target) credible observation. **The mechanism is not established** — see C1. Two candidates remain
live, and they differ in whether a lens could ever capture the gain:

- **H_path** — at finite ε the truth is a secant, not a tangent; averaging approximates the
  mean-value operator. δ-dependent, so no fixed matrix captures it.
- **H_denoise** — the exact operating point is atypically ill-conditioned and any small average
  regularises it. Largely δ-independent, so a fixed operator could capture it.

A third candidate was raised in review and is not excluded: **numerical**. `J_loc` is a bf16
central difference at ε_ref = 0.01·median‖h‖, so part of the gain could be finite-difference
truncation, quantisation, or smoothing noise acting as dither rather than model nonlinearity.

![Smoothing mechanism](figures/fig4_smoothing_mechanism.png)

The measured grid, 60 sites × 300 directions × 51 held-out prompts per layer, σ_decomp = 0.2:

| layer | ε | σ\* | J̄ | T_sec | `J_loc` σ=0 | `SG_par` ∥δ only | `SG_orth` ⊥δ only | `SG_iso` | `T_IG` |
|---|---|---|---|---|---|---|---|---|---|
| L16 | 0.05 | 0.05 | .092 | .259 | .649 | .217 | .823 | .818 | .899 |
| L16 | 0.2 | 0.2 | .114 | .406 | .490 | .445 | .705 | .704 | .882 |
| L16 | 1.0 | 0.5 | .170 | .370 | .207 | .345 | .319 | .310 | .592 |
| L31 | 0.05 | 0.05 | .171 | .325 | .600 | .557 | .855 | .856 | .883 |
| L31 | 0.2 | 0.2 | .194 | .380 | .592 | .654 | .864 | .868 | .899 |
| L31 | 1.0 | 0.5 | .295 | .463 | .372 | .527 | .546 | .546 | .780 |
| L46 | 0.05 | 0.05 | .379 | .484 | .753 | .741 | .935 | .936 | .949 |
| L46 | 0.2 | 0.2 | .394 | .524 | .744 | .804 | .933 | .934 | .957 |
| L46 | 1.0 | 0.5 | .530 | .612 | .584 | .710 | .739 | .738 | .911 |

**The `SG_orth` vs `SG_iso` columns carry no information about mechanism** (C1): those two
perturbation sets differ by 0.01% in direction. They are retained only as a record of what was run.
`SG_par` is not a valid path control either, because its radius was not matched to the path length.

What the grid *does* support: σ\* tracks ε monotonically at all three layers (0.05→0.05, 0.2→0.2,
1.0→0.5), so the optimal amount of averaging scales with the size of the effect being predicted.
That is consistent with both H_path and H_denoise and does not separate them.

**The correct discriminator, not yet run**, is a parallel-energy sweep with matched radial
distributions — draw `u_ρ = σ(ρ·s·δ̂ + √(1−ρ²)·q)` with `q ⊥ δ`, `s = ±1`, over
ρ ∈ {0, 0.25, 0.5, 0.75, 1}, holding `K` and the radius distribution fixed so only the parallel
*fraction* varies. H_path predicts the gain rises with ρ; H_denoise predicts it is flat in ρ.
Pairing that with the ε_ref convergence sweep and an fp32 replication separates all three
candidates including the numerical one.

### The positive control, and a bug it caught

`T_IG = δ·mean_t J(h+tδ)` must equal the true effect exactly, by the fundamental theorem of
calculus. First implementation integrated t over [0,1] and scored **0.81/0.84/0.62** — a control
that should read ~1.0. The cause was a mismatched identity, not a broken framework: ∫₀¹ gives
`Δ_one = F(h+δ)−F(h)`, while the target is the antithetic `Δ_odd = [F(h+δ)−F(h−δ)]/2`, which
equals the integral over the **symmetric** interval [−1,1]. Sampling t symmetrically:

| L31 | one-sided (wrong identity) | symmetric (correct), M=16 |
|---|---|---|
| ε=0.05 | 0.811 | **0.921** |
| ε=0.2 | 0.839 | **0.947** |
| ε=1.0 | 0.622 | **0.879** |

The residual gap is discretisation (largest at ε=1.0, where the path is longest and most curved)
plus finite-difference noise. **The framework recovers the true finite effect at ~0.95 where the
path is short, and that is the ceiling every other number competes against.**

The general lesson kept: a positive control that reads 0.84 when theory says 1.0 is not "close
enough" — it was pointing at a real mismatch between the estimand and the target.

---

## Result 5 — split leakage, and the operator that actually generalises

Added after review (C2). Calibrate/held-out was split by **base prompt**, which does not separate
(a→b) from (b→a) — different bases, exactly negated deltas — nor two arguments in one template.
Re-fitting under four nested split levels, held-out D4, ε=0.2 antithetic:

| T_sec | base | unordered-pair | **template** | **category** |
|---|---|---|---|---|
| L16 | 0.430 | 0.530 | **0.088** | **0.014** |
| L31 | 0.402 | 0.557 | **0.073** | **0.021** |
| L46 | 0.560 | 0.700 | **0.153** | **0.060** |

and the internal control that makes this leakage rather than distribution shift — `J̄` and `R̄` were
never fitted on this data and are flat across all four levels:

| L16 | base | unordered-pair | template | category |
|---|---|---|---|---|
| J̄ | 0.107 | 0.110 | 0.119 | 0.112 |
| R̄ | 0.121 | 0.124 | 0.133 | 0.126 |

**`T_sec`'s in-family advantage was template memorisation.** Under a category-disjoint split it
scores 0.014–0.060 against `J̄`'s 0.11–0.40 — an order of magnitude worse than the fixed lens it
was supposed to beat. This is a cleaner and stronger statement of the G-SECANT negative than the
learning-curve extrapolation the first version relied on, and it supersedes it as the primary
argument.

**Anchored operators.** The identity
`A + (C_Δδ − A·C_δδ)(C_δδ+λI)⁻¹ = (C_Δδ + λA)(C_δδ+λI)⁻¹` means a J-anchored operator is the
existing anchored solve with `anchor = J̄`. The first version anchored only to `R̄`, the weaker
baseline. J-anchoring degrades gracefully — at L46/category, `T_J@λ=1` scores 0.243 where `T_sec`
scores 0.070 — but **still loses to plain `J̄` (0.393)** out-of-category, so the learned correction
is actively harmful off-distribution at every λ tested.

**The affine blend is the only fitted thing that generalises.** Fitting three scalars
`a·J̄ + b·R̄ + c·I` by least squares beats `J̄` at every layer and is **stable across all four split
levels** (L16: 0.122 / 0.125 / 0.138 / 0.131 vs `J̄` 0.107 / 0.110 / 0.119 / 0.112). Small, but it
is the only positive fitted result in the study that survives a category-disjoint split.

**Relative error changes the reading of every operator.** Cosine hides magnitude, and
`relerr = ‖pred − true‖/‖true‖ = 1.0` is what predicting *zero* scores:

| L46, native one-sided | cos | rel. err |
|---|---|---|
| J̄ | 0.373 | **0.96** |
| R̄ | 0.378 | 0.96 |
| T_sec (base split) | 0.826 | 0.55 |
| T_J@0.03 (base split) | 0.824 | 0.55 |

**The released J-lens transport is no better in relative error than predicting no effect at all**
(0.96–1.00 at every layer). It carries direction, weakly, and essentially no magnitude. Any claim
that a lens "transports" a representation should be read against that.

**Target ecology.** The primary target is now the **native one-sided** patch
`Δ_one = F(h + (h(x′)−h(x))) − F(h)`, since for a natural delta `h+δ` is a state the model reaches
while `h−δ` is an extrapolation and 0.2·median‖h‖ is not the natural magnitude. Under the native
one-sided target every operator scores higher (L46 base split: T_sec 0.826 vs 0.560) but the
template/category collapse is unchanged (0.226 / 0.070).

---

## What this does *not* establish

- **`J_own` was never computed.** `T_sec` is compared against the *released* `J̄`, which was fitted
  under its own convention (`skip_first=4`, pile-10k, n=25). A convention-matched refit is the
  controlled comparison and it was not run — cost, not principle. The ε→0 secant (`fit@0.01`) is a
  partial stand-in and is *worse* than released J̄ at L16/L31.
- **M2 cannot include `J_loc` or SmoothGrad-J.** Their probe vectors need the local Jacobian as a
  matrix (5,120 JVPs per site). So the behavioural gate ranks the *fixed* operators only, and the
  two best predictors in this study are absent from it. This is a real gap in the argument, not a
  detail: the operator that best predicts Δ has not been shown to best identify a causal direction.
- **`J̄ > R̄` on M2 here** (+0.269 [+0.073,+0.554]) is **not** a refutation of the R-lens. Different
  protocol: 40 single-token probe-swap items, Δ log-prob, 3-layer band — versus the post's 30
  multihop questions, first-half band, autorater. It means the R anchor's value is unestablished on
  *this* metric, which is why `T_RC ≈ T_sec` is unsurprising.
- **Three layers, one model, one δ-family as primary.** Depth trends rest on 3 points. Nothing here
  has been checked on a second model or a standard-attention architecture.
- **The causal endpoint is layer 62, not the model's output.** `forward_from` stops at the
  released lens's target row and the primary target sums residual change over current-and-future
  positions. That matches the estimator being tested, but "causal fate" should ultimately be read
  at the logits — final-position logit change, or output KL. Not done.
- **Better transport does not imply a better readout, and this study never tested a readout.** A
  lens applies `h ↦ U·T·h`; everything here scores `T·δ`. Derivative accuracy says nothing about
  affine offsets or semantic basis alignment, so an operator can win here and produce worse
  vocabulary tokens. Every candidate needs three separate gates — finite-effect prediction,
  vocabulary-readout quality, semantic causal specificity — and only the first was run.
- **One shared 96-token prefix** is used by every prompt in the bank, so prefix diversity is zero
  and nothing here speaks to generalisation across contexts at the document level.
- **The builder enforces equal total token length** but does not enforce single-token arguments or
  a single contiguous differing block, contrary to how the method section described it.
- **One model, and an unusual one.** Qwen3.6-27B is a GatedDeltaNet hybrid (48 of 64 blocks are
  linear-attention). No standard-attention or fp32 replication was run, so no general lens claim
  is supported.
- **The bank's natural deltas span 555 of 5,120 dimensions.** Better than 117, still 11%. Every
  cross-family number remains identifiability-limited and is reported with its captured fraction.

---

## Where this leaves the project

Both fitted-operator branches are closed (G-SECANT fail, G-CONDVIABLE fail), and after C2 the
G-SECANT negative is stronger and simpler: the secant's apparent advantage was template
memorisation, and it loses to `J̄` by an order of magnitude across a category-disjoint split. The
only fitted object that generalises is the 3-parameter affine blend `a·J̄ + b·R̄ + c·I`.

**We have a strong causal-transport study and we do not have a better lens.** The gap between those
two is the honest headline. What we have is a validated measurement framework (`T_IG` ≈ 0.95
positive control, exact `forward_from`), a well-characterised negative on fitted transport, and one
large unexplained observation — spherical averaging around the operating point predicts finite
effects far better than the local finite difference.

Run order, revised after review:

1. **Offline, done** — leak-repaired splits, native one-sided targets, J-anchored operator, affine
   blend (`B7_splits_targets_anchors.py`). This is Result 5.
2. **Numerical validation, blocking** — `ε_ref` convergence sweep
   {1e-4 … 3e-2} on 8–16 sites, plus an fp32 replication. If the SmoothGrad advantage shrinks in
   fp32 or depends strongly on probe scale, it is partly numerical rather than model biology, and
   nothing downstream is worth running.
3. **The real mechanism discriminator** — parallel-energy sweep at matched radii,
   `u_ρ = σ(ρ·s·δ̂ + √(1−ρ²)·q)`, ρ ∈ {0, .25, .5, .75, 1}. Flat in ρ → denoising; rising in ρ →
   path. This replaces the retracted orth/iso test.
4. **`J_own`** — a convention-matched `E_{our contexts}[J_loc(x)]`, which finally separates
   context-averaging from corpus/convention/estimation mismatch, and is also the correct baseline
   for step 5.
5. **B6, rewritten** — compare fixed sketched `J̄_own^σ` against fixed sketched `J̄_own^0` on
   held-out *templates and prefixes*, with the sketch basis built from calibration data only, σ
   chosen on an inner split, and shared-energy estimated by the cross-product of two independent
   smoothing estimates `⟨C̄^A, C̄^B⟩ / mean_i⟨C_i^A, C_i^B⟩` so Monte-Carlo noise does not bias it
   toward "not shared". Report convergence over K ∈ {2,4,8,16}.
6. **Only if 5 passes** — fit `J̄_σ = E_{x,u}[J(h_x+u)]`, or a low-rank smoothed correction that
   falls back to `J̄` outside the measured span.
7. **Lens gate** — frozen intermediate-recovery/readability sets, wrong-concept controls, and
   output-level causal tests. Nothing here has tested a readout.
8. **Replication** — one smaller standard-attention model in fp32.

**Pre-registered decision rule for step 6:** proceed to a full smoothed lens fit only if the fixed
sketched `J̄_cal^σ` beats **both** `J̄_cal^0` **and** the released `J̄`, on template- and
prefix-disjoint data, for one-sided native patches, on **both** relative residual error and
final-logit prediction. If it improves only per-input SmoothGrad, or only cosine on antithetic
normalised directions, then what we have is a better local causal estimator — not a better lens.

## Reproduce

```
research/experiments/A0_verify_bank.py        bank integrity (blocking)
research/experiments/B1_bank_broad.py         broadened bank
research/experiments/B2_patch_bank_bases.py   make banks self-describing + verify
research/experiments/A1_A6_fits.py            per-ε fits, J_loc/J̄        [no model]
research/experiments/A2_A5_model.py           Stein control, in-span transfer
research/experiments/A4_m2_ablation.py        behavioural M2
research/experiments/C2_conditional.py        conditional low-rank model
research/experiments/B3_learning_curve.py     learning curves            [no model]
research/experiments/B4_smoothing_mechanism.py  σ×ε grid, par/orth, T_IG
research/experiments/B5_figures.py            figures                    [no model]
```

Raw outputs (`research/outputs/`, ~4.5 GB) are gitignored and regenerable; each carries `utc`, git
hash, and its configuration.
