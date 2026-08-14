# Ekko-Lens: Findings Ledger

**Project.** Which linear transport operator best recovers causally active intermediate
representations in a language model? J-lens (averaged Jacobian), R-lens (LRP-modified backward),
input-specific Jacobians, fitted secant/anchored operators, and smoothed/path operators are treated
as competing hypotheses about one object, ranked on a directly measured causal benchmark rather
than on readout aesthetics.

**Model & conventions (fixed throughout).** Qwen/Qwen3.6-27B — 64 layers, d_model 5120, hybrid
GatedDeltaNet (48 `linear_attention` + 16 `full_attention` blocks, interval 4). Released-lens
convention from `camilablank/workspace-lenses` provenance: `target_layer=62`, `skip_first=4`,
corpus `NeelNanda/pile-10k`, `n_prompts=25`, `max_seq_len=128`. Bank layers 16/31/46 (26/50/74% of
the target row). Perturbation scales ε ∈ {0.01, 0.05, 0.2, 1.0}×median‖h‖ plus native magnitude.
Primary target: downstream residual change summed over current-and-future positions (matches the
J̄/R̄ estimand); native one-sided patch promoted to primary after external review. Statistics:
paired bootstrap (2,000 resamples) over base prompts / split-level groups, never over deltas.

**Sources.** Gurnee et al., *Verbalizable Representations Form a Global Workspace in LMs*
(Transformer Circuits, 2026); Bhatia, Blank & Nanda, *Meta-tokens in the J-space* (AF, 2026);
Blank, Bhatia & Nanda, *R-lens* (LW/AF, 2026). Code: `senku14x/jacobian-lens`, branch `ekko-lens`
(experiments in `research/experiments/`, compact metrics committed under `research/artifacts/data/`).

**Evidence levels used below.** *Observation* → *recurring pattern* → *supported claim* →
*interpretation/hypothesis*. A margin is only quoted as real if it clears both a bootstrap CI and,
where applicable, the twin-fit estimator noise floor (F8).

---

## Part 0 — Instrument validation

### F0.1 — Can true finite effects be measured exactly, autograd-free, on this architecture?

**Question.** The model is a GatedDeltaNet hybrid; `torch.func` and double-backward were untested
on it. Can we measure `F(h+δ) − F(h)` exactly without autograd?

**Experiment.** `000_probe_arch.py`, `001_stage0_checks.py`. Block kwargs (`position_embeddings`,
`position_ids`, `attention_mask=None`) were probed for hidden-state dependence; `forward_from(h, ℓ)`
replays blocks ℓ+1..62 with kwargs captured from one clean forward. Checks: wrapper identity
(`unembed(h_L)` vs model logits), round-trip substitution at 25/50/75% depth, batch-of-8
self-consistency, known-answer factual reads.

**Findings.** Max error **0.000e+00** on every check, both 4B and 27B. Known-answer reads sensible
(boot-country → *Italy* rank 1 @L36; web-spinner → *spider* rank 4 @L39). Every JVP in Phases A–C
is a central difference through this exact replay.

**Status.** Gate passed; the measurement chain is exact by construction, not approximately.

### F0.2 — Does the published R>J readability gap replicate here?

**Question.** If R>J does not hold on this model, R is not a meaningful anchor and the harness is
suspect.

**Experiment.** `002_calib_passk.py`. 59-item calibration slice (10 items/category, seed 0, held
out of the frozen eval sets), released J vs R, pass@10 = min-over-layers rank ≤ 10 per the eval-set
README.

**Findings.** All layers: J 0.479 → R **0.534** (+0.055). First half: J 0.089 → R **0.150**
(+0.061). Mean first layer reaching top-10: 43.5 → **39.7**. Direction non-negative in all six
categories (typo +0.30 first-half; multihop/association 0.00).

**Status.** Recurring pattern (n=59, no CIs; validates the instrument, not a headline). The
published examples (e.g. sushi→Japan at L2 vs L14) did **not** reproduce on reconstructed prompts —
uninformative, since only one of four prompts was verbatim.

### F0.3 — Does the framework recover a known ground truth? (positive control)

**Question.** The path-integrated operator `T_IG = mean_t J(h+tδ)·δ` must equal the true finite
effect by the fundamental theorem of calculus. Does it?

**Experiment.** `B4_smoothing_mechanism.py`, M=8→16 path points, 60 sites × 3 layers.

**Findings.** First run: **0.81/0.84/0.62** — a control that must read ~1.0. Cause: ∫₀¹ gives the
one-sided effect, but the target was the antithetic effect, which is the integral over the
*symmetric* interval [−1,1]. With symmetric sampling: **0.921 / 0.947 / 0.879** (L31, ε=0.05/0.2/1.0),
residual gap = path curvature discretisation + finite-difference noise.

**Status.** Supported: the framework's ceiling is ≈0.95 at ε ≤ 0.2, and every operator below is
scored against it. Lesson kept: a positive control reading 0.84 where theory says 1.0 is a real
estimand/target mismatch, not "close enough."

---

## Part 1 — The transport-operator ladder

### F1 — Does a directly-fitted secant operator beat J̄/R̄? (G-SECANT)

**Question.** Fit `T_sec = C_Δδ (C_δδ + ηI)⁻¹` on measured finite responses. Does it transport
held-out perturbations better than the released lenses?

**Experiment.** `003_bank.py` / `B1_bank_broad.py` (bank: token-aligned minimal pairs on a 96-token
pile prefix; broadened to 10 categories × 40 templates × 6 args = 240 bases, 30,096 deltas/layer,
90,288 total; D4 effective rank 117 → **555**), `A1_A6_fits.py` (per-scale fits, ridge chosen on
calibrate), `A2_A5_model.py` (Stein control, in-span transfer), `A4_m2_ablation.py` (behavioural),
`B3_learning_curve.py`, `B7_splits_targets_anchors.py` (nested splits).

**Findings.**
- *In-family* (held-out D4, ε=0.2, base-prompt split): `T_sec − J̄` = **+0.294 / +0.180 / +0.142**
  at L16/31/46, CI-clear — the number that made the secant look alive.
- *Span-immune checks, all negative:* Stein control `T_sec − SmoothGrad-J` = **−0.306 / −0.469**
  (L16/L31); in-span transfer (held-out D1/D2 projected into the identified subspace, true effects
  re-measured) `T_sec − J̄` = **−0.119** (D1) / **−0.199** (D2) at L16, CI-clear, margins *widened*
  as the bank grew; behavioural M2 `T_sec − J̄` = **−0.861** [−1.358, −0.445].
- *The decisive correction (split leakage):* base-prompt splits do not separate (a→b) from (b→a) —
  exactly negated deltas — nor two arguments in one template (608 reverse pairs crossed the base
  split; 0 cross the template split). Under nested splits, `T_sec` at ε=0.2:

  | layer | base | unordered-pair | template | category |
  |---|---|---|---|---|
  | L16 | 0.430 | 0.530 | **0.088** | **0.014** |
  | L31 | 0.402 | 0.557 | **0.073** | **0.021** |
  | L46 | 0.560 | 0.700 | **0.153** | **0.060** |

  while `J̄`/`R̄` — never fitted on this data — are flat across all four levels (L16 J̄:
  .107/.110/.119/.112). **The in-family advantage was template memorisation.**
- *Learning curves:* fitting on 15→120 prompts, nothing plateaus; cross-family closes at
  ~+0.03/doubling; parity with J̄ needs ~6 more doublings (~64× corpus) under an extrapolation that
  must saturate.

**Status.** **Supported negative.** A bounded, quantified negative at this data scale — not a proof
of impossibility. The methodological corollary is itself a finding: any transport-transfer result
reported without an effective-rank / energy-captured diagnostic is uninterpretable (two of three
gates would have been called wrong on the narrow bank).

### F2 — Does anchoring the fit to R̄ (or J̄) rescue it?

**Question.** `T_RC = (C_Δδ + λR)(C_δδ + λI)⁻¹`: does regularising toward a working lens keep its
generalisation while adding causal accuracy?

**Experiment.** `A1_A6_fits.py` (R-anchor), `B7_splits_targets_anchors.py` (J-anchor via the
identity `A + (C_Δδ − A·C_δδ)(C_δδ+λI)⁻¹ = (C_Δδ + λA)(C_δδ+λI)⁻¹`; affine blend `aJ̄+bR̄+cI` by
least squares).

**Findings.** J-anchoring degrades gracefully (L46/category: 0.243 vs `T_sec` 0.070) but **still
loses to plain J̄ (0.393)** out-of-category at every λ — the learned correction is harmful
off-distribution. On M2, `T_RC` is barely above random (+0.016 vs −0.005). **The only fitted object
that generalises is the 3-scalar blend** `aJ̄+bR̄+cI`: beats J̄ at every layer, flat across all four
split levels (L16: 0.122/0.125/0.138/0.131 vs J̄ 0.107/0.110/0.119/0.112).

**Status.** Supported. The signal: the viable family is *small numbers of causally-grounded
components, adaptively mixed*, not free d×d matrices (26.2M parameters from ~1,700 effective
observations).

### F3 — How much does one fixed matrix lose to input-specific transport? (G-CONTEXT)

**Question.** Is context-averaging the dominant failure of the fixed lens?

**Experiment.** `A1_A6_fits.py` (J_loc = central secant at ε_ref=0.01·median‖h‖, same directions,
paired rows), then **isolated** in `D2_fit_jown.py`/`D4_jown_analysis.py` with a convention-matched
own fit (see F8) so corpus/convention/estimation mismatch is controlled.

**Findings.** `J_loc/J̄` = **4.34 [4.14,4.53] / 3.17 [3.05,3.29] / 1.95 [1.90,2.00]** at L16/31/46
(ε=0.2 antithetic) — monotone in depth, worst early, the best-determined effect in the study.
Isolated: `J_loc − J_own` = **+0.174 (native one-sided) / +0.412 (antithetic)** — 15× / 47× the
noise floor — and `J_loc − J_own` ≈ `J_loc − J_rel` to 0.001, so the gap is context dependence, not
corpus mismatch. Correction: on the ecological native one-sided target the ratio is **1.82×**, not
3.17× — the antithetic normalised target overstates the context-averaging cost.

**Status.** **Supported claim — the one large, robust effect in the project.** Consistent with the
workspace picture: representations approach context-independence as they approach the readout.

### F4 — Is the context dependence low-rank and shared? (G-CONDVIABLE)

**Question.** If `J_loc(x) − J̄` were low-rank-shared, a lens could carry a few correction modes.

**Experiment.** `C2_conditional.py`. Sketch `J_loc(x_i)·S` on top-192 eigenvectors of pooled
`C_δδ`, 56 sites (29 held out); PCA of deviations; **oracle** rank-r reconstruction (coefficients
fit on the held-out site's own deviation — an upper bound no deployable model reaches).

**Findings.** Top-16 deviation variance: .606/.525/.495 (2/3 straddle the pre-registered >50% bar —
the proxy does no work). Direct measurement: oracle rank-16 recovers **23.2% / 40.0% / 85.2%** of
the `J̄→J_loc` gap — recovery is **anti-correlated with need** (works only at L46 where the gap is
smallest). Rank-32 adds +2.4/+0.4 points.

**Status.** Supported negative for the *continuous low-rank* conditional family. A failing oracle
bounds every predicted-coefficient variant, so those were correctly not run. Discrete-regime
(cluster atlas) conditioning remains untested (see Open, O5).

### F5 — Smoothing the operating point helps a lot. Why is open.

**Question.** SmoothGrad-J (average J over a small ball around h) vs the local estimator J_loc:
which mechanism, path-averaging (unfittable by any fixed matrix) or denoising (fittable)?

**Experiment.** `B4_smoothing_mechanism.py`: σ×ε grid, parallel/orthogonal decomposition, T_IG
ceiling. 60 sites × 300 directions × 51 held-out prompts per layer.

**Findings.** SmoothGrad-J beats J_loc by **+0.17…+0.27**, CI-clear at every layer and ε ≤ 0.2
(e.g. L31 ε=0.2: SG 0.868 vs J_loc 0.592 vs J̄ 0.194; T_IG 0.899). σ* tracks ε (0.05→0.05,
0.2→0.2, 1.0→0.5). **The orth-vs-iso discriminator was retracted:** for isotropic u in d=5120 only
1/d ≈ 2e-4 of energy is parallel to δ, so cos(u_iso, u_orth) = **0.99990** — the "9 of 9 conditions"
result compared two near-identical perturbation sets and was guaranteed by geometry. The parallel
arm was separately mis-specified (radius 4× the path length at ε=0.05). A third mechanism was added
by review and then directly supported: **the bf16 central difference at ε_ref=0.01 has cos ≈ 0.93
to the true forward-mode JVP** (0.26 at ε=0.001; 0.995 at 0.05), so J_loc carries ~7% direction
error by construction and SmoothGrad averages 8 such estimates — part of the gain may be
measurement-variance reduction, not model geometry.

**Status.** Supported observation (spherical averaging around the operating point substantially
improves finite-effect prediction); **mechanism open** — three candidates (path / conditioning /
numerical), discriminators specified but not run: Richardson ε_ref sweep at a fixed point, fp32
replication, matched-radius parallel-energy ρ-sweep.

### F6 — Do readout-side modifications help? (reference anchoring, shrinkage, fusion)

**Question.** Three review-proposed "cheapest wins" for a better fixed lens.

**Experiment.** `C1_reference_anchor.py` (readout `μ_F + βJ(h−μ_h)` with intercept control),
`C3_spectral_fusion.py` (spectral shrinkage of J; mode-wise J/R fusion with a random-basis
control), all under leak-repaired splits, with `C5_stress_tests.py` (12 estimator checks, all pass;
closed-form == explicit least squares to 1.9e-8; planted-operator recovery 4.7e-8).

**Findings.** All three negative, each with a measured reason:
- *Anchoring:* the motivation is real (‖μ_h‖=59.0 vs median ‖h‖=74.4 — ~79% of a typical
  activation's norm is corpus mean), and the intercept control is clean (`unembed(μ_F)` alone scores
  0.0000 everywhere — not the tuned-lens pathology). But β=1 destroys early-layer readout
  (first-half pass@10 → 0.0000, three operators collapse to an identical .4096), and the best β=0.25
  buys aggregate pass@10 (+0.051, itself optimistic — β swept on the eval items) **by halving
  first-half pass@10 (.0890 → .0381)** — a divergence from the North Star, not a win. Centring alone
  is worse than nothing (.3588 vs .4788).
- *Spectral shrinkage:* never beats J; **(J−I) carries only 7.5–14.4% of its energy in its top 64 of
  5120 modes** — no concentrated unreliable structure to shrink.
- *Fusion:* the fit drives ~65–70% of R−J modes to α=1 and ~15–33% to α=0 (R's correction is
  genuinely non-uniform), but the selection scores at or below taking all of R; the 512-coefficient
  random-basis control shows no gain, so the small effect is basis-specific, not free parameters.
- Two properties survive the failures: constrained (≤d-parameter) operators are *flat across split
  levels* (the leakage-driven overfitting is fixed, it just buys nothing), and **every fixed
  operator sits at relerr 0.95–1.01** (but see F10.2 for what that does and does not mean).

**Status.** Supported negatives. Together with F1/F2/F4: the fixed-linear-operator class on this
model is effectively exhausted at ≈R̄'s level, with the blend as its cheap equal.

### F7 — What do the R-lens backward rules actually do? (block-level)

**Question.** Test each rule where it is cheap and interpretable — one block — before any lens fit,
including the rules R-lens omits (q/k norms, attention gate, routing, GatedDeltaNet gates).

**Experiment.** `D3_block_rules.py` + `research/ekko/rules.py` (forward-preserving stop-gradient
surrogates, verified against the actual `modeling_qwen3_5.py`; forward-mode JVP — the first working
autograd on this architecture here; eager attention forced after SDPA's missing forward-mode
derivative). One full-attention + one GatedDeltaNet block, 24 sites, both delta families, ε sweep.
The first run omitted the R baseline entirely and was redone (the addendum).

**Findings.**
- **R itself is the headline: it is a finite-displacement operator.** Worse than J at every
  ε ≤ 0.2 and better at ε = 1.0 by **+0.020…+0.023 in all four block × delta-family conditions**
  (e.g. full-attn D4: J .9264 → R .9468 at ε=1.0; J .9949 → R .9833 at ε=0.05). The R recipe trades
  tangent accuracy for secant accuracy — the regime a lens reading a whole activation operates in.
- **Q/K-norm LN-rule — the most direct extension of R's own logic — is negative at every ε and both
  families** (−0.009…−0.012 on natural deltas).
- **GatedDeltaNet gate/state rules do nothing** (±0.0012) despite governing 48 of 64 layers — the
  predicted highest-upside direction is empirically inert.
- The rules that help at ε=1.0 (+0.005…+0.010, natural deltas only, not isotropic) are all
  routing/branch-gradient damping: `attn_gate_half`, `cp_value_only`, `softmax_temper`,
  `gdn_out_half`. **What pays is damping routing sensitivity, not repairing normalisations.**
- Three bugs each silently produced a plausible null before being caught: SDPA has no forward-mode
  derivative (NaN → eager required); `Qwen3_5RMSNorm` uses `.eps` and `(1.0 + weight)`;
  `b.sigmoid()` is a tensor method, so patching `torch.sigmoid` intercepted nothing.

**Status.** R's crossover: recurring pattern (4/4 conditions, but n=24 sites, no CIs, one block per
type). New rules: observations at ≤0.011 on cos ≈0.98 — no rule earns a full-stack fit *on
block-level evidence alone* (but see O1: compounding is exactly how R's own +0.02/block becomes a
lens-level gap, and a full-stack fit now costs 1.3 h).

### F8 — The estimator noise floor, and what it retires

**Question.** Every J/R comparison in the literature and in this project compares single n=25 fits.
How large must a margin be to mean anything?

**Experiment.** `D1_time_fit.py` (cost: **183 s/prompt**; n=25 ≈ 1.3 h — the "fitting is
infeasible" reading was a stale process holding 72 GiB), `D2_fit_jown.py` (fit `J_own` at exactly
the released convention, as two disjoint half-corpus fits, merged), `D4_jown_analysis.py`.

**Findings.**
- `J_own` vs released `J`: **−0.0011 [−0.0025,+0.0001] (native) / +0.0001 (antithetic)** — neither
  CI-clear. Corpus/convention mismatch contributed nothing; every "vs released J" comparison in the
  study stands. (Matrix level: relF 0.328, inside the scaled twin floor ≈0.40.)
- **Twin-fit floor:** two fits of the *same* estimator on disjoint halves differ by **0.0113 cos
  (native) / 0.0088 (antithetic)**; 0.56 relative Frobenius.
- Against it: `R − J` in effect prediction = **+0.0155 / +0.0059** — 1.4× the floor in one target,
  *below* it in the other, despite bootstrap CIs excluding zero. The bootstrap resamples evaluation
  prompts, not the fit, so it cannot see this variance. `J_loc − J_own` = +0.17/+0.41 clears the
  floor 15–47×.

**Status.** **Supported claim, and the most transferable methodological result:** the R-lens's
effect-prediction advantage over the J-lens does not clear the estimation noise of the estimator it
modifies. Any fixed-operator margin under ≈0.01 cos at n=25 is unresolved, however tight its CI.
This retires several minor margins in F1–F6 without changing any verdict.

### F9 — Whose token directions are causally load-bearing? (M2, behavioural)

**Question.** Span-immune gate: ablate each operator's folded intermediate-token direction
(`v = Tᵀ(γ⊙u_t)`, unit-normalised, same band {16,31,46} and same fold for every operator) on 40
probe-swap items; measure Δ log-prob of the correct answer; matched-norm random controls;
general-damage check on pile text.

**Experiment.** `A4_m2_ablation.py` (judge-free variant of the R-lens post's ablation protocol),
run on both original and broadened-bank refits.

**Findings.** J̄ **+0.856** [+0.437,+1.354] > R̄ +0.587 > logit lens +0.022 > T_RC +0.016 >
T_sec −0.006 ≈ random −0.005. Top-1 retention on pile .97–.99 for every operator — nobody wins by
breaking the model. `T_sec − J̄` = −0.861 CI-clear; the earlier "worse than random" for T_sec^D4
does not survive the refit (−0.005, not CI-clear) — "indistinguishable from random" does.
**`J̄ − R̄` = +0.269 [+0.073,+0.554] CI-clear — J beats R on this protocol.** Scope: 40 single-token
items, 3-layer band, Δ log-prob — *not* the post's protocol (30 multihop, first-half band,
autorater); no contradiction claimed, but the post's causal claim does not carry to a neighbouring
protocol unchanged.

**Status.** Supported under this protocol. Known gap: J_loc and SmoothGrad-J cannot enter M2
(their probe vectors need the local Jacobian as a matrix), so the two best effect-predictors are
absent from the behavioural ranking.

---

## Part 2 — Independent audit (2026-08-14)

### F10.1 — Are the negatives implementation artifacts?

**Question.** The study is LLM-executed; the user asked whether the code does what the plan meant.

**Experiment.** Independent end-to-end audit: `harness.py`/`rules.py` read against the fetched
`modeling_qwen3_5.py` source; bank targets checked against the J̄ estimand; fit/split/bootstrap
logic re-derived; `Qwen3_5RMSNorm` parameterisation settled from the cached model weights
(final-norm weight mean **+0.9619**, zero-init offsets, forward multiplies by `(1.0 + weight)` —
source-verified); every headline number in reports 007/008/009 cross-checked against the committed
metrics JSONs.

**Findings.** **The negatives are real.** All spot-checked report numbers match the committed
artifacts exactly (B7 collapse table, D4 floor and deltas, A4 rankings, D3 crossover 4/4). The
surrogate math in `rules.py` is correct (including the subtle point that the half-rule on a
both-factors-dependent product equals 0.5× the ordinary product gradient). One real,
conclusion-safe bug found: **every hand-folded probe vector uses `γ⊙u_t` where this architecture's
`(1+w)` norm makes the ranking-exact fold `(1+γ)⊙u_t`** (in `003/B1` D2 family and `005/A4` M2
probes). Measured impact through the released Jᵀ: cos(used, correct) = **0.938 / 0.920 / 0.960** at
L16/31/46 (min 0.72) — a uniform mild under-measurement that cannot reorder M2 gaps of 30–100×, but
should be fixed before quoting M2 magnitudes.

**Status.** Audit-supported. Cosmetic issues: stale header comment in `rules.py`; dead loop in
`apply_rule`.

### F10.2 — One framing correction: "J̄ carries essentially no magnitude" overstates

**Question.** Report 007 reads relerr ≈ 0.96–1.00 as "no better in magnitude than predicting zero."

**Finding.** At cosine c, the best relative error achievable by *any* rescaling is √(1−c²). At L46
(c = 0.373) that floor is 0.928; J̄ scores 0.96 — its magnitude calibration is close to the
direction-limited optimum. relerr ≈ 1 is mostly *forced by direction error*; direction, not
magnitude, is the binding failure of the fixed-operator class.

**Status.** Correction to the report's wording; no verdict changes.

---

## Part 3 — Synthesis

### F11 — Why does the R-lens work? (the conservation / secant-from-zero account)

**The puzzle.** R reads out clearly better (F0.2) while transporting perturbations no better than J
at the noise floor (F8) and strictly worse at tangent scale (F7). "A more faithful backward pass"
cannot explain that dissociation — the chain rule is exact and R is measurably not a better
derivative.

**The identity.** Each LRP rule is exactly the linear operator that *reproduces its component's
output when applied to the full activation* (LRP's conservation property, imported intact):

| component | R rule backward | applied to the full input gives | true Jacobian applied to the full input |
|---|---|---|---|
| SiLU | σ(z) | σ(z)·z = SiLU(z) **exactly** | σ(z)z + z²σ′(z) ≠ SiLU(z) |
| RMSNorm | diag(1+w)/r | x⊙(1+w)/r = RMSNorm(x) **exactly** | ≈ **0** — the norm's exact Jacobian annihilates the radial direction, which is most of what the activation is |
| gated product a⊙b | ½(b·da + a·db) | ab **exactly** (Euler, degree-2 homogeneous) | 2ab — double-counted |
| attention (value path, A frozen) | A·dV | A·V(x) = output **exactly** (V linear) | adds (dA)V, reproduces nothing |

So J answers "what happens if I nudge h"; R answers "what is the downstream stack doing with *this*
h." A readout `W_U norm(T·h)` asks the second question. R works because its rules are exact for the
readout question and wrong for the perturbation question — the precise double dissociation
measured in F0.2 + F7 + F8.

**What the account explains.** Early layers improve most and trash tokens vanish (the tangent's
norm-annihilation and gate terms compound once per block; ~120 norms sit between L4 and the
readout); the R>J gap growing with model scale (depth = number of compounding traversals — their
unexplained scale trend gets a mechanism); the meta-tokens finding that J-lens vectors are often
non-causal while centroid *differences* (finite-displacement objects) are; and D3's rule pattern —
the rules that helped at ε=1.0 are exactly the remaining output-reproducing homogenisations
(frozen-A, tempered routing, GDN output half-rule), while the one "more faithful derivative" rule
(q/k-norm) hurt, as predicted when the readout-optimal routing treatment is to freeze A entirely.

**Status.** Interpretation, supported by: exact per-component identities; the 4/4 block-level
crossover at activation-scale displacement; the noise-floor equality at perturbation scale. Not
established: composition through the full stack is *not* exact (residual adds and cross-position
mixing), one model, block-level n=24. Falsifiable predictions in O1–O3.

### F12 — Is the R-lens post overclaiming?

**Assessment.** Split three ways:
- **The observation stands** — early-layer readability gain, independently replicated here (F0.2);
  the released lenses made this whole audit possible.
- **The mechanism narrative does not** — "error accumulation" / "relevance collapse" implies the
  gradient is lossy and LRP repairs it; the measurement says R is a *different* operator answering
  the readout question (F11), and its directly-measured faithfulness edge is at the noise floor
  (F8).
- **The title's "faithful" is the overclaim** — it rests on one n=30 autorater ablation whose sign
  flips under an adjacent judge-free protocol (F9), plus probe bounds the authors themselves
  discount, with no fit-noise control anywhere.

The honest title would be "making J-lens more *legible* on early layers" — and that post would be
fully supported.

---

## Part 4 — What is closed, what is open

### Closed on this model (do not re-run without new structure)

Free d×d secant fits and their anchored/low-rank variants (F1, F2); continuous low-rank conditional
transport (F4); spectral shrinkage and mode-wise J/R fusion (F6); reference-anchored readout as
tested (F6); q/k-norm LN-rule and GDN gate-freezing rules (F7); backward-rule micro-tuning at
tangent scale (tautologically — the exact Jacobian is optimal as ε→0).

### Open, ranked by expected information per hour

- **O1 — The fully-homogenised lens (the F11 theory's sharpest prediction).** Full-stack fit of
  R + frozen-A/value-only + `gdn_out_half` (+ tempered softmax variant). Rationale: R's own
  block-level +0.02 compounds into its entire published readability gap, and fitting now costs
  1.3 h (F8). Score on first-half pass@10, M2, and a skip-ahead guardrail. A negative here is also
  informative: it localises where composition breaks the conservation account.
- **O2 — Richardson ε_ref sweep + fp32 replication (blocking for F5).** If the SmoothGrad gain is
  largely truncation/quantisation noise, the smoothed-operating-point lens branch dies before it is
  fitted; if it survives, run the matched-radius ρ-sweep, then the pre-registered smoothed-fit gate.
- **O3 — Protocol-matched replication of the R-lens ablation** (their 30 multihop items,
  first-half band, penultimate-token position, judge-free scoring) — settles the J-vs-R causal
  question under matched conditions instead of dueling protocols (F9 vs their result).
- **O4 — Probe-fold fix + A4 re-run** (`(1+γ)⊙u_t`; ~30 min) — hygiene before quoting M2 anywhere
  (F10.1).
- **O5 — Input-adaptive blend** `α(h)J̄ + β(h)R̄ + γ(h)I` with gating restricted to cheap layer-ℓ
  observables (norm, gate-saturation, position). The minimal conditional lens, inside the only
  family that generalised (F2), immune by construction to the C2 and B7 failure modes. Companion:
  the deferred discrete-atlas oracle test (cheap kill).
- **O6 — Sparse-frame readout.** The workspace paper builds the nonneg-cone machinery and never
  uses it for the readout; its own result says the J-space ~7% of variance carries the causal load.
  Input-adaptive with zero training. Tractable via candidate-set + nonneg lasso.
- **O7 — Depth-vs-width test of F11** on the Qwen3.5 size ladder (block-level, cheap): the R>J gap
  should track depth (norm/gate traversals), not parameters.
- **O8 — Radial-annihilation check** (minutes): on stored activations, compare ‖J_ℓ·h‖ vs ‖R_ℓ·h‖
  decomposed against h's own direction — the most direct test of F11's central identity.

### Open aspects of the three source posts (beyond the above)

- **Workspace paper:** the early "sensory band" — instrument artifact or real absence? (F3's
  depth-monotone context cost supports "partly instrument"; decisive test = a causally-validated
  early readout.) Single-token vocabulary (named by the meta-tokens authors as *the* binding
  constraint; multi-token/phrase readout untouched by anyone). Attention treatment relegated to a
  robustness appendix while F7 shows routing treatment matters at finite displacement. Generality
  beyond Claude models (CKA "less clean" on Qwen).
- **Meta-tokens post:** the causal-handle gap — J-lens vectors detect but are often not causal;
  nobody has built a lens whose per-token vectors are causally calibrated (the M2 harness is
  exactly the trainer/evaluator for a small vocabulary of them). Autoresearch recall limited by the
  vocab constraint.
- **R-lens post:** attention rules (AttnLRP) unimplemented anywhere including here; the scale-trend
  mechanism (candidate: F11/O7); no ground truth for "represented at layer K" — this project's
  patching + M2 benchmark is the closest thing to an answer and is worth distilling into a
  reusable evaluation standard; on hybrid models, 48/64 layers currently have *no* effective
  transport treatment at all (F7's GDN nulls).

---

## Appendix A — Plan-conformance audit (2026-08-14)

The question this answers: independent of whether the numbers are bug-free, did the implementation
cover what the two plan documents (`beyond_jlens_rlens_research_ideas.md`, `ekko_lens.md`)
specified? Four buckets.

**A.1 Implemented as designed (the trunk).** The perturbation bank (D4-primary/D1/D2, both
targets, summed-over-future estimand), closed-form `T_sec`/`T_RC`, both mandated circularity
controls, the `J_loc` and `J+λI` baselines, M1 with the full baseline set, prompt-level bootstrap,
the pre-registered Stage-2 gate honestly evaluated and failed, Stages 3–4 correctly skipped per the
decision tree, C2's cheap test before any conditional lens, effective-rank diagnostics, data
separation (with the logged calibration-slice carve).

**A.2 Deviations — all logged, all defensible.** Release convention (`skip_first=4`, pile-10k)
over the doc's 16/WikiText so the R anchor lives in the right space; science directly on 27B
(user-approved); judge-free Δlogp M2 instead of an autorater; twin floor from halves of one n=25
rather than a disjoint FIT-25b; ε grid extended beyond spec (0.01 + native); J-anchoring added only
after review; the noise floor arriving last despite "mandatory before any sweep comparison" (009
owns this); Stage 5 executed as discrete block-level rules rather than the continuous θ wrapper.

**A.3 Specified and never implemented.**
1. **Swap and clamp interventions** (M2b + the paper-Methods battery) — ablation is the only
   causal intervention in the codebase; no pinv-swap code exists. All "direction identification"
   claims are ablation-only.
2. **M4 skip-ahead guardrail** — never built; mandatory before any readout claim (O1/O6).
3. **The D3 concept-vector delta family** — admitted "not built"; transfer ran on D1/D2 only.
4. **Continuous α/β/γ/λ interpolation sweeps** (ideas doc Priority 1; "very strong result A") —
   only endpoints and discrete rules were tested. The interior-optimum question is untested, not
   refuted.
5. γ_q/γ_k split; AttnLRP softmax rule.
6. **Full-stack θ_R endpoint equivalence** — J side eventually verified (J_own ≈ released J); the
   R recipe verified at block level only.
7. Shuffled-J / random-matrix control on M1; the clean small-ε sanity check (the fit@0.01
   stand-in uses the bf16 FD later shown to carry ~7% error at exactly that scale).
8. §3.2 tiny-model brute-force test (upstream jlens tests only).
9. Stage-6 items (content/routing, robust aggregation, joint-span ablation, sparse-frame readout),
   logit-level causal endpoints, second model / fp32 replication.

**A.4 Faithful to a spec that was wrong.** The probe fold `Tᵀ(γ⊙u_t)` is the plan's own formula,
implemented to the letter; this architecture's `(1+w)` RMSNorm makes the correct fold
`Tᵀ((1+γ)⊙u_t)` (F10.1). The implementation error is upstream, in the plan.

**Net effect.** No missing item props up a headline negative — those rest on A.1 machinery. The
gaps genuinely limit: M2-based claims (ablation-only), the interior-optimum hypothesis (still
open), and any future positive result (currently lacking its specified M4 guardrail and the swap
leg of the causal battery).

---

## Bottom line

**A strong causal-transport study; still no better lens.** What the project bought: an exact,
validated measurement framework (F0.1, F0.3) with a ~0.95 ceiling; a supported negative on the
entire fitted-fixed-operator class with the leakage and identifiability mechanisms named (F1–F6);
the first fit-noise floor in this line of work, which retires the R-lens's published faithfulness
margin (F8); one large robust positive effect — context dependence, worst early (F3); a mechanistic
account of what the R-lens actually is, with exact identities and falsifiable predictions (F7,
F11); and a short, cheap, theory-ordered queue of what to run next (O1–O8).
