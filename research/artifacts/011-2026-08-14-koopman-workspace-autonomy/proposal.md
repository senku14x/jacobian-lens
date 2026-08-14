# 011 — Koopman lift of the residual stream: is the workspace autonomous?

**Status: PARKED — proposal only, no experiments run.** Written 2026-08-14, during the
E1/E2 homogenized-lens runs. Priority sits below the blackmail causal follow-up, the
NLA three-instrument comparison, and multi-token readout. Parked deliberately after a
skeptical pass (§7): the formalism is elegant, the *decision-relevant* payoff is one
specific measurement — workspace autonomy — and that is what this document scopes.

---

## 1. Motivation: what the measured negatives license

This project established (007/009, FINDINGS.md F1–F8) that on Qwen3.6-27B:

- No **fixed linear map on the state** `h_ℓ ↦ T·h_ℓ` beats the R-lens beyond the
  twin-fit noise floor: the fitted secant, anchored, low-rank, spectral, and fusion
  variants all fail off-distribution; the only generalising fitted object is the
  3-scalar blend `aJ̄+bR̄+cI`.
- The dominant error of any fixed matrix is **context dependence**: the input-specific
  operator beats the averaged one by 1.8–4.3×, worst early (F3), and the deviation is
  not low-rank-shared (F4).

Two responses exist. The H-lens (E1) keeps state-space linearity and changes *which*
operator is averaged (conservation instead of tangent). The Koopman lift is the other
response: **give up linearity in the state, buy linearity in a lifted space of
observables.** The measured failure of fixed d×d state maps is precisely the
situation this tool was built for.

## 2. The mathematics

### 2.1 The Koopman operator

Let `f: X → X` be any (nonlinear) map on a state space, and let observables be
functions `g: X → R`. The Koopman operator acts on observables by composition:

    (K g)(x) := g(f(x)).

**K is exactly linear**, for any f:

    K(a·g1 + b·g2)(x) = (a·g1 + b·g2)(f(x)) = a·g1(f(x)) + b·g2(f(x))
                      = a·(K g1)(x) + b·(K g2)(x).

No approximation has been made; the nonlinearity has been moved into the (infinite)
dimension of the function space K acts on.

### 2.2 Eigenfunctions and invariant subspaces

If `K φ = λ φ`, then along any trajectory `φ(x_{k+1}) = λ·φ(x_k)`: eigenfunctions
evolve geometrically. `|λ| = 1` ⇒ conserved/persistent content; `|λ| < 1` ⇒ decaying.

A finite set `Ψ = (ψ_1, …, ψ_m)` spans an **invariant subspace** if each `K ψ_i` is a
linear combination of the ψ_j. Then there exists `A ∈ R^{m×m}` with

    Ψ(f(x)) = A · Ψ(x)        exactly, for all x,

i.e. *exact finite linear dynamics in the lifted coordinates*, however nonlinear f is.
The entire practical question is whether a small, meaningful Ψ is (approximately)
invariant.

### 2.3 Data-driven estimation: (E)DMD

Given snapshot pairs `(x_k, x_k')` with `x_k' = f(x_k)`, fit

    A* = argmin_A Σ_k ‖Ψ(x_k') − A·Ψ(x_k)‖²
       = G' · G⁺,      G = Σ_k Ψ(x_k)Ψ(x_k)ᵀ,   G' = Σ_k Ψ(x_k')Ψ(x_k)ᵀ.

This is the same ridge/normal-equation machinery as `T_sec = C_Δδ C_δδ⁻¹` (007 §3.5),
living in observable space. With the trivial dictionary `Ψ(x) = x` this is DMD — and,
applied across layers, it is exactly the correlational (tuned-lens-style) state
predictor. **The entire potential gain over what already exists is carried by the
nonlinearity of the dictionary.**

### 2.4 Adaptation to a transformer — and the honest deflations

"Time" is depth: `x_ℓ = h_ℓ` (residual at layer ℓ, one position), `f_ℓ` = block ℓ.
Three deviations from the textbook setting, each with a consequence:

1. **Non-autonomy in depth.** The 64 blocks are different maps applied once each, so
   there is a sequence `A_ℓ`, not one K. Consequence: the deep spectral theory
   (invariant spectra, ergodic averages) does **not** apply. What survives is per-step
   structure: an eigenvalue of `A_ℓ` near 1 means only "this observable barely changes
   at this step" — a closure/persistence statement, not a global invariant. Band-level
   products `A_{m}⋯A_{ℓ}` form a cocycle of small matrices (Oseledets analysis becomes
   tractable here, unlike on 5120×5120 Jacobian products).
2. **Non-autonomy in position.** Attention injects other positions, so the
   single-position depth dynamics has external forcing. The standard fix is DMD with
   control (DMDc):

       Ψ(h_{ℓ+1}) ≈ A_ℓ·Ψ(h_ℓ) + B_ℓ·u_ℓ,

   with `u_ℓ` a feature summary of the attended context. The fitted `‖B‖/‖A‖` balance
   per layer is itself a content-vs-routing decomposition (ideas doc, Idea G).
3. **Dictionary dependence.** With a rich enough Ψ anything fits; the result is only
   informative if a *small, meaningful* dictionary approximately closes. The
   self-diagnostic is the closure residual (§3); if closure needs thousands of
   generic features, the answer is "not autonomous in interpretable coordinates,"
   which is itself the finding.

## 3. The one question this answers that nothing else on the board does

**Is the workspace autonomous?** The workspace paper establishes that verbalizable
content in mid-layers is causally load-bearing for flexible computation. It does not
establish whether that content *propagates itself* layer-to-layer (a medium with its
own closed dynamics) or is *re-derived at each layer* from non-verbalizable state.

Formal statement. Let `Ψ_tok(h)` be token/concept coordinates (e.g. reference-lens
logits for a fixed token set, baseline-z-scored per E6). Define the per-layer
**closure residual**

    ρ_ℓ = ‖Ψ_tok(h_{ℓ+1}) − A_ℓ·Ψ_tok(h_ℓ)‖ / ‖Ψ_tok(h_{ℓ+1})‖

(fit `A_ℓ` on generic text, evaluate held-out), and its forced variant ρ_ℓ^ctx with
the `B_ℓ·u_ℓ` term. Autonomy over the workspace band means: ρ_ℓ small and
`‖B_ℓ‖/‖A_ℓ‖` small for ℓ in the band, with a visible transition at the
sensory→workspace boundary (≈L20 on this model).

**Why it matters (decision consequence, not aesthetics).** For persona/anomaly
monitoring — the project's North Star:

- If token coordinates are approximately autonomous over the band, a monitor may
  legitimately track *only* workspace coordinates: cheap, deployable, and personas /
  weird cognition appear as trajectory anomalies in an m-dimensional space (m ≈ 10³),
  not a 5120-dimensional one.
- If they are not autonomous, any workspace-coordinate monitor is blind by
  construction — content is steered from outside the verbalizable subspace — and
  full-state instruments (NLA-class) are required.

Either outcome changes what monitoring system gets built. Secondary outputs: the
near-unit-eigenvalue count of `A_ℓ` as a per-layer "broadcast dimensionality" profile,
and an independent band detector to set against the paper's four (kurtosis, top-1
autocorrelation, effective dimensionality, CKA).

## 4. The Koopman lens (secondary, explicitly speculative)

Read layer ℓ by propagating observables, not states:

    scores(h_ℓ) = [A_{L−1} ⋯ A_ℓ · Ψ(h_ℓ)]_token block.

Nonlinear in h (via Ψ), linear and inspectable per step. Known weaknesses, stated up
front: correlational fit (tuned-lens pathologies — the M4 skip-ahead guardrail from
E2 is mandatory); readout restricted to the dictionary's token set (worse than
full-vocabulary lenses for discovery); multi-step products compound fit error. This
secondary use is only worth pursuing if §3 finds substantial closure.

## 5. Experimental plan (E7, when unparked)

Everything reuses existing infrastructure: `ekko.harness.capture`, the E6 calibration
stats, and the E2 metric battery.

**Step 1 — data (≈1 GPU-hour).** ~50 generic pile documents (disjoint from the 25
fitting docs and all eval items), capture `h_ℓ` at all 64 layers at positions 16..127
→ ≈5.6k snapshots per layer-pair.

**Step 2 — dictionary.** Primary: `Ψ_tok` = z-scored released-R lens logits for the
~1–2k most active tokens (reuses E6 stats verbatim). Ablations: (a) trivial `Ψ = h`
projected to 1–2k PCA dims (the "is nonlinearity doing anything" control — if this
closes equally well, the token-coordinate framing adds nothing); (b) + pairwise
products of the top ~40 coordinates; (c) cone/sparse-inference coefficients if the
sparse-frame readout exists by then.

**Step 3 — fits.** Ridge `A_ℓ` per layer (closed form, CPU-feasible); DMDc variant
with `u_ℓ` = attention-weighted context summary (needs one extra capture of per-layer
attention outputs). Train/held-out split by document.

**Step 4 — read the two curves.** ρ_ℓ vs depth (with the PCA-control curve overlaid)
and near-unit eigenvalue count vs depth. Also `‖B‖/‖A‖` vs depth for the forced fit.

**Decision rules (pre-registered here):**
- ρ_ℓ in the band ≲ 0.3 with a clear boundary transition, and the token dictionary
  beats the PCA control ⇒ autonomy supported → build the trajectory-anomaly monitor
  prototype on the blackmail scenario (connects to artifact 010).
- ρ_ℓ ≳ 0.7 everywhere, or PCA control closes equally well ⇒ token coordinates are
  not privileged / not autonomous ⇒ record the negative; the monitoring conclusion
  ("workspace-only monitors are blind") is the deliverable. Do NOT proceed to the
  Koopman lens (§4).
- Anything between: one dictionary iteration (products, cone coefficients), then stop
  regardless — the "keep enriching the dictionary until it closes" loop is exactly
  the vacuity failure mode of §2.4.3 and is capped at one iteration by design.

**Costs.** Step 1: ~1 GPU-h. Steps 2–4: CPU + <1 GPU-h. Total well under half a day.

## 6. Relation to prior work

- Tuned lens = DMD with the trivial dictionary, fit to outputs; this proposal differs
  by (i) nonlinear token-coordinate dictionaries, (ii) fitting *layer-to-layer*
  closure rather than layer-to-output prediction (which is what makes autonomy — not
  readout accuracy — the measured quantity), (iii) the DMDc forcing split.
- Koopman/DMD has been applied to NN *training* dynamics and to descriptive analyses
  of residual streams ("iterative inference" line). Transfer record to interpretability
  practice is mixed — mostly descriptive. This proposal survives that base rate only
  because its deliverable is a decision (monitoring design), not a formalism.
- The workspace paper's own band diagnostics are correlational statistics of single
  layers; ρ_ℓ is a *dynamical* statement about layer-to-layer propagation, which none
  of the four existing detectors measure.

## 7. Priors and the skeptical record (from the 2026-08-14 discussion)

Stated before any data, so the eventual result is read against them:

- P(at least one non-redundant number, i.e. the autonomy profile differs informatively
  from existing band metrics) ≈ 0.6.
- P(changes what we build next — via the monitoring-design consequence) ≈ 0.2–0.25.
- P(yields a better lens directly) — low; §4 is explicitly secondary.
- Named failure modes: dictionary vacuity (capped by the one-iteration rule),
  spectral pollution from finite data (use residual-validated DMD practices), the
  possibility that closure merely re-measures smoothness of the residual stream
  (the PCA control exists precisely to catch this).
- Deflations already conceded: the deep Koopman spectral theory does not apply to a
  64-step non-autonomous system; per-step near-unit eigenvalues are persistence
  statements measurable without operator language; the formalism's marginal value
  over "fit closure regressions and read the residuals" is organisational, not
  mathematical.

**Unpark condition:** after the E5 causal follow-up and the NLA three-instrument
comparison are done — or immediately, if a monitoring prototype becomes the priority
and the autonomy question blocks its design.
