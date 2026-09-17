# 004 — Causal geometry: does a phrase-conditioned lag-specific gradient edit a latent intermediate more selectively than constituent J?

Status: **draft v1, 2026-09-17 — awaiting agreement.** Supersedes the stub. Written after 003a-controls (Branch B).
This is the make-or-break experiment for Phrase-J: readout is settled as mostly compositional; the only remaining
hypothesis is that the phrase-conditioned gradient has different (better) *write* geometry.

## 1. Question

Given equal readout conditions and matched perturbation size, does the Phrase-J `lin` object (a phrase-conditioned,
lag-specific covector; 003a-controls step 5) produce a more selective latent-intermediate edit A → B than the
constituent-token J coordinates, under one and the same swap operator?

## 2. Why now

003a and 003a-controls: `lin` vectors converge at L52–60 (ρ 0.91–0.98), read at parity with J-sum, are outside the
constituent-J span, and are a different estimand from J rows (single target lag; cos 0.36–0.73 to the all-target
object). Two objects with equal readout can have different intervention geometry; that is untested and decides
whether Phrase-J exists as a method.

## 3. Objects and coordinate systems (all at L52 and L56; vectors from 003a / 003a-controls, natural regime, at t′)

| method | coordinate directions for pair (A, B) in family F with shared prefix p |
|---|---|
| **PJ-lin** | u_A = v_cond^{lin}(A), u_B = v_cond^{lin}(B) (suffix-conditional; n = 20, or 40–60 where available) |
| **PJ-lin centred** | same, minus the family mean over members (removes the shared component, 003a §3.3) |
| **J-const** (constituent J) | one coordinate per constituent token of A and B: Jᵀq_p (shared), Jᵀq_{a_suffix}, Jᵀq_{b_suffix}; the swap permutes the two suffix coordinates and leaves the prefix coordinate |
| **J-first** (negative control) | Jᵀq_p only: identical for A and B, so the swap is the identity; establishes that nothing happens without suffix information |
| **random, norm-matched** | isotropic random pairs with ‖δh‖ matched to each method's δh per item |
| template rows | none of the geographic pairs has both members in the released vocabulary; not run (recorded) |

A covector is a sensitivity, not necessarily the concept's activation representation; the design tests it *as* an
edit direction under a Euclidean budget and says so. A Mahalanobis-budget arm (Σ_λ u) is deferred: 001 found the
covariance duals unstable (split-half 0.78–0.92), so that arm would test the estimator, not the hypothesis.

## 4. The one swap operator (paper's lens-coordinate patch, generalised)

For a dictionary V = [u_1 … u_m] and a permutation σ that exchanges the A and B coordinates: c = V⁺h,
h′ = h + α·V(σ(c) − c). The component of h orthogonal to span(V) is untouched; correlated columns are handled by the
pseudoinverse (Gram matrix), so a shared prefix coordinate is not double-counted. α is the dose. Applied ("clamped")
at every layer in the band {L52, L56} at the chosen positions, with the lower-layer edit propagated and the upper-layer
edit re-applied to the propagated state. Every method uses this operator; only V changes.

Positions: primary = the final prompt token (the latent read position, matching the `at t′` estimand); secondary = all
prompt positions (the paper's convention).

## 5. Items (Dataset C; admission is mechanical and lens-independent)

- Source: 001's admitted latent two-hop prompts for the four geographic families (New, South, San, North), routes
  continent / language / currency / hemisphere, four frames, cue rotation.
- Pair (A, B) admitted for a route only if A's and B's answers **differ** on that route (the swap stratum; e.g.
  New Zealand → New York on continent and hemisphere, not on language or currency).
- Item admitted only if the clean model's top-1 next token equals the first token of A's answer (the model must be
  right before we try to change it). Reported: admission rate per family.
- Both directions (A → B on A-prompts, B → A on B-prompts) so reversal is built in.
- Expected size: ≈ 4 families × 2–3 sibling pairs × 2 routes × 4 frames × 2 directions ≈ 100–150 items.

## 6. Measurements (per item, method, dose α ∈ {0.5, 1, 2, 3}, band L52+56, position = final token)

1. **Target**: log P(first token of Y_B) − clean; **top-1 flip** to Y_B's first token.
2. **Suppression**: log P(Y_A first token) − clean.
3. **Damage**: KL(clean ‖ edited) of the full next-token distribution at the edited position; and, on 100 held-out
   pile positions with the same-norm edit applied, top-1 retention and mean KL (unrelated-output preservation).
4. **Perturbation size**: ‖δh‖ per layer, reported; all cross-method comparisons at matched ‖δh‖ (interpolating each
   method's dose curve to a common norm grid) and at matched damage.
5. **Selectivity index** = Δ log P(Y_B) − Δ log P(Y_A) at matched ‖δh‖, per item; the primary statistic is its paired
   difference PJ-lin − J-const with a bootstrap CI over items (and over families).
6. **Reversibility**: B → A on B-prompts must produce the mirror effect; asymmetric methods are flagged.
7. **Local first-order control (positive control; run first).** On 10 emission contexts per phrase where the `lin`
   functional f = q_suffixᵀh_{62,t_i} is defined: perturb h_{ℓ,t′} by α·d for d ∈ {v̂, random} and α small; check
   Δf ≈ α·vᵀd (report the slope and R²). Pass: slope within 20% of 1 and R² > 0.9 for α ≤ 0.1·‖h‖. Fails → the object
   is not locally causal and the rest of 004 is not run.
8. **M4-style guard**: an edit that raises Y_B by lifting *all* answers (entropy collapse) is not a swap; report
   entropy change.

## 7. Pre-registered readings (harsh by design)

- PJ-lin (or centred) beats J-const on the selectivity index at matched norm **and** matched damage, CI excluding 0,
  in both directions, on ≥ 3 of 4 families → **Phrase-J survives as a write instrument**: "readout is compositional,
  phrase-conditioned gradients give better causal write geometry." Proceed to a 40-concept intervention benchmark
  (the 003b compute goes here instead).
- Parity (CI includes 0 on the pooled index and no family shows a CI-clear advantage) → practical case for Phrase-J is
  weak; keep `lin` as a diagnostic object, stop developing it as a method.
- J-const better → stop Phrase-J as the main method; write up 001–004 as: late phrase information is compositional in
  J coordinates for both reading and writing on this model.
- J-first (identity swap) or norm-matched random producing comparable flips → the harness is broken or the effect is
  nonspecific; fix before interpreting.
- Local first-order control fails → 004 does not run; report and reconsider the object.

## 8. Kill criteria

Fewer than 60 admitted items after §5; or the first-order control fails; or the random norm-matched arm flips ≥ 50% of
what J-const flips at matched norm (nonspecific regime).

## 9. Cost

Forwards only after the vectors: ≈ 150 items × 4 methods × 4 doses × (1 position set) ≈ 2.4k forwards, plus random
arms (×3 seeds) and pile damage checks ≈ 6k forwards ≈ 1 h on the A100. First-order control: ≈ 400 forwards.
Optional secondary all-positions arm doubles it.

## 10. Outputs

`results/004-causal-geometry/{items.json, first_order_control.json, swaps.json, analysis.md}`, dose-response plots per
method and family, matched-norm selectivity plot; `report.md`.
