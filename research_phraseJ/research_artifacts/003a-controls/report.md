# 003a-controls — report

Run 2026-09-17. Design: `design.md` (agreed). Code: `scripts/003a_controls_cpu.py` (steps 1, 3), `scripts/003a_controls_gpu.py`
(steps 2, 4, 5; 5 min GPU). Raw: `results/003a-controls/` (`symmetric_2x2`, `residual_screen`, `single_token_control`,
`convergence`, `positive_control_decomp` JSON; `analysis_cpu.md`, `analysis_gpu.md`). Step 6 is a wording decision (design).

## Answers to the three uncertainties

### 1. Is the comparison symmetric? No; corrected. On the matched comparison Phrase-J is at parity or below, except San at L52.

Latent pairwise AUC, `lin` at t′ (n=20), mean over sibling pairs:

| phrase | L | PJ own | PJ diff | J-sum own | **J-sum diff** | ceiling | probe |
|---|---|---|---|---|---|---|---|
| New Zealand | 56 / 60 | 0.68 / 0.87 | 0.97 / 0.97 | 0.84 / 0.86 | 0.96 / 0.96 | 0.95 / 0.94 | 0.99 / 0.99 |
| South Korea | 56 / 60 | 0.95 / 0.75 | 0.99 / 0.82 | 0.85 / 0.86 | 0.99 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| North Carolina | 52 / 56 / 60 | 0.64 / 0.66 / 0.77 | 0.89 / 0.83 / 0.86 | 0.69 / 0.80 / 0.68 | 0.98 / 0.99 / 0.99 | 0.98 / 0.99 / 0.99 | 0.97 / 0.99 / 1.00 |
| San Diego | 52 / 56 / 60 | 0.75 / 0.73 / 0.62 | **0.84** / 0.79 / 0.69 | 0.65 / 0.64 / 0.63 | 0.63 / 0.76 / 0.79 | 0.74 / 0.79 / 0.79 | 0.90 / 0.96 / 0.96 |

- Contrastive scoring raises J-sum by +0.1 to +0.3; the post hoc Phrase-J "difference" gain of 003a was largely
  restoring symmetry. On diff-vs-diff, Phrase-J is at parity (NZ; SK at L56) or below (NC; SK at L60).
- The one exception is San Diego at L52: matched Phrase-J 0.84 vs J-diff 0.63 and vs the constituent ceiling 0.74,
  still below the probe 0.90. One family, one layer, n=20 (ρ 0.78 there). A lead, not a result.

### 2. Is the mid-band gap multi-token-specific? No.

Six single-token countries in the same two-hop harness (90 prompts): Δ_single = probe − J at L8 = **+0.24**, L12 +0.18
vs Δ_multi (001) +0.24, +0.20. Identical where the gap is largest. At L20–28 Δ_multi (+0.16/+0.10) exceeds
Δ_single (+0.07/+0.03), but the single-token task is easier (probe 0.95 vs 0.82), so this is not evidence of phrase
specificity. At L52–62 the single-token task is at ceiling (1.00) and uninformative. Conclusion: the early probe-over-J
gap is a generic early-J property (002: same-token rows on different prompts have cos 0.03 at L8); it is not a
phrase-J target.

### 3. How many San-like families? One.

Δ_residual = full probe − constituent ceiling (leave-one-cue-out, bootstrap CI over cue groups), L ≥ 44:

| family | L52 | L56 | L60 | L62 |
|---|---|---|---|---|
| New | −0.02 [−0.06, +0.02] | +0.02 [−0.01, +0.04] | +0.04 [+0.02, +0.07] | +0.03 [+0.01, +0.05] |
| South | −0.01 | 0.00 | −0.01 | 0.00 |
| North | −0.01 | +0.01 | 0.00 | 0.00 |
| United | +0.04 [−0.01, +0.10] | +0.07 [+0.03, +0.11] | +0.05 [+0.03, +0.08] | +0.04 [+0.01, +0.08] |
| **San** | **+0.10 [+0.05, +0.17]** | **+0.13 [+0.09, +0.18]** | **+0.17 [+0.12, +0.22]** | **+0.15 [+0.11, +0.19]** |

At L8–20 every family shows +0.13 to +0.26, which step 2 attributes to the generic early gap. Late, San is the only
family with a substantive residual; United is marginal and is the family the 001 addendum flagged as not a clean
two-hop set (UN cues point to New York); New is at +0.02–0.04.

## Supporting results

### 4. Convergence (`lin` at t′, natural contexts; ρ / own / diff)

| phrase | n | L44 | L52 | L56 | L60 |
|---|---|---|---|---|---|
| New Zealand | 20 / 40 / 60 | 0.63/0.92/0.88 · 0.69/0.88/0.86 · 0.74/0.85/0.84 | 0.90/0.97/0.93 · 0.93/0.95/0.90 · 0.94/0.91/0.89 | 0.93/0.68/0.97 · 0.96/0.66/0.97 · 0.97/0.66/0.97 | 0.95/0.87/0.97 · 0.98/0.86/0.96 · 0.98/0.86/0.96 |
| South Korea | 20 / 33 | 0.63/0.78/0.84 · 0.70/0.79/0.83 | 0.86/0.95/0.96 · 0.91/0.93/0.95 | 0.88/0.95/0.99 · 0.94/0.94/0.98 | 0.92/0.75/0.82 · 0.95/0.80/0.86 |

Vectors converge at L52–60 (ρ 0.91–0.98) and the AUCs settle; the unstable-high L52 estimates regress modestly
(NZ own 0.97 → 0.91). L36–44 does not converge at n=60 (ρ 0.35 / 0.74).

### 5. Positive-control decomposition (first token; NZ / SK)

| | L36 | L52 | L60 |
|---|---|---|---|
| released J row vs all-target v_lin on 20 pile seqs | 0.99 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 |
| released J row vs all-target v_lin on 20 natural contexts | 0.78 / 0.85 | 0.93 / 0.96 | 0.99 / 0.99 |
| released J row vs single-lag g₁ (source-mean) | 0.13 / 0.21 | 0.29 / 0.32 | 0.70 / 0.64 |
| lag effect (single-lag vs all-target, same contexts) | 0.36 / 0.38 | 0.47 / 0.44 | 0.73 / 0.69 |
| context effect (all-target natural vs pile) | 0.78 / 0.85 | 0.93 / 0.96 | 0.99 / 0.99 |

Twenty generic sequences reproduce the released row to 0.99+; the low 003a positive control is the **target-lag
definition** (single position at t_i vs the fitter's sum over all t′ ≥ t), with a secondary context effect that fades
by L52. The 003a `lin` phrase object is a lag-specific covector, a different estimand from a J row, and less reliable
(single-lag split-half 0.44–0.97 vs 0.86–1.00 all-target). Not variance.

### 6. Constituent-span geometry

The ceiling features are ⟨h, Jᵀq_t⟩ / rms(Jh); the raw covector span used in 003a is exactly the feature-defining span
up to a per-sample scalar. 003a's "2–17% of v_cond energy inside the constituent span" stands as stated.

## Decision (design branches): **Branch B**

- The mid-band gap is generic early-J (2). San is the only substantive late residual (3). Matched Phrase-J is at parity
  with J-sum where stable and above it only on San at L52 (1). The vectors do converge late (4), and the object is a
  lag-specific covector rather than a J row (5).
- **Drop the generic multi-token readout claim.** The honest readout statement: on this model, late-layer phrase
  identity is compositionally available from ordinary constituent-token J coordinates for the families tested, with San
  as the one measured exception.
- **Pull 004 forward.** The remaining value question is intervention geometry: does a `lin` difference vector (or a
  family-centred `lin` vector) swap a latent intermediate more selectively than a J-sum composite or a template row,
  where first-token rows cannot? Test on the four geographic families at L52–56 with dose-response, norm-matched random
  controls, unrelated-output preservation, reversal, and the constituent-J intervention as the direct comparator.
- San remains the single San-type case; any 003a2-style readout follow-up needs more such families first (a CPU screen
  over new candidate families by Δ_residual, before any gradient is computed).

## What this does not establish

- Not causal. n = 20–60 contexts; four (readout) / five (screen) families; one model; single-token control is at
  ceiling late and easier than the multi-token families mid-band.
- The San L52 matched advantage rests on three sibling pairs at ρ 0.78.
