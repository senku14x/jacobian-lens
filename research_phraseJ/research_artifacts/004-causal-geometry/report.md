# 004 — Causal geometry: report

Run 2026-09-17 on Qwen3.6-27B (bf16, one A100). Design: `design.md` v1 (agreed). Code: `scripts/lib/edit_harness.py`,
`scripts/004_first_order.py` (gate), `scripts/004_swaps.py` (primary arm + pile damage), `scripts/004_eval.py`,
`scripts/004b_allpos.py` (secondary all-positions arm). Raw: `results/004-causal-geometry/{first_order_control, items,
swaps, pile_damage, summary, swaps_allpos}.json`, `analysis.md`. Figure: `plots/004-causal-geometry/dose_response.png`.

**Design amendment recorded at run time.** The San family's members share every answer on the four 001 routes, so a
`state` route (California vs Texas; four frames) was added for San only; without it the one family with beyond-token
phrase information (003a-controls) could not enter the swap stratum. Admission rules were otherwise unchanged.

## 1. Question

At matched perturbation norm and matched damage, under one swap operator, does the phrase-conditioned lag-specific
gradient (`lin`, at t′) edit a latent intermediate A → B more selectively than the constituent-token J coordinates?

## 2. Scope

Gate: 4 phrases × 10 emission contexts × 2 layers × 21 perturbations. Items (Dataset C): 311 candidates from the 001
two-hop harness (New, South, North: continent/language/currency/hemisphere; San: state), pairs admitted only where
answers differ, items admitted only where the clean top-1 equals the first token of A's answer → **115 items** (New 40,
South 45, North 19, San 11), both directions. Operator: c = V⁺h, h′ = h + V(σ(c) − c), applied at L52 and L56
(clamped), primary position = final prompt token. Methods: PJ (v_cond^lin A, B), PJc (family-centred), Jc (prefix row +
two suffix rows, suffix coordinates swapped), random (3 seeds), Jfirst (identity by construction). Doses norm-targeted
at r = ρ·median‖h_L52‖, ρ ∈ {0.05, 0.1, 0.2, 0.4} (r = 6.5–52; native ‖δ(α=1)‖ medians PJ 7.2, Jc 9.9). Damage: 40
pile sequences, 4 representative pairs.

## 3. Results

### 3.1 Gate (first-order local control): PASS
Measured Δf vs predicted vᵀδ for ρ ≤ 0.02: L52 slope 1.006, R² 0.969; L56 slope 1.016, R² 0.978 (n = 600 each);
per-ρ slopes 0.98–1.02 out to ρ = 0.05; random directions predict ≈ 0 and give ≈ 0. The `lin` object is locally causal
on its own estimand.

### 3.2 Primary arm (final token, L52+56): J-const is the better write direction at matched norm and matched damage

Pooled selectivity Δlog P(Y_B) − Δlog P(Y_A), mean over 115 items, and the paired difference with bootstrap 95% CI:

| ρ | PJ | PJc | Jc | random | PJ − Jc | PJc − Jc | Jc − random |
|---|---|---|---|---|---|---|---|
| 0.05 | +0.04 | +0.09 | +0.37 | −0.01 | −0.33 [−0.50, −0.16] | −0.28 [−0.45, −0.11] | +0.38 [+0.20, +0.57] |
| 0.1 | +0.13 | +0.19 | +0.74 | 0.00 | −0.61 [−0.86, −0.40] | −0.55 [−0.79, −0.33] | +0.74 [+0.51, +0.99] |
| 0.2 | +0.30 | +0.11 | +0.74 | +0.01 | −0.44 [−0.90, −0.03] | −0.63 [−1.09, −0.22] | +0.73 [+0.33, +1.16] |
| 0.4 | +0.41 | +0.38 | +0.53 | +0.11 | −0.12 [−0.41, +0.20] | −0.14 [−0.50, +0.18] | +0.42 [+0.19, +0.65] |

- Jc beats PJ and PJc with CIs excluding zero at ρ ≤ 0.2; parity only at ρ = 0.4, where random also starts to move.
- Both are specific: PJ − random is positive at ρ ≥ 0.1; Jfirst is the identity (no effect) by construction.
- **Matched damage** (selectivity interpolated at common pile KL): at KL 0.003, PJ +0.18, PJc +0.17, Jc +0.71; at 0.009,
  PJ +0.31, Jc +0.53. Jc is also *less* damaging at matched norm (pile KL 0.003 vs 0.006 at ρ = 0.2; top-1 kept 0.99 vs
  0.97).
- **No method is a working swap at this position and band.** Top-1 flips ≤ 3% for all methods; the clean margin
  log P(A) − log P(B) is ≈ 4.9 nats (median) and the edits move 0.4–1.6 nats. Mechanism of the edits differs: at
  ρ = 0.2 Jc mostly suppresses A (−0.51) with modest B gain (+0.23); PJ is symmetric and smaller (+0.16 / −0.14).
- **Per family** (PJ − Jc, CI): New −0.26 to −0.63 (all CIs below 0); South −0.35 to −1.18 (below 0 at ρ 0.1–0.2);
  North mixed (−0.75 at ρ 0.05, +0.55 at ρ 0.2, else CI includes 0); **San +0.16, +0.46, +0.80, +2.25 with all four CIs
  above zero** (n = 11, one route). On San, Jc is at or below zero at every dose (−0.02 to −0.31) while PJ rises
  monotonically to +1.94.
- **Reversibility** (ρ = 0.2): Jc is sign-consistent in both directions on most pairs; PJ is asymmetric on several,
  notably every pair involving New Delhi (n = 5 contexts for its vector) and San Antonio ↔ San Diego (A→B −0.69,
  B→A +1.67). Thin sibling vectors are a real limitation of the PJ arm.

### 3.3 Secondary arm (all prompt positions, L52+56; per-position norm target; PJ, Jc, random × 2 seeds)

| ρ | PJ sel / flip / KL | Jc sel / flip / KL | random | PJ − Jc [CI] | per family PJ / Jc (New, South, San, North) |
|---|---|---|---|---|---|
| 0.1 | +0.22 / 0.00 / 0.03 | +0.81 / 0.01 / 0.12 | +0.02 | −0.59 [−0.84, −0.35] | +0.04/+0.65, +0.24/+1.36, **+0.91/−0.07**, +0.18/+0.35 |
| 0.2 | +0.54 / 0.00 / 0.10 | +0.79 / 0.01 / 0.17 | +0.08 | −0.25 [−0.64, +0.13] | +0.00/+0.65, +0.57/+1.27, **+2.01/−0.07**, +0.74/+0.44 |
| 0.4 | +0.92 / 0.02 / 0.41 | +1.34 / 0.03 / 0.24 | +0.31 | −0.43 [−0.90, +0.05] | +0.19/+1.48, +0.92/+1.64, **+4.62/−0.02**, +0.29/+1.13 |

- Editing every prompt position roughly doubles both methods' selectivity but does not change the ordering (Jc ≥ PJ
  pooled; CI-clear at ρ = 0.1) and **still does not produce flips** (≤ 3% for all methods at up to 0.4·‖h‖ per
  position over two layers). Random also grows (+0.31 at ρ = 0.4), so the largest dose is entering the nonspecific
  regime; PJ's item-KL (0.41) exceeds Jc's (0.24) there.
- **San**: PJ − Jc = +0.99 [+0.55, +1.46], +2.08 [+1.12, +3.15], +4.64 [+3.07, +6.36] at ρ = 0.1/0.2/0.4; Jc stays at
  zero on San at every dose and position set. The San dissociation is robust to the position convention.
- The flip failure is therefore not a position limitation within this band; it is that two-layer edits of this size
  do not overcome a ≈ 5-nat clean margin. The paper's swaps clamp across a wide layer band; a band sweep is the natural
  follow-up if flips are needed, but it would not change the PJ-vs-Jc ordering, which is stable across doses,
  positions, and damage levels.

## 4. Reading against the pre-registered table (§7 of the design)

- The gate passed; the harness is valid (random ≈ 0, Jfirst = identity, Jc − random CI-clear at every dose).
- **Pooled: "J-const better."** Under the design's harsh criterion this ends Phrase-J as the main method: for the
  families tested, constituent-token J coordinates are the better write direction at matched norm and at matched damage,
  as they were the equal-or-better read direction in 003a and 003a-controls.
- **San is the exception in the same direction as readout.** The one family where phrase information sits outside the
  constituent-J span (003a-controls: residual +0.10 to +0.17) is the one family where the phrase-conditioned gradient
  is the only direction that moves the answer the right way, with CI-clear margins at every dose. This is exactly the
  read/write consistency the project was looking for, but on one family, one route, eleven items.
- Neither method achieves flips at the final-token position over two layers, and the all-positions arm (§3.3) shows
  this is not a position limitation within the band: the clean margin (≈ 5 nats) exceeds what two-layer edits of
  ≤ 0.4·‖h‖ move. The PJ-vs-Jc ordering is unchanged by the position convention.

## 5. What this does not establish

- One model; four families; the San result rests on 11 items and a single new route.
- Edits at the final token only (primary); two layers; norm-targeted doses up to 0.4·‖h‖. Larger bands or doses may
  behave differently (see §3.3).
- PJ vectors for several siblings are thin (New Delhi 5, North Dakota 8, San Jose 14 contexts), which plausibly drives
  PJ's reversibility failures; not controlled for.
- Damage is measured on pile next-token distributions at the edited position; no downstream generation was scored.

## 6. Verdict and what changes

**Phrase-J as a general method (read or write) is closed on this model**: constituent-token J is the equal-or-better
reader and the better writer for compositional phrases. The surviving object is narrower and more interesting:
**for concepts whose phrase-level information lies outside the constituent-token coordinates (San-type), the
phrase-conditioned gradient carries information that reads (003a-controls) and writes (004) where token atoms do not.**
The next question is therefore not "improve Phrase-J" but "how common are San-type concepts, and what distinguishes
them" — a screen over many new candidate families by Δ_residual (CPU after one activation capture), followed by a
pre-registered replication of the 004 comparison on whatever that screen finds. If the screen finds none beyond San,
the project's honest close is: late multi-token identity on Qwen3.6-27B is compositional in J coordinates for reading
and writing, with a single documented exception.
