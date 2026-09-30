# 001 — Lag-bucket (distance-matched) J-lens on planned rhymes: report

**Date** 2026-09-30 · **Verdict: KILL (pre-registered)** · **Design** `designs/001-lag-bucket-poetry.md`, Amendments 1–5,
each committed before the data it governs · **Code** `scripts/001_lag_bucket_poetry_v2.py` (run),
`scripts/001_analysis.py` (decision rule, committed before any aggregate), `scripts/001_reporting.py` (Amendment-5
reporting metrics) · **Raw** `v2/cells.jsonl`, `v2/summary.json`, `v2/reporting.json`, `v2/qualitative.md`,
`v2/wsb_readouts/`, `v2/wsb_judged/`; vectors in `outputs/001_v2_vecs/` (gitignored).

## Question
The J-lens almost never shows the rhyme word a model has planned (WorkspaceBench poetry: J ≈ .07). J̄ sums a state's
influence on *every* later token, weighted toward the next few. Does a lens that counts only influence at the rhyme's
distance (4–8 or 9–16 tokens ahead) surface the planned rhyme that J̄ misses? This is the user's spec II.5.

## Setup
- Qwen3.6-27B @6a9e13bd.
- **Items:** 50 WorkspaceBench poetry couplets, a seeded random sample stratified 25/25 by source. All gated on
  Qwen3.6-27B, all single-token rhymes. The rhyme is predicted 7–11 tokens after the read point.
- **Read point:** the newline ending line one (WSB's `line_one_newline`). Layers 40/48/56.
- **Estimator:** matrix-free J-lens (validated to reproduce J_CB at 0.997–0.999 in 000). 32 fresh pile hosts × 8
  stratified source positions, each perturbed alone (exact single-source), true fp32 through blocks ℓ+1..62. Distance
  windows B0 = 0, B1 = 1–3, B2 = 4–8, B3 = 9–16, B4 = 17–32, plus WITEM = the item's own rhyme distance ± 1.
- **Comparators:** ordinary J̄ on the same hosts with the same code (FULL_U, primary), J_CB, J_NP (n=1000), the logit
  lens, and (reporting) WSB's official J arm J_NP_cos.
- Nothing fitted. Total GPU ≈ 40 min for the 45 new items plus 5 reused.

## Observations
Median rank of the rhyme token (of 248,320), fraction of items with the rhyme in the top-10, and mean decoy AUC (0.5 =
not item-specific):

| layer | B2 (4–8) | B3 (9–16) | WITEM | FULL_U (J̄) | J_CB | J_NP | logit |
|---|---|---|---|---|---|---|---|
| L40 | 72,147 · 0.00 · 0.37 | 58,827 · 0.00 · 0.49 | 68,094 · 0.00 · 0.45 | 52,030 · 0.00 · 0.42 | 46,247 · 0.00 · 0.43 | 18,644 · 0.00 · 0.48 | 35,167 · 0.00 · 0.49 |
| L48 | 21,704 · 0.00 · 0.46 | 23,985 · 0.00 · 0.51 | 23,770 · 0.00 · 0.50 | 13,726 · 0.00 · 0.47 | 12,684 · 0.00 · 0.47 | 5,456 · 0.00 · 0.50 | 63,646 · 0.00 · 0.46 |
| L56 | 3,016 · 0.02 · 0.74 | 9,171 · 0.04 · 0.69 | 2,960 · 0.02 · 0.74 | 1,828 · 0.02 · 0.78 | 1,771 · 0.02 · 0.79 | 2,197 · 0.00 · 0.76 | 10,070 · 0.02 · 0.77 |

## Pre-registered decision (per layer, paired, Holm over 6 tests)
- **No window beats FULL_U at any layer.** Every window ranks the rhyme *worse*: the median-rank ratio FULL_U / window
  is 0.09–0.88 across all B0–B4 × 3 layers. **The kill rule (all < 1.5) is met.**
- At L48, B2 and B3 are significantly *worse* than FULL_U (Holm-corrected Wilcoxon p = 0.0007 and 0.011).
- Top-10 hit gains are 0.00–0.02 everywhere (the success rule needed ≥ 0.10).

## Controls
- **Positive control** (can the instrument read lag-specific content at all?). It passes at L48 and L56: B0 ranks the
  actual next word far better than B3 does (mean log10-rank difference −1.03 and −2.10, p ≈ 1e-12 and 1e-15), and B1
  ranks words 1–3 ahead better than B3 (p ≈ 5e-7 and 1e-8). At L40 it passes only partly (B0-vs-B3 p = 0.27), so
  **the L40 null is weak evidence.**
- **Reliability.** Split-half Spearman over items for B3 is 0.23 / 0.47 / 0.65 (L40/48/56), vs FULL_U 0.83 / 0.89 /
  0.98. The far windows are noisy, as the design anticipated.
- **Specificity.** At L40/48 the decoy AUC is 0.37–0.51 for **every** readout, including J_CB/J_NP. No lens ranks an
  item's own rhyme above other items' rhymes there. At L56 all readouts reach 0.69–0.79.
- **Guardrail.** The windows' top-10 are not dominated by shared generic tokens more than FULL_U's (the generic-slot
  share is lower for the windows).

## Reporting metrics (Amendment 5; established protocols; not the decision rule)
- **Anthropic §A.6** (pass@k at any of the 3 layers, normalized AUC over log k): B2 0.117, WITEM 0.100, J_CB 0.092,
  FULL_U 0.087, B3 0.083, J_NP 0.068, logit 0.033. B2 − FULL_U, 95% CI [−0.001, +0.066]: not significant. Any-layer
  favours noisier readouts (the reason it is not the decision rule).
- **WorkspaceBench's own poetry judge** (the same 3 layers for every arm): every arm passes 1–2/50 (B2 .04, B3 .04, WITEM
  .02, FULL_U .02, J_CB .02, J_NP .04, J_NP_cos (WSB's official J arm) .04, logit .04). Judge spend $0.52. Consistent
  with WSB's published J ≈ .07 over its full 11-layer grid.

## What this supports
- **Supported negative** (Qwen3.6-27B, these 50 items, layers 48/56 where the positive control passes): re-weighting
  J̄ by distance over generic web-text hosts does **not** surface planned rhymes. The distance windows are worse
  than J̄, not better.
- **Observation:** at the line-one newline in layers 40–48, *no* tested lens, J included, is item-specific for the
  planned rhyme (decoy AUC ≈ 0.5). Item-specificity appears only by L56, and even there no lens puts the rhyme in its
  top-10 more than 4% of the time.

## What this does not establish
- **That the rhyme plan is absent** from the activation. The hosts are generic web text; a plan that only acts inside
  a poem would not push the rhyme at lag ~10 in unrelated text. That was stated before the run. Domain-matched (poem)
  hosts are the untested follow-up.
- Anything at L40 (the positive control only partly passed), at other layers, on other models, or for other planned
  content (002 tests free text).
- Anything about other lens definitions (horizon, chat hosts), which are separate estimands.

## Decision
- Close "distance re-weighting over generic hosts" for planned rhymes.
- Poem-host lenses are a conditional follow-up, not scheduled. Given this result and 002 pending, the next priority is
  the brew write-direction test (003).
- 002 (future words in the model's own text) decides whether distance-matching fails generally or only for rhyme
  plans.
