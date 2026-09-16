# 003 — Gradient-template 2×2 and the first multi-token readout

Status: **draft, 2026-09-16.** Awaiting agreement. Runs after 002 passes.

## 1. Question

Can corpus-averaged gradients of multi-token sequence verbalization (phrase-J objects) recover
latent multi-token intermediates better than first-token J-vectors, and does using a gradient
estimator on the template lens's own passages remove the template lens's skip-ahead?

## 2. Why now

This is the first real multi-token number and the cleanest test of the workspace paper's own
conjecture (Appendix: "modifying the template lens to include causal measurements" might fix
skip-ahead). The passages are released, so the estimator-vs-data-distribution decomposition costs
one backward per passage.

## 3. Design

### 3.1 The 2×2

| | w-conditioned passages (released) | generic contexts (regime i/ii) |
|---|---|---|
| mean-difference (whitened) | released template row t_w | — (no label) |
| gradient | **gradient template** (phrase-J regime iii) | phrase-J regime (ii) surprisal-binned |

Regime (ii) contexts: pile-10k positions binned by whole-phrase mean surprisal −(1/m) log P(w|c)
into three bins (low / medium / naturally plausible), ≥ 25 contexts per bin per phrase. Report
cos(v_w^(bin a), v_w^(bin b)) and cos(v_w^(iii), v_w^(ii)) per object: direction stability across
regimes is a primary result, not a side note.

### 3.2 Concept set (~40, tokenizer-verified, all in the template vocabulary)

- 10 single-token controls (frequent and rare).
- 10 ordinary 2-token concepts.
- 10 ordinary 3–4-token concepts.
- 10 prefix-collision family members from 001 (first-class stratum).

Phrase-fitting contexts (passages) and evaluation prompts are entity-disjoint.

### 3.3 Objects per phrase

`v_lin` (single tokens), `v_seq`, `v_cond`, `v_PB` (siblings from the family where one exists,
else the 5 most probable same-prefix completions from the model), `v_mean`; per-context gᵢ saved.
Contexts: start at 8, estimate split-half cosine ρ_w, add contexts (to 50) until ρ_w > 0.9 or the
cap; phrases that never stabilize are reported and excluded from accuracy, not silently dropped.

### 3.4 Evaluation set (Dataset A; admission is lens-independent)

- 59-item calibration slice (single-token, existing) for the single-token controls.
- New multi-token two-hop set: for every multi-token concept ≥ 5 prompts with the concept as
  unspoken intermediate and a defined counterfactual target within the same family or category.
  Mechanical admission: intermediate and aliases absent from the prompt; answer absent; identical
  template across counterfactuals; changing the cue changes intermediate and answer; no relation
  template naming the answer relation. Model correctness recorded as a covariate.
- **Held-out context style**: evaluation prompts are two-hop questions; fitting passages are
  narrative text. Skip-ahead is never measured on passage-style text.
- Dataset B (post hoc stratum): items where the single-token timing test (intermediate swap earlier
  than answer swap, released J) supports a separable intermediate. Reported alongside, never used
  for admission.

### 3.5 Readout scoring

- Fixed universe: the 13,731 template rows; every method scores every row (first-token J and
  summed-constituent J score a phrase by its first token / constituent sum).
- Calibration: for each vector and layer, μ_w, σ_w from 10k null pile activations; z-score
  (empirical-CDF variant if tails are heavy); calibration is done within phrase-length strata.
  Raw scores reported too.
- Metrics, per layer and best-over-band: exact-phrase rank; within-family accuracy
  (prefix-collision stratum); length-stratified rank; pass@10; intermediate-vs-answer trajectory.
- Skip-ahead, pre-defined: Δ_skip(ℓ) = rank(answer) − rank(intermediate) averaged over
  L8–L36; a method "skips ahead" if the answer outranks the intermediate at ≥ 50% of items there.
- Noise floor: split-half phrase vectors give a twin floor for every rank metric.

### 3.6 Methods compared

first-token J (released); constituent-aggregation baselines J-sum and J-mean (and R-sum/R-mean, logit-sum); released R (first token); logit lens; released
template t_w; gradient template (`v_seq`, `v_cond`, `v_PB` on passages); phrase-J regime (ii)
objects; word head (FF) as an optional readout-only baseline trained with same-first-token hard
negatives, if time permits.

## 4. What we will learn (pre-registered)

- Phrase objects must beat **J-sum/J-mean**, not only first-token J (001 establishes whether aggregation already suffices).
- `v_seq` beats first-token J on ordinary multi-token items but `v_cond` does not beat chance
  within families → the gain is first-token evidence; multi-token information is not linearly
  present in the gradient. Labelling branch.
- `v_cond` beats chance within families at workspace depth → the original J-lens's multi-token
  limitation is an output-basis limitation, not absence of linearly accessible phrase information.
  Proceed to 004 with `v_cond`/`v_PB` vectors.
- `v_PB` > `v_cond` within families → sibling contrast adds discriminative value; use `v_PB` for
  auditing-style phrase dictionaries.
- Gradient template shows less skip-ahead than the template row on the same passages → estimator
  is the cause; the paper's conjecture holds. If both skip ahead equally → data distribution is
  the cause; regime (ii) vectors become the default.
- Cross-regime cosine < 0.5 → phrase vectors are context-specific; the averaged-lens logic does
  not transfer to phrases without regime control.
- Template ≥ gradient on readout while gradient ≥ template on swap (004) → read/write dissociation
  reproduced; use each for its leg.

## 5. Cost

Passages: 30 phrases × ≤ 50 contexts × ~4 backwards ≈ 6k backwards; regime (ii): 30 × 75 × 4 ≈ 9k;
≈ 15k scalar backwards ≈ 15–20 min backward time plus forwards; evaluation forwards ≈ 300 prompts.
Under 2 h GPU including Σ-calibration nulls.

## 6. Kill criteria

Single-token controls disagree with 002's compat rows (implementation regression) → stop.
`v_cond` within-family accuracy at chance at every layer with ρ_w > 0.9 → phrase-J as a
direction-finder is dead for this model; write the negative with the 2×2 as the mechanism.

## 7. Outputs

`results/003-.../{vectors_meta.json, readout.json, stability.json, skipahead.json, analysis.md}`,
per-context gradients in `outputs/003/` (gitignored), plots of rank vs layer per method and
stratum, cross-regime cosine matrices, skip-ahead curves; `report.md`.
