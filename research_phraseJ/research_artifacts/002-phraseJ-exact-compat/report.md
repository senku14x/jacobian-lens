# 002 — Phrase-J implementation validation: report

Run 2026-09-17 on Qwen3.6-27B (bf16, one A100-80GB). Design: `design.md` with Amendments 1–2 (both written
before the corresponding results printed; commits `52286af`, `fd61167`). Code: `scripts/002_phraseJ_compat.py`
(compat + smoke, 86 min wall-clock, of which the 3-sequence fresh fit at `dim_batch=4` took 71 min),
`scripts/002b_same_shape_check.py` (decomposition + row reproducibility, 4 min). Raw:
`results/002-phraseJ-exact-compat/{compat,same_shape_check,smoke}.json`, per-context gradients in
`outputs/002/` (gitignored). Figure: `plots/002-phraseJ-exact-compat/compat_cos_norm_by_layer.png`.

## 1. Question

Does a scalar-cotangent backward that follows the released fitting convention reproduce, for single tokens,
the row `J_freshᵀq_t` of a Jacobian fitted on the same sequences? Do the multi-token phrase objects compute,
and are they reliable on a handful of contexts?

## 2. Scope

3 pile-10k sequences (128 tokens, `skip_first=4`, target layer 62, 13 source layers L8–L60 step 4);
25 single tokens (frequent, rare, place names, punctuation; the multilingual/diacritic candidates were not
single tokens); fresh J fitted with `jlens.fit` at `dim_batch=4`; q_t = (1+γ)⊙W_U[t], the final-norm gain
verified on the loaded module (Amendment 2). Smoke: 5 family members × 8 emission-natural contexts plus two
null "phrases" (` of the`; junk pair) on New Zealand's contexts.

## 3. Results

### 3.1 Implementation gate (Amendment 2, step 1): PASS

- Fitter-row contraction Σ_d q_d J_fresh[d,:] vs `J_freshᵀq`: max relerr 1.4e-6 (identity).
- `v_lin` (batch 1) vs `J_freshᵀq` at L44–60: min cosine over 25 tokens 0.9999 (L44) and 1.0000 (L48–60);
  max relerr 0.016, 0.009, 0.006, 0.004, 0.003.
- Stronger than required: `v_lin_batched` (dense q, prompt replicated ×4 = the fit's graph shape) vs
  `J_freshᵀq` has median cosine **0.9999 at L8** and ≥ 0.9999 at every layer, relerr 0.016 → 0.002.
- The design's original all-layer strict rule (cos > 0.999, relerr < 1% at every layer) fails at L8–L40 on
  cosine and at L8–L44 on relerr; recorded as `gate_cos=False`, `gate_norm_strict_1pct=False`.

### 3.2 Numerical measurement (Amendment 2, step 2): the early-layer discrepancy is batch shape, only

Medians over tokens (3-sequence averages), from `same_shape_check.json` and `compat.json`:

| L | dense B=4 vs fit (cos / relerr) | dense B=1 vs fit | B=1 repeat | reps=1 path vs B=1 | one-hot row B=1 vs B=4 | same token, different prompts |
|---|---|---|---|---|---|---|
| 8 | 0.9999 / 0.016 | 0.928 / 0.41 | 1.0000 / 0 | 1.0000 / 0 | 0.993 / 0.13 | 0.03 |
| 16 | 0.9999 / 0.014 | 0.964 / 0.28 | 1.0000 / 0 | 1.0000 / 0 | 0.995 / 0.10 | 0.11 |
| 24 | 1.0000 / 0.010 | 0.993 / 0.12 | 1.0000 / 0 | 1.0000 / 0 | 0.999 / 0.05 | 0.25 |
| 36 | 1.0000 / 0.006 | 0.9995 / 0.030 | 1.0000 / 0 | 1.0000 / 0 | 0.9995 / 0.028 | 0.45 |
| 44 | 1.0000 / 0.004 | 0.9999 / 0.014 | 1.0000 / 0 | 1.0000 / 0 | 0.9999 / 0.015 | 0.61 |
| 56 | 1.0000 / 0.002 | 1.0000 / 0.004 | 1.0000 / 0 | 1.0000 / 0 | 1.0000 / 0.005 | 0.86 |
| 60 | 1.0000 / 0.002 | 1.0000 / 0.002 | 1.0000 / 0 | 1.0000 / 0 | 1.0000 / 0.002 | 0.95 |

Per-sequence (not averaged) B=1 vs B=4 dense floor from the main run: min cos 0.71 @L8, 0.96 @L20, 0.998
@L36, 1.000 @L60; max relerr 0.77 @L8 → 0.003 @L60.

- **Observation.** Dense-cotangent vs one-hot-cotangent backwards agree to 1e-2 relative error at every layer
  when run at the same batch shape. Two backwards at the same shape are bit-identical. The batched code
  path with `reps=1` equals batch 1 exactly. The dominant source of discrepancy is **batch size 1 vs 4**:
  bf16 kernels differ by shape, and the difference grows monotonically toward early layers (relerr 0.41 at
  L8 for 3-sequence dense averages; 0.13 for single one-hot rows). A residual 1–2% relative error remains
  between dense and one-hot contractions at the same shape at L8 (0.2% at L60).
- **Observation.** Single J rows are more shape-stable (cos 0.993 @L8) than dense contractions of 5120 rows
  (cos 0.928 @L8), consistent with contraction accumulating correlated per-row shape errors.
- **Observation.** Genuine per-prompt variation dwarfs shape noise: the same token's batch-1 row on two
  different sequences has cosine 0.03 at L8, 0.45 at L36, 0.95 at L60. Early-layer J rows are averages of
  nearly orthogonal per-prompt vectors (the ekko-lens F3 context-dependence result, seen from the row side).
- Released J (25 prompts, unknown `dim_batch`) vs `v_lin`: median cos 0.47 @L8 → 0.995 @L60. Reported as
  external consistency only; the gap is corpus size plus shape, not code.
- `v_logit,agg` vs `v_lin`: median cosine 0.78 @L8 → 0.93 @L56, norm ratio ≈ 0.19–0.22 at all layers. The
  last block + final norm + logit machinery changes the functional substantially; both objects are kept.

### 3.3 Smoke: phrase objects compute, but log-probability targets saturate

`per_token_fast` (one forward, retained graph) equals `per_token` (re-forward per token) exactly on all 7
items: max |Δ log p| = 0, min cos 0.9999997, relerr 0. Retained-graph backwards are safe for 003.

Per phrase (8 contexts, L56; log-probs are means over contexts; siblings = other family members):

| phrase | mean log p(suffix \| c, prefix) | target vs best sibling | ‖v_cond‖/‖g₁‖ | split-half cos seq / cond / PB (L56) | (L36) |
|---|---|---|---|---|---|
| New Zealand | −0.01 | −0.01 vs −9.97 | 0.002 | 0.69 / 0.26 / 0.76 | 0.35 / 0.02 / 0.21 |
| United States | −0.00 | −0.00 vs −9.95 | 0.001 | 0.32 / 0.32 / 0.29 | 0.10 / 0.10 / 0.10 |
| ice cream | −0.35 | −0.35 vs −15.9 | 0.047 | 0.86 / 0.66 / 0.00 | 0.46 / 0.37 / 0.01 |
| San Diego | −0.44 | −0.44 vs −4.78 | 0.082 | 0.50 / 0.42 / 0.54 | 0.13 / 0.18 / 0.19 |
| South Korea | −0.50 | −0.50 vs −6.71 | 0.034 | 0.76 / 0.63 / 0.86 | 0.37 / 0.20 / 0.52 |
| null: of the | −2.16 | n/a | 0.146 | 0.93 / 0.65 / 0.00 | 0.56 / 0.21 / 0.00 |
| null: junk | −18.8 | n/a | 0.249 | 0.92 / 0.88 / 0.00 | 0.37 / 0.19 / 0.00 |

- **Observation.** In emission contexts the suffix is nearly determined by the prefix (log p ≈ 0 for New
  Zealand and United States), so ∇log p ≈ 0: the conditional object `v_cond` has 0.1–8% of the first-token
  gradient's norm and split-half reliability 0.26–0.66. Both nulls have *larger* and *more reliable*
  `v_cond` (0.65, 0.88) because their suffixes are improbable. Norm and reliability track improbability, not
  concept content.
- **Observation.** `v_PB` (log-softmax over sibling completions) collapses the same way whenever the target
  beats its siblings by several nats (ice cream: 15.9 nats → split-half 0.00); it survives where siblings
  retain probability (South Korea 0.86, New Zealand 0.76).
- **Observation.** At L36 every object's split-half reliability falls to 0.1–0.5 on 8 contexts.

## 4. What this establishes, and does not

- **Established (implementation):** the scalar backward reproduces the released estimator's row exactly at the
  fit's own graph shape at every layer; 003 phrase-J numbers are attributable to the objects, not the code.
- **Supported claim (numerical, this model, bf16, one A100):** early-layer Jacobian rows and dense contractions
  are batch-shape dependent (relerr 0.13–0.41 at L8), deterministic within a shape, and the effect vanishes
  by ~L44. Not tested: fp32 (does not fit), other hardware, whether the effect is in the GatedDeltaNet
  fallback kernels or the attention blocks.
- **Observation (smoke, n=8):** log-probability-based phrase objects saturate in emission contexts. This is a
  property of the target, not of the model's representations; it says nothing yet about whether phrase
  information is linearly present in the gradient (003's question).
- Not established: anything about readout or intervention quality of phrase objects.

## 5. What changes downstream (proposed 003 Amendment 1, awaiting agreement)

1. **Non-saturating targets.** Replace log-prob suffix objectives with pre-softmax logit objectives:
   `v_cond,logit` = ∇ Σ_{i>k} logit_{w_i}(t′+i−1) and `v_PB,logit` = ∇[logit_suffix − logit_best-sibling]
   (a logit difference does not vanish at confidence). Keep the log-prob versions as recorded variants.
2. **Uncertainty-conditioned contexts.** Regime (ii) contexts binned by suffix surprisal become the primary
   fitting distribution for conditional objects; emission passages (regime iii) are kept for the 2×2 but
   their conditional objects are expected to be degenerate and are reported as such.
3. **One fixed graph shape.** Every fit and every phrase object in 003 is computed at the same batch size;
   objects compared across shapes carry the measured shape floor (relerr 0.41 @L8 → 0.004 @L56).
4. **Reliability-gated context counts.** Split-half cosine per phrase *per layer* with adaptive context
   counts; a layer is reported only where ρ > 0.9; expect mid-band to need far more than 8 contexts.
5. **Null phrases in every readout table** (non-word bigram; junk) as the reliability and norm reference.
6. `logging.basicConfig(level=INFO)` in fit-class scripts so `jlens.fit` progress is visible; `dim_batch=4` is
   the ceiling on this card (a batch-1 graph already peaks at 59 GB).
