# 002 — Phrase-J implementation validation: exact compatibility with the released J

Status: **draft, 2026-09-16.** Awaiting agreement. Runs after 001 (needs only its families file).

## 1. Question

Does a scalar-cotangent backward that reproduces the released fitting convention give, for
single tokens, the same row as the released J matrix? And do the five phrase objects compute
without error on a few multi-token smoke cases?

## 2. Why now

Every later phrase-J number rests on the implementation. If m = 1 does not reproduce the released
row, a phrase-J failure has five implementation explanations before any scientific one. This is
the gate.

## 3. Objects (saved separately; "phrase-J" never means an unspecified one)

For phrase w = (w₁…w_m) with common family prefix p = (w₁…w_k), context c, insertion at t′
(teacher-forced w at positions t′+1…t′+m−1), source position t ≤ t′:

| object | scalar target | note |
|---|---|---|
| `v_lin` | Σ_{j∈T} q_tᵀ h₆₂,j with q_t = (1+γ)⊙W_U[t], T = all valid target positions, then average over valid source positions exactly as `jlens/fitting.py` | single tokens only; the compatibility object |
| `v_logit` | logit_t(t′) through layer 63 and the final norm | single tokens; measures what the last block adds |
| per-token `g_i` | ∇ log P(wᵢ | c, w₍<ᵢ₎) at position t′+i−1, i = 1..m | stored per context; everything below is a combination |
| `v_seq` | Σᵢ gᵢ | complete-phrase verbalization |
| `v_cond` | Σᵢ>ₖ gᵢ = ∇ log P(w_{k+1:m} | c, p) | phrase information beyond the shared prefix |
| `v_PB` | ∇[ log P(w_{k+1:m} | c, p) − log Σ_{u∈S(p)} P(u_{k+1:} | c, p) ] | this completion vs siblings; needs one extra backward |
| `v_mean` | (1/m) Σᵢ gᵢ | length-normalized variant for calibration checks |

Implementation: `scripts/lib/phrase_objective.py` with an enum of these names; each saved gradient
carries its objective name, target layer, mask convention, dtype, and context id in metadata.

## 4. What we run

**A. Estimator-matched compatibility (the gate).** 3 pile sequences; for 30 single tokens
(frequent, rare, multilingual, punctuation) compute `v_lin` by scalar backward under the exact
released convention (`target_layer=62`, `skip_first=4`, cotangent at all valid targets, average
over valid sources, bf16 model, fp32 accumulation). On the same 3 sequences run `jlens.fit` to get
a fresh J (≈ 10 min). Compare `v_lin` to `J_freshᵀ q_t` row by row: cosine and relative norm error
‖v − Jᵀq‖/‖Jᵀq‖, per layer on the 13-layer grid. Then compare to the released fp16 J on its own
25 sequences using the released `n_prompts` averaging.

Pass: cos > 0.999 and relative norm error < 1% against the fresh J (implementation); cos > 0.99
against released J (storage + precision). Fail → stop, fix, rerun.

**B. `v_logit` vs `v_lin`** on the same tokens: cosine per layer. Reported, not gated.

**C. Multi-token smoke.** 5 family members from 001 (e.g. New Zealand, South Korea, San Diego,
ice cream, general relativity) × 8 contexts from their held-out passages (regime iii): compute
gᵢ, `v_seq`, `v_cond`, `v_PB` (siblings = the family), `v_mean`. Checks: shapes; `v_seq` equals
Σgᵢ to fp32 tolerance; ‖v_cond‖/‖g₁‖ reported (the norm-ratio diagnostic); split-half cosine over
4/4 contexts for each object.

## 5. What we will learn

- A passes → phrase-J numbers in 003 are attributable to the objects, not the code.
- A fails on cosine but not norm → mask/averaging mismatch; fails on norm only → dtype/scale;
  fails against released but not fresh → storage precision, acceptable if cos > 0.99.
- C: if ‖v_cond‖ ≪ ‖g₁‖ (ratio < 0.1) already on 8 contexts, note it; 003 decides.
- C split-half cosines < 0.5 on 8 contexts → phrase objects are noisy; 003 needs adaptive context
  counts from the start.

## 6. Cost

Fresh 3-sequence J fit ≈ 10 min; scalar backwards: 30 tokens × 3 seqs + 5 phrases × 8 contexts ×
~4 objectives ≈ 300 backwards ≈ 1 min. Total < 20 min GPU.

## 7. Outputs

`results/002-phraseJ-exact-compat/{compat.json, smoke.json, analysis.md}`;
`plots/002-.../compat_cos_norm_by_layer.png`; `report.md`.
