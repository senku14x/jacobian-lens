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

---

## Amendment 1 (2026-09-17, runtime rebuilt after crash; written while the run was in progress, before the gate line printed)

Changes to §4 made after the ideas-ledger review and the environment check (`scripts/000_env_verify.py`: a
batch-1 128-token graph from L8 peaks at 59 GB, so `dim_batch=8` cannot fit beside the 53.8 GB model):

1. **Fresh fit uses `dim_batch=4`** (env `DIM_BATCH`), with one automatic halve-and-retry on OOM. Recorded in `compat.json`.
2. **A bf16 batch-shape noise floor is measured before the gate.** `v_lin` (batch 1) is compared with
   `v_lin_batched` (the prompt replicated `dim_batch` times, i.e. the graph shape `jlens.fit` uses) for the
   first 3 compat tokens on the same 3 sequences; per layer, min cosine and max relative norm error are stored
   as `floor`. Rationale: the fit and the scalar backward run different graph shapes in bf16, so part of any
   discrepancy is precision, not implementation.
3. **Gate rule.** The design's rule (cos > 0.999 and relerr < 1% at every layer) is kept as the *strict*
   verdict and recorded (`gate_norm_strict_1pct`, and `gate_cos`). The gate that decides whether 003 may
   proceed is: cos > 0.999 **and** max relerr < max(1%, 3 × floor) at every layer.
   *Set after the floor printed but before the gate printed:* the measured floor is itself far below 0.999 in
   the early layers (min cos 0.71 at L8, 0.77 at L12, 0.87 at L16, 0.96 at L20–24, 0.97 at L28, 0.994 at L32,
   ≥ 0.998 from L36; max relerr 0.77 at L8 falling monotonically to 0.003 at L60). A cos > 0.999 gate is
   therefore unattainable at L8–L32 for reasons that have nothing to do with the code. The implementation
   verdict at a layer is: **pass if cos(v_lin, J_freshᵀq) ≥ floor cos at that layer (min over tokens vs min over
   the 3 floor tokens) and relerr ≤ 3 × floor**; the strict rule applies unchanged at layers where the floor
   itself satisfies it (L44–L60). A post-run check computes `v_lin_batched` for the 3 floor tokens and compares
   it with `J_freshᵀq` directly (same graph shape); that number is the cleanest implementation test at early
   layers and is reported alongside.
4. **Smoke additions.** `per_token_fast` (one forward, retained graph, m backwards) replaces `per_token`; on
   the first context of every phrase the two are compared (max |Δ log-prob|, min cos and max relerr of the
   source-mean gradients) and recorded in `smoke.json["fast_vs_slow_per_token"]`. Two null "phrases" are
   teacher-forced after New Zealand's 8 emission-natural contexts: the frequent non-word bigram ` of the` and a
   fixed junk pair (` spider`, `ünd`); their norm ratios and split-half cosines are the reference for the real
   phrases. `v_PB` is skipped (zeros) for nulls, which have no siblings.
5. **Extra records.** `compat.json` stores the first 80 characters of each pile prompt, `dim_batch`, the floor,
   and per layer the ratio ‖v_logit‖/‖v_lin‖ (the §7.6 norm-hygiene measurement).

Interpretation rule added: if the early-layer floor is confirmed by the post-run same-shape check, then a
single-prompt J row at L8–L28 in bf16 is not a reproducible object on this model, which bears on every
early-layer claim in the ekko-lens ledger and on the paper's "sensory band". That is an observation to carry
into 003's layer choices, not a claim of this experiment.
