# Plan decisions from the two review rounds (2026-09-16)

Condensed record of what changed in the plan after two external reviews of the ideas ledger. The
reviews themselves may be added verbatim as `docs/06a_review_1.md`, `docs/06b_review_2.md`.

## Adopted

- **Five phrase-J objects, saved separately, never conflated:** `v_lin` (exact compat: gradient
  of the linear functional (1+γ)⊙W_U[t] on the penultimate residual under the released
  source/target aggregation), `v_logit` (through the last block and final norm), `v_seq`
  (Σ per-token log-prob gradients), `v_cond` (log P(suffix | c, shared prefix), any prefix length
  k), `v_PB` (prefix-balanced: suffix vs sibling suffixes). Per-token gradients gᵢ are stored per
  context so seq/cond are recoverable for any family definition.
- **Exact compatibility is the gate (002):** cos > 0.999 and relative norm error < 1% against a
  fresh J fit on the same contexts, then cos > 0.99 against the released fp16 J.
- **Dataset A / Dataset B:** benchmark admission is lens-independent and mechanical (intermediate
  and answer absent from prompt, identical templates across counterfactuals, cue change changes
  intermediate and answer, no relation template naming the answer). The intermediate-vs-answer
  timing test is a post-hoc stratum, never an admission rule. Y1-as-admission dropped.
- **Prefix-collision entity families** (New York / New Zealand / …) built from
  `tok.encode(" " + phrase)` with recorded common-prefix length k; central benchmark stratum.
  First-token J is at chance within a family by construction.
- **Read/write duality as a hypothesis (004):** r = gradient covector; u = Σ_λ r is the
  minimum-Mahalanobis-cost perturbation with unit effect on r. Tested under both Euclidean and
  Mahalanobis budgets. Universal SS whitening dropped (would double-whiten gradient rows).
- **Σ infrastructure:** ≥ 500 × 128 pile tokens, two disjoint halves, shrinkage sweep
  λ ∈ {0.01, 0.1, 0.3}; a dual is usable only if stable across halves and λ. Σ_pile·t is the
  "pile covariance-dual", never called the template mean-difference (Σ_template not shipped).
- **Labelled discrimination in 001:** emission condition from held-out template passages; latent
  condition from two-hop prompts. Unlabelled pile positions are not a discrimination test.
- **Length calibration:** sum, mean, and z-calibrated scores within length strata; fixed
  13,731-row universe for all methods.
- **Split-half reliability per phrase before any accuracy**, with adaptive context counts.
- **Cross-regime cosine** (passages vs surprisal-binned generic contexts) is a primary result.
- **Skip-ahead pre-defined:** Δ_skip = rank(answer) − rank(intermediate) over L8–L36, on held-out
  prompt-style contexts only.
- **Word head (FF):** readout-only baseline with same-first-token hard negatives; not causal.
- **JJ (local re-rank):** ceiling/forensics instrument, not a lens.
- **Reportability:** lens score, spontaneous rate, and elicited rate reported separately.
- **DDD removed:** Qwen3.6-27B has no MTP module.
- **Stein diagnostic (K) deferred to 001b** with matched estimands and ESS reported.
- **Lemma/concept merging:** report exact-phrase rank and class rank separately.

## Added after review round 3 (001 v3)

- **Cross-validated linear-probe ceiling** (Σ+λI)⁻¹(μ_a−μ_b) alongside the template direction, so
  a template failure is not read as absence of linear information. Template-low / probe-high is the
  highest-value phrase-J target.
- **Emission-natural** condition from corpus occurrences (marked missing when unavailable, never
  replaced by synthetic text); emission-template = held-out passages with a prefix-leak filter.
- **Covariance-aware pair geometry** ρ_ab and D_ab; Euclidean cosine alone is not "same direction".
- **Constituent-aggregation baselines** J-sum/J-mean (and R, logit) in 001 and 003.
- **≥ 20 latent prompts per member** from ≥ 5 independent clue routes; leave-one-route-out AUC;
  correctness as a post-hoc stratum.
- **Covariance halves split by sequence**; frequency-stratified row sampling for stability checks.
- **Template scoring convention locked** before any full-universe rank; within-family accuracy is
  primary.
- Template vocabulary is lowercase-normalized (2,647 rows multi-token only for that reason); flagged
  and checked in 001.

## Learned in 001 (2026-09-16)

- Random k-fold CV is invalid for frame-matched latent designs (identical frames across classes):
  use grouped leave-one-cue/frame/route-out. Pooled out-of-fold scores need fold-standardization.
- Released template passages leak the first word of multi-word phrases in 12–16% of cases; filter.
- Only United States / United Nations exists as a two-member family in the released template
  vocabulary; the vocabulary is lowercase-normalized (1,298 lowercase-artifact rows).
- `blackmail` is a single Qwen token. Los Angeles / Las Vegas share no token.
- Σ from 74k positions gives covariance-duals with split-half cosine 0.78–0.92; not usable for
  intervention yet.
- J rows and released template rows are near-orthogonal on Qwen3.6-27B (cos 0.01–0.15).
- Qwen3.6 emits `<think>` after `Q: … A:` frames; avoid that frame. The second-word-letter-count
  route does not elicit name content; drop it.

## Learned in 002 setup (2026-09-17)

- The final norm on Qwen3.6-27B is exactly (1+γ)·x/rms(x) (tested on the loaded module, max |Δ| 1e-6);
  q_t = (1+γ)⊙W_U[t] stands.
- **Correction to "DDD removed":** the checkpoint ships an MTP module (`mtp.fc`, `mtp.layers.0.*`,
  `mtp.norm`; config `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`). HF's
  `Qwen3_5ForCausalLM` does not load it, but the weights exist. It is one transformer block conditioned on the
  next-token embedding (DeepSeek-style), so it is a nonlinear conditional t+2 readout, not a linear word head.
  DDD is reopened at low priority, after 003.
- A dense scalar backward (`v_lin`) at batch 1 vs the prompt replicated ×4 differs strongly at early layers in
  bf16 (min cos 0.71 @L8, 1.00 @L60). This measures dense-cotangent graph-shape sensitivity, not J-row
  reproducibility (002 design, Amendment 2).

## Frozen after 003a + 003a-controls (2026-09-17)

- **Readout conclusions are closed.** On Qwen3.6-27B, late-layer (L52–62) multi-token latent identity is, for the
  tested families, compositionally recoverable from ordinary constituent-token J coordinates (constituent-J subspace
  ceiling = full probe on New, South, North; United marginal and messy); San is the single measured exception
  (residual +0.10 to +0.17). No generic Phrase-J readout benchmark (003b) will be run.
- **The mid-band probe-over-J gap is generic early-J**, not phrase-specific (single-token countries show the same
  +0.24 at L8). L8–L32 is not a Phrase-J target.
- **`lin` is the Phrase-J gradient object**; `logp` is a control only. `lin` converges at L52–60 with 33–60 contexts;
  it is a phrase-conditioned, lag-specific covector, distinct from a J row (target lag, not variance).
- **Every readout comparison is contrastive on both sides** (PJ diff vs J-sum diff); own-vector vs difference
  comparisons are not reported as headlines.
- **004 is make-or-break**: one swap operator (pseudoinverse two-coordinate patch) for every method, matched
  perturbation norm and damage, first-order local control first, harsh pre-registered criterion against J-const.

## Decided after 004 (2026-09-17)

- **Phrase-J as a general method is closed on this model.** Constituent-token J coordinates are the equal-or-better
  reader (003a, 003a-controls) and the better writer at matched norm and matched damage (004: PJ − Jc −0.33 to −0.61
  nats, CIs below zero; all-positions arm same ordering). The `lin` object is locally causal (gate slope 1.01, R² 0.97)
  but not a better edit direction for compositional phrases.
- **The San dissociation is the project's positive result**: on the one family where the full probe beats the
  constituent-J ceiling, the phrase-conditioned gradient is the only direction that moves the answer the right way
  (PJ − Jc CI-clear at every dose and both position conventions, up to +4.6 nats), and J-const does nothing.
  Read/write consistent; n = 11 items, one route, one family.
- **No two-layer edit flips answers** at ≤ 0.4·‖h‖ against a ≈ 5-nat clean margin; flips need a wider band and are
  not required for the comparison.
- **Next question is about concepts, not the lens**: how common are San-type concepts and what distinguishes them.
  Screen new candidate families by Δ_residual (CPU after one capture), then replicate 004 on hits.

## Order

001 template geometry → 002 exact compat → 003a objective screen → 003a-controls (Branch B) → **004 causal
geometry** (first-order control, swap harness, matched-norm comparison) → decide the project on 004.
