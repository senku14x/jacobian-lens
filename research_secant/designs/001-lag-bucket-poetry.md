# 001 — Lag-bucket matrix-free lens on planned rhymes (II.5) (design, written before any run)

**Date** 2026-09-29 · **Model** Qwen3.6-27B · **Depends on** 000 (G0 must pass; G1 or the G2 fallback per 000's rules)
**Status** pre-registered

## Why
The J-lens reads WorkspaceBench poetry at ≈.07: the planned rhyme word is almost never surfaced at the newline ending
line one. J̄ sums a source position's effect over *all* later targets with triangular lag weights (lag δ weight ∝
123−δ), so it is dominated by short lags: next-token and format content. The rhyme word is emitted 8–12 tokens after
the newline. If the residual at the newline carries a representation that pushes the rhyme word at *that* lag, a lens
restricted to the matching lag window should rank it far higher. This is a cheap kill test of the idea that "J's
estimand, not the representation, hides planned content".

## What we want to find out
At the newline, does a lag-restricted matrix-free lens (J̄ restricted to target lags in a bucket) rank the committed
rhyme word higher than the full-window J̄ computed the same way on the same hosts?

## Estimator (no fitting)
- Hosts: fresh pile-10k docs 100–131 at full 128-token length, positions P = [4, 126].
- For host k and sign draw r: sₚ ∈ {±1} i.i.d. Perturb x_p ← x_p ± ε·sₚ·h for all p ∈ P, with fp32 blocks
  ℓ+1..62. Then Δy = y(+) − y(−).
- **Lag-δ estimate:** L_δ(h) = Σ_{t'∈P, t'−δ∈P} s_{t'−δ}·Δy_{t'} / (2ε·N_p). Its expectation is
  (1/N_p) Σ_p J_{p+δ,p}·h, the lag-δ slice of J̄h. Cross terms have zero mean.
- **Buckets:** B0 = {0}, B1 = [1, 3], B2 = [4, 8], B3 = [9, 16], B4 = [17, 32]. FULL = Σ over all δ ≥ 0, from the same
  forwards. It is reported as the sign-noise reference only.
- **FULL_U** = a uniform perturbation of every p ∈ P on the same hosts (one extra forward pair per host). It is the
  exact first-order J̄h, with no cross-lag sign noise. **FULL_U is the primary comparator.** (Amended before any run:
  the signed FULL carries cross-term noise, which would make it an unfairly weak comparator.)
- Readout: rank of token w under U·L(h), where U = (1+γ)⊙W_U. This is ranking-identical to W_U·norm(·).
- Draws: K = 32 hosts × R = 2 sign draws = 64 forward pairs per (item, layer).

## Items, target, positions, layers
- WorkspaceBench `evals/poetry/items.json` (100 couplets, gated on Qwen3.6-27B). The target is `intermediates[0]`
  (the committed rhyme word), scored as the single token `" " + word` when that is one Qwen token. Items whose rhyme is
  multi-token are reported separately and excluded from the primary metric; I expect most to be single-token.
- Read position: the newline token ending line one (WSB's `line_one_newline`), plain render, no chat template.
- Layers ℓ ∈ {40, 44, 48, 52, 56}.

## Baselines
Paired, on the same items and cells:
- FULL (the matrix-free J̄ on the same hosts; the primary comparator);
- J_CB standard readout;
- J_NP (Neuronpedia n=1000, the WSB J arm);
- logit lens.

## Metrics
- Primary: per item, best rank over layers of the rhyme token. Compare median log-rank, B3 vs FULL_U, paired (Wilcoxon
  signed-rank), and top-10 hit rate (any layer), B3 vs FULL_U (McNemar).
- Secondary: per-bucket curves of median rank vs δ; per-layer hit rates.
- Noise: split hosts into halves (16 each); report the split-half agreement of ranks for B3 and FULL.
- Qualitative: top-10 tokens per bucket at L48 for 10 randomly sampled items (seed 0) plus the 3 best and 3 worst
  B3-vs-FULL items. Saved to `results/001-*/qualitative.md`.

## Registered predictions and decision rule
- **User's prediction** (spec II.5): B3 (9–16) ranks the rhyme word well above J̄.
- **User's kill:** no bucket improves the rhyme word's rank.
- **My prior** that B3 beats FULL by the success criterion: 25%. The reason: phrase_J found lag-specific covectors are
  distinct from J rows and noisier, and a planned rhyme's representation in *generic pile* contexts may not push the
  word at a consistent lag.
- **Success:** B3 (or B2) top-10 hit rate exceeds FULL_U's by ≥ 0.10 with McNemar p < 0.05, AND median best-rank
  improves by ≥ 2×.
- **Kill:** no bucket improves the median best-rank over FULL_U by ≥ 1.5×.
- In between: report as inconclusive, and increase R before any claim.
- **Guardrail:** if a bucket's gain comes with the top-10 filling with generic tokens (the same tokens across items),
  report it. Measured as the fraction of items sharing ≥ 5 of their top-10 tokens.

## Amendment 1 (registered 2026-09-29 while run v1 was in flight, before any analysis)

Only the per-item log lines of the first ~10 items had been seen, at one layer. No aggregate was computed. The
amendment adds controls and changes no success or kill threshold. It was prompted by the question "how are we not
fooling ourselves?", which exposed four gaps in the original design.

1. **No validation of the lag estimator.** 000 validated FULL_U, not the sign-randomised lag decomposition, which is
   new code (a shift-direction or masking bug would silently scramble lags).
   - **Gate G_lag** (before the main run): at L48, for 2 item activations on 8 hosts, compute the exact per-bucket
     estimate by perturbing one source position at a time (all p ∈ P, ± pairs). Compare with the signed estimator at
     R ∈ {1, 2, 8}.
   - Pass: B3 cos(signed_R8, exact) ≥ 0.9, and agreement increasing in R (unbiased and noise-limited).
2. **No positive control, so a null would be uninterpretable.** The tokens of line two are known, and so are their
   lags from the newline (1 to lag_rhyme−1).
   - **PC:** for each bucket, the mean log-rank of line-two tokens whose true lag lies inside the bucket vs outside it
     (diagonal enrichment).
   - Pass: B0/B1 rank lag-1–3 tokens better than B3 does (at minimum). Otherwise a B3 kill is reported as "instrument
     insensitive", not "no plan".
3. **No specificity null.** A readout that boosts all short, common words would improve the rhyme's rank without
   reading the plan.
   - **Decoy null:** at every cell, rank all 100 items' rhyme tokens. Per item, AUC_own = the fraction of the other
     items' rhyme words ranked below its own rhyme word (0.5 = no item specificity).
   - **Added to the success rule:** B3's gain over FULL_U must also hold on per-item AUC_own (paired Wilcoxon,
     p < 0.05).
4. **Forking paths / replication.** Report everything separately on the two WSB sources: `jlens_source` (51,
   Anthropic-derived) and `staging*` (49). A success must hold in the same direction in both halves.

Engineering: save every readout vector (fp16, `outputs/001/vecs.pt`, gitignored), so later analyses are post hoc on
stored data rather than recomputed. Run v1 (`ranks.jsonl`, own-rank only) is superseded by v2 and kept as a record.

**Known limitation, stated before results.** Hosts are generic pile text. A rhyme plan may only drive the rhyme word
inside a couplet, so on web-text hosts its effect at lag ~10 may be ≈0 even if the plan is linearly present. A B3 kill
with a passing PC therefore licenses only "lag re-weighting over generic hosts does not surface plans". The follow-up
it would motivate is **domain-matched hosts** (other couplets as hosts, perturbed at their own newline). That would be
a separate design, shown to the user before any run.

## Amendment 2 (registered 2026-09-30, after stopping v1 at 101/500 cells, before running v2, no aggregate analysis)

Disclosure: two per-item log lines of v1 had been seen ("led" B3 rank 194,306; "game" B3 rank 51,720). The amendment
fixes an estimator defect found while writing the G_lag gate. It does not change the question, the predictions or the
success/kill thresholds.

1. **Defect in the all-positions sign estimator.** With every p ∈ P perturbed with sign sₚ, the bucket sum
   Σ_{t'} s_{t'−δ}·Δy_{t'} contains, for every t', the **self term** s_{t'−δ}·s_{t'}·J_{t't'}h.
   - J_{t't'} carries the residual identity path. It is typically far larger than any cross-position block
     J_{t'+δ, t'}.
   - These |B|·N_p self terms have random signs but nearly the same direction (≈ J_self·h). The noise is therefore
     unbiased but *structured*: a random-signed multiple of one high-norm direction, with amplitude ~√(|B|·N_p)·‖J_self h‖
     against a signal ~|B|·N_p·‖J_lag h‖.
   - Token ranks then depend on the sign of that noise.
2. **Replacement: the spaced-source estimator.**
   - Per host and draw, sources are at o, o+40, o+80, with o ~ U[4, 43] and independent signs.
   - Per source p: m_δ ← s_p·Δy_{p+δ}/(2ε) for δ = 0..32 (p+δ ≤ 126).
   - No target window contains another source, so there are no self terms. The residual leakage is from earlier
     sources at lags ≥ 40, with random signs and zero mean.
   - Bucket_B = Σ_{δ∈B} ((N_p−δ)/N_p)·m̄_δ, which is J̄'s own weighting of lag δ.
   - Budget: K = 32 hosts × R = 4 draws × 3 sources = 384 samples per lag, plus FULL_U (uniform, all positions) for the
     standard J̄h.
3. **Lag indexing made explicit.** y_{t'} predicts token t'+1. The rhyme token (index len(ids)) is therefore predicted
   at δ_item = len(ids) − 1 − nl ∈ [7, 11] (median 9): in B3 for δ_item ≥ 9, in B2 for 7–8. The success rule already
   read "B3 (or B2)", and stays unchanged.
   - Pre-registered secondary readout: the item-matched window W_item = [δ_item − 1, δ_item + 1], defined from each
     item's geometry, not tuned.
   - Positive control, with the corrected lags: the line-two token at index q is predicted at δ = q − 1 − nl. So the
     first word of line two is a B0 (lag-0) token, and the next three are B1 tokens.
4. **Matched comparator.** FULL32 = Σ_{δ=0}^{32} ((N_p−δ)/N_p)·m̄_δ, from the same samples as the buckets (the same
   noise level). FULL_U stays the primary comparator; it is conservative against the hypothesis, because it is less
   noisy than the buckets.
5. **Gate G_lag, revised** (L48, run before the main v2 run):
   - **A, implementation/leakage.** 2 item activations × 8 hosts × 2 draws. The spaced estimator vs exact single-source
     perturbations of the *same* (host, source) samples. Pass: cos ≥ 0.95 for B1–B4 and FULL32.
   - **B, documentation of the defect.** Split-half cos of B3 (16 vs 16 hosts) for the old all-positions estimator
     (K=32, R=2) vs the spaced estimator (K=32, R=4).
   - **C, reliability.** Spaced B3 split-half cos at the planned budget. If < 0.7, double R before the main run
     (pre-committed).

## Amendment 3 (registered 2026-09-30, after the G_lag gate and before the main run; the user approved the estimator change)

No v2 outcome data exists. The gate recorded estimator agreement and reliability only, no rhyme ranks. v1 is invalid
(gate B) and will not be analysed.

1. **Estimator: exact single-source.** Per host, 8 source positions stratified over [4, 126] (one uniformly random
   position per stratum of ~15 positions, seeded per item and layer). Each source is perturbed alone (± pair), and its
   Δy_{p+δ} is read for δ = 0..min(32, 126−p). There is no cross-source leakage by construction. That makes gate A
   moot, because this *is* the gate's reference.
   - m̄_δ is the mean over samples.
   - Buckets use J̄'s lag weighting (N_p−δ)/N_p.
   - 32 hosts × 8 = 256 samples per cell, plus FULL_U.
   - **Preflight (gate C):** at L48 on items 0–1, B3 host split-half vocab cos ≥ 0.7. Otherwise the positions per host
     double to 16 (pre-committed).
2. **Layers {40, 48, 56}.** These are the layers validated in 000. The run is resumable per (item, layer), so 44 and 52
   can be added later at incremental cost only.
3. **Primary analysis is per layer (replaces "best rank over layers").** Best-over-layers favours noisier readouts
   (more chances at a lucky rank), and the user's spec rules require per-layer reporting.
   - At each layer, compare B2 and B3 against FULL_U, paired on items (single-token rhymes):
     - log-rank, Wilcoxon signed-rank;
     - top-10 hit, McNemar.
   - Holm correction over the 6 tests (2 buckets × 3 layers).
   - **Success:** at some layer, a bucket passes both tests after Holm, with ≥ 0.10 hit-rate gain and ≥ 2× median-rank
     improvement. At the same layer it must also improve per-item decoy AUC_own over FULL_U (Wilcoxon, p < 0.05), and
     the direction must hold in both source halves.
   - **Kill:** no bucket at any layer improves the median rank over FULL_U by ≥ 1.5×.
   - The thresholds are unchanged from the original design.
4. **Secondary: cross-validated best layer.** Choose each item's layer on host-half A, score it on half B, and vice
   versa. WITEM and FULL32 are reported per layer.
5. Positive control, decoy null and source split exactly as in Amendment 1, with the lag indexing of Amendment 2.

## Amendment 4 (registered 2026-09-30, user decision, before any aggregate analysis)

The only look at v2 data was a user-requested descriptive glance at the first 4 items. No decision was taken on it,
and no threshold changes here.

1. **50 items instead of 100.** `results/001-lag-bucket-poetry/v2/subset50.json`: a seeded random sample (seed
   20260930) of 25 `jlens_source` items and 25 `staging`/`staging_b3` items. Stratification keeps the both-halves
   replication rule usable. 5 of the 15 already-computed items fall in the subset and are reused (seeds are tied to
   the bank index, so reused and new cells are consistent).
2. **Power, stated in advance.** At n≈50 the paired tests detect roughly ≥ 15–20-point hit-rate gains, not 10. The
   thresholds are unchanged.
   - Significant success: claim at the scoped level.
   - Effect size above the thresholds but not significant: "borderline". The remaining 50 are then added (resumable,
     no recomputation), and the analysis is rerun on 100.
   - Kill (effect-size based): unaffected by n.
3. **Poetry is no longer the only test.** Design 002 (future words in the model's own text) runs alongside, as the
   generality test of the same hypothesis. The interpretation of 001 is read jointly with 002 (see 002's decision
   table).

## Amendment 5 (registered 2026-09-30, user decision, after the run finished and before any analysis)

The data were complete (50/50). No aggregate had been computed or viewed; only the per-item log lines were visible. The
**decision rule (Amendment 3's per-layer primary) is unchanged.** Two established metrics are added as **reporting
metrics** for comparability with the field (the user wants what is "already out there"):

1. **Anthropic paper metric (§A.6).**
   - An item counts as recovered at k if the rhyme token is in the top-k at any evaluated layer (40/48/56).
   - pass@k for k ∈ {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}.
   - The normalized AUC of pass@k against log10 k over [0, 3] (trapezoid, divided by 3). Paired bootstrap (2,000
     resamples over items) for each arm minus FULL_U.
   - Caveat: any-layer favours noisier readouts. That is why it is reporting only.
2. **WorkspaceBench's own poetry protocol.** Each arm's top-10 tokens (with scores) at the read cell, for layers
   40/48/56, are judged by WSB's pinned poetry judge (`google/gemini-3.8-flash`, bank prompt version). An item passes if
   any of the 3 layers names the rhyme. Items = `subset50`.
   - Flagged as a **subset** of WSB's 11-layer grid: every arm, J included, is judged on the same 3 layers.
   - Arms: B2, B3, WITEM, FULL_U, J_CB, J_NP (standard readout), **J_NP_cos** (WSB's official J arm: cosine readout,
     raw W_U, recomputed from the saved vectors), LOGIT.
   - Paired McNemar of each arm vs FULL_U and vs J_NP_cos.
   - Reference floor: WSB's frozen poetry prompt-only baseline (0.71 on its 100 items) is quoted, not re-measured.

## Cost
100 items × 5 layers × 64 forward pairs of 128 tokens through ≤ 22 fp32 blocks. Estimated after 000's measured
throughput; if it exceeds 2 GPU-hours, drop to layers {44, 52} and state it.
