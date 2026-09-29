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

## Cost
100 items × 5 layers × 64 forward pairs of 128 tokens through ≤ 22 fp32 blocks. Estimated after 000's measured
throughput; if it exceeds 2 GPU-hours, drop to layers {44, 52} and state it.
