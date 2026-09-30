# phrase-lens (chen-jan/phrase-lens)

Commit ea85583 (single "initial research release", 2026-09-05). Clone: `/workspace/repos/phrase-lens`. A portable
extraction of another author's project. It is **not** the original workspace: CPU tests only, and no numerical
reproduction is claimed. **No results are shipped.** No open-source licence for the project code (see `LICENSING.md`);
the upstream WSB (MIT) and jlens (Apache) notices are retained.

## Methods [REPO] (Qwen3.6-27B; layer l = output of block l)
- **Template lens (rebuilt):** t_w = (Σ+λI)⁻¹(μ_w−μ), λ = ρ·tr(Σ)/d, ρ=0.001.
  - Exactly 200 passages per entry; 13,389 entries meet the bar (of 14,327).
  - 2.68M entry passages + 61.7k background passages. No held-out split in the production fit.
- **PJ-lens:** for phrase tokens, F_w(c) = Σ_i raw logit z[w_i] with earlier tokens teacher-forced (no log-softmax, no
  length normalisation).
  - Gradients averaged over 128 generic WikiText contexts in the released J aggregation geometry.
  - Single-token rows are exactly `W_U[token] @ J[l]`.
  - The library has 11,544 ST + 1,845 phrase rows; the causal bank has 283 cased entries.
- **Gradient-anchored template:** β_w = (Σ+λI)⁻¹(μ_w−μ+λ g'_w), with ρ=1, g' = unit(PJ row)·‖t_w‖. Controls:
  - ridge-only (g'=0);
  - random-prior (fitted data term retained, so **not** a chance control);
  - mean-difference;
  - an η sweep separating prior strength from ridge.
- **Readout:** template-shaped banks use cosine; token J uses final norm + unembedding.
- **Interventions:** normalized directions, all prompt positions during prefill, L24–47, α=1. The clamp is recomputed
  relative to the clean residual at every layer. Generated positions are not edited.
- **Workspace band diagnostics:** next-token agreement, kurtosis, persistence vs shuffled null, dimensionality.
  - The late transition is near L47–48; onset estimates range L20–36; L24 is a compromise.
  - The authors state the band choice was **not blinded** (an earlier band outcome was seen before the diagnostics).

## Benchmarks [REPO] (the valuable part for us)

| bank | ST | MT | metric |
|---|---|---|---|
| association | 100 | — | top-1 any-layer concept recovery |
| multihop readout | 100 | 100 | same (MT targets can include the **answer**, not only the intermediate) |
| multilingual | 100 | 100 | same |
| poetry | 100 | — | same (final prompt token, not the last-newline convention) |
| typo | 100 | 100 | same |
| directed modulation | 100 | 94 | rank-derived proxy over a mixed think/dont_think/suppress bank |
| **multihop swap + ablation** | 96 | 67 | target-answer success / source-answer survival (lower = stronger) |
| **flexible broadcast** | 192 (164 headline) | 192 (112 headline) | target-answer success across functions |

- The readout banks are byte-copied from WorkspaceBench `baseline_evals` @ 84e3a2d.
- The causal banks were drafted with Claude Opus help, then gated on Qwen3.6-27B: strict greedy correct + ≥8/10
  sampled at T=0.7.
- MT broadcast families: composers/thinkers, landmarks, dishes (cities-by-continent supplemental).
- Stats: Wilson intervals; macro = Bonferroni-adjusted Clopper-Pearson; `paired_bootstrap_diff` on aligned item IDs.
- Qualifications the authors list:
  - the causal vocabulary favoured template-covered concepts (a selection effect);
  - 16 MT `gate_scrutiny` flags;
  - broadcast trials share concepts, so trialwise CIs understate uncertainty;
  - a passed capability gate ≠ a correct clean answer under the intervention prompt.

## [MINE] takeaways
- **This is the swap/ablation/broadcast leg ekko-lens never built, ready-made, with random-direction controls and a
  Qwen3.6-27B capability gate.** It's CPU-inspectable (`phrase-lens benchmarks show …`). It pairs naturally with WSB
  (readout) and ekko M2/M4.
- Watch-outs:
  - MT multihop targets include answers, so the RHUT answer-steering trap applies. Score intermediates separately.
  - A band of L24–47 with α=1 at all positions is one convention; phrase_J 004 shows swap success is band-sensitive.
  - No shipped results: we don't know how PJ/anchored did in the original project, and the README carefully avoids
    headline claims.
- Its PJ objective (raw logit sum, 128 WikiText contexts) differs from phrase_J's `lin`/`logp` objects. Don't treat
  their PJ banks as the same object.
