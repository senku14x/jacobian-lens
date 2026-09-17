# 003a — Which phrase objective recovers the known late-layer phrase representation? Report

Run 2026-09-17 on Qwen3.6-27B (bf16, one A100-80GB). Design: `design.md` (agreed 2026-09-17). Code:
`scripts/003a_fit.py` (682 contexts, 5,456 scalar backwards at batch 1, ~50 min plus ~30 min of context building),
`scripts/003a_eval.py`, `scripts/003a_posthoc.py` (post hoc, see §5). Raw: `results/003a-phrase-objective/`
(`items`, `contexts`, `surprisal`, `reliability`, `readout` JSON; `analysis.md`; `posthoc.md`); gradients in `outputs/003a/`
(gitignored). Figure: `plots/003a-phrase-objective/reliability_and_readout.png`.

## 1. Question

For multi-token names whose latent representation exists (001/001f), which per-token gradient target (`logp`, `lin`,
`logit`, `odds`) yields a conditional vector `v_cond` that is reliable, reads latent two-hop states, and carries
information beyond the constituent-token J coordinates?

## 2. Scope

Six primary phrases (New Zealand, South Korea, San Diego, North Carolina; ice cream, credit card), 20 natural contexts
each plus 20 generic-medium contexts (random pile insertion, middle surprisal tercile: −10 to −13 nats per token, far from
natural); 11 siblings at natural regime (5–20 contexts); 13 nulls (two surprisal-matched non-name bigrams per primary,
5 of 12 within ±0.3 nats, plus junk). Layers L36/44/52/56/60. Reductions `at` (t′, pre-registered primary) and
`mean` (source-mean). Evaluation on 001's latent two-hop activations (final prompt token) and held-out emission
contexts. Baselines: first-token J, J-sum/J-mean, constituent-J subspace ceiling (leave-one-cue-out LDA on the family's
constituent lens logits), full-residual probe, template rows where they exist.

## 3. Results (numbers in `analysis.md`; all `at` reduction unless stated)

### 3.1 Reliability (split-half cosine of v_cond, n=20, natural regime)

| objective | primaries @L44 | primaries @L56 | nulls @L56 |
|---|---|---|---|
| logp | 0.10–0.59 | 0.15–0.85 | 0.85–0.97 |
| **lin** | 0.48–0.72 | **0.78–0.97** (5/6 ≥ 0.88; ρ ≥ 0.9 for NZ, NC, ice cream, credit card) | 0.89–0.99 |
| logit | 0.24–0.72 | 0.26–0.93 | 0.89–0.97 |
| odds | 0.22–0.57 | 0.22–0.86 | 0.87–0.98 |

- `lin` is the most reliable object at L56–60 and the only one that passes ρ ≥ 0.9 on several phrases at n=20; at L44
  nothing is reliable at this context count. `logp` behaves as the saturation control predicted (norm ratio 0.02–0.13 on
  primaries; lowest reliability).
- **Nulls are as reliable as or more reliable than real phrases for every objective**, including the non-saturating
  ones. Reliability is not evidence of phrase specificity.
- Cross-regime cosine (natural vs generic-medium) is 0.55–0.79 for `lin` at L56 and 0.14–0.69 for the others: the
  fitting distribution changes the vector substantially; `lin` is the most regime-stable.
- Positive control: `g₁^lin` (source-mean, 20 single-lag emission contexts) vs the released J row: cos 0.13–0.32 at
  L36 rising to 0.62–0.85 at L60. Estimator variance (20 single-lag contexts vs thousands of pairs on generic text;
  002: cross-prompt row cosine 0.45 at L36), not a bug (002 established the estimator at the fit's own shape).

### 3.2 Latent readout (fixed own vector of the phrase; mean pairwise AUC vs siblings)

| phrase | best `lin` own (L) | `lin` headline (unstable → 0.5), best | J-sum (same L) | constituent ceiling | full probe |
|---|---|---|---|---|---|
| New Zealand | 0.97 (L52, ρ 0.63) | 0.87 (L60) | 0.96–0.97 | 0.94–0.98 | 0.91–0.99 |
| South Korea | 0.95 (L52–56, ρ 0.88) | 0.75 (L60) | 0.98–1.00 | 1.00 | 1.00 |
| San Diego | 0.75 (L52, ρ 0.78) | 0.62 (L60) | 0.63–0.79 | 0.74–0.79 | 0.90–0.96 |
| North Carolina | 0.77 (L60, ρ 0.93) | 0.77 (L60) | 0.98–0.99 | 0.98–0.99 | 0.97–1.00 |

- Under the pre-registered headline (ρ < 0.9 counts as 0.5), **no objective beats J-sum on any family at any layer**.
  `lin` reaches J-sum's level only where it is unstable (NZ L52, SK L52–56) and falls well below where it is stable.
- Within-family z-scored accuracy for `lin`: 0.42–0.79 against chance 0.25–0.33; below J-sum's 001 numbers.
- **The constituent-J ceiling equals the full probe on three families** (NZ, SK, NC at L52–60), so on those families all
  phrase identity is already inside ordinary token coordinates. **San is the exception**: full probe 0.90–0.96 vs
  ceiling 0.74–0.79 and J-sum 0.63–0.79 at L52–60; there is linear phrase information beyond the constituent atoms.
  On San, `lin` own 0.75 / `odds` 0.84 at L52 sit between J-sum and the probe but do not close the gap and are not
  stable (ρ 0.78 / 0.22).
- `v_cond` lies almost entirely **outside** the constituent-J span (2–17% of its energy, all objectives, all four
  families). Combined with the readout, that energy is not phrase-discriminative.

### 3.3 Post hoc (§5): shared component and difference directions

- Cross-item cosine of `v_cond^lin` at L56: within-family siblings 0.39, between families 0.32, primaries vs nulls
  0.01, nulls among themselves 0.27; top principal component carries 30% of variance across the 30 item vectors;
  the mean vector has 0.41 of a typical item's norm. A large component is shared across unrelated phrases (and a
  different one across nulls).
- Using the difference direction v_a − v_b instead of the phrase's own vector raises `lin` to 0.97/0.97 (NZ L56/60),
  0.99 (SK L56), 0.83–0.86 (NC), 0.79–0.84 (SD L52–56), i.e. on par with J-sum for NZ/SK, still below J-sum for NC,
  and above J-sum but below the probe for San. Not pre-registered; noted as a lead, not a result.

## 4. Reading against the pre-registered table

- `logp` ≈ 0 / unreliable; `lin` stable at L56–60: saturation was the failure of 002's object. **Confirmed.**
- `lin` reads latent states but **not above J-sum or the constituent ceiling** under the headline. The pre-registered
  reading for "reliable but no better than the ceiling" applies, with the twist that the vectors are *outside* the
  constituent span rather than inside it: the extra energy is a context-generic, phrase-nonspecific component (§3.3),
  not new phrase information.
- Full probe ≫ constituent ceiling holds only for San. There, no objective closes the gap at n=20.
- Kill criterion (no non-saturating object reaches ρ ≥ 0.9 at n=20 on any geographic phrase at any layer ≥ L44):
  **not triggered** (`lin` reaches 0.93 on NZ and NC at L56).

**Verdict.** As a corpus-averaged direction fitted on 20 contexts, phrase-J with a non-saturating target is reliable
at L56–60 but does not add readout value over summing constituent-token J rows on these families, and the one family
where phrase-level information demonstrably exists beyond token atoms (San) is not recovered. The objects are
dominated by a shared component that the own-vector readout does not remove.

## 5. What this does not establish

- Not causal; nothing here tests intervention, which is where the paper's template result lives.
- n = 20 contexts; L44 and below are unresolved at this budget; three families have thin siblings (New Delhi 5,
  North Dakota 8, San Jose 14 contexts).
- The shared-component and difference-direction analyses are post hoc.
- One model, one regime for siblings, four families; San's probe advantage rests on 3 sibling pairs.
- The generic-medium regime is random insertion at −10 to −13 nats per token; it is not "natural but uncertain".

## 6. What changes downstream (for agreement)

1. **003b as planned is not justified** on readout grounds: on these families J-sum already reaches the constituent
   ceiling, which equals the probe except on San. A 40-concept readout benchmark would mostly re-measure that.
2. **The live question is intervention.** 001f showed one shared late-band phrase direction; 003a shows `lin` vectors
   are reliable there. Whether a `lin` (or difference) phrase vector can *swap* a latent intermediate where first-token
   and J-sum swaps cannot (the paper's Tchaikovsky result) is untested and is the only place the method can still add
   value. Proposal: pull the swap-validity harness (004 prerequisite) forward and run a small swap test on the four
   geographic families with `lin` difference vectors vs J-sum-composite vectors vs template rows where available.
3. If readout is pursued at all, target the San-type case: families where the full probe beats the constituent ceiling.
   A pre-screen over the 001 families with the ceiling-vs-probe gap as the admission statistic costs CPU only.
4. Any further phrase-vector work should remove the shared component (difference directions or projection out of the
   item-mean) as a pre-registered step, with the null items as the specificity reference.
