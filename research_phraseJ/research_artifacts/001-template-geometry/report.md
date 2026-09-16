# 001 — Template geometry and the linear ceiling for multi-token concepts: report

Run 2026-09-16 on Qwen3.6-27B (bf16, one A100). Design: `design.md` (v3). Code: `scripts/001a–001e`.
Raw results: `results/001-template-geometry/` (`families.json`, `contexts.json`, `discrimination.json`,
`geometry.json`, `covariance_stability.json`, `template_full_universe_rank.json`, `qualitative_samples.md`).
Plots: `plots/001-template-geometry/`.

## 1. Question

For multi-token names sharing a first token (New York / New Zealand, …): is the distinction between
family members linearly detectable (a) when the model is about to emit the name and (b) when the
name is a silent two-hop intermediate; does the released template lens recover it; and what is the
geometry between J-lens rows, template rows, and their covariance-duals?

## 2. What was run (scope)

- **Families (tokenizer-verified, `" " + phrase`):** New {York, Zealand, Delhi, Jersey}; South {Africa,
  Korea, Dakota, Carolina}; North {Korea, Carolina, Dakota}; United {States, Kingdom, Nations, Arab
  Emirates}; San {Francisco, Diego, Antonio, Jose}; plus World {Cup, War}, hot {dog, water} (emission
  only) and the non-family pair photosynthesis / photograph. Dropped: Los Angeles / Las Vegas (no
  shared token); blackmail / blackboard (`blackmail` is a single Qwen token).
- **Only one family has two members in the released template vocabulary** (United States / United
  Nations). The released template arm is therefore evaluable on that family, the non-family pair,
  and the vocabulary controls only. The template vocabulary is lowercase-normalized; 1,298 of its
  rows are multi-token under Qwen only because of lowercasing (`" j"+"une"`).
- **Contexts.** Emission-template: held-out 20% of released passages, minus passages that leak the
  phrase's first word (12–16% of multi-word entity passages do; New Zealand drops to n=13).
  Emission-natural: pile-10k text immediately before the phrase, ≤128 tokens of left context,
  n=40 cap (n≥29 for all geographic members except New Delhi 10, North Dakota 16, UAE 10; those
  docs excluded from Σ). Latent: 349 admitted two-hop prompts, ~19 per member, 5 routes ×
  4 frames × rotating cues (`latent_facts.json`), phrase/alias/answer absent from the prompt.
- **Σ infrastructure:** 600 × 128-token pile sequences (natural-context docs excluded), disjoint
  halves A/B of 36,900 positions each, 13-layer grid L8–L60 step 4 + L62, shrinkage
  λ ∈ {0.01, 0.1, 0.3}. 10k null activations stored.
- **Instruments.** Probe: (Σ_λ+τλI)⁻¹(μ_a−μ_b), held out (emission: 5-fold with fold-standardized
  scores; latent: leave-one-cue-index-out primary, leave-one-frame-out and leave-one-route-out
  secondary). Template: t_a − t_b (released rows, δ=0 alignment). First-token J; J-sum / J-mean
  (exact readout logits of constituent tokens through released J, final norm with (1+γ)); R and
  logit-lens aggregates. Metrics: pairwise AUC (bootstrap CI over contexts / groups), within-family
  argmax accuracy under the same folds.

## 3. Results

### 3.1 Latent condition (silent two-hop intermediate) — the main result

Pairwise probe AUC, leave-one-cue-out (probe never saw the held-out surface cues), best layer:

| family | L | cue-out AUC [95% CI] | frame-out | route-out | released template | J-sum | J-mean | within-family acc: chance / probe / J-sum |
|---|---|---|---|---|---|---|---|---|
| New (4) | 56 | 0.98 [0.96, 1.00] | 1.00 | 0.98 | — | 0.96 | 0.96 | 0.26 / 0.83 / 0.71 |
| South (4) | 62 | 1.00 [1.00, 1.00] | 1.00 | 1.00 | — | 0.96 | 0.96 | 0.25 / 0.88 / 0.83 |
| North (3) | 60 | 1.00 [1.00, 1.00] | 1.00 | 0.99 | — | 0.99 | 0.99 | 0.33 / 0.89 / 0.91 |
| United (4) | 62 | 0.99 [0.97, 1.00] | 1.00 | 1.00 | 0.97 (US\|UN) | 0.89 | 0.91 | 0.28 / 0.81 / 0.72 |
| San (4) | 62 | 0.94 [0.89, 0.98] | 0.99 | 0.99 | — | 0.81 | 0.81 | 0.25 / 0.76 / 0.54 |

Layer profile, mean over the five families (probe cue-out vs J-sum pairwise AUC):

| L8 | L12 | L16 | L20 | L24 | L28 | L32 | L36 | L40 | L44 | L48 | L52 | L56 | L60 | L62 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.79 / 0.55 | 0.76 / 0.57 | 0.72 / 0.60 | 0.83 / 0.67 | 0.81 / 0.71 | 0.76 / 0.69 | 0.73 / 0.71 | 0.80 / 0.80 | 0.83 / 0.83 | 0.81 / 0.82 | 0.79 / 0.81 | 0.95 / 0.87 | 0.98 / 0.91 | 0.98 / 0.92 | 0.98 / 0.92 |

Read position −2 (token before the final one) is much weaker (best cue-out 0.51–0.63), so the
content is concentrated at the final prompt token.

### 3.2 Emission conditions

Emission-natural (best layer): probe AUC 0.92–1.00 for every family; J-sum 0.89–1.00; within-family
accuracy probe 0.70–0.94 vs J-sum 0.64–0.91 (probe higher in 6 of 7 families by 3–14 points).
Emission-template (T-members only): United probe 0.99, released template 0.83 (L56), template
within-family accuracy by cosine 0.97; photosynthesis / photograph: everything at 1.00.

### 3.3 Template geometry (Part A)

United States vs United Nations rows: Euclidean cosine 0.25–0.35 in the band, score correlation
ρ 0.34–0.43, D² above the unrelated-pair median. Not collinear by the pre-set rule. Random
same-prefix vocabulary pairs have median cosine 0.015 at L40; unrelated 0.000. Template layer
alignment: δ=0 (jlens block-output convention) by a small margin (0.080 vs 0.075 mean cosine).

### 3.4 Covariance infrastructure and instrument geometry (Part C)

- Covariance-dual stability, median over 200 frequency-stratified single-token rows:
  cos(Σ_A t, Σ_B t) = 0.78–0.92 across layers at λ=0.1 (0.83–0.92 at λ=0.3). **Below the 0.95 bar**
  at every layer. Duals are not usable as intervention vectors in 004 with this Σ.
- Released J rows vs released template rows: median cos(Jᵀq_t, t_w) = 0.01–0.15 across layers;
  pile-dual comparison 0.15–0.20. At L62 (J = I) the unembedding row and the template row have
  cosine 0.10. The template rows (ridge 0.001) live mostly in low-variance directions.

### 3.5 Qualitative (see `qualitative_samples.md`)

"Geography note: the city of Auckland can be found on the continent called" → model says ` Oce`;
J-lens top-8 at L56: ` Zealand`, `...`, ` Auckland`, `.nz`, `NZ`; template top-1 of 13,731: `New
Zealand` (cos 0.30). The suffix token of the latent name is promoted in the readout even though
the answer is a different word. The `second_word_letters` route produces no name content and a
blank next token; `Q: … A:` frames trigger Qwen's `<think>` token.

## 4. Reading against the pre-registered table

| | template latent | probe latent | decision |
|---|---|---|---|
| United | high (0.97) | high (0.99) | template already finds it; phrase-J must match causally |
| New, South, North, San | no rows | high (0.94–1.00) | probe-high with no template available; phrase-J target |
| mid-band L8–L48, all families | no rows | 0.72–0.83 vs J-sum 0.55–0.83 | **linear phrase information exists where no lens-style instrument reads it; highest-value phrase-J target** |

- **Supported claim (this model, these families, n≈19/member):** phrase identity within a
  first-token family is linearly decodable as latent content at L52–62 near ceiling, and it
  generalizes across surface cues, frames, and clue routes.
- **Supported claim:** constituent-token aggregation (J-sum) recovers most of it only at L52–62;
  in the mid band the probe leads J-sum by 0.1–0.25 AUC. The bar for phrase-J at late layers is
  J-sum, not first-token J.
- **Observation:** the one released template family works latently (0.97), consistent with the
  paper's Tchaikovsky/Beethoven result on an open model.
- **Observation:** J rows and template rows are nearly orthogonal; the whitening heuristic gets no
  geometric support here (caveat: Σ_template ≠ Σ_pile).

## 5. What this does not establish

- Not causal. Nothing here shows the probe direction is used by the model; 004 is the causal test.
- Cue-category confound at early layers: a member's five cues may share features (all UAE cues are
  Arabic names), so early-layer cue-out AUC may partly reflect cue category rather than the entity.
  The surface-form invariance test and route-diverse cues address this later.
- Small n (15–19 latent prompts per member, 3–4 per route); San's CI reaches 0.89.
- Low behavioral correctness (greedy 0.30–0.47) means the correctness stratum is thin; two frames
  (`Q:…A:`, `second_word_letters`) are poor and should be replaced in 003's item set.
- The released-template arm is one family; no claim about the template lens in general.
- One model.

## 6. What changes downstream

- 003 item set: keep the geographic families as the prefix-collision stratum; drop the
  `second_word_letters` route and the `Q:…A:` frame; add cues from more categories per member.
- 003 baselines: J-sum and J-mean at L52–62 are the bar; report mid-band separately.
- 004: increase Σ data (≥ 200k positions) or raise λ before using covariance-duals; add the
  Mahalanobis OOD check only once cos(Σ_A t, Σ_B t) > 0.95.
- 001b (Stein) remains deferred.

## 7. Methodological notes

- Random k-fold CV on the frame-matched latent design gives below-chance AUC (0.03–0.25 at L24)
  because held-out items' identical-frame partners remain in the other class's training mean.
  Grouped leave-one-frame/cue/route-out is required; recorded in `docs/06_plan_decisions.md`.
- Pooled out-of-fold scores from different discriminants must be fold-standardized before AUC.
- Passage prefix leakage (first word of a multi-word phrase at the passage tail) must be filtered.


---

## Addendum (2026-09-16, after external review): 001f transfer, and two corrections

### 001f — emission → latent transfer (`scripts/001f_transfer.py`, `results/.../transfer.json`, `plots/.../transfer_emission_latent.png`)

Direction (Σ_λ+τλI)⁻¹(μ_a − μ_b) fit **only on emission-natural contexts** (text right before the phrase is
spoken), evaluated with no refitting on the latent two-hop prompts; and the reverse.

| family (members) | emission→latent AUC [95% CI] (best L) | latent→emission best | multiclass emission→latent best (chance) | emission→latent mean over L8–L32 |
|---|---|---|---|---|
| New (3) | 0.98 [0.95, 1.00] (L56) | 0.98 | 0.74 (0.33) | 0.65 |
| South (4) | 1.00 [1.00, 1.00] (L62) | 0.99 | 0.87 (0.25) | 0.70 |
| North (2) | 1.00 [1.00, 1.00] (L62) | 1.00 | 0.92 (0.50) | 0.82 |
| United (3) | 0.96 [0.89, 1.00] (L62) | 0.97 | 0.78 (0.33) | 0.68 |
| San (4) | 0.96 [0.91, 0.99] (L62) | 0.89 | 0.55 (0.25) | 0.51 |

Layer profile of emission→latent pairwise AUC, mean over families (L8 … L62): 0.65 | 0.64 | 0.59 | 0.73 | 0.76 | 0.69 | 0.64 | 0.77 | 0.81 | 0.79 | 0.79 | 0.93 | 0.97 | 0.97 | 0.98.

**Reading.** At L52–62 a direction learned from "about to say the phrase" separates "silently
thinking the phrase" at 0.96–1.00, in both directions, for every geographic family. So at late
layers there is one shared linear phrase representation across emission and latent use, which is
the object phrase-J should match causally. In the mid band (L8–L32) transfer is 0.6–0.8, below the
in-condition cue-out probe (0.72–0.83): part of the early latent signal is entity information that
is not yet in verbalization form, consistent with the cue-category caveat. Supported claim, scoped
to this model, these families, n≈19 latent / 29–40 emission per member; not causal.

### Corrections to §3–§5

- **"frame-out" was mislabelled.** The reported `loo_frame` held out one (route, frame) *pair*, a
  paired same-text comparison, not a frame style across routes. It is now stored as `loo_rf`; a
  true `loo_frame` (hold out a frame across all routes) is reported alongside after the rerun.
- **"greedy correctness" is first-token match.** The field is renamed `first_token_match_rate`; it
  compares the model's first predicted token with the first token of the expected answer. Full
  answer generation is scored in 003.
- **United States / United Nations is not a clean two-hop family.** The UN cues ("the place where
  the Security Council is located") point to New York as the *place*, and the second-word-letter
  route targets the string "United Nations" while the frame asks about a place. The template AUC
  0.97 there is an observation that the UN-associated direction fires on UN-associated prompts, not
  a replication of the paper's latent-template result. The country and city families are the clean
  ones.
- **Mid-band wording.** The probe–J-sum gap is concentrated in L8–L32 (0.72–0.83 vs 0.55–0.71); by
  L36–48 J-sum has caught up (0.80–0.83 vs 0.79–0.83); after L52 both are strong. Phrase-J's
  opportunity is earlier than §4 stated, and at late layers its job is cleaner readout or better
  intervention, not recovering something J-sum cannot read.
