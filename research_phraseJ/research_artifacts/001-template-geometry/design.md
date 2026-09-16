# 001 — Template geometry and the linear ceiling for multi-token concepts

Status: **v3, 2026-09-16 — frozen pending go-ahead for code.** Changes from v2: cross-validated
linear-probe ceiling (blocking fix); natural-occurrence emission condition; covariance-aware pair
geometry; constituent-aggregation baselines; ≥20 heterogeneous latent prompts per member with
leave-one-relation-out; sequence-level covariance splits and frequency-stratified row sampling;
exact template scoring convention locked before ranks; passage prefix-leak filter; lowercase-vocab
check. Stein deferred to 001b.

## 1. Question

For multi-token concepts that share a tokenizer prefix: (a) is the distinction between family
members linearly detectable at all, under a supervised probe, at emission and as a latent
intermediate; (b) does the released template lens recover it; and (c) what is the geometry between
J-lens rows, template rows, and their pile covariance-duals?

## 2. Why now

Every multi-token method under consideration searches for a linear direction per phrase. This
experiment separates four states that imply different next actions: information not detectable
under these instruments; information linearly present but missed by the template estimator;
template sees it only during production; template sees it as latent content. If the template rows
cannot separate `New York` from `New Zealand` at emission, either the template estimator failed to
recover the distinction or the distinction is not linearly detectable under these conditions; the
supervised linear ceiling distinguishes these cases. The **template-low / probe-high** cell is the
highest-value target for phrase-J: it asks whether a causal gradient construction can recover
phrase-level information that exists linearly but a correlational template misses.

Σ infrastructure built here is reused by 003 (calibration), 004 (duals, Mahalanobis OOD), and the
swap certificate.

## 3. What we run

### 3.1 Inputs

- `templates+phrases_v3.safetensors` (13,731 × 64 × 5120, pre-whitened, `ridge_c=0.001`) and the
  `templates.safetensors` base stack; `template_words+phrases_v3.txt` for row → text.
- Released J and R lenses; logit lens (`use_jacobian=False`).
- Template passages: for every phrase we use, the **last 20%** (the build's held-out split) as
  `emission-template` contexts, after the leak filter (§3.3).
- `NeelNanda/pile-10k` (and, if occurrence counts are insufficient, a larger web-text sample to be
  approved separately) for: Σ_ℓ, μ_ℓ estimation; null activations; `emission-natural` contexts.
- Qwen3.6-27B, bf16, one A100.

### 3.2 Item families (`001_build_families.py`)

1. Tokenize each candidate phrase as `tok.encode(" " + phrase, add_special_tokens=False)`; group
   by common token prefix; record prefix length k and per-member suffix length. Assert identical
   `ids[:k]` across members. Also record the tokenization of the lowercase form, and flag any row
   where the template vocabulary's lowercase entry tokenizes differently from the natural
   capitalized form (the vocabulary is lowercase-normalized; 2,647 rows are multi-token only for
   that reason).
2. Two membership classes per family: **T** (in the template vocabulary, so a template row and
   passages exist) and **N** (not in the vocabulary; probe and phrase-J only). Families need ≥ 2
   members; families with ≥ 2 T-members are used for template evaluation, all families for the
   probe ceiling.
3. Candidates: New York / New Zealand / New Delhi / New Jersey; South Africa / South Korea / South
   Dakota / South Carolina; North Korea / North Carolina / North Dakota; United States / United
   Kingdom / United Nations / United Arab Emirates; San Francisco / San Diego / San Antonio / San
   Jose; Saint Petersburg / Saint Louis; Golden Gate Bridge / Golden Retriever; ice cream / ice
   age; credit card / credit score; general relativity / general election; World Cup / World War;
   hot dog / hot water; plus ordinary pairs without a shared prefix (blackmail / extortion) kept as
   non-family pairs. Extended by Vishesh.
4. Controls: 20 random same-prefix T-pairs from the vocabulary (excluding lowercase artifacts);
   50 random unrelated T-pairs (floor).

### 3.3 Contexts

- **emission-template**: held-out passages, dropping any passage whose trailing tokens match any
  prefix of the phrase's token sequence (12–16% of multi-word entity passages leak the first word).
  ≥ 30 per T-member after filtering, else the condition is marked insufficient for that member.
- **emission-natural**: naturally occurring positions immediately preceding the full phrase in
  the corpus, ≥ 20 per member where available; otherwise marked missing (never replaced by more
  synthetic text). Documents used here are excluded from the Σ corpus.
- **latent**: two-hop prompts, **≥ 20 per member** from ≥ 5 independent clue/relation routes × ~4
  surface paraphrases (e.g. capital → country → currency; landmark → country → language; city →
  country → hemisphere; national symbol → country → capital; historical clue → country →
  currency). Mechanical admission: intermediate string and aliases absent from the prompt; answer
  absent; identical template across family members; changing the cue changes the intended
  intermediate and answer; no relation template naming the answer relation. Greedy correctness is
  recorded as a covariate. Read positions: final prompt token and last content token.
- **null**: 10k random pile positions (for calibration and AUC floors).

### 3.4 Part A — vector geometry (CPU, all 64 layers, T-members)

Per pair (a, b) and layer: Euclidean cos(t_a, t_b); **on-distribution score correlation**
ρ_ab = t_aᵀΣt_b / √(t_aᵀΣt_a · t_bᵀΣt_b); **natural variance of the difference**
D_ab² = (t_a − t_b)ᵀΣ(t_a − t_b), reported relative to the null distribution of D² for random
unrelated pairs (Σ on the 13-layer grid; Euclidean cosine on all 64). Also cos(t_a, t_prefix) where
the prefix is a row, and cos(t_a, J_ℓᵀq_p) with q_p = (1+γ)⊙W_U[first token]. High Euclidean cosine
is not read as "same direction" unless ρ_ab is also high and D_ab is at the unrelated-pair floor.

### 3.5 Part B — labelled discrimination (GPU, 13-layer grid)

For each family and each condition (emission-template, emission-natural, latent), each layer:

| direction | definition | asks |
|---|---|---|
| template | t_a − t_b | does the released template recover the distinction? |
| **probe ceiling** | (Σ_pile + λI)⁻¹(μ_a − μ_b), class means on training folds, 5-fold CV; λ from {0.01, 0.1, 0.3}·τ chosen on training folds only | is the distinction linearly decodable under a sensitive instrument? |
| first-token J | J_ℓᵀq_p (identical for all members) | chance within family by construction; floor |
| J-sum / J-mean | Σᵢ or (1/m)Σᵢ score of constituent token i via J | does aggregating constituent evidence suffice? |
| R-sum / R-mean, logit-sum | same with released R and logit lens | |

Metrics: pairwise AUC of the scalar score on held-out contexts (folds by context; for latent, also
**leave-one-relation-route-out**); within-family accuracy (argmax over members); paired bootstrap
CIs over contexts (2,000 resamples). Stratified afterwards by greedy correctness; never selected on
it. Full-universe rank (13,731 rows) is **secondary** and computed only under the released
implementation's exact scoring convention (§3.7).

### 3.6 Part C — covariance infrastructure and instrument geometry

- Σ_ℓ, μ_ℓ from ≥ 500 × 128-token pile sequences on the 13-layer grid; halves A/B are **disjoint
  sequences**; pooled estimate stored fp32 in `outputs/`.
- Shrinkage Σ_λ = (1−λ)Σ̂ + λτI, τ = tr(Σ̂)/d, λ ∈ {0.01, 0.1, 0.3}.
- Stability: for every family member and for 200 single-token template rows **stratified by token
  frequency decile** (20 per decile, frequency from the pile corpus), report cos(Σ_A t, Σ_B t) and
  the norm ratio, across λ. A dual is usable in 004 only if cos > 0.95 at λ = 0.1 and stable across λ.
- Instrument geometry per layer: cos(J_ℓᵀq_t, t_w) and cos(Σ_pile J_ℓᵀq_t, Σ_pile t_w). The
  latter is the **pile covariance-dual** comparison, not a template mean-difference recovery.

### 3.7 Scoring convention (locked before any rank is computed)

Score s_w(h) is defined exactly as the released implementation computes it (to be read from the
`global_workspace.template_lens` code if available, else from the safetensors metadata): which
activation normalization/centering is applied, whether template rows are unit-normalized, and at
which residual (pre- or post-norm). This is recorded in `results/001/.../scoring_convention.md`
before Part B runs. Pairwise AUC is intercept-free and robust to this; full-universe rank is not.

## 4. What we will learn (pre-registered)

Thresholds: AUC ≥ 0.85 high; ≤ 0.65 low; "collinear" requires Euclidean cos ≥ 0.95 **and**
ρ_ab ≥ 0.95 **and** D_ab at the unrelated-pair floor.

| Template latent | Probe latent | Decision |
|---|---|---|
| high | high | strong phrase-J target; template already finds the distinction; phrase-J must match it causally |
| low | high | **highest-value phrase-J target**: estimator problem, not linear-access problem; these families lead 003 |
| high | low | inspect for template/data leakage before use |
| low | low | no detectable linear separation with these instruments and data; deprioritize as primary benchmark, keep a few as hard negatives |

Emission-template high but emission-natural low → template-distribution specialization, not robust
phrase geometry; weight natural and latent results over synthetic.

Latent AUC high pooled but low leave-one-route-out → direction tracks cue families (Wellington-like),
not a general intermediate; report both and use only route-robust families as primary.

J-sum/J-mean ≈ template on families → phrase instruments add nothing beyond constituent aggregation
on this model; phrase-J must beat J-sum, not first-token J, in 003.

## 5. Cost

Σ pass ≈ 3 min. Contexts: ≤ 20 families × 4 members × (30 + 20 + 20) ≈ 5.6k forwards ≈ 15 min.
Probe fitting and bootstraps on CPU. Under 1.5 h wall-clock.

## 6. Outputs

`results/001-template-geometry/`: `families.json`, `scoring_convention.md`, `geometry.json`,
`discrimination.json` (per family/condition/layer/direction: AUC, CI, accuracy, per-route AUC),
`covariance_stability.json`, `analysis.md`. `plots/001-template-geometry/`: pair geometry vs layer
(cos, ρ, D); AUC vs layer per condition and direction per family; covariance-dual stability vs λ.
`research_artifacts/001-template-geometry/report.md`.

## 7. Deferred to 001b

Stein diagnostic with matched estimands and ESS.
