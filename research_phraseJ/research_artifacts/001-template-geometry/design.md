# 001 — Template geometry and the linear ceiling for multi-token concepts

Status: **v2 draft, 2026-09-16.** Supersedes `001-template-pretest` (renamed; the Stein diagnostic
moved to 001b; the unlabelled pile-position test removed). Awaiting agreement before code.

## 1. Question

Do the released template-lens rows for Qwen3.6-27B linearly separate multi-token concepts that
share a tokenizer prefix, (a) when the phrase is about to be emitted and (b) when it is a latent
two-hop intermediate that is never emitted? And what is the geometric relationship between J-lens
rows, template rows, and their covariance-duals?

## 2. Why now

Every multi-token method on the table (phrase-J objects, gradient template, word head) searches for
a linear direction per phrase. The released template lens is the best existing linear phrase
instrument on this model. If its rows cannot separate `New York` from `New Zealand` even at
emission, the whitened representation does not carry that information linearly and the programme
is a labelling problem. If it separates at emission but not latently, the template describes phrase
production and the latent question is still open for phrase-J. Either way this sets the ceiling
and the item set that 002–004 use. Cost is minutes of GPU.

The covariance infrastructure built here (Σ_ℓ with split halves and shrinkage sweep) is reused by
004 (dual write vectors), by the swap-validity certificate (Mahalanobis OOD distance), and by
calibration in 003.

## 3. What we run

### 3.1 Inputs (no fitting)

- `templates+phrases_v3.safetensors`: 13,731 rows × 64 layers × 5120, pre-whitened (`space=raw`,
  `ridge_c=0.001`). Σ_template is not shipped.
- Released J and R lenses.
- Template passages (`passages/*.json`): per phrase, ~300 passages cut immediately before the
  phrase; the build used the first 80% for estimation, last 20% held out. We use the **held-out
  20%** as labelled emission contexts.
- Qwen3.6-27B for: one forward pass over ≥ 500 × 128-token `NeelNanda/pile-10k` sequences (≈64k
  token positions) to estimate μ_ℓ and Σ_ℓ; forwards over held-out passages and two-hop prompts.

### 3.2 Item families

Built by a script (`001_build_families.py`) that:

1. Tokenizes each candidate phrase as `tok.encode(" " + phrase, add_special_tokens=False)` (the
   in-prose boundary convention) and groups by **common token prefix**, recording the common-prefix
   length k (≥ 1) and the per-member suffix length.
2. Keeps only members present in the template vocabulary (so a template row and passages exist).
3. Asserts `ids[0..k-1]` identical across every member of a family; drops and lists failures.

Candidate families (to be extended by Vishesh): New York / New Zealand / New Delhi / New Jersey;
South Africa / South Korea / South Dakota / South Carolina; North Korea / North Carolina / North
Dakota; United States / United Kingdom / United Nations / United Arab Emirates; San Francisco /
San Diego / San Antonio / San Jose; Saint Petersburg / Saint Louis; Golden Gate Bridge / Golden
Retriever; ice cream / ice age; credit card / credit score; general relativity / general election;
World Cup / World War; hot dog / hot water; blackmail / blackboard (if tokenized with a shared
prefix; else listed as "no shared prefix" and dropped from families but kept as an ordinary pair).

Controls: (i) 20 random same-prefix pairs sampled from the template vocabulary by the same
grouping (cherry-picking control); (ii) 50 random unrelated-row pairs (floor).

### 3.3 Part A — vector geometry (CPU, all 64 layers)

For every pair (a, b) within a family, per layer: cos(t_a, t_b); cos(t_a, t_prefix) where the
prefix string is itself a template row; cos(t_a, J_ℓᵀq_p) where q_p = (1+γ)⊙W_U[first token] is the
first-token J-lens direction. Report distributions for families vs random-same-prefix vs
random-unrelated.

### 3.4 Part B — labelled discrimination (GPU, standard 13-layer grid L8–L60 + L62)

Score s_ab(h) = (t_a − t_b)ᵀ h at the final position of each context. Two labelled conditions:

- **Emission.** Held-out passages for a (label a) vs held-out passages for b (label b), ≥ 40 per
  member. Metric: AUC of s_ab, per layer.
- **Latent.** Two-hop prompts we write, ≥ 5 per member, where the intended intermediate is the
  member and neither the phrase nor the answer appears in the prompt ("The currency of the country
  whose capital is Wellington is"). Read at the final prompt position and at the position of the
  last content token. Metric: AUC of s_ab per layer, plus the rank of each member's template row in
  the full 13,731 universe (exact-phrase rank) and the rank of the correct member among the family
  (within-family accuracy).

Admission for latent prompts (mechanical, lens-independent): intermediate string and aliases absent
from prompt; answer string absent; same template across family members; changing the source cue
changes the intended intermediate and answer; the model answers correctly (greedy) — recorded, not
required, and reported as a covariate.

Baselines in Part B: first-token J-lens direction (identical for all members, so within-family
accuracy is chance by construction; it sets the floor), logit lens, released R.

### 3.5 Part C — covariance infrastructure and read/write geometry

- Σ_ℓ, μ_ℓ from the pile corpus, computed on two disjoint halves (A, B) and pooled, for the 13-layer
  grid (store fp32; ≈1.4 GB in `outputs/`).
- Shrinkage Σ_λ = (1−λ)Σ̂ + λτI for λ ∈ {0.01, 0.1, 0.3}, τ = tr(Σ̂)/d.
- For 200 single-token template rows with single-token ids and for every family member: compute
  the pile covariance-dual u = Σ_λ t and report cos(Σ_A t, Σ_B t) and the norm ratio across halves
  and across λ. A direction that changes with λ or between halves is not ready to be a write vector.
- Geometry between instruments, per layer: cos(J_ℓᵀq_t, t_w) and cos(Σ_pile J_ℓᵀq_t, Σ_pile t_w).
  Terminology: Σ_pile t_w is the **pile covariance-dual**, not the template mean-difference, since
  Σ_template is unavailable.

## 4. What we will learn

| Emission AUC | Latent AUC | Reading | What it changes |
|---|---|---|---|
| high | high | template direction tracks phrase identity in latent reasoning | phrase-J's job is to match or beat this causally; use these families as Dataset A stratum 2 |
| high | low | template describes production, not latent content | latent multi-token direction is open; phrase-J regime (iii) may share the failure; test v_cond on latent prompts in 003 |
| low | high | surprising; inspect construction | re-check passages and tokenization before anything else |
| low | low | whitened rows do not separate the pair | labelling problem; pivot to trie/beam labelling over first-token J and drop phrase-J for this family |

Part A: if within-family cosines are ≥ 0.95 across the band the rows are essentially the prefix
direction and Part B's result is expected to be "low"; if ≤ 0.8 there is room.

Part C: if cos(Σ_A t, Σ_B t) < 0.95 at λ = 0.1, the covariance-dual is not usable for 004 without
more data; if cos(Jᵀq_t, t_w) is high, J rows and LDA rows agree on this model (heuristic support
for the whitening reading, without a Stein claim).

## 5. Cost

Σ pass: 500 forwards of 128 tokens ≈ 3 min. Part B: ≤ 20 families × 4 members × (40 passages + 5
prompts) ≈ 3.6k forwards ≈ 10 min. CPU for the rest. Under one hour wall-clock.

## 6. Kill criterion / decision rule

Thresholds fixed now: within-family template cosine ≥ 0.95 → "collinear"; AUC ≥ 0.85 → "high";
AUC ≤ 0.65 → "low". No hypothesis is protected; the table in §4 is the pre-registered reading.

## 7. Outputs

- `results/001-template-geometry/`: `families.json` (members, ids, prefix length, admission
  flags), `geometry.json`, `discrimination.json` (per pair, per layer, per condition AUC and ranks),
  `covariance_stability.json`, `analysis.md`.
- `plots/001-template-geometry/`: within-family cosine vs layer; emission vs latent AUC per family
  per layer; covariance-dual stability vs λ.
- `research_artifacts/001-template-geometry/report.md`.

## 8. Deferred to 001b

Stein diagnostic (K): compare a future-aggregated probability-weighted Stein estimate against the
J row with matched source/target weighting (or a same-position estimate against a self-only J),
with effective sample size reported per token. Not needed for the 003 decision.
