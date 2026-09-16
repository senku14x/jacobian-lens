# 001 — Template-lens pre-test: is there a linear direction for multi-token concepts to find?

Status: **DRAFT, awaiting agreement.** Written 2026-09-16 before any code.

## 1. Question

Before spending GPU on phrase-J or the word head: do the released template-lens rows for
Qwen3.6-27B already separate multi-token minimal pairs (blackmail / blackboard) at workspace depth,
and how far do whitened (read-side) directions sit from unwhitened (write-side) directions for the
same concept?

## 2. Why now

Every multi-token method we are considering (phrase-J, gradient template, word head) is a search for
a *linear* direction per phrase. The released template lens is the best existing linear phrase
instrument on this model. If its rows for `blackmail` and `blackboard` are near-collinear in the
band, the whitened representation does not separate them and no linear lens will either; the
programme becomes a labelling problem (fragment → word) rather than a direction-finding problem.
That changes which experiments are worth running next. This costs almost nothing, so it goes first.

Second, the ideas ledger (SS) and the ideas review (E.4) both rest on the conjecture that J-lens
rows behave like whitened (LDA) directions. We can measure the read/write dissociation angle
directly for every single-token concept once we have Σ. That number is reused by every later swap.

## 3. What we run

**Inputs (all on disk, no fitting):**
- `templates+phrases_v3.safetensors` — 13,731 rows × 64 layers × 5120, pre-whitened
  (`space=raw`, `ridge_c=0.001`).
- Released J lens (`j-lens/lens.pt`) and R lens.
- Qwen3.6-27B for one forward pass over 25 × 128-token `NeelNanda/pile-10k` sequences (the released
  lenses' own fit corpus) to estimate per-layer residual mean μ_ℓ and covariance Σ_ℓ. Cost: one
  forward pass, ~1 minute; store μ, Σ (64 × 5120² fp32 ≈ 6.7 GB → keep only the 13 layers on the
  standard grid, ≈1.4 GB, in `outputs/`).

**Part A — minimal-pair separability (template rows only).**
- Pair list (proposal, to be edited): the paper's own hard cases plus common fragment collisions
  present in the template vocabulary. Each pair shares a first token in Qwen's tokenizer
  (verified at build time):
  `blackmail / blackboard`, `photosynthesis / photograph`, `New Zealand / New York`,
  `ice cream / ice age`, `credit card / credit score`, `general relativity / general election`,
  `Golden Gate Bridge / Golden Retriever`, `hot dog / hot water`, `World Cup / World War`,
  `drug overdose / drug store`, plus 10 same-first-token pairs sampled at random from the vocabulary
  (control for cherry-picking). Pairs whose members are not both in the vocabulary are dropped and
  listed.
- For each pair and each layer: cos(t_a, t_b). Also the cosine between each row and the template
  row of its shared first token (`black`, `photo`, …) if present, and the cosine to the J-lens row of
  the first token, `Jᵀ((1+γ)⊙W_U[first])`, to see how much of a phrase template is just its
  first-token direction.
- Also for each pair, the **discriminability on real activations**: on 20 natural pile-10k
  positions where the model's top-1 next token is the shared first token, project the residual onto
  t_a − t_b and record where it sits relative to the pair's template midpoint. This is the only
  part that needs the model beyond the Σ pass and it is 20 forwards.

**Part B — read/write dissociation angle (single-token concepts).**
- For the 13,174 single-token-or-word rows with a single-token Qwen id: J-lens row
  `v_t = J_ℓᵀ((1+γ)⊙W_U[t])`, template row `t_w`, and the *unwhitened* mean-difference direction
  `m_w = Σ_ℓ t_w` (recovering it from the stored whitened row; exact up to the ridge).
- Report per layer the distributions of: cos(v_t, t_w), cos(v_t, m_w), and ∠(m_w, Σ⁻¹m_w). The
  last is the dissociation angle from SS.
- Stein diagnostic (K): from the same 25 sequences, `s_t = Σ⁻¹ E[p_t(x)(x − μ)]` for the 2,000 most
  frequent single tokens; cos(s_t, v_t) per layer. Where it is low the Gaussian approximation
  behind "J ≈ whitened" fails.

**Layers:** the standard 13-layer grid L8–L60 step 4, plus L62.

## 4. What we will learn

| Outcome | Reading | What it changes |
|---|---|---|
| A: pair cosines ≥ 0.95 in the band for most pairs, and real activations do not separate on t_a − t_b | Whitened representations do not distinguish these phrases before emission | Phrase-J and word head are unlikely to find a direction either. Pivot: labelling methods (trie beam F, template E), and test whether separation exists only *after* the first token is emitted (a positional question, not a lens question). |
| A: pair cosines ≤ 0.8 in the band and real activations separate | A phrase-specific direction exists in the whitened metric | Phrase-J is worth running; its target is to find the same or a more causal direction. Record which pairs separate for use as the 003/A item set. |
| A: separable at late layers only | Disambiguation is a late computation | Phrase-J gains should be confined to late band; set the layer range for 002/003 accordingly. |
| B: cos(v_t, t_w) high (>0.7) and dissociation angle large | J rows are effectively whitened directions | SS protocol adopted: read whitened, swap unwhitened. The "privilege" results in the paper are partly a whitening story; E.4 confirmed cheaply. |
| B: cos(v_t, t_w) low, cos(v_t, m_w) higher | J rows are closer to mean-difference than to LDA | The whitening reading is wrong; keep J rows for both read and swap; SS not needed. |
| B: Stein cosine low where B's cosines are high | The J≈whitened agreement is not via the Gaussian route | Report; does not change the plan. |

Either branch of A is a decision. B is a measurement that every later swap uses.

## 5. Cost

One forward pass corpus (25 × 128 tokens), 20 extra forwards for Part A discriminability, CPU for
everything else. Under 30 minutes wall-clock, ~2 minutes of GPU.

## 6. Kill criterion

None; this is a measurement with no hypothesis to protect. The pre-registered *interpretation* rule
is the table above; the cosine thresholds (0.95 / 0.8) are set now, before looking.

## 7. Baselines and controls

- Random same-first-token pairs alongside the chosen ones (cherry-picking control).
- Random-row pairs (no shared first token) as the floor for pair cosine.
- Dissociation angle for random Gaussian directions under the same Σ as the null.

## 8. Outputs

- `results/001-template-pretest/analysis.md` and `pairs.json` (per pair, per layer cosines and
  discriminability), `dissociation.json` (per layer summary stats), `stein.json`.
- `plots/001-template-pretest/`: pair cosine vs layer (chosen vs random), dissociation-angle
  distribution per layer, Stein-vs-J cosine histogram.
- `research_artifacts/001-template-pretest/report.md`.

## 9. Open points for Vishesh

1. Edit the pair list. Anything you specifically want separated (auditing-relevant phrases?) goes in.
2. Σ from pile-10k (matches the lens fit corpus) or from the template passages (matches the template
   lens's own whitening)? Proposal: pile-10k as primary, passages as a check on 3 layers.
3. Confirm the 13-layer grid.
