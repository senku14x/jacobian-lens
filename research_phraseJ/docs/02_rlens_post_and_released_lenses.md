# Notes: "R-lens: Making J-lens More Faithful on Early Layers" (Blank, Bhatia, Nanda; LessWrong, 2026-08-05) + released artifacts (HF `camilablank/workspace-lenses`, last modified 2026-08-10)

Read: full post text (via LW GraphQL), all result figures, the HF README, the template-lens passages README, and the template vocabulary file.

---

## 1. What R-lens is

- **Definition.** Identical estimator to J-lens (mean Jacobian from layer ℓ to the target layer, averaged over positions and prompts), but the backward pass uses **Layer-wise Relevance Propagation (LRP) rules** instead of raw autograd. Borrowed from Relevance Patching (RelP, arXiv:2508.21258), which does the same swap inside attribution patching.
- **Three rules (dense models):**
  1. **LN-rule:** detach the RMSNorm normalization factor (rsqrt of variance) so the norm is linear in the backward pass.
  2. **Identity-rule:** detach the nonlinear factor of SiLU/GELU so the activation's backward is per-element linear (gradient becomes sigmoid(z), not the full SiLU derivative).
  3. **Half-rule:** split relevance evenly between the gate and up branches of the SwiGLU product instead of double-counting through the bilinear term.
- **Not modified:** linear layers (LRP 0-rule = ordinary gradient), attention, q/k norms.
- **MoE additions:** apply the MLP rules to all routed experts; freeze router weights, scoring, router logits, gate projection; scale the always-on shared expert's gradient by a swept constant (4 for the released DeepSeek and Qwen3.6-35B-A3B arms). DeepSeek-V4-Flash's mHC residual-mixing coefficients are also frozen.
- **Cost:** stop-gradients only; forward pass bit-identical; J and R are a matched pair on the same forward pass.
- **Motivation:** J-lens early-layer readouts are noisy/uninterpretable. The paper left open whether that is estimator degeneracy or genuine absence of content. R-lens is a bet on the former: gradients accumulate high-curvature error terms through many nonlinear layers; LRP's conservation-preserving rules suppress them.

## 2. Evals and metric

- Five categories adapted from workspace-paper A.6: multihop, multilingual, association, typo, poetry. Order-of-operations dropped. Filtered to questions each model answers correctly (multihop, multilingual).
- **Metric: mean per-layer pass@10** (is the intermediate in the top-10 readout at the target position at layer x), averaged over layers and categories. Reported for "first half of layers" and "all layers." Note this differs from the paper's pass@k AUC over *any* layer; per-layer averaging rewards a lens that shows the concept across more depth, not just once.
- Eight models: DeepSeek-V4-Flash (~280B MoE, 13B active), Gemma-3-27B-it, Qwen3.6-27B, Qwen3.5-27B, Qwen3.5-9B, Qwen3.5-4B, Qwen3.5-122B-A10B (MoE), Qwen3.6-35B-A3B (MoE).

## 3. Quantitative results (from the figure)

Mean per-layer pass@10, R / J / logit:

| Model | First half: R / J / logit | All layers: R / J / logit |
|---|---|---|
| DeepSeek-V4-Flash | 0.23 / 0.03 / 0.04 | 0.29 / 0.16 / 0.15 |
| Gemma-3-27B | 0.10 / 0.04 / 0.02 | 0.21 / 0.13 / 0.15 |
| Qwen3.6-27B | 0.12 / 0.09 / 0.00 | 0.16 / 0.14 / 0.04 |
| Qwen3.5-27B | 0.12 / 0.08 / 0.01 | 0.16 / 0.14 / 0.04 |
| Qwen3.5-9B | 0.09 / 0.07 / 0.01 | 0.12 / 0.10 / 0.05 |
| Qwen3.5-4B | 0.05 / 0.06 / 0.02 | 0.08 / 0.09 / 0.03 |
| Qwen3.5-122B-A10B | 0.02 / 0.01 / 0.05 | 0.16 / 0.14 / 0.11 |
| Qwen3.6-35B-A3B | 0.09 / 0.13 / 0.04 | 0.18 / 0.20 / 0.05 |

- Advantage grows with model size; largest on DeepSeek-V4-Flash (first-half 0.23 vs 0.03). No advantage on the smallest dense (4B) and the A3B MoE; the 122B MoE first-half numbers are near zero for both (that arm ships "paper-minimal" RelP, MoE rules off).
- Absolute values are small everywhere (max 0.29). Per-layer averaging over 60+ layers where most layers don't show the concept drags this down; it is not comparable to the paper's AUC.
- Logit lens is competitive with J-lens on all-layers for DeepSeek and Gemma, which is a reminder that on these open models most J-lens value is late-layer.

## 4. Qualitative claims

- Earlier surfacing: "against" at rank 1 at L4 on a typo where J-lens never surfaces it at that position; "Japan" at L2 vs L14 on "sushi"; "basketball" at L4 vs L20 on "Jordan". Hero figure (Qwen3.6-27B, "ramen" → Japan/Tokyo at L11–16 in R-lens; J-lens shows 尷, 音乐, ?datasetId, aplenty, Taiwanese until "noodles" at L22).
- Fewer "trash tokens" (锁定, 尷, ＊＊＊＊, euw, tav, zinho) in early layers; early R-lens readouts show structure (current token, translations, neighbours: 颜色的 after "color", "fifth" after "fourth"). Quantified in a figure not reproduced here.
- Early "interpretative meta-tokens" (是什么呢, "what is this?") at L6 during multihop, linking to a separate LW post on meta-tokens in the J-lens.
- Authors' inference: "the fact that R-lens works shows [that early layers contain no verbalizable content] is clearly false." This is overstated; see §7.

## 5. Causal ablation

- Setup: 30 multihop prompts; ablate the projection onto the intermediate's R-, J-, or logit-lens direction at the penultimate prompt position; first-half-of-layers vs all layers; 8 samples per prompt; accuracy loss judged by GPT-5.4-nano; random-direction control.
- **First half only:** R > J on every model except the two MoEs (122B: all ≈ 0; A3B: J ≈ 0.23 > R ≈ 0.19). E.g. DeepSeek R 0.16 vs J 0.02; Gemma 0.29 vs 0.08; Qwen3.6-27B 0.15 vs 0.06; Qwen3.5-9B 0.28 vs 0.18. Error bars are wide (n=30). Random control ≈ 0.
- **All layers:** R ≥ J by a small margin everywhere (e.g. Qwen3.6-27B 0.43 vs 0.37; Qwen3.5-27B 0.48 vs 0.41); the gap is mostly within error bars except Gemma (0.48 vs 0.18). Logit lens far below both except Gemma and DeepSeek.
- Reading: the early-layer R-direction for an intermediate is more causally load-bearing than the early-layer J-direction; at all layers the two are close, consistent with R and J converging in late layers.

## 6. Other analyses

- **Concepts J-lens never surfaces:** "Italy" at rank 1 around L5 on "Verona" (rank >1000 in J-lens); "physicists" in top-10 at L6–17 on an Einstein mention. Authors' own caveat: typically the concept reappears later and J-lens catches it, so R-lens is mainly for *tracking* computation across depth, less for *detection*.
- **Probe upper bound:** 10-way linear probes within a category (athletes, authors, cities, landmarks, musicians) on simple recall prompts; earliest layer with 0.99–1.00 accuracy = crude bound on how early a lens could read the entity. Median first-recovery@10 deficit vs bound (Qwen3.6-27B, 64 layers): musicians R +15 / J +27 / logit +47; landmarks R −11 (beats the bound) / J ≈ 0 / logit +36; cities R +6 / J +17 / logit +54; authors R +5 / J +17 / logit +54; athletes R +11 / J +19 / logit +52. "Never" rates: R 8–25%, J 3–28%, logit 16–81%. Note this is *entity identity* readout at the entity token, a much easier target than an unspoken intermediate.
- **CKA:** R-space CKA on Qwen3.6-27B shows 2–3 bands vs 4–5 for J-space. R and J differ most in early layers.
- **MLP gain (appendix):** on Qwen3.6-27B, J and R directions are amplified similarly across layers, both well above MLP neuron directions in later workspace layers. So R-lens changes early-layer readout without changing the broadcast-structure picture.

## 7. Critical read

1. **The metric change matters.** Per-layer pass@10 rewards breadth across depth. R-lens surfacing a concept earlier automatically raises the first-half average even if the *same* set of prompts is recovered somewhere. The paper's AUC-over-any-layer metric would be the apples-to-apples comparison and is not reported. The "concepts J-lens never surfaces" section concedes recovery is mostly the same set.
2. **Are R-vectors still "the J-space"?** LRP rules produce a different linear map, not the true tangent map. Detaching the RMSNorm denominator and the SiLU derivative yields something between a gradient and a "relevance" flow. The workspace paper's J-space claims (privilege, capacity, broadcast) were made about the tangent-map directions. Showing R-vectors are more causally load-bearing early (§5) partially answers this, but nothing in the post tests swap/patching in R-coordinates, occupancy, variance explained, or the J-vs-non-J privilege decomposition. The MLP-gain match is the only structural check.
3. **Ablation confound.** R-directions and J-directions differ in norm and in how correlated they are with the residual at early layers. "Ablate the projection" removes more of the activation when the direction is more aligned with it. A norm-matched or variance-matched control (as in the paper's experiential-report controls) is absent; the random-direction control is not matched to projection magnitude.
4. **Early-layer content vs estimator faithfulness.** The post's strong claim ("clearly false" that early layers lack verbalizable content) conflates "a better linear map reads the current token, its translation, and near neighbours at L0–L10" with "workspace-like content exists early." The paper's criterion for the workspace was abstract, persistent, non-input, non-output content. Current-token echoes and "fifth after fourth" are exactly the sensory-regime content the paper attributed to early layers. The Japan-at-L2 and Italy-at-L5 examples are the interesting ones and are entity-attribute lookups at the entity token, plausibly early-MLP factual-recall features rather than workspace posting. This post moves the question forward but does not settle it in the direction claimed.
5. **Judge-based ablation accuracy with n=30 and wide CIs.** The first-half effect on Gemma and Qwen3.5-9B is solid; several other bars overlap.
6. **What is genuinely useful:** a drop-in, zero-cost change that makes early-layer readouts less noisy on most models ≥9B, with matched J/R pairs released on 8 open models so anyone can check. For our purposes the *release* is the main value.

## 8. Released artifacts (HF `camilablank/workspace-lenses`, MIT)

**J-lens / R-lens pairs (matched, same forward pass, same recipe):** DeepSeek-V4-Flash, Qwen3.5-4B/9B/27B, Qwen3.6-27B, Gemma-3-27B-it, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B. Format: `lens.pt` dict with `J` (stacked per-layer Jacobians, target row = I), `n_prompts`, `source_layers`, `d_model`, `provenance` (model_id, target_layer, skip_first, n_prompts, dataset_id, config_json). Sizes 0.33–3.5 GB each.

**Recipe (differs from the paper's Sonnet lens in two ways):** `target_layer = n_layers − 2` (penultimate, matches paper), **`skip_first = 4`** (paper code: 16), **n = 25 prompts** from `NeelNanda/pile-10k` (paper: 1000; paper says ≥10 suffices). MoE arms: DeepSeek and Qwen3.6-A3B ship the swept "all experts + shared-scale 4" variant; Qwen3.5-122B ships dense-rules-only RelP.

**Template lens for Qwen3.6-27B (this is the item most relevant to multi-token work):**
- `templates.safetensors`: 13,174 rows (common words + curated entities/multi-token concepts); `templates+phrases_v3.safetensors`: 13,731 rows (adds "v3 phrase registry": SAE/seed/gap-mined multi-word phrases). Shape `[64, n_rows, 5120]` bf16, all 64 layers; `word_ids` maps row → vocab index; `.txt` files give row → text.
- Scoring: cosine of per-layer residual against each template direction (no unembedding).
- Vocabulary tail (rows 13,691–13,730) shows the phrase set: "Isaac Newton", "Romeo and Juliet", "blood pressure", "merci beaucoup", "sorry for your loss", "general relativity", "Golden Gate Bridge", "green fluorescent protein", "runtime error", "human oversight", "liver toxicity", "Eiffel Tower", "drug overdose", "button battery", "ethylene glycol", "New Zealand", "ice cream", "what this means", "this sentence"... i.e. a mix of named entities, safety-relevant phrases, and meta-phrases.
- **Passages released too** (`passages/`): generated by gpt-5.4-nano, ~300 passages per word/phrase target, with a "logprob leak filter + continuation verifier"; 12,742 base words / 1.89M passages; 601 entities / 148k; three phrase registries (834 + 33 + 10 phrases). Build: first 80% of passages per word for template estimation, last 20% held out for a ridge validation split; activations at the word's final residual position; function words skipped. Merged 14,219 keys / 2.14M passages.
- This is the paper's A.9.1 template-lens recipe (mean activation before the word, centered, whitened with ridge) re-implemented on an open model with an extended phrase vocabulary and the training passages public. It is directly reusable as a baseline and as a source of contrastive passages for any new multi-token lens variant.

## 9. Implications for today's multi-token work

- **Baseline exists.** A template lens with 13.7k rows on Qwen3.6-27B is released, with passages, so any proposed multi-token extension can be compared against it on the same model without regenerating data. The template lens's known pathologies (tuned-lens-style skip-ahead; 67% final-layer next-word top-10 on Sonnet; a few ubiquitous junk phrases) are the things to beat.
- **R-lens is orthogonal to the multi-token problem** (it changes the backward pass, not the vocabulary), but two things carry over: (a) any Jacobian-based multi-token extension (e.g. gradient of a multi-token sequence's log-prob under teacher forcing, or of a phrase-template direction) can be fit with LRP rules in the same way, and the released R/J pairs make that a one-line switch; (b) the ablation-based causal eval in §5 (projection ablation at penultimate position, judge-scored accuracy loss, random control) is a cheap protocol to reuse, with the caveat in §7.3 that it needs a norm-matched control.
- **Model choice.** Qwen3.6-27B is the only model with all three lenses (J, R, template) released. It is the natural testbed. DeepSeek-V4-Flash shows the biggest R-lens gains but has no template lens and is heavy.
- **Open question to keep in view:** whether the "workspace" properties the paper established for J-vectors transfer to R-vectors or template vectors is untested by anyone. Whatever lens variant is built today, the minimal check is: does a swap in the new coordinates redirect a two-hop answer, and does the effect appear at the intermediate's layer band rather than the answer's.

## 10. Links

- Post: https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers
- Artifacts: https://huggingface.co/camilablank/workspace-lenses/tree/main
- RelP: arXiv:2508.21258; LRP: arXiv:1604.00825
- Meta-tokens post referenced: https://www.lesswrong.com/posts/6ek6n7yZ5DzfarJHy/towards-surfacing-model-algorithms-with-meta-tokens-in-the-j
- Source repos named in provenance: `agu18dec/relp-jlens-matrix`, `agu18dec/qwen3.6-27b-relp-jlens`, `camilablank/deepseek-elens-jlens`
