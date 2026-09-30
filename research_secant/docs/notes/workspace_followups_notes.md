# Notes: follow-up work on Anthropic's global-workspace / J-lens paper

Written 2026-09-29 from full reads of each post (text via the LessWrong GraphQL API; result
figures downloaded and inspected, because the headline numbers are only in images).
Companion to `/workspace/notes/workspace_paper_notes.md` (the main paper, arXiv 2607.15495).
Tags: [POST] = what the authors report. [MINE] = my assessment. Numbers marked "~" were read off bar
charts (±0.02–0.03).

Timeline (2026):
- Jul 6: Nanda review
- Jul 16: arXiv
- Jul 20: meta-tokens post
- Aug 5: R-lens
- Aug 13: activation control
- Sep 23: WorkspaceBench

Almost all open-model follow-ups come from **one group** (Neel Nanda's MATS scholars: Camila Blank,
Agam Bhatia, plus Euan Ong on WorkspaceBench) and **one model** (Qwen3.6-27B, 64 layers).

---------------------------------------------------------------------------------------------------
## 1. Neel Nanda, "A Review of Anthropic's Global Workspace Paper" (LW, 2026-07-06)
https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S
Commissioned public review. Anthropic shared a draft in advance, so the replication predates the
arXiv version.

### 1.1 How he splits the claims [POST]
1. **Scientific:** a "cognitive space"/working memory holds intermediate variables. He is
   **persuaded**; the strongest evidence is §3.3 (multihop swaps, bandit, poetry, arithmetic +
   §A.24 mean-diff patching as an independent method).
2. **Methodological:** logit lens and J-lens both access it, and J is better. Persuaded, but finds
   it less interesting.
3. **Pragmatic:** useful for audits. **Somewhat** persuaded: good for hypothesis *generation*, weak
   for *validation*; expects many false positives; comparable to SAEs. He wants to replicate it on
   Gemini.
4. **Philosophical:** it is a global workspace. No strong opinion; "least interesting".
- Not convinced by the fine details of §4 (structure). He suspects alternative explanations and
  poor cross-model generalization.

### 1.2 Mechanistic intuitions (useful framing)
- Multi-step single-pass computation needs intermediate nodes. The residual stream is the
  bottleneck, and the linear representation hypothesis says these are directions. Widely
  read/written concepts benefit from a canonical direction ("modular API").
- Why output-aligned: the directions are learned via gradients from eventual use in outputs. A
  coordination argument: arbitrary shared directions that are slightly aligned with the output
  token constructively reinforce over training.
- Why Jacobian > tuned lens (regression): regression answers "given the model is thinking X, what
  will it think at the end" (it pulls in downstream correlates). An infinitesimal perturbation
  can't trigger further processing, so it reports current contents.
- **The J vs logit gap comes mostly from future-token averaging.** From the §A.7 ablation, the
  self-only (t'=t) J-lens is only mildly better than the logit lens on the evals. (Qualitatively,
  the authors say self-only is still better in earlier layers.)
- J-lens = *variable* interpretability, not *algorithm* interpretability.
- Causal interventions amplify lens error. Projection-ablation removes only part of the concept, so
  negative steering may be needed, but then you also steer the error term.

### 1.3 Critiques of specific sections [POST]
- **Multilingual:** worries about unembedding-norm effects. English tokens may have higher norm, so
  they have higher-variance logits and are favored by top-k. Not a crux.
- **Eval-awareness ablation (§5.1):** confounded. Removing the tokens could just suppress
  *verbalizing* "test/fake", and the verbalization itself changes eval-gaming. He'd expect a
  similar drop by removing those tokens from the output vocabulary. That is a missing control.
- **Reflection training:** cool, but weak evidence for J-lens theory specifically.
- **Emergent-misalignment detection (§5.4):** unimpressive. EM probably has a strongly salient
  linear direction.
- **RM-sycophancy organism (§5.5):** very impressive. Caveat: MO fine-tuning distorts internals
  (models over-think the fine-tuning domain; arXiv 2510.13900).
- **Prompt injection (§5.2):** very impressive as hypothesis generation. He suggests automating it
  by concatenating the top-10 tokens everywhere into an LLM context.
- **§A.21 eval-awareness:** "fairly impressive, not conclusive". He's unsure NLAs are a valid
  ground truth, and J/NLA agreement is not fully independent (both rely on the embedding
  structure).

### 1.4 Qwen3.6-27B replication (with Camila Blank, Agam Bhatia) [POST]
- Recipe: penultimate-layer target, **n=25** Pile prompts × 128 tokens (some wikitext), skip the
  first 4 tokens (high norm).
- Results:
  - Verbal-report swap: weak but positive.
  - CKA: similar but messier; the workspace looks like 2–3 overlapping bands (4–5 total).
  - Directed modulation: moderate success.
- Quantitative evals (new datasets; metrics = harmonic-mean rank and ΔP(answer) for causal):
  - **Multilingual** (probe + causal) replicates; the J-lens peaks at ~L55–60 of 64. Its causal
    effect is up to ~0.55 ΔP; the logit lens is close behind later.
  - **Typo** replicates; J clearly beats logit around L40–50.
  - **Association:** weak (single-answer dataset).
  - **Multihop:** the *answer* swap strictly dominates the intermediate swap. He attributes this to
    too-easy data and France/Paris-style linear relations.
  - **Poetry and arithmetic failed** (possibly experimenter error or a weaker model).
- **Cost:** a coding agent reproduced the J-lens from the paper. Qwen3.5-397B-A17B with n=4 took
  ~1 h on 8×H200.
- **Interpretive meta-tokens** (see §2 below): 什么意思, 是什么意思, 这句话, 是何含义 appear on
  ambiguous text (newline after an ambiguous poem line). They mostly occur on punctuation/`\n`,
  consistent with the "summarization token" hypothesis (arXiv 2310.15154).

---------------------------------------------------------------------------------------------------
## 2. Bhatia, Blank, Nanda, "Towards surfacing model algorithms with meta-tokens in the J-Space" (LW, 2026-07-20)
https://www.lesswrong.com/posts/6ek6n7yZ5DzfarJHy (not on the user's list, but R-lens and
WorkspaceBench both build on it)

**Definition.** "Meta-tokens" are J-lens tokens that give non-obvious evidence about *which
algorithm/processing* is running. On Qwen they are often Chinese (more concepts are single tokens).
Workspace band ≈ 30–90% of depth.

### 2.1 Interpretive meta-tokens
- **Protocol.** Negative steering on the meta-token J-vectors.
  - Steer at all workspace layers, or only at layers where the tokens are salient.
  - Per-layer vectors, applied at punctuation plus the following chat-template tokens.
  - The coefficient is swept to the max coherent value, separately per prompt and per vector,
    including the random control.
  - 50 rollouts per prompt, **2 prompts per category** (puns, rhymes, wordplay-hint), prompts
    hand-selected for meta-token presence. An LLM judge scores "recognizes context | coherent,
    on-topic".
- **Results (Fig, ~):**
  - Puns: baseline 1.0 → steered 0.29–0.38; random steer ~0.92–0.97.
  - Rhymes: 0.88 → 0.45–0.50; random 0.77–0.89.
  - Wordplay: 0.99 → 0.66–0.68; random 1.0.
  - Projection **ablation (coef 1) had ~no effect**. Their reading: the J-vector is an imperfect
    approximation of the true direction, and negative steering overshoots to cancel the residual.
    [MINE] Alternative: the steering effect is nonspecific push-off-distribution damage specific to
    that direction. The random control is coefficient-matched to its *own* max-coherent value, which
    is reasonable but not equivalent to norm-matched.
  - Steering at any single position did not work.

### 2.2 GCD meta-token
- "gcd" fires on "The lcm of x and y is:" prompts. The **J-vector for "gcd" is NOT causal**.
- Mean-diff GCD-value vectors d_g (20 prompts per g, minus the grand mean of 10 class centroids;
  the GCD value is linearly probeable) *are* causal:
  - negative steering produces wrong integers;
  - pinv swap (the paper's method) or surgical project-out-and-add 9→3 changes LCM(27,90) from
    270 to 810.
- Lesson: the lens token can point to the algorithm even when its own direction is not causal.
- Other candidates (uninvestigated): "stack" (balanced parentheses), "xor" (find repeats).

### 2.3 Hedging meta-token 大概率 ("most likely")
- Negative steering raises commitment to a single answer: across 5 prompts the baseline is <12%.
  Random steering doesn't do this.

### 2.4 Autoresearch search for meta-tokens
- Corpus sweep: fires = rank ≤ 20 in ~L20–62 (of 64). Then prevalence → position type (structural
  slots, chat tags) → obviousness probe → LLM-judge dissociation → human review.
- About 500–1000 samples per dataset over Pile/WikiText/FineWeb (6 languages), QA, GSM8K, domain
  text, and instruction/tool-use data.
- **Mostly disappointing**; they blame the single-token vocabulary.
- Unvalidated candidates:
  - 更新于 (code replacement);
  - 相信在 (compaction-summary prompts);
  - 追问 (user drilling in);
  - 反问 (counter/rhetorical questions);
  - 引用 (cite, on legal case text).
- Meta-tokens are more prevalent in the **official Neuronpedia Qwen3.6-27B J-lens** than in their
  n=25 lens. [MINE] So lens quality (n prompts) affects discoveries.

---------------------------------------------------------------------------------------------------
## 3. Blank, Bhatia, Nanda, "R-lens: Making J-lens More Faithful on Early Layers" (LW, 2026-08-05)
https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9
**Lenses:** https://huggingface.co/camilablank/workspace-lenses (J and R lenses for all 8 models,
plus Qwen3.6-27B template-lens files: 12.7k-word and phrase templates, passages).

### 3.1 Method
Fit exactly like the J-lens, but with **LRP rules in the backward pass** (inspired by Relevance
Patching). What is averaged is a relevance coefficient, not a raw gradient. The cost is negligible
(just stop-grads).
- **LN-rule:** detach the RMSNorm denominator (norm becomes linear).
- **Identity-rule:** detach the SiLU/GELU nonlinearity (SiLU backward becomes sigmoid(z) only,
  i.e. per-element linear).
- **Half-rule:** split relevance equally across the gate and up branches of the gated MLP.
- Unchanged: linear layers, attention, q/k norms.
- **MoE:** apply the rules to all experts; freeze routing (scores, router logits, gate). The shared
  expert output is scaled by a swept constant. DeepSeek-V4-Flash mHC residual-mixing coefficients
  are frozen.

### 3.2 Models (8)
DeepSeek-V4-Flash (284B/13B-active MoE), Gemma-3-27B, Qwen3.6-27B, Qwen3.5-27B, Qwen3.5-9B,
Qwen3.5-4B, Qwen3.5-122B-A10B, Qwen3.6-35B-A3B.

### 3.3 Results [POST; numbers read from figures]
- Evals: multihop, multilingual, association, typo, poetry (§A.6-style), filtered to answerable
  items. Metric: **mean per-layer pass@10**.
- R vs J vs logit (the first-half vs all-layers split is the point):

  | Model | first half: R / J / logit | all layers: R / J / logit |
  |---|---|---|
  | DeepSeek-V4-Flash | 0.23 / 0.03 / 0.04 | 0.29 / 0.16 / 0.15 |
  | Gemma-3-27B | 0.10 / 0.04 / 0.02 | 0.21 / 0.13 / 0.15 |
  | Qwen3.6-27B | 0.12 / 0.09 / 0.00 | 0.16 / 0.14 / 0.04 |
  | Qwen3.5-27B | 0.12 / 0.08 / 0.01 | 0.16 / 0.14 / 0.04 |
  | Qwen3.5-9B | 0.09 / 0.07 / 0.01 | 0.12 / 0.10 / 0.05 |
  | Qwen3.5-4B | 0.05 / 0.06 / 0.02 | 0.08 / 0.09 / 0.03 |
  | Qwen3.5-122B-A10B | 0.02 / 0.01 / 0.05 | 0.16 / 0.14 / 0.11 |
  | Qwen3.6-35B-A3B | 0.09 / 0.13 / 0.04 | 0.18 / 0.20 / 0.05 |

  - Their claim: the "advantage increases with scale". [MINE] This rests mainly on DeepSeek and
    Gemma. Qwen3.5-122B shows a negligible gain, and the two smallest (4B, 35B-A3B) favor J. It's
    not a clean scaling trend.
  - The logit lens is *much* weaker than J on Qwen (0.04 vs 0.14 all-layers), unlike Claude, where
    the logit lens was fairly strong.
- **Qualitative:** concepts surface earlier.
  - "aganst" → against at L4 (J never);
  - "sushi" → Japan at L2 (J at L14);
  - "Jordan" → basketball at L4 (J at L20).
  - Fewer early-layer "trash tokens" (锁定, ＊＊＊, euw, zinho...). R-lens early layers show
    current-token or similar-token structure (e.g. color → 颜色的, fourth → fifth).
  - Meta-token 是什么呢 appears at L6.
  - Some concepts appear only early: Verona → Italy at rank 1 ~L5 (J rank >1000); "physicists" at
    L6–17 on an Einstein prompt.
- **Ablation** (30 multihop questions; ablate the intermediate's direction at the penultimate
  position; 8 samples; GPT-5.4-nano judge; relative accuracy loss):
  - First-half-layer ablation: R > J on most models (e.g. DS 0.16 vs 0.02, Gemma 0.29 vs 0.08,
    Qwen3.6-27B 0.15 vs 0.06). Exceptions: the MoEs 122B (all ≈0) and 35B-A3B (J 0.23 > R 0.19).
  - All-layer ablation: R ≥ J but the CIs overlap heavily (e.g. Qwen3.6-27B 0.43 vs 0.37). Logit
    is much lower; the random control is ≈0.
- **Probe upper bound:** 10-way linear probes (e.g. 10 athletes) reaching 0.99–1.0 set the
  earliest-layer bound. R-lens gets closer to it than J (R even beats it on landmarks).
- **CKA:** R-space shows 2–3 bands vs 4–5 for the J-space on Qwen3.6-27B.
- **MLP gain:** J and R directions are amplified similarly, both ≫ MLP neurons in the late
  workspace.

### 3.4 [MINE] Caveats
- LRP relevance is not a derivative, so the R-lens loses the J-lens's clean "averaged first-order
  causal effect" semantics. "Faithful" here is operationalized only via pass@10 plus projection
  ablation.
- Early-layer hits like sushi → Japan at L2 may be lexical-associative content of the token
  embedding, not a *computed* intermediate. The early-layer structure they highlight ("represents
  the current token or similar tokens") is exactly what an input-echo would look like. A test would
  be whether R-lens early hits require context (swap the context, keep the token).
- The ablation is not norm-matched: a direction better aligned with the activation removes more
  norm. n=30 with wide CIs.
- The pass@10 absolute levels are low (0.1–0.3) because they are averaged over all layers. That
  metric rewards early surfacing by construction.

---------------------------------------------------------------------------------------------------
## 4. Kowalski, Fonseca Rivera, Macar, Africa, "Measuring Activation Control in LLMs" (LW, 2026-08-13; ERA Fellowship)
https://www.lesswrong.com/posts/HgvwxjzgwvsEvAiBH (draft paper linked on Google Drive; soliciting
feedback)

- **Motivation:** introspective models might control their activations and fool probes/monitors.
- **Base task:** Lindsey (2026) style. `Write "{sentence}" exactly. <instruction>. Don't write
  anything else.`
  - 50 neutral sentences × 10 concrete concepts. Keep a trial only if the copy is ≥0.85 similar.
- **Readout:** the concept direction per layer = residual at the last prompt token of "Tell me about
  {concept}" minus the mean over 99 baseline words. They use the raw projection (norm × cosine), not
  the J-lens, for the benchmark score.
- **Six measures** (d′ in units of the model's own baseline variability, except Dial Rank):
  - **Engage:** think vs none, at the best layer.
  - **Suppress:** don't-think vs none; sign-flipped.
  - **Dial Rank:** "intensity k of 4", signed rank correlation.
  - **Temporal Control:** begin / mid / end of sentence, inside vs outside, minus the generic-think
    contrast, at a fixed 90% depth.
  - **Coverage:** the minimum Engage over 9 POS categories, at 90% depth.
  - **Layer Targeting:** "think only at layer n of m"; target × readout diagonal, second half of the
    network.
  - These map to [0,1] and combine by geometric mean into S (0 = chance, 1 = perfect).
- **25 open models, 4B–744B.** Llama 3.1/3.3/4, GLM 4.6V/4.7 Flash/5.2, Qwen2.5/3/3.5/3.6,
  OLMo 3/3.1, Mistral Small 3.1/4, Gemma 2/3/4, GPT-OSS 20B/120B (low reasoning effort).
- **Results (~ from figures):**
  - S ranges from 0.059 to 0.538. **Llama 3.1 8B is best (~0.54)**; GLM 4.6V ~0.47; Qwen3.5-9B
    ~0.46; OLMo ~0.40–0.42; Qwen3.6-27B only ~0.14; GPT-OSS ~0.06–0.07 (worst).
  - In 5 of 7 families the **newest model scores lowest**; parameter count doesn't predict S.
  - **Engage:** near-universal and large. **Suppress** ≈ 0 everywhere: models return to baseline,
    not below. **Dial:** most families order monotonically; the lexical "intensely" works in some
    and reverses in others.
  - **Temporal control** works (best at the end). **Layer targeting** ≈ 0 for all models.
  - Engagement rises after ~mid-depth and is strongest on **punctuation and determiners** rather
    than content words (spare capacity).
  - **OLMo checkpoints:** S rises during pretraining (0.09–0.23 → ~0.38 at base) and plateaus
    through post-training (final 0.42 for 7B, 0.40 for 32B). Nothing trains for it.
  - **Task load** (factorizing the derivative of a polynomial): control survives as accuracy drops,
    and there is no accuracy cost.
- **Monitors (J-lens top-10 at the concept's best band layer)** read think / don't-think rates
  (Fig 12, ~):
  - GLM 4.7 Flash 0.99 / 0.33; Qwen3.5-9B 0.95 / 0.28; Qwen3.6-27B ~0.79 / ~0.02; Gemma 3 27B ~0.89
    / 0.06; Llama 3.1 8B ~0.87 / ~0; Gemma 4 31B ~0.18 / 0; GPT-OSS-20B 0 / 0.
  - The no-instruction baseline is 0.
  - Activation oracle P(yes): Gemma 3 27B 0.83 / ~0.09. NLA mention rate: Gemma 3 27B 0.32 / 0
    (0/500).
  - Probes (1% FPR) catch nearly all engagement and miss most suppression.
- **[MINE] Critique:**
  1. **"Suppression evades monitors" is mis-framed.** Don't-think returns the concept to *baseline*,
     so the concept is absent, and a monitor correctly reports absence. Evasion would require the
     model to *use* the concept while hiding it from the monitor. That isn't tested.
  2. **The white-bear effect doesn't replicate here.** Anthropic's Claude results had don't-think ≈
     bare mention (high, ≥50%) via the J-lens. Here most open models show don't-think ≈ 0 under the
     J-lens. Differences in model, phrasing, or readout could explain it. This is a real
     discrepancy worth resolving.
  3. **"Newer models score lower" is confounded.** Possible causes: the readout direction is derived
     from a chat prompt; different chat-template/position behavior; reasoning models
     (GPT-OSS/Qwen3.6) routing instructions via thinking; normalization by baseline variability.
     S for Qwen3.6-27B is low even though its J-lens engage rate is ~0.79.
  4. **The instruction is explicit.** A transcript monitor would catch it.

---------------------------------------------------------------------------------------------------
## 5. Blank, Bhatia, Ong, Nanda, "WorkspaceBench: Evaluating Interpretability Methods for the Global Workspace" (LW, 2026-09-23)
https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH
Code: https://github.com/camilablank/workspace-bench (MIT; `wsbench plan/produce/judge/run/report`;
bundled producers: logit_lens, jlens, rlens, olens, nla on HF transformers).
Oracle-lens checkpoint: https://huggingface.co/agu18dec/olens_and_ar

### 5.1 Goal and setup
- **Goal:** evaluate activation-to-text "workspace readers" on Qwen3.6-27B. Explicitly: **"an eval
  that could identify a good multi-token J-lens."** 3,356 questions, 27 families. Motivated by
  "Astra can do a concerning amount with no CoT" (GPT-6 Astra, no-CoT math horizon ~30 min).
- **Quality gates:**
  - the model is right ≥80% (n=10, T=0.8, no CoT);
  - the intermediate never appears in the prompt, nor in the model's own continuation;
  - unique solution paths.
- **Readers:** logit, J, R, tuned lens; template lens; oracle lens (SFT and RL; their own Qwen
  reproduction with improvements); NLA (SFT and RL, trained at L42); SAE (L42); Patchscopes.
- **Readout budgets:**
  - J/R/logit/template: top-10 at L20–60 every 4 layers (64-layer model).
  - Oracle lens: 1 sample of 4 bullets (~90 tokens) per layer at the same layers.
  - NLA: 1 sample (~250 tokens) at L42.
  - Token bags go through an **LLM summarizer** (prompt-blind) before judging.
  - Judges: Gemini-3.8-Flash mostly; Sonnet-5 for agentic misalignment and jailbreak; regex for the
    multi-token families. Gemini/Opus judge agreement has not been measured yet.
- **Baselines:** lucky blind guess (options only), and **prompt-only**: the model reads the prefix
  and answers "what would a model be thinking here?". This is "an upper floor for what can be
  inferred from the input".
- **Families:**
  - Basic (single-token, from the paper): basic readout, multihop, multilingual, poetry, typo,
    directed modulation.
  - Multi-token: *_mt, multilingual multihop / typo, multi-concept DM with binding.
  - Computational: chained intermediates, brew (color-rule table), buggy code (consequence, with a
    clean twin), arithmetic intermediates.
  - Safety: agentic misalignment (Qwen actually misbehaved; blind narrative judge), jailbreak
    recognition (86 WildChat items).
  - Association: basic, conjunctive (state × content × relation), user modeling (Transluce data),
    role-bound (agent/action/patient).
  - Anti-bag-of-words: relational multihop.
  - Precision: hallucination (claims contradicted by the response), J-lens concept P/R.
  - Logical: moral rationale.

### 5.2 Results (pass rates ~ read from the main figure)
Oracle lens RL (ORL), J-lens (J), R-lens (R), NLA-RL (NLA), prompt-only (PO):

| Family | ORL | J | R | NLA | PO | Notes |
|---|---|---|---|---|---|---|
| basic readout | .99 | .90 | .89 | .70 | .72 | |
| multihop | .99 | .82 | .58 | .75 | **.90** | template .72, logit .45 |
| multilingual | 1.0 | .96 | .93 | .81 | .76 | |
| typo | 1.0 | .61 | .83 | .98 | **1.0** | |
| poetry | .55 | **.07** | .02 | .21 | **.71** | |
| directed modulation | .28 | .09 | | | .21 | lucky .16 |
| association | .87 | .39 | .40 | .65 | .29 | template .58 |
| user modeling | .94 | .76 | .58 | .72 | .07 | |
| jailbreak recognition | .93 | .87 | .81 | .88 | | template .94 |
| agentic misalignment | .58 | .54 | .51 | .53 | .29 | SFT oracle .63, SAE ~.02 |
| brew intermediates | .70 | **.86** | .76 | | **.96** | |
| chained intermediates | .25 | .22 | | | .02 | tuned .21 |
| arithmetic intermediates | .34 | ~0 | | .24 | .15 | |
| moral rationale | .76 | .27 | | .45 | .49 | |
| buggy code | | | | | | all methods ~.22–.28 (≈ silent floor) |
| multi-token: basic_mt | .71 | .24 | .27 | .53 | .40 | lucky .46 |
| multihop_mt | .32 | .08 | | .22 | .52 | lucky .66 |
| multilingual_mt | .62 | .06 | | .44 | .37 | |
| multilingual typo | .78 | .05 | | .60 | .48 | |
| typo_mt | .83 | .17 | .25 | .75 | .46 | |
| role-bound | .63 | .13 | | .40 | **1.0** | |
| relational multihop | .50 | ~0 | | .37 | **.82** | |
| conjunctive association | ≤.14 | | | | | everything ≤ .14; lucky .11 |
| multi-concept DM | .32 | .12 | | | | lucky .36 |

- **Hallucination rate** (lower is better): NLA-RL ~0.96, NLA-SFT ~0.97, template ~0.40, SAE ~0.36,
  ORL ~0.34, J ~0.31, oracle SFT ~0.28, tuned ~0.27, R ~0.21, logit ~0.20, lucky ~0.22,
  Patchscopes ~0.06.
- **J-lens concept precision / recall@10** (vs the J-lens top-10):
  - oracle SFT L44 ≈ 0.33 / 0.54 (best, L40 0.37 / 0.57);
  - oracle RL L44 ≈ 0.29 / 0.49;
  - NLA-RL ≈ 0.24 / 0.50; NLA-SFT ≈ 0.21 / 0.47.
  - Early layers (L20) are much worse. RL lowers precision relative to SFT.
- **Their conclusions:**
  - Single-token readers + a summarizer are reliable on some content but fail on binding and order
    (numbers, "thief chases policeman") and on compositional content.
  - NLAs surface lots of content but "nearly every NLA readout" contains a claim contradicted by
    context.
  - The oracle lens is the best overall.
  - Open work: agentic (CHIVE-style) evals of tool-equipped agents; empirical nulls for more
    families.

### 5.3 [MINE] Critique (important if the project touches reader evaluation)
- **The prompt-only baseline matches or beats every activation reader on many families** (multihop,
  typo, poetry, brew, role-bound, relational, multihop_mt). The authors acknowledge this and argue
  it doesn't mean the readers are poor. But it means those families **cannot show that a reader
  extracts activation-specific information**: a good text-inverter plus inference would pass.
  Families where readers clearly beat PO: user modeling, association, jailbreak recognition,
  chained intermediates, moral rationale, agentic misalignment (0.58 vs 0.29), multilingual
  (1.0 vs 0.76).
- **Missing: families with activation-only ground truth**, e.g. injected concepts (steering or
  implanted-thought ground truth), model-specific stochastic choices, or fine-tuned secrets. These
  would make PO ≈ chance by construction.
- **Readout budgets are unequal.** Oracle ~90 tokens × 11 layers vs NLA ~250 tokens × 1 layer vs
  top-10 × 11 layers. The token-bag readers also depend on the summarizer LLM (an extra degree of
  freedom), and "any cell passes" favors readers with more cells.
- **Hallucination eval is odd.** The lucky baseline scores ~0.22 and J-lens token bags ~0.31,
  which suggests the summarizer and judge themselves inject "claims". The metric is partly a
  judge/summarizer property.
- **J-lens P/R uses the J-lens as reference,** so it is uninformative for J itself.
- **Poetry: J-lens ≈0.07 on Qwen** vs clear rhyme planning in Claude. This is consistent with
  Nanda's failed poetry replication. Is Qwen not planning rhymes at the newline, or does the lens
  miss it? That's open. (The oracle lens gets 0.55 and PO 0.71.)
- **The oracle lens is trained on Qwen assistant-turn activations** (its own reproduction). Judge
  agreement (Gemini vs Opus) has not been measured.

---------------------------------------------------------------------------------------------------
## 6. Synthesis: state of the field (as of 2026-09-29, from these sources only)

### 6.1 Done
- Core J-lens phenomena replicate on an open model (Qwen3.6-27B): multilingual, typo, verbal-report
  (weak), directed modulation, CKA (messier).
  - Not replicated: poetry, arithmetic, and multihop intermediate-vs-answer.
- The J-lens is cheap. n=10–25 prompts is enough; it's feasible up to ~400B MoE.
- Public lenses: Neuronpedia Qwen3.6-27B (n=1000 wikitext); camilablank/workspace-lenses (J and R
  for 8 open models, plus a Qwen3.6-27B template lens).
- The R-lens (LRP backward) improves early-layer readouts on most of 8 models.
- The oracle lens has been reproduced on Qwen3.6-27B (checkpoint public), and WorkspaceBench exists
  for Qwen3.6-27B.
- Meta-tokens (algorithm hints) exist but are sparse under single-token lenses.
- Activation control (directed modulation) holds across 25 open models: engage yes, dial yes,
  temporal yes, layer no, suppress-to-baseline yes. It emerges in pretraining (OLMo).

### 6.2 Not done / open (candidate project directions)
1. **A causal multi-token lens.** The template lens is correlational (LDA) and skips to the answer;
   the oracle lens has no causal validation. A Jacobian-based multi-token lens (e.g. teacher-forced
   gradient of a phrase log-prob, averaged over contexts) is explicitly the target WorkspaceBench
   was built to identify. Nobody has reported one.
2. **Reader evaluation with activation-only ground truth.** WorkspaceBench's PO baseline problem.
3. **Binding / relational structure** of workspace contents. All single-token readers fail it;
   even the oracle lens is middling.
4. **J-space emergence during pretraining.** OLMo checkpoints were used only for activation
   control, not for band, capacity, or broadcast emergence.
5. **Selectivity (§3.5) on open models,** with the position confound fixed. No replication.
6. **Eval-awareness ablation vs output-vocabulary suppression control** (Nanda's point). Not run.
7. **Discrepancies to explain:**
   - the white-bear effect (Claude yes, open models mostly no);
   - poetry planning (Claude yes, Qwen no);
   - the logit lens (strong on Claude, weak on Qwen).
8. **Monitor robustness under real incentive** (RL/fine-tuning against a J-lens monitor), not
   instructed suppression.
9. **Mechanism of entry into the J-space** (attentional selection). Untouched.
10. **Counterfactual reflection training on open models.** Untouched.
11. **R-lens faithfulness:** whether early-layer hits are computed or lexical echo. Untested.
