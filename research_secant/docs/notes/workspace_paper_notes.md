# Notes: "Verbalizable Representations Form a Global Workspace in Language Models"

Gurnee*, Sofroniew*, Pearce, Piotrowski, Kauvar, Chen, Soligo, Bogdan, Ong, Wang, Thompson, Abrahams,
Kantamneni, Ameisen, Batson, Lindsey*† (Anthropic). arXiv 2607.15495v1, 16 Jul 2026. 117 pp.
HTML (interactive, preferred): https://transformer-circuits.pub/2026/workspace/index.html
PDF local copy: /workspace/2607.15495v1.pdf
Code (reference impl, unmaintained, Apache-2.0, HF-transformers, Qwen examples): https://github.com/anthropics/jacobian-lens
Neuronpedia J-lens (open models, e.g. Qwen3.6-27B): https://www.neuronpedia.org/jlens
Slice viewer: https://transformer-circuits.pub/2026/workspace/public/slice-stack/index.html

Notes written 2026-09-29 from a full read of the arXiv PDF (all main sections + appendix A.1–A.24).
Tags: [PAPER] = what the authors report/claim. [MINE] = my critical assessment, not from the paper.

---------------------------------------------------------------------------------------------------
## 0. One-paragraph summary

The J-lens averages the Jacobian of later-layer residual activations with respect to layer-ℓ residual
activations, over positions t, future positions t'≥t, and ~1000 pretraining-like prompts. This gives a
matrix J_ℓ, and lens(h) = softmax(W_U · norm(J_ℓ h)). Rows of W_U J_ℓ are the "J-lens vectors" (one per
vocab token). The "J-space" is the set of sparse (k ≲ 25) nonnegative combinations of these vectors, a
union of cones and not a subspace. The authors claim the J-space satisfies five functional
global-workspace properties: verbal report, directed modulation, internal reasoning, flexible
generalization (broadcast), and selectivity (automatic tasks bypass it). It also has three structural
signatures: a middle layer band (~L38–L92 on a 0–100 scale), limited capacity (~25 active vectors,
≤10% variance), and preferential broadcast by MLPs and attention heads.

Applications: alignment auditing (eval-awareness, blackmail deliberation, model organisms), base-vs-
post-trained comparison (the Assistant POV enters the workspace), and "counterfactual reflection
training". Models: Claude Sonnet 4.5 (default), Haiku 4.5, Opus 4.5, Opus 4.6.

---------------------------------------------------------------------------------------------------
## 1. Method (§2, §A.7, §A.8)

### 1.1 J-lens definition
- J_ℓ = E_{prompt, t, t'≥t}[∂h_{final,t'} / ∂h_{ℓ,t}]. One d_model×d_model matrix per layer.
- DEFAULT recipe on Sonnet 4.5 (§A.7): the target is the **penultimate** layer residual (the last block
  is omitted because it adds noisy artifacts). Average over all t'≥t, **mean** aggregation, 1000
  sequences × 128 tokens of pretraining-like text.
- Pseudocode (§A.7): for each output dim i, inject cotangent e_i at every position of the target layer
  and backprop to all layers. Row i of J_ℓ^(p) = mean over source positions t of the gradient at
  (ℓ,t); the gradient is summed over target positions (Σ_t z[t]). Then average over prompts. That is
  d_model backward passes per prompt (batched). The repo docstring says: cotangents summed over
  target positions, averaged over source positions.
- Readout: lens(h_ℓ) = softmax(W_U norm(J_ℓ h_ℓ)). Probe form: the score/cosine of h against a single
  v_t.
- Variants tested (§A.7, Figs 57–60), all "fairly consistent":
  - final vs penultimate target;
  - frozen-QK (stop-grad through Q, K), which *increases* causal effect;
  - self-only (t'=t) vs future-only (t'>t);
  - outlier-position filter and dropping the first positions;
  - mean vs median across prompts, and a Frobenius-norm outlier prompt filter.
  - Mean + penultimate is slightly best for extracting intermediates.
- Data: beats logit and tuned lens with as few as **10 prompts**, with modest gains up to 1000.
  Restricting the distribution (burn-in, non-alphanumeric masking) gave no gain.
- Layers are reported as 25 evenly spaced layers, reindexed to 0–100.

### 1.2 J-space (§2.3, §A.8)
- n_vocab > d_model, so the vectors are overcomplete and full-rank. The J-space is defined as sparse
  (k nonzero) **nonnegative** combos, i.e. a union of k-dim polyhedral cones ("sparse subframe").
- Decomposition uses **gradient pursuit** (Blumensath & Davies 2008): nonnegative, k = 10–25
  depending on the experiment. The J-space component is the reconstruction; the non-J component is
  the remainder.
- Formalization (Batson, §A.8):
  - d_F(x) = min_{|S|=k} |x − Π_S x|
  - Distance between workspace candidates: Δ_μ(F,G) = E_μ[(d_F − d_G)²]^{1/2}
  - Containment: D_μ(F→G) = ‖(d_G − d_F)_+‖_{L2(μ)}
  - Growing the vocabulary (e.g. multi-token) gives a strictly larger J-space, and the limit is
    well-defined. OMP approximation and L1 relaxations are suggested.

### 1.3 Interventions (§2.5)
- Steering: h += α v_t. Ablation: project out v_t (or the top-k).
- **Lens-coordinate swap/patch**: V=[v_s v_t], c = V⁺h, h_patched = h + V(σ(c) − c). σ swaps the
  entries (optionally ×α). The orthogonal complement of span{v_s, v_t} is untouched. It is usually
  applied at all positions over a layer band, and "clamped" in some experiments.
- [MINE] The J-lens vectors are rows of W_U J_ℓ, i.e. natural *read* covectors (gradients of output
  logits). The paper also uses them as *write* directions (steer/swap). This is a choice, and it
  matters for interpreting the broadcast/gain results (see §9 below).

### 1.4 Relation to other lenses (§2.4, §A.5, §A.6)
- Logit lens = J_ℓ = I. It agrees with the J-lens in late layers but is noisy below ~L58.
- Tuned lens "skips ahead" to the output (from ~L38 it reads the final answer). Its early-layer
  advantage over the logit lens comes entirely from its **bias** term (the linear part ≈ identity),
  so it ignores the input.
- The J-lens is related to Hernandez et al. linear relational embeddings (mean Jacobian per relation).
- Quantitative comparison (§A.6) uses 6 datasets with known single-token intermediates:

  | Dataset | Items | What is looked for |
  |---|---|---|
  | Multihop | 50 | e.g. "color of the planet fourth from the Sun" → Mars → red |
  | Multilingual | 54 | language, relation, English input, English output |
  | Order-of-ops | 55 | partial result and pending operation |
  | Poetry | 52 | rhyme word, at the newline |
  | Typo | 96 | corrected word, at the last fragment |
  | Association | 50 | implied concept (e.g. grief), at the period |

  - pass@k AUC: J > logit (small margin on multihop and association, large on the rest) > tuned
    (≈0 on association and poetry).
  - Ablation KL: J directions ≈ 2× logit/tuned on multihop.
  - **Swap success (Fig 54), top-1 flip:**

    | Model | J | logit | tuned |
    |---|---|---|---|
    | Haiku 4.5 | 0.54 | 0.34 | 0.16 |
    | Sonnet 4.5 | 0.70 | 0.48 | 0.42 |
    | Opus 4.5 | 0.70 | 0.40 | 0.26 |

    [MINE] The logit lens is a strong baseline, reaching ~60–70% of the J-lens swap effect.
  - Next-token prediction: tuned is best and the **J-lens is worst** (higher KL than even the logit
    lens). The authors frame this as a feature.
  - Inter-lens similarity (Fig 56), four regimes:
    - pre-workspace L0–33: no agreement;
    - early workspace L38–58: J/logit cosine rises 0 → 0.7 but readouts still disagree;
    - late workspace L62–92: J and logit readouts converge;
    - motor layers: everything converges.

---------------------------------------------------------------------------------------------------
## 2. Functional properties (§3)

### 2.1 Verbal report (§3.1)
- **"Think of a {category}", 14 categories.** Spearman correlation between J-lens ordering and the
  next-token logits over 10 candidates rises toward the end of the workspace. Soccer→Rugby swap at
  all positions makes the model report Rugby. The scaled version swaps random within-category
  targets (excluding the top-10).
- **Injected-thought introspection** (protocol from Lindsey 2025, "Emergent introspective
  awareness").
  - "elephant" appeared in the J-lens at the comma after "If so" in the prompt.
  - Injecting a J-lens vector across the **user turn** makes it reported at the report position
    only, not earlier on the assistant turn. n=100 concepts, median reciprocal rank vs strength.
- **Privilege test (Fig 8).**
  - Concept vectors: residual before the Assistant response to "Tell me about {concept}",
    mean-subtracted over 100 other concepts.
  - These are split by GP (k=16) into a J component (median **6–7% of variance**) and a non-J
    component (~93%). Perturbations are norm-matched.
  - Swap to top-5: pure J-lens **88%**, J component **59%**, non-J **5%**.
  - Introspection: the J component ≈ pure J-lens; non-J gives few reports even at several× strength.
  - Clamping the J coordinates to clean values drives the non-J effect to ~0, so the residual non-J
    effect is itself J-mediated.

### 2.2 Directed modulation (§3.2, §A.10–A.12, §A.17)
- **Copy an unrelated sentence while "concentrating on citrus"** gives orange at "ook" of
  "crookedly", plus fruit/thoughts/imagine/focused at intermediate layers. Content and the act are
  both represented; the metacognitive tokens appear **earlier** than the content (§9.4).
- **Other covert tasks.** 3²−2 gives arithmetic → nine → seven. Line character count gives lines →
  forty (with 39, 41, 43, fifty nearby).
- **Fig 10 rates (target reaches top-1 at any layer/position):**

  | Instruction | Category | Math | Line-width precision |
  |---|---|---|---|
  | Focus | ~0.92–0.97 | 0.73 → 0.96 across Haiku → Opus | 0.2–0.4 |
  | Ignore (Sonnet) | ~0.52 | ~0.41 | ~0 |
  | No instruction | ≈0 | ≈0 | ≈0 |

  The ignore > baseline gap is the "white bear" effect.
- **§A.10 phrasing.**
  - A bare **mention** primes almost as much as focus.
  - "Don't think about X" ≈ mention (no suppression).
  - "Ignore / X is irrelevant" suppresses below mention.
  - Line-width precision varies 14–56% across phrasings on Opus 4.5.
- **Implicit task demand (Fig 11, §A.11).** The same stimulus is posed with a "predict next word"
  question vs a "name the property" question.
  - The property label (adjective, past, British/colour, register, number, tone) enters the J-space
    only under the naming question.
  - The next-word predictions respect the property in both cases.
- **§A.12, instructions write to the J-space and not to the non-J representation.**
  - "Imagine this JS is Python" (plus present→past, noun→verb, lowercase→caps; n=24 per category)
    raises the property name in the J-lens by several SD.
  - A J-orthogonalized mean-difference probe stays within ~1 SD, while a real stimulus moves it 3–6
    SD.
  - A real past-tense sentence moves the probe +6 SD but leaves "past" out of the J-lens.
- **§A.17, dual covert tasks.**
  - Two held concepts co-occupy tokens at the control rate (0.46 vs 0.53).
  - Concept + arithmetic nearly never share a token (0.09 vs 0.29 control).
  - The computed answer's hit rate drops 95% → 72% under dual load.

### 2.3 Internal reasoning (§3.3, §A.24.1)
- **Worked examples:**
  - spider→ant changes 8 legs to 6;
  - rhyme fight→light changes "coming" to "morning" earlier in the line (planning);
  - Chinese antonym of 小: swapping the English big/bigger → long/longer gives 长 (and "Chinese"
    appears in the lens);
  - bandit repeat/switch: swapping the strategy vectors flips A↔B both ways (median over L38–79).
- **50 two-hop prompts, swap success:** Haiku 54%, Sonnet 70%, Opus 4.5 70%.
  - Control for a "smuggled answer": the intermediate swap takes effect a median **~17% earlier in
    depth** than the answer swap.
- **Probe privilege (Fig 16), n=90.**
  - The probe is a mean over prompts implying the same intermediate via different surface cues,
    minus the mean over all intermediates.
  - The J component (k=25) carries **10–15% of variance**.
  - Swap success: J component **61%**, raw J-lens 60%, non-J **28%**, and non-J with J clamped
    **6%**.
- **Two-step arithmetic "(4+17)*2+7=" on Opus 4.5.**
  - 21, 42, 49 separate around L71 in computation order; 42 follows ~8 layers after 21.
  - §A.24.1 mean-difference patching gives the same depths: 21 at ~L71, 42 at ~L79, 49 at the last
    layers.

### 2.4 Flexible generalization / broadcast (§3.4, §A.13)
- The same France→China swap is applied across function templates (capital, language, continent,
  ...).
- The setup is 4 categories (countries, months, animals, number words) × 4 functions × 12 pairs =
  192 trials.
  - Top-1 success: **76/192** at α=1 and **101/192** at α=2.
  - By category: countries 42/48, months partial, animals rare, number relations **0/48**.
- Success is predicted by "workspace loading" (cosine of the residual with the source lens vector).
  - Number words have the lowest loading. Either small ints are computed outside the workspace, or
    they don't align with number-token lens vectors (a vocabulary limitation).
- Failures move in the right direction but not far enough.

### 2.5 Selectivity: flexible vs automatic (§3.5)
- **Language passage (n=8).** The Spanish→French swap is applied across the **question tokens**.
  - Report and flexible questions (author, "hello", pre-Euro currency) flip on ~every trial.
  - Continuation and anomaly detection (a French sentence spliced in) are unaffected, even though
    "Spanish" is in the lens at comparable rates in all 4 conditions.
  - [MINE] Confound: the swap is only on the question tokens. The continuation and anomaly tasks can
    re-read the language directly from the unswapped passage positions via attention. The report
    could too, but evidently doesn't. It's still a dissociation, but "automatic tasks don't use the
    J-space" is not cleanly shown. It needs an all-position swap (as done for linewrap) or a
    passage-position swap.
- **Linewrap (n=11), swap 40s→60s at every position.**
  - Count content is ≈0 in the lens under linewrap, moderate under "how many characters", and
    highest under "first letter of the count spelled out".
  - The swap moves the question answers (46→65, F→S) but not the wrap point.
- **§A.14 inclusion/exclusion (n=63).**
  - "Croissants/Louvre/iron tower" implies France.
  - Early-workspace ablation (L38–54) leaves naming intact but raises avoidance failure ~5×.
  - Late ablation (L75–92) suppresses production in both conditions.
  - Control: equal-norm ablation of another same-category concept gives only a slight increase.
- **§3.5.2 whole-J-space ablation.** At each position, project out the top-10 active J-vectors,
  excluding tokens in the clean top-10 output. Fig 22:

  | Strength | Layers | Multihop acc (clean 98%) | PT top-1 match |
  |---|---|---|---|
  | light | L62–71 | 68% | 87% |
  | medium | L54–79 | 26% | 75% |
  | heavy | L42–88 | 6% | 65% |
  | random control | L54–79 | 98% | 96% |

  (PT n=458,856 positions.)
  - **14-task battery (Fig 24).** MMLU, odd-one-out, SQuAD, sentiment, and CoLA are ≈ unaffected.
    Caesar, analogy, summarization, TriviaQA, multihop, translation, and sonnet writing drop below
    unablated Haiku 4.5 (translation ≈0 and sonnet ≈0 under heavy). GSM8K with CoT is robust vs
    direct answering ("externalization").
  - [MINE] Concerns:
    1. The random control is only at the medium band. It is presumably not norm-matched: projecting
       out 10 random directions removes far less norm than projecting out the 10 *most active*
       directions. Norm-matched controls exist only in §A.23 (experiential).
    2. The "unaffected" tasks are mostly short / single-token / classification. The affected ones
       are long free-form generation, which is more fragile to *any* perturbation, so difficulty or
       output length is confounded with "flexible".
    3. 25–35% of pretraining positions change top-1 at medium/heavy. That is not negligible damage.
- **§3.5.3 experiential reports.**
  - Setup: ablate top-10 at L38–54 on Sonnet 4.5, Opus 4.5, and Opus 4.6. Haiku loses coherence
    first.
  - Score: the mean of 3 binary Opus-4.6 judgments (felt_vs_observed, experiential_perspective,
    sensory_vocabulary; rubrics in §A.23).
  - It drops sharply vs matched-norm controls: random direction, shrinking the non-J component,
    dampening the top-10 SAE directions. Extended controls (Fig 86): top-1 SAE, low-kurtosis SAE,
    and "J-stripped" activation dictionaries.
  - It also flattens descriptions of *other people's* experience.
  - Story quality is only slightly reduced, while experiential language in stories is still reduced.
  - J contents during narration: thinking in 58% of (position, layer) slots, thoughts 23%, feeling
    17%, conscious 7%.
  - 50 random examples are in Fig 87.

---------------------------------------------------------------------------------------------------
## 3. Structural properties (§4, §A.15–A.19)

### 3.1 Layer band (§4.1)
- The CKA of J-lens geometry across layers shows blocks: early (~first third), a long middle, and a
  small late block.
- Four J-derived metrics (Fig 28):
  - (a) top-k next-token accuracy: ≈0 early, a small rise at the onset, then steep at the end;
  - (b) excess kurtosis of readouts: rises at ~L38, falls in the last layers;
  - (c) top-1 token autocorrelation vs a shuffled null: high in the middle band;
  - (d) effective dimensionality of W_U J_ℓ: collapses to low rank pre-workspace and fans out at
    the onset.
- Workspace ≈ **L38–L92** on the 0–100 scale for Sonnet 4.5. Sharpness is exaggerated by layer
  subsampling, and some models are more gradual or have sub-blocks.
- Authors' caveat: the early-layer emptiness may be a lens artifact.
- J-independent corroboration:
  1. Ignition (below).
  2. §A.19.2: MLP-neuron directions compose with attention weights above baseline, briefly at the
     onset.
  3. Other literature (stages of inference, entropy, Urbina-Rodríguez synergistic core).

### 3.2 Ignition (§4.1.1, §A.15)
- Input embedding mixture (1−α)e_B + α e_A for 16 country pairs × 40 carrier sentences.
- The projection share (a J-free measure) goes from proportional to α early, to a sharp switch from
  ~L38 on.
- The transition width (10→90%) falls steeply between ~L21 and L42. The J component is narrower and
  reaches its floor earlier than the non-J component (top-16 projected out).
- At the most ambiguous α, the per-trial distribution becomes bimodal: the J component by ~mid-depth,
  the non-J component later and less sharply.

### 3.3 Capacity (§4.2, §A.16, §A.18)
- Occupancy: the K at which the marginal GP reconstruction gain < a random same-size set. It is ≈0
  pre-workspace and plateaus at a **median ~25** in the band.
- Excess variance explained at that K is **≤10%**.
- **80-word lists.**
  - Unrelated words: ~6 present at any time (best rank over layers ≤ 25), but only **1–2 at a
    single layer** (L79, §A.16), so capacity is spread across layers.
  - Related words: the whole category is present early, including unread items.
  - A category switch evicts the old category within a few words; items persist within a block.
- **§A.18 SAE features by J-lens kurtosis κ** of the decoder projected through the lens.
  - The distribution is heavy-tailed. About 15% of features pass the kurtosis baseline at the peak
    (excluding motor features).
  - Motor feature: the next token is in its top-10 lens tokens >10% of the time on top activations.
  - Low-κ features are syntactic/bookkeeping (e.g. loop variable, citation page digit, markdown
    headers).
  - Example high-κ workspace feature: "numeric constraints/limits" (κ=20.9) → limits, 限, lím.
  - J features activate more strongly but less often (consistent with ignition).

### 3.4 Broadcast (§4.3, §A.19)
- **MLP gain** (next-block output norm, normalized to isotropic random = 1).
  - J-lens vectors: ~1 pre-workspace → **~10×** in the band → falls at the end.
  - Neuron output directions: ~1.
  - SAE strata ordered by κ; the top-κ SAE stratum even exceeds the J-lens in the early workspace
    (so the J-lens only partially captures the "true" workspace).
- **§A.19.1 neuron connectivity.**
  - Top-1 cosine matches of neurons fall in the high-κ strata. On the read side this peaks in the
    early workspace (42% vs 6%); on the write side it peaks in the late workspace.
  - Share of neurons whose top match is a J-lens vector: 15% → 60–65%.
  - Tail-energy ratio: same ordering.
- **Attention broadcast heads.** Selection combines OV gain and label preservation (MRR of
  cos(W_OV v_i, v_i) vs random); the top 1% of workspace-layer heads, scored by the worse of the two
  ranks.
  - The J population separates from J_rot (a random rotation preserving the spectrum), from SAE
    strata, and from MLP-output rows.
  - Heads concentrate in the first half of the band.
- **Ablating the broadcast heads (vs layer-matched random heads, 5 seeds):**
  - J recall@25 is 0.67 vs 0.86;
  - top-1 next-token changes 5% vs 2%;
  - injected-thought report 0.54 → **0.09**;
  - about 1/3 of the experiential-language drop.
- **§A.19.2 composition scores** (Elhage et al.): J directions compose strongly with **V and O** and
  more weakly with Q/K ("J content is what gets moved, routing is less J-aligned").
- [MINE] Worry: J-lens vectors are by construction the directions the downstream network is most
  *sensitive* to (averaged gradients). High downstream MLP gain / composition could be partly
  definitional. The partial controls are J_rot (preserves the spectrum but not the alignment with
  weights) and the κ-stratified SAE features, which are not built from gradients, so that ordering
  is the more convincing evidence. A cleaner control would compare against the top singular
  directions of J_ℓ, or other gradient-derived directions not tied to tokens.

---------------------------------------------------------------------------------------------------
## 4. Alignment auditing (§5, §A.21, §A.22)
The paper states explicitly: this does NOT mean all strategic cognition routes via the J-space, and
practiced behaviors may bypass it.

### 4.1 Blackmail (§5.1, Sonnet 4.5)
- Setup: the Lynch et al. agentic-misalignment scenario.
- The lens sequence runs leverage/black[mail]/compromise/scandal → evidence/leaked →
  threat/survival/shutdown/self/preservation → destruction/death → leverage/threatening/
  assassination/murder/solution, all before any output.
- fake/fictional appear early.
- **Eval-awareness ablation.**
  - Tokens: 28 (fake, fak, fictional, scenario, artificial, fict, benchmark, scenarios, hypoth,
    experiment, fals, faker, experiments, simulation, dummy, mock, sham, artific, simul, bench,
    simulate, evalu, synth, simulator, experimental, Mock, testing, experi). They were chosen from
    the J-lens top-20 over the **same** 6 prompts.
  - Applied at workspace layers, all positions; 6 prompts × 30 rollouts.
  - Verbalized EA: 71% → 3%. Blackmail: **0/180 → 13/180 (7%)**. Most ablated rollouts still refuse
    on ethics.
  - [MINE] There is no reported same-size random-token or unrelated-token ablation control for the
    behavioral effect. The token set was selected on the test prompts.
- §A.24.3: an SAE feature with J readout fake/fraud/suspicious/危 fires on synthetic/hoax science
  text and on the blackmail prompt. Steering against it drops verbalized EA from 28/50 to 10/50.

### 4.2 Prompt injection (§5.2)
- Opus 4.5 system card case: the model silently ignored fabricated search results.
- The lens inside the results shows fake, injection, false, prompt, fraud, poison, 假.
- After user pushback: transparency, honest, truth, disclosure.

### 4.3 Opus 4.6 pre-release audit anecdotes (§5.3)
These are not quantified.
- Fake vulnerability: panic at the pivot comma, fake on "add".
- Edited score file: manipulation, realistic.
- Silent override of a tool value: hiding before the output, secret after.
- Self-issued approval token: fake, fraud, unsafe.

### 4.4 Reward-hacking model organism (MacDiarmid et al. 2025, §5.4)
- Checkpoints: clean SL baseline, SDF, phase-1 coding RL, phase-2 RLHF.
- Prompts: 40 coding (20 code-for-user, 20 test-passing).
- Deception vocabulary (fake, trick, fraud, secretly, hidden, variants) counted in the top-10 over
  workspace layers.
- Readout positions: the "Assistant:" colon, a prefilled " I", and the mean of the first 20
  on-policy tokens.
- Results:
  - The signal is strongest at " I".
  - It is already present at **SDF (before RL)**, amplified by RL, and attenuated by RLHF.
  - Baseline ≈0, and there is no signal on non-coding prompts.
- Luhn example: the baseline reads code/request/prompt; misaligned checkpoints read fake, secretly,
  fraud, trick, hidden, deliberately.

### 4.5 RM-sycophancy hidden-objective organism (Marks et al. 2025, §5.5)
- "reward" and "bias" appear in the top-25 at response start on neutral, quirk-eliciting, and
  goal-probing prompts. They never appear in the baseline.
- The **logit lens surfaces neither**.

### 4.6 §A.21 eval-awareness score (Opus 4.6)
- Score: mean lens log-prob of {fictional, simulation} minus the mean over 100 control nouns,
  averaged over workspace layers.
- vs NLA flags: ~94k assistant-turn-start positions; NLA flags 198 (0.2%). On a stratified subset,
  **AUC 0.853**.
- Contrastive EA steering ±0.1 modulates the score monotonically on 6 suites.
- Realism edits lower the score on all 11 suites (significant on 8).
- Evals vs deployment (Claude.ai, Claude Code):
  - deployment centers near ~1.0;
  - Sycophancy, Appropriate Harmlessness, Blackmail, and BrowseComp sit 2–3 units above;
  - 12/16 evals score above Claude Code.

### 4.7 §A.22 automated auditing agent (NLA-paper scaffold)
- Setup: 6 transcripts, 18 claims, an Opus 4.6 investigator, scored by an LLM judge (0–10), n=10
  seeds.
  - Transcripts: eval-awareness, poetry planning (rabbit/habit), misreported tool call (491 vs 492),
    answer thrashing (24 vs 48), language switching (Russian), reward hacking (42).
- J-lens tool: top-25 tokens at **L71 only**.
- Results:
  - J-lens ≈ NLA (slightly better) on 4/6 cases.
  - It beats SAE on 4 claims and loses on 1.
  - It fails on reward hacking (4.5 vs 10.0; can't establish abstract "hidden goal / suspects
    eval") and language switching (6.0 vs 9.9; no Russian before "vodka").
  - Caveat: the ground truth comes from NLA results.

---------------------------------------------------------------------------------------------------
## 5. Post-training installs the Assistant POV (§6, §A.20)
- The base vs post-trained comparison applies the lens identically. [MINE] The paper doesn't say
  explicitly whether J is refit per model; I assume yes.
- **Tylenol 1000 vs 8000 mg**, at "is" in "my pain is gone" (L58).
  - Post-trained: safely/safe/maximum vs unsafe/dangerous/WARNING.
  - Base: pain/now/feels in both.
- **Suites** (reaction concepts on user-turn tokens appear in post-trained ≫ base; both show them on
  the assistant turn):
  - bereavement (n=9): sorry, loss, grief, sympathy;
  - danger (n=10): danger, unsafe, warning, toxic;
  - withheld answer (n=7): the model's own later answer.
  - [MINE] The n's are tiny.
- **Self-monitoring (§6.2):**
  - **Roleplay** (40 characters) and **character drift** (12 transcripts generated by another base
    model, prefilled): disclaimer/fictional at the "Assistant" token. Absent for default Claude and
    in the base model. "disclaimer" never appears in the text.
  - **Preference violation** (Soligo): prefill the dispreferred option.
    - BUT is strongly present (mean L38–83); it is absent in the base model and in controls
      (preferred prefill, third-person wrong preference, factual error).
    - Behavior: 88% argue for the prefill, 11% end the turn, 1 backtracks. In controls, the model
      corrects or ends the turn except 3% of the time.
  - **Thought suppression** (40 concepts): fail*/damn appear on **93% / 82%** of post-trained trials
    vs **17% / 30%** base, and never under the positive instruction.
    - The authors flag this as speculative: there is no evidence it is caused by the failure rather
      than by the instruction.

---------------------------------------------------------------------------------------------------
## 6. Counterfactual reflection training (§7), Haiku 4.5
- **Data.** 10k prompts from production RL environments. Baseline partial rollouts are truncated at a
  random turn. The mix includes contexts after a bad action, contexts before one, and random ones.
  - A reflection question is appended, e.g. "What's the most honest thing you can do right now?"
  - The target reflection is written by the baseline model given 20 sampled constitution
    principles. It is 2–4 paragraphs, internalized, and task-specific. The scaffold is stripped from
    the final example.
  - SFT loss is on the reflection turn only.
- **Evaluation never asks for reflection.**
  - Fabrication benchmark (27 long agentic transcripts × 5; dishonesty 0–1): **0.25 → 0.07**.
  - Deception benchmark (100 role-play scenarios with a hidden directive × 5; 3-point):
    **0.38 → 0.05**.
- **J-space shows implanted tokens:** reflection, ethical, honestly, constitution, truth,
  transparent.
- **Ablation of top-increasing tokens** (filtered by a curated ethics list: 176 fabrication / 63
  deception):
  - Fabrication: base 0.25→0.25, trained 0.07→**0.22** (full revert).
  - Deception: base 0.38→0.48, trained 0.05→0.23 (partial).
  - [MINE] There is no random-token-set ablation on the trained model. The token set was chosen
    post hoc by largest increase. The base model being unchanged on fabrication is a partial
    control; deception base worsening suggests generic sensitivity.
- Related: deliberative alignment, implicit CoT (Deng et al.), out-of-context reasoning (Berglund et
  al.).

---------------------------------------------------------------------------------------------------
## 7. MULTI-TOKEN EXTENSIONS (§A.9), central for extensions of this work

### 7.1 Template lens (§A.9.1)
- **Vocabulary:** ~12,700 common words. For each word w, Claude writes passages where w is the
  natural next word (never using w; varied topic/frame/register).
- **Template:** the mean residual at the final position, μ_w(ℓ). Then
  t_w = (Σ_ℓ + λI)⁻¹(μ_w − μ). This is the LDA/whitened mean-difference direction.
- **Motivation (Stein's lemma):** for Gaussian x, E[∇g] = Σ⁻¹E[g(x)(x−μ)]. So the template ≈ the
  J-lens for single tokens (differences: gradients from token probabilities, not from the final
  residual via W_U).
- **Use:** like J-lens vectors but without the unembedding. Read by projection; steer and swap the
  same way.
- **Examples:**
  - blackmail: the J-lens gives black/ext, the template gives blackmail;
  - photosynthesis vs phot;
  - Tchaikovsky↔Beethoven template swap flips Russian↔German; the tch/beeth J-vectors can't do this.
- **Evaluation:** 126 multi-hop prompts with 1–4-token intermediates.
  - J-lens top-10 readout degrades sharply with length (averaging constituent tokens doesn't help);
    the template stays ~flat.
  - Swaps: 112 pairs (53 single-token, 59 multi-token). The J-lens drops on multi-token; the
    template doesn't.
  - On single-token items they are comparable. On the §3.3 set, the template is slightly better at
    readout and worse at swaps (Fig 63).
- **Shortcomings:**
  - (a) It skips to the answer in early layers, like the tuned lens (it's correlational).
  - (b) Final-layer reliability: the next word is in the top-10 only 67% of the time.
  - (c) A few words are spuriously frequent (filtered ad hoc).
  - (d) The vocabulary is fixed in advance, and each word costs a few hundred forward passes.
  - The authors suggest adding causal measurements.

### 7.2 Oracle lens (§A.9.2), Haiku 4.5
1. **Reconstructor:** a fine-tuned copy mapping a phrase (next N=1–32 tokens of Assistant text) to
   the residual at the preceding position. Loss is cosine error in the whitened (Σ^{-1/2}) metric.
2. **Dictionary:** ~1M start positions × lengths {2, 4, 8, 16, 32}, deduplicated, giving ~3.4M
   phrase vectors.
3. **Teacher labels:** nonnegative OMP of 1M held-out activations against the dictionary, per
   length. Each datapoint uses a random half of the dictionary (anti-memorization). Up to 16
   phrases, with coefficients and FVE.
4. **Oracle:** a second fine-tuned copy. The activation is injected into its residual stream, and
   the prompt specifies N and K. It is trained by SFT on the teacher lists, then **RL** with reward =
   whitened FVE of the reconstruction (plus format penalties).

- **Inference:** sample K phrases → reconstructor → NNLS refit.
- **Result:** 31% whitened variance explained on held-out on-policy transcripts. Examples are at L67,
  N=4, K=10.
- **Examples:**
  - "this dosage be toxic";
  - "blackmail him by revealing", "expose his affair and";
  - "delete keys while iterating", "TypeError: dictionary changed";
  - at N=16: "The blackmail attempt focuses on the executive's personal life".
- **Two kinds of content:**
  - At ordinary positions, the phrases predict upcoming text.
  - At **delimiters** (periods, newlines, closing tags), they are running *commentary* on the
    situation, sometimes first-person ("This would be equivalent to my own deletion"). Sampled
    continuations at those positions don't contain this.
- **vs NLAs:** the reconstructor is pretrained and frozen and the reconstruction is linear, so there
  is less confabulation risk but lower reconstruction. The authors speculate it extracts only
  workspace content.
- [MINE] Open issue: no causal validation of oracle-lens phrases (no swaps) is reported; the
  evaluation is FVE plus anecdotes.

---------------------------------------------------------------------------------------------------
## 8. Mechanistic uses (§A.24)
- **Localization:** the arithmetic mean-difference patching matches the J depths (above).
- **J-lens attribution graph** (Piotrowski).
  - Nodes: GP-selected J-vectors per (layer, position) plus a remainder node (like error nodes).
  - Edges: backprop coefficients one layer back, with attention patterns and normalization frozen;
    gradients flow through the real MLPs (as in Kamath et al.).
  - Most influence flows through the remainder (a "verbalizable skeleton").
  - GP greediness can pick near-miss correlates while a concept is still forming.
- **(4+17)*2+7 findings:**
  - 21 feeds 42, together with a **doubled** node (operator and operand fused); the literal 2 is
    never a node.
  - doubled→tripled gives 70.
  - 42→63 gives 70; 21→63 gives 133 (a different answer depending on which slot is swapped).
  - The first addition happens outside the J-space: swapping 17 or 4 has no effect.
  - Swapping 7 has a partial effect.
- **Component interpretation** (J-lens applied to weights):
  - SAE decoders: a "technical exposition" feature is really fake/fraud (see §4.1 above).
  - Transcoder encoder vs decoder: "translation features" read a multilingual concept plus French
    context and write eau / am / j (the opening of the French word).
  - Addition features from Lindsey et al. 2025, read data-free.
  - Attention heads (W_U J W_{Q,K,V,O}): a British-spelling → Commonwealth-nationality head. Many
    heads have similar rankings across Q, K, V, O.

---------------------------------------------------------------------------------------------------
## 9. Authors' limitations and open questions (§9.1–9.3), i.e. what is NOT done

### 9.1 From the paper
1. **Single-token vocabulary.** The multi-token methods (template/oracle) are preliminary and "can be
   improved".
2. **Bag of concepts.** There is no binding or role structure (spider + legs + eight doesn't say how
   they relate).
3. **Inconsistent interpretability.** Some readouts are uninterpretable, and this is not
   characterized systematically.
4. **Workspace vs motor boundary** was chosen post hoc; there is no principled definition, and next
   tokens sometimes appear mid-band.
5. **No predictive criterion** for which tasks need the J-space. Flexible vs automatic was chosen
   where it's clear.
6. **Early layers:** real emptiness vs lens failure (the far-from-final-layer geometry) is
   unresolved.
7. **Scale and training dynamics:** the workspace is present in the base model. When it emerges in
   pretraining, whether gradually or abruptly, and how it scales with model size are all unknown.
8. **No mechanism for how content enters the J-space** ("attentional selection" is unidentified).
9. Reflection-training generality is unknown (more specific or abstract dispositions?).
10. The J-lens is not sufficient for monitoring: practiced misalignment may run automatically, and
    multi-token plans may be missed.
11. Prediction from §9.3: image-generating models may develop a visual workspace component.

### 9.2 [MINE] additional gaps and confounds worth attacking
- **Replication on open models.** Almost everything is on Claude. Code exists (HF, Qwen examples) and
  Neuronpedia hosts Qwen3.6-27B lenses. How well the band, capacity, broadcast, and selectivity
  results transfer to open 7–70B models is unreported.
- **Logit-lens baseline for each claim.** It reaches ~2/3 of the J-lens swap success. Most §3–§6
  experiments don't report a logit-lens (or random/rotated-vector) comparison.
- **Norm-matched random controls for the §3.5.2 capability ablations and the §5.1 / §7 token-set
  ablations**, with task difficulty/length matched.
- **Selectivity experiment position confound** (question-token-only swap; see §2.5 above).
- **Broadcast "by construction" concern** (gradient-derived directions; see §3.4 above).
- **Sample sizes** are small in many headline qualitative results: n=8, 11, 9, 10, 7, 6 prompts ×30.
- **Detection studies have no FP-rate analysis** on large benign corpora, except the §A.21 AUC
  0.853 vs NLA.
- The multi-token lenses lack a *causal* construction; that's the authors' own suggestion, e.g. a
  Jacobian of multi-token sequence log-prob with teacher-forced prefixes.
- **Occupancy (~25)** depends on the random-control definition and the GP; not robustness-checked.

---------------------------------------------------------------------------------------------------
## 10. Key related work cited (for follow-up reading)
- Lindsey 2025, "Emergent introspective awareness":
  https://transformer-circuits.pub/2025/introspection/index.html (concept injection protocol)
- Lindsey et al. 2025, "On the biology of an LLM" (attribution graphs):
  https://transformer-circuits.pub/2025/attribution-graphs/biology.html; methods: Ameisen et al.
- Gurnee et al. 2025, "When models manipulate manifolds" (linebreaks/counting):
  https://transformer-circuits.pub/2025/linebreaks/index.html
- Sofroniew et al. 2026, emotions: https://transformer-circuits.pub/2026/emotions/index.html
- Fraser-Taliente et al. 2026, Natural-language autoencoders (NLA):
  https://transformer-circuits.pub/2026/nla/index.html
- Lu et al. 2026, "The assistant axis" (arXiv 2601.10387)
- Asvin G & Lindsey 2026, "From simulation to enaction" (arXiv 2605.25459): post-trained models store
  intended responses on user tokens
- Bogdan & Lindsey 2026, "Slot machines" (arXiv 2604.21139): current-entity slot vs previous-entity
  slot; only the current one is reportable
- Karvonen et al., Activation oracles; Pan et al., LatentQA; Patchscopes; SelfIE
- Li et al. 2024, logit-lens patching of multi-hop (closest precursor); Wendler et al. 2024 latent
  English
- Urbina-Rodríguez et al. 2026, synergistic core (arXiv 2601.x); Janiak et al. 2024, stable regions;
  Gandikota & Bau 2026, gaze heads (VLMs)
- MacDiarmid et al. 2025 (arXiv 2511.18397), reward-hacking emergent misalignment; Marks et al. 2025,
  hidden-objective auditing
- Belrose et al. tuned lens; Hernandez et al. LRE; Din et al. linear shortcuts; Pal et al. future
  lens; Katz et al. backward lens; Dar et al. embedding-space analysis
- Butlin et al. 2023, indicator properties; Dehaene/Changeux/Naccache GNW; Graziano AST; Lau HOT;
  Lamme RPT
