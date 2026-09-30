# 002 — Do distance-matched lenses read the words the model will write later? (design, written before any run)

**Date** 2026-09-30 · **Model** Qwen3.6-27B · **Depends on** 000 (validated matrix-free lens), 001 Amendment 3
(exact single-source estimator, same code path) · **Status** pre-registered · **Nothing is fitted.**

## Why
001 tests the lag-bucket idea (spec II.5) on one family: planned rhymes. A result there licenses at most "planned
rhymes". This eval tests the **general** claim behind the idea: *the standard J-lens counts a state's influence on
every later token, weighted toward the next few, and so hides what the model will say several tokens later.* If that
is true, it should show on ordinary text too, and most cleanly on the model's own text, where the future words are by
definition the ones this model went on to produce.

## What we want to find out
At a position inside one of Qwen3.6-27B's own responses, does the window counting influence 4–8 (or 9–16) tokens ahead
rank the words Qwen actually wrote 4–8 (9–16) tokens later higher than ordinary J̄ does? Is that item-specific, rather
than a generic boost to content words?

## Data (no model generation needed)
- WorkspaceBench `evals/hallucination/items.json`: 149 real chat prompts (LMSYS etc.), each with Qwen3.6-27B's own
  response (text).
- **Tokens:** the **verbatim token ids** from WorkspaceBench `evals/hallucination/capture_rows.json`. These are the
  chat-templated prompt (thinking off) plus the response exactly as Qwen3.6-27B sampled it (HF `generate`, T=1.0,
  seed 7, ≤512 new tokens). `prompt_len` marks the response start. Re-tokenising the response text is avoided because
  it could change token boundaries, which would silently break "the model's own tokens". (Chosen before any run, after
  reading the bank's README.)
- **Items:** a seeded random 50 (seed 20260930) among items whose response has ≥ 48 tokens.
- **Read position t:** uniform (same seed) in [response_start + 8, len − 34], so 33 future tokens exist inside the
  response.
- **Targets:** for δ = 0..32, the token at t+δ+1 (the token y_{t+δ} predicts).
  - **Eligible "future words":** word-initial tokens (decoded string starts with a space), alphabetic after stripping,
    ≥ 3 letters, not in the fixed stopword list in the script, whose token id does **not** occur in ids[0..t] (no
    copying from context). Only the first occurrence within the window counts (a unique lag).
  - Lag classes: L0 = {0}, L1 = 1–3, L2 = 4–8, **L3 = 9–16**, L4 = 17–32.
- **Decoy targets (specificity null):** a seeded derangement pairs each item with another item. The partner's
  eligible future words, with their own lags, are ranked under this item's readouts.

## Estimator and readouts
Identical to 001 v2:
- exact single-source matrix-free lens;
- 32 fresh pile hosts (docs from index 100, full 128 tokens), 8 stratified positions per host;
- layers 40/48/56.

Readouts:
- windows B0–B4 (lag weighting (N_p−δ)/N_p, as in J̄) and FULL32;
- FULL_U (ordinary J̄h, the **primary comparator**);
- J_CB, J_NP, the logit lens.

Ranks are over the full 248k vocabulary under U = (1+γ)⊙W_U. Readout vectors are saved (fp16, gitignored).

## Pre-registered analysis (unit = item/position, paired)
- **Per item, class c and readout r:** mean log10(rank + 1) of that item's eligible targets in class c.
- **Primary:** for c ∈ {L2, L3} at each layer ∈ {40, 48, 56}, the matching window (B2 for L2, B3 for L3) vs FULL_U.
  Paired Wilcoxon over items that have ≥ 1 target in c. Holm over the 6 tests.
- **Success** at some (c, layer): Holm p < .05, AND a geometric-mean rank improvement ≥ 2× (mean log10-rank difference
  ≤ −0.30), AND specificity at the same (c, layer). Specificity means the item's own targets outrank its decoy targets
  (same class) by more under the window than under FULL_U: paired Wilcoxon p < .05 on the per-item AUC
  own-vs-decoy (window minus FULL_U), with a positive mean difference.
- **Kill:** no (c ∈ {L2, L3, L4}, layer) shows a matching-window improvement ≥ 1.5× (mean log10-rank difference
  ≤ −0.176).
- **Positive control:** at each layer, B0 ranks the L0 target (the actual next word, when eligible) better than B3
  does (paired Wilcoxon p < .05). If this fails, a null is reported as "instrument insensitive".
- **Secondary:**
  - all tokens (including function words and fragments), the same comparisons;
  - host split-half reliability;
  - a per-lag curve (median rank vs δ for B-windows and FULL_U);
  - 10 random items + 3 best + 3 worst (L3, L48) as qualitative samples in `results/002-*/qualitative.md`.

## Registered predictions
- **Mine:** the positive control passes (next words are easy). For L2/L3, I expect **no success**: prior 15% for L2,
  10% for L3. What the model "knows" about a word 4–16 tokens ahead in open-ended chat is usually weak (the future-lens
  literature finds decodability falls fast with distance), and generic web-text hosts don't help. A success would be
  surprising and important.
- **If 001 (poetry) wins and 002 is null:** planning-specific. Rhymes are committed early, and free text isn't.
- **If both are null:** distance re-weighting of J̄ does not surface future content. Close this branch, and consider
  domain-matched hosts only if 001's positive control passed.

## Decision table (read with 001)

| 001 poetry | 002 own-text | reading | next |
|---|---|---|---|
| win | win | J̄'s lag weighting broadly hides future content | replicate (Anthropic poetry set; a second model), then adopt distance-matched readouts as a standard option |
| win | null | specific to planned content | test other planned content (idioms, constrained generation) |
| null | win | distance-matching works for generic future words but not rhyme plans on web hosts | domain-matched (poem) hosts for 001 |
| null | null | re-weighting by distance doesn't help | close the II.5 branch |

## Cost
50 items × 3 layers × ≈17 s ≈ 45 min on the H100, forward passes only. It runs after 001's remaining 45 items (≈38
min), sequentially on the one GPU.
