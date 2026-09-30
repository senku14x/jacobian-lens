# 002 — Distance-matched lenses on the words the model later wrote: report

**Date** 2026-09-30 · **Verdict: KILL (pre-registered)** · **Design** `designs/002-future-words-own-text.md` (committed
before the run) · **Code** `scripts/002_future_words.py` (run), `scripts/002_analysis.py` (committed before any real
data; dry-run on a 1-item smoke file only) · **Raw** `cells.jsonl`, `items.json`, `summary.json`, `qualitative.md`;
vectors in `outputs/002_vecs/` (gitignored). GPU ≈ 44 min. Nothing fitted; no judge.

## Question
The general version of the lag-bucket idea (spec II.5). At a position inside Qwen3.6-27B's own response, does a lens
counting only influence 4–8 (or 9–16) tokens ahead rank the words Qwen *actually wrote* that far ahead higher than
ordinary J̄ does? Is that item-specific?

## Setup
- **Data:** 50 seeded responses from WorkspaceBench's hallucination bank, using the **verbatim sampled token ids**
  (Qwen3.6-27B, T=1.0).
- **Read position:** one seeded position per response, with ≥ 33 response tokens after it.
- **Targets:** content words (word-initial, alphabetic, ≥ 3 letters, not a stopword, not present earlier in context,
  first occurrence) at lags 0–32. Median 8 per position.
- **Decoy:** another response's future words at the same lags.
- **Estimator identical to 001 v2:**
  - exact single-source matrix-free lens;
  - 32 fresh pile hosts × 8 positions;
  - true fp32;
  - layers 40/48/56.
- **Comparator:** ordinary J̄ on the same hosts (FULL_U). J_CB, J_NP and the logit lens are also recorded.

## Pre-registered decision
Mean per-item log10-rank difference, matching window − FULL_U (**positive = window worse**; −0.30 would be 2× better):

| layer | L2 (4–8), window B2 | L3 (9–16), window B3 |
|---|---|---|
| L40 | +0.233 (Holm p 0.005) | +0.423 (Holm p 2e-7) |
| L48 | +0.106 (n.s.) | +0.149 (Holm p 0.015) |
| L56 | +0.245 (Holm p 0.015) | +0.402 (Holm p 6e-5) |

- **Kill rule met.** No (class, layer) improves on J̄ by ≥ 1.5×. Every matching window is *worse*: 1.3–2.6× in
  geometric-mean rank, and 2.1–3.5× for lags 17–32.
- **No specificity gain** (own-vs-decoy AUC differences within ±0.06, n.s.).

## Controls
- **Positive control passes at all three layers.** B0 finds the actual next word far better than B3 (all tokens: mean
  log10-rank difference −1.31 / −1.41 / −1.67, p ≤ 3e-7; content-only next words, n = 12: p ≤ 0.034).
- **Reliability** (split-half Spearman): B2 0.83–0.94, B3 0.75–0.87, FULL_U ≥ 0.98. The windows are measured reliably
  here, so the deficit is not noise.
- **All-token secondary:** the same direction, all six comparisons significantly worse.

## Descriptive (L48; median over items of mean log10 rank)
Future content words 4–8 and 9–16 tokens ahead sit at log10 rank ≈ 3.4–3.8 (rank ~2,500–6,000) under **every** lens,
J̄ and the released lenses included. The next word (lag 0) is at 1.7–2.5 (rank ~50–300).

## What this supports (Qwen3.6-27B, 50 responses, layers 40/48/56)
- **Supported negative:** re-weighting J̄ toward a target distance does not surface the words the model later writes at
  that distance. The matching window is reliably worse than J̄ itself.
- **Observation:** no linear lens tested (J̄, the released J_CB/J_NP, logit) ranks words 4+ tokens ahead anywhere
  near the top of the vocabulary. Whatever the model "knows" about its words 4–16 tokens ahead, these lenses do not
  expose it.

## Joint reading with 001 (002's pre-registered decision table)
Poetry null + own-text null, with positive controls passing (001 at L48/56; 002 at all layers): **distance
re-weighting of J̄ does not help. Close the II.5 lag-bucket branch.**

## What this does not establish
- That future words are *absent* from the activation. A probe or a decoder could still read them; that is II.7's
  question.
- Anything about other hosts (chat/poem), other lens definitions (horizon, report lens), other layers or models.
