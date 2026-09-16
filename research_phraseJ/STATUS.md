# STATUS — research_phraseJ

Updated 2026-09-16.

## Decided

- Model: Qwen3.6-27B only. Scale to other models only if a method beats the released template lens.
- Baseline for multi-token claims: released template lens (`templates+phrases_v3`, 13,731 rows).
- Baselines for single-token claims: released J, released R, logit lens.
- Development eval: 59-item calibration slice (seed 0, 10/category) from `data/evaluations/`;
  frozen sets untouched until final claims.
- Design doc agreed before any experiment code.
- Branch stays clean off `main`; `ekko-lens` is read-only reference; borrowed code lives in
  `scripts/lib/` with attribution.

## Environment

Verified end to end 2026-09-16 (see CLAUDE.md): model + released J lens load and reproduce the
web-spinner → `spider` readout at L40–46.

## Candidate first experiments (designs to be written and agreed)

| # | slug | question | GPU |
|---|---|---|---|
| 001 | template-pretest | Are minimal pairs (blackmail/blackboard …) separable in the released template rows at workspace depth? Stein-vs-J cosine; dissociation angle ∠(v, Σ⁻¹v). | ~0 (one forward-pass corpus for Σ) |
| 002 | phraseJ-single-token-control | Does phrase-J with m=1 on released passages reproduce the released J row (lag-matched)? | ~1 h |
| 003 | gradient-template-2x2 | On identical passages, does the gradient estimator remove the template lens's skip-ahead? Single- and multi-token words. | ~1–2 h |

## Running

Nothing.

## Next

Write `research_artifacts/001-template-pretest/design.md` and agree it.
