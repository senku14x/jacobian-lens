# STATUS — research_phraseJ

Updated 2026-09-16 (evening).

## Decided

- Model: Qwen3.6-27B only; scale only if a method beats the released template lens.
- Baselines: released J, R, logit lens, released template lens (`templates+phrases_v3`).
- Development eval: 59-item calibration slice for single-token; new multi-token two-hop Dataset A
  built under lens-independent mechanical admission (see `docs/06_plan_decisions.md`).
- Design agreed before any code. Branch clean off `main`; `ekko-lens` read-only.
- Plan revised after two review rounds; decisions in `docs/06_plan_decisions.md`.

## Designs

| # | slug | state |
|---|---|---|
| 001 | template-geometry | **done** — report in `research_artifacts/001-template-geometry/report.md` |
| 002 | phraseJ-exact-compat | draft — awaiting agreement |
| 003 | gradient-template-2x2 | draft — awaiting agreement |
| 004 | causal-geometry | stub — full design after 003 |
| 001b | stein-diagnostic | not written; deferred |

(`001-template-pretest` v1 was superseded before running and removed.)

## Environment

Verified 2026-09-16: model + released J load; web-spinner → `spider` rank 4 @L40.

## Running

Nothing.

## Next

002 (exact-J compatibility gate) on go-ahead. 001 findings: latent phrase identity within a first-token family is linearly decodable at L52–62 (cue-out AUC 0.94–1.00) and at 0.72–0.83 in L8–L48 where J-sum reads 0.55–0.83; covariance-duals fail the 0.95 stability bar; J rows and template rows near-orthogonal.
