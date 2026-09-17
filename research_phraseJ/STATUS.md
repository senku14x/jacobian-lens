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
| 002 | phraseJ-exact-compat | **done** — gate PASS (Amendment 2); report in `research_artifacts/002-phraseJ-exact-compat/report.md` |
| 003a | phrase-objective | **done** — `lin` reliable at L56–60 but no readout gain over J-sum; constituent ceiling = probe except San; report in `research_artifacts/003a-phrase-objective/report.md` |
| 003a-controls | symmetry / San-likeness / mid-band specificity | **done** — Branch B: mid-band gap is generic early-J (single tokens show the same +0.24 at L8); San is the only late residual; matched Phrase-J ≤ J-sum except San L52; `lin` converges at L52+ and is a lag-specific covector; report in `research_artifacts/003a-controls/report.md` |
| 003 | gradient-template-2x2 | superseded — generic readout benchmark dropped (Branch B) |
| 004 | causal-geometry | stub — full design after 003 |
| 001b | stein-diagnostic | not written; deferred |

(`001-template-pretest` v1 was superseded before running and removed.)

## Environment

Runtime crashed after 001; rebuilt 2026-09-17 (`scripts/000_env_verify.py`, infra, no design doc by
agreement): model 52 GB at `/content/models/Qwen3.6-27B`, released lenses 24 GB at `/content/lenses/qwen3.6-27b`
(j, r, both template stacks, all passage files), pile-10k cached; smoke test reproduces (`spider` rank 4 @L40,
rank 8 @L46); one `v_lin` backward = 2.3 s at 59 GB peak. `outputs/` is empty: 001b (Σ, nulls, token
frequencies) must be rerun before 003.

## Running

Nothing.

## Next

Branch B confirmed by 003a-controls. Next: 004 design (intervention geometry) — swap-validity harness first, then
`lin` difference / family-centred vectors vs J-sum composites vs template rows as latent-intermediate swaps on the four
geographic families at L52–56, with dose-response, norm-matched random, unrelated-output preservation, reversal, and
constituent-J intervention as comparator. Cache of all outputs (001/002/003a) is at HF `senku21x/phraseJ-cache`
(private).

002 findings: implementation exact at the fit's graph shape (cos 0.9999 at every layer incl. L8); the
early-layer discrepancy between batch-1 and batch-4 bf16 backwards is real, deterministic per shape, and
vanishes by ~L44 (relerr 0.41 @L8 for dense contractions, 0.13 for single rows); same-token rows across
prompts have cos 0.03 @L8 → 0.95 @L60; log-prob phrase objects saturate in emission contexts (nulls are more
reliable than real phrases); `per_token_fast` is exact. 001 findings: latent phrase identity within a first-token family is linearly decodable at L52–62 (cue-out AUC 0.94–1.00) and at 0.72–0.83 in L8–L48 where J-sum reads 0.55–0.83; covariance-duals fail the 0.95 stability bar; J rows and template rows near-orthogonal.
