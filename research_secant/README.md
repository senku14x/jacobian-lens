# research_secant — matrix-free J-lens, secant decomposition, and method upgrades

Branch `research_secant` of `senku14x/jacobian-lens` (local clone `/workspace/ekko-lens`), forked from `ekko-lens` at 89ae076.
Read `STATUS.md` first (what was run and why), then `docs/00_user_spec_secant_and_upgrades_2026-09-30.md` (the method
spec, verbatim) and `docs/01_spec_review_2026-09-29.md` (our review of it).

## Layout
```
research_secant/
  README.md  STATUS.md  requirements.txt
  docs/            00 = the method spec (verbatim), 01 = our review of it
    notes/           reading notes: the workspace paper, follow-up posts, prior repos (ekko-lens, phrase_J, phrase-lens,
                     RHUT), seeded WorkspaceBench screening subsets
    workspace_bench/ our local WorkspaceBench patches (mojibake + OOM fixes) and findings for its authors
  designs/         NNN-slug.md    written and committed BEFORE a run: why, what, predictions, decision rules
  results/NNN-slug report.md after the run + compact JSON/JSONL outputs
  scripts/         NNN_slug.py;   lib/engine.py (bf16 capture, fp32 upper-stack replay), lib/mfl.py (matrix-free lens)
  plots/           figures
  outputs/         gitignored: logs, large arrays
```

## Conventions
- Qwen/Qwen3.6-27B @6a9e13bd. Layer ℓ = output of block ℓ. J_CB = `camilablank/workspace-lenses`
  `qwen3.6-27b/j-lens/lens.pt` (target 62, t_max 128, skip_first 4, the first 25 pile-10k docs, no BOS; the docs were
  confirmed in 000).
- Readout covector U = (1+γ)⊙W_U (the fold was verified in 000).
- Finite differences run in true fp32 (TF32 off) through blocks ℓ+1..62. Never in the bf16-autocast regime (000 G3).
- Every run: design committed first, then shown to the user, launched only on their go, then a report.

## Index
| # | slug | state | one line |
|---|---|---|---|
| 000 | matrix-free-lens | done | reproduces J_CB at cos .997/.998/.999 (L40/48/56); fit corpus identified |
| 001 | lag-bucket-poetry | done: **KILL** | distance-matched windows do not surface planned rhymes; worse than J̄ at every layer |
| 002 | future-words-own-text | done: **KILL** | nor the words Qwen actually wrote 4–16 tokens later; with 001, the lag-bucket branch is closed |
