# Artifacts

One directory per analysis: `NNN-YYYY-MM-DD-slug/report.md` plus `figures/`. Start from
[`_TEMPLATE.md`](_TEMPLATE.md). Numbers must be reproducible from `research/experiments/NNN_slug.py`
at the commit named in the report.

Sequence numbers are permanent. A superseded analysis keeps its number and gets a status line
pointing at what replaced it.

| # | Date | Title | Phase | Status | One-line result |
|---|------|-------|-------|--------|-----------------|
| [000](000-2026-08-11-environment-baseline/report.md) | 2026-08-11 | Environment and fit-cost baseline | execution | current | Fit path runs on the H100; measured per-prompt cost sets the compute budget for any lens fit |
| [007](007-2026-08-12-transport-operators/report.md) | 2026-08-12 | Which operator transports an intermediate representation? | validation | current | Revised after external review. Fitted secant fails 3 span-immune gates AND collapses to ~0 under a template-disjoint split (the in-family win was memorisation); released J-lens loses 2–4× to input-specific transport and scores relerr ≈1.0 (no better than predicting zero); SmoothGrad-J beats local FD transport by +0.17…+0.27 but the **mechanism claim is retracted** — the orth/iso discriminator is vacuous in d=5120 |
| [008](008-2026-08-12-lens-improvements/report.md) | 2026-08-12 | Four proposed lens improvements, tested | validation | current | Reference anchoring, spectral shrinkage and mode-wise J/R fusion are all negative; R̄ and the 3-scalar blend aJ̄+bR̄+cI remain the best fixed operators under a category-disjoint split |
| [009](009-2026-08-12-jown-and-backward-rules/report.md) | 2026-08-12 | A fitted lens, an estimator noise floor, and the backward rules R-lens omits | validation | current | First fitted lens (J_own, 183 s/prompt): matches released J to within noise, so the uncontrolled-comparison worry was nil. Twin-fit noise floor ≈0.01 cos — the R>J effect-prediction advantage does not clear it, while context averaging clears it 15–47×. Q/K-norm and attention-gate rules negative or nil; routing-desaturation rules show the predicted finite-ε crossover but at floor magnitude. bf16 finite difference at ε_ref=0.01 has cos 0.93 to the true JVP |
