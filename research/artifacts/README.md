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
