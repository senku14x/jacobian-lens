# Artifacts

One directory per analysis: `NNN-YYYY-MM-DD-slug/report.md` plus `figures/`. Start from
[`_TEMPLATE.md`](_TEMPLATE.md). Numbers must be reproducible from `research/experiments/NNN_slug.py`
at the commit named in the report.

Sequence numbers are permanent. A superseded analysis keeps its number and gets a status line
pointing at what replaced it.

| # | Date | Title | Phase | Status | One-line result |
|---|------|-------|-------|--------|-----------------|
| [000](000-2026-08-11-environment-baseline/report.md) | 2026-08-11 | Environment and fit-cost baseline | execution | current | Fit path runs on the H100; measured per-prompt cost sets the compute budget for any lens fit |
