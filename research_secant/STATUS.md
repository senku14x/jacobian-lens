# STATUS — research_secant (what is running, why, and what we expect)

Updated 2026-09-30. Branch `secant` of `/workspace/ekko-lens`. Layout and conventions are in `README.md`.

## The big picture in three sentences

Your spec's central tool is a **matrix-free J-lens**: compute "J̄ times a vector" directly from forward passes on
ordinary text, instead of using the stored 5120×5120 J matrix. Once that tool reproduces the released lens exactly,
the *definition* of the lens can be changed at no cost, with no refit. Examples: count only effects 9–16 tokens ahead,
count only effects inside this sequence's horizon, or use chat text instead of pile text. Today's runs are (000)
proving the tool is exact; (001) using it for the cheapest decisive test in your spec, namely whether the J-lens misses
planned rhyme words only because its definition is dominated by short-range effects; and (002) the general version of
that test, on the words Qwen actually wrote later in its own responses.

Nothing is being fitted. Every quantity is a forward-pass average over host documents.

---

## 000 — Is the matrix-free lens exactly J_CB? DONE, all gates pass

- **Design:** `designs/000-matrix-free-lens.md`, committed before the run (04c69ba).
- **Report:** `results/000-matrix-free-lens/report.md`.

**Why.** Every later quantity (horizon gap, lag buckets, secant lens) is built on this tool. If it didn't reproduce the
released lens, every gap downstream would include a definitional mismatch. Your spec called this "the single most
important check".

**What it does.** For each test vector v (random, a real activation, or a difference of activations) at layers
40/48/56:
- add +εv and −εv at every valid position of a host document;
- run the rest of the network in true fp32;
- difference the summed block-62 outputs, and average over documents;
- compare against J_CB·v.

**What we expected (pre-registered).**
- ε and ε/2 agree (90% prior).
- It matches J_CB at cos ≥ 0.99 if we guess the right 25 fit documents (60% prior, since the checkpoint doesn't
  record which docs were used).

**What we found.**
- Precision gate: pass (cos 0.9999996+).
- **It reproduces J_CB at cos 0.997 / 0.998 / 0.999** (L40/48/56), with norm ratio ≈ 1.00, on the first 25 pile-10k
  docs with no BOS. That also identifies J_CB's undocumented fit corpus.
- On 31 fresh documents it gets 0.97–0.995. That's the sampling floor for any later comparison.
- Side result: the spec's original plan (finite differences at ε = 1e-3 under RHUT's bf16-matmul regime) gives
  cos 0.80. It would have been ~20% wrong in direction, which is why the true-fp32 upper stack is needed.

---

## 001 — Lag-bucket lens on planned rhymes (your spec II.5): RUNNING (50-item subset)

Design `designs/001-lag-bucket-poetry.md`. Amendments 1–4 are each committed before the runs they govern.

- **v1** (the all-positions ± estimator) was stopped at 101/500 cells. The gate showed its 9–16-token window was
  mostly noise (split-half 0.61 / 0.11). It is kept as a record; **do not analyse it**.
- **Gate G_lag** (`results/001-lag-bucket-poetry/gate_lag_report.md`): the spaced estimator failed its leakage check,
  so the main run was not launched then.
- **Amendment 3** (you approved):
  - exact estimator, one perturbed position per forward pair;
  - layers 40/48/56;
  - per-layer primary analysis (your spec's rule) instead of best-over-layers;
  - reliability preflight passed (0.83 / 0.85).
- **Amendment 4** (you asked for 50 items):
  - a seeded random 50, split 25/25 by source (`v2/subset50.json`); 5 are already done, so 45 remain;
  - thresholds unchanged. At n≈50 only ~15–20-point gains are detectable. Effect-size-but-not-significant counts as
    "borderline", in which case we add the other 50.
- Analysis script `scripts/001_analysis.py`, committed before any aggregate was viewed.

## 002 — Future words in the model's own text (generality test of II.5): RUNNING after 001

Design `designs/002-future-words-own-text.md`, committed before the run.

- **Why:** a poetry result only covers rhymes. This asks the general question: at a position inside Qwen's own
  response, does the 4–8- or 9–16-tokens-ahead window rank the words Qwen *actually wrote* that far ahead better than
  ordinary J̄?
- **Data:** 50 seeded responses from WorkspaceBench's hallucination bank, using the verbatim sampled token ids. Only
  content words not already in the context count; a decoy null uses another response's future words. There's a
  next-word positive control.
- **Same estimator and hosts as 001. No judge, nothing fitted.**
- **Predictions:** my priors are 15% for 4–8 and 10% for 9–16. The decision table reading 001 and 002 together is in
  the design.
- Analysis script `scripts/002_analysis.py`, committed before any real data (dry-run on a 1-item smoke file only).

**Chain:** `scripts/run_001subset_002.sh`. Logs: `outputs/logs/001_v2_subset.log`, then `outputs/logs/002.log`.
About 38 + 45 min.

## Queued (not started; each needs your go after I show the spec)

1. **Part I Stage 1, revised:** span-level P / A / curvature / P^int×P^ψ on RHUT's 12 two-hop items, plus the horizon
   gap G_hz.
2. **II.1:** write directions (filter vs pattern vs Fisher) on the ekko M2 set.

## Where things are

| what | path |
|---|---|
| review of your spec | `docs/01_spec_review_2026-09-29.md` |
| designs, written before runs | `designs/NNN-*.md` |
| reports, after runs | `results/NNN-*/report.md` |
| code | `scripts/NNN_*.py`, shared `scripts/lib/{engine,mfl}.py` |
| logs, large outputs (gitignored) | `outputs/` |
| environment | `/workspace/ekko-lens/.venv-secant` (torch 2.14 cu126, transformers 5.17), `requirements.txt` |
