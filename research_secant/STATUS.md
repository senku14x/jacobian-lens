# STATUS — research_secant (what is running, why, and what we expect)

Updated 2026-09-29. Branch `secant` of `/workspace/ekko-lens`. Layout and conventions are in `README.md`.

## The big picture in three sentences

Your spec's central tool is a **matrix-free J-lens**: compute "J̄ times a vector" directly from forward passes on
ordinary text, instead of using the stored 5120×5120 J matrix. Once that tool reproduces the released lens exactly,
the *definition* of the lens can be changed at no cost, with no refit. Examples: count only effects 9–16 tokens ahead,
count only effects inside this sequence's horizon, or use chat text instead of pile text. Today's runs are (000)
proving the tool is exact, and (001) using it for the cheapest decisive test in your spec: whether the J-lens misses
planned rhyme words only because its definition is dominated by short-range effects.

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

## 001 — Lag-bucket lens on planned rhymes (your spec II.5): PAUSED at the estimator gate

Design `designs/001-lag-bucket-poetry.md`, with Amendment 1 (controls) and Amendment 2 (estimator fix), both
committed before the runs they govern.

- **v1** (the all-positions ± estimator) was stopped at 101/500 cells. The gate showed its 9–16-token window was
  mostly noise (split-half 0.61 / 0.11). It is kept as a record; **do not analyse it**.
- **Gate G_lag** (report: `results/001-lag-bucket-poetry/gate_lag_report.md`). The new spaced estimator is implemented
  correctly: it agrees with the exact reference at 0.98–1.00 for short lags. But it **failed the pre-registered
  leakage check** at 9–16 tokens (0.945 on one item) and at 17–32 (0.85–0.93). So the main run was **not launched**.
- **Proposed Amendment 3** (awaiting your go): use the exact estimator, one perturbed position per forward pair,
  which has no leakage by construction. 256 samples per cell. About 2.3 h for 5 layers, or 1.4 h for 3 layers.

Everything else in the earlier description of 001 (why, predictions, success/kill rules, the controls) is unchanged.

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
