# ekko-lens (branch `ekko-lens`): what it did and what carries over

Read 2026-09-16 from `research/CLAUDE.md`, `FINDINGS.md`, `STATUS_2026-08-12.md`,
`expanded_templates.md`, `ekko/harness.py`, `ekko/rules.py`, `E5_blackmail.py`, `002_calib_passk.py`.
Not merged into `phrase_J`.

## What the project was

Which linear transport operator best recovers causally active intermediate representations? J-lens,
R-lens, input-specific Jacobians, fitted secant/anchored operators, smoothed/path operators, all
ranked on a directly measured causal benchmark (perturbation bank with exact finite effects) rather
than readout aesthetics. Model Qwen3.6-27B, released-lens convention, bank layers 16/31/46.

## Findings that matter for phrase-J

- **F3 (supported, the one large effect).** Input-specific Jacobian beats the averaged lens in
  effect prediction by 4.3× / 3.2× / 1.9× at L16/31/46 (antithetic; 1.8× on the ecological one-sided
  target). Context dependence is worst early, and the deviation is **not low-rank-shared** (F4:
  oracle rank-16 recovers 23%/40% of the gap at L16/L31). → Context-matched gradients (phrase-J
  regime iii) are the cheapest way to use this; a fitted correction matrix is not.
- **F1/F2/F6/F13 (supported negatives).** Free d×d secant fits, anchored fits, low-rank conditional
  transport, spectral shrinkage, J/R fusion, reference-anchored readout, and the fully homogenised
  H-lens all fail with named mechanisms (template memorisation under leaky splits; 26M parameters
  from ~1,700 observations; per-component conservation does not compound through attention/GDN).
  → Do not build fixed-operator variants. Phrase-J is per-phrase vectors from scalar backwards, a
  different class.
- **F7/F8/F9/F11 on the R-lens.** R is a finite-displacement operator whose LRP rules exactly
  reproduce each component's output when applied to the full activation (SiLU: σ(z)·z; RMSNorm:
  the tangent annihilates the radial direction, R keeps it). It reads better (F0.2: first-half
  pass@10 0.089 → 0.150 on the 59-item slice) but transports perturbations no better than J at the
  twin-fit noise floor (R−J = +0.016/+0.006 vs floor 0.011/0.009), worse at tangent scale, and
  **J beats R on ablation Δlogp** (+0.856 vs +0.587, CI-clear) on 40 probe-swap items. → R is a
  readout baseline, not a causal one. Any J-vs-R margin under ≈0.01 at n=25 is unresolved.
- **F14/F15.** Patchscopes is depth-flat (first-half 0.217 vs R 0.140); probes read early concepts
  at ~0.99; no matrix reads early layers. Their diagnosis: the early→vocabulary translation *is* the
  computation of layers 8–30. → Relevant to FF (word head) and to the P experiment; note the R-lens
  post's "error accumulation" story is a third candidate they reject on their evidence.
- **F10.1 bug.** Qwen3_5 RMSNorm is `(1+w)`; every hand-folded probe vector used `γ⊙u_t` instead of
  `(1+γ)⊙u_t` (cos 0.92–0.96 to the correct fold). Fix before quoting any folded-direction number.
- **Fit cost.** 183 s/prompt for a full-d_model J fit on the 27B (n=25 ≈ 1.3 h). A scalar-cotangent
  backward is ~36 ms per 128-token sequence with the same machinery. Phrase-J at 40 phrases × 300
  passages ≈ 12k backwards ≈ 10 min of backward time plus forwards.

## What was never built there

Swap/clamp interventions (ablation is the only causal op in that codebase), the M4 skip-ahead
guardrail exists only in `E2_eval_equal.py`, no multi-token anything, no sparse-frame readout, no
second model.

## Borrowed into `scripts/lib/`

`ekko_harness.py` (exact `forward_from` replay verified at 0 error on this architecture, lens
loading, rank helpers), `ekko_rules.py` (LRP surrogates for the R arm), and two reference scripts
for the calibration-slice carve and the matched-grid evaluator with M4.

## Methodological lessons kept

- Split leakage: base-prompt splits leak (a→b) vs (b→a) and shared templates; use template- or
  category-level splits and report an effective-rank diagnostic with any transfer result.
- A positive control reading 0.84 where theory says 1.0 is an estimand mismatch, not "close".
- bf16 central differences at ε=0.01 carry ~7% direction error; check numerics before attributing
  a gain to geometry.
- Bootstrap over evaluation items cannot see fit noise; a twin-fit floor is separate and required.
