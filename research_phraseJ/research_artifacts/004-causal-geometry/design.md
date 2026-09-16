# 004 — Causal geometry: read covectors vs covariance-dual write vectors

Status: **stub, 2026-09-16.** Full design after 003 reports and the swap-validity harness exists.

## Question

For phrase vectors that read well in 003, which primal direction intervenes best: the raw gradient
r_w, its pile covariance-dual u_w = Σ_λ r_w, the template row t_w, or the template dual Σ_λ t_w?

## Hypothesis (explicitly a hypothesis about the model, not a theorem)

If Mahalanobis distance under the natural activation distribution is the appropriate intervention
cost, the minimum-cost perturbation with unit effect on the read functional r is
δh* = Σr / (rᵀΣr). Whether that cost is the right one for a transformer residual is empirical.

## Design outline

- Swap-validity certificate first (`scripts/lib/swap_certificate.py`): source loading naturally
  substantial on the intervention items; target direction naturally active in matched contexts;
  Mahalanobis distance of h_patched − h_clean relative to natural differences; clamp / re-entry
  control; matched-norm random directions; unrelated-task damage (pile top-1 retention);
  dose-response monotonicity; reversibility. Supernode handling for correlated prefix rows.
- Interventions normalized in **two separate arms**: ‖δh‖₂ = C and δhᵀΣ⁻¹δh = C. A direction that
  wins only under its own metric is a weaker result than one that wins under both.
- Items: Dataset A multi-token two-hop set from 003; counterfactual target = sibling within the
  family where possible.
- Metrics: top-1 flip rate to the implied counterfactual answer; Δ log-prob; effect layer band
  (intermediate vs answer timing) per direction.
- Reconstruction control (from R): NNLS reconstruction of activations using r-vectors vs u-vectors
  as atoms; prior is that u are the better atoms and r the better decoders.

## Pre-registered readings

- Σr wins under both norms → activation-metric geometry governs intervention; adopt read/write
  pair as the standard object.
- Σr wins only under Mahalanobis → metric-dependent; report both, no default.
- r wins under both → Euclidean geometry; the whitening story does not carry to intervention.
- Template t_w reads best in 003 but Σt_w or r intervenes best here → the read/write dissociation
  from Marks & Tegmark reproduced on phrase vectors.
