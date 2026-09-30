# ekko-lens — transport-operator study (senku14x/jacobian-lens, branch `ekko-lens`)

Commit 89ae076, last activity 2026-08-14, closed by user decision. Clone: `/workspace/ekko-lens`. Canonical docs:
`research/FINDINGS.md` (ledger F0–F15, open list O1–O8, plan-conformance appendix), `research/CLAUDE.md` (working
agreement), `research/RESEARCH_LOG.md`, and `research/artifacts/{007..012}/`.

**Model and conventions [REPO].** Qwen3.6-27B, 64 layers, d 5120, 48 GatedDeltaNet + 16 full-attention blocks.
Released-lens convention: target 62, skip_first 4, pile-10k, n=25. Bank layers 16/31/46. ε ∈ {0.01, 0.05, 0.2, 1.0} ×
median‖h‖. Paired bootstrap over base prompts.

## Question
Which linear transport operator best recovers causally active intermediate representations? Candidates: J̄, R̄,
input-specific J, fitted secant/anchored, and smoothed. They were ranked on a directly measured causal benchmark
(predicting finite downstream effects of perturbations), not on readout aesthetics.

## Findings [REPO] (evidence levels as the ledger states them)
- **F0 instruments.**
  - Autograd-free exact replay (`forward_from`) with 0.0 error.
  - Positive control T_IG ≈ 0.92–0.95. The first run read 0.81 because of a one-sided vs antithetic target mismatch,
    which was caught.
  - R>J readability replicates on a 59-item slice: pass@10 all 0.479→0.534, first-half 0.089→0.150.
- **F1 secant fit fails** (supported negative). The in-family win (+0.29) was template memorisation: under a
  template/category-disjoint split it collapses to 0.01–0.15, while J̄/R̄ stay flat. Stein, in-span and behavioural
  (M2 −0.86) gates all fail.
- **F2 anchoring** doesn't rescue it. Only the 3-scalar blend aJ̄+bR̄+cI generalises (it beats J̄ at every layer and is
  flat across split levels).
- **F3 context dependence is the one large effect.** J_loc/J̄ = 4.34 / 3.17 / 1.95 at L16/31/46 on the antithetic
  target; 1.82× on the native one-sided target. Isolated from corpus mismatch, it is 15–47× the noise floor.
- **F4** the deviation J_loc − J̄ is not low-rank-shared: oracle rank-16 recovers 23% / 40% / 85%, anti-correlated with
  need.
- **F5 SmoothGrad-J** beats J_loc by +0.17…+0.27, but the mechanism is open: path, conditioning, or bf16 FD numerical
  error (FD at ε=0.01 has cos 0.93 to the true JVP). The orth/iso discriminator was retracted (vacuous in d=5120).
- **F6** reference anchoring, spectral shrinkage and J/R mode fusion are all negative.
  - Corpus mean ≈79% of activation norm.
  - β=1 anchoring destroys early readout.
  - (J−I) top-64 modes hold only 7.5–14.4% of energy.
- **F7 block-level backward rules.**
  - **R is a finite-displacement operator**: worse than J at ε≤0.2, better at ε=1.0 by +0.02 (4/4 conditions).
  - The Q/K-norm LN-rule is negative; GDN gate rules are inert.
  - Only routing-damping rules help, at floor magnitude.
- **F8 twin-fit noise floor ≈0.011 cos** (native). R−J effect prediction (+0.016 / +0.006) does not clear it, so R's
  "faithfulness" margin is inside estimator noise. J_own ≈ released J.
- **F9 M2 behavioural ablation** (40 single-token items, band {16,31,46}):
  - Ordering: J̄ +0.86 > R̄ +0.59 > logit +0.02 ≈ fitted operators ≈ random.
  - **J̄ − R̄ = +0.27, CI-clear**: J beats R on this protocol, the opposite of the R-lens post's autorater result.
- **F10 audit.** The negatives are real. Probe fold bug: `γ⊙u` should be `(1+γ)⊙u` (cos 0.92–0.96 impact).
  "relerr≈1" is forced by direction error, not a magnitude failure.
- **F11 conservation account of R** (interpretation). Each LRP rule reproduces its component's output on the full
  activation (SiLU→σ(z)z exactly; RMSNorm LN-rule → RMSNorm(x); half-rule → Euler; frozen-A value path). So J = "nudge
  h", R = "what the stack does with this h". This explains R reading better while not transporting better.
- **F12** the R-lens post: the observation stands; the mechanism narrative ("error accumulation") and the title's
  "faithful" overclaim.
- **F13 H-lens** (R + frozen attention + GDN output half-rule, a full 4.1 h fit): pre-registered H>R first-half
  **falsified** (−0.038, ~9× floor). It is the least skip-ahead-prone operator (M4 0.000).
- **F14 identity patchscope** is depth-flat: first-half 0.217 beats every lens (R 0.140). Its zeros on semantic sets
  are template artifacts.
- **F15 close-out diagnosis** (interpretation): the early readout gap is computational. There is no global linear
  early→vocabulary map; the instruments that cross the gap are per-concept (probes) or run the model (patchscopes,
  NLA/AO).

Later runs at close:
- **E5**: a blackmail-scenario concern-token sweep, logit/J/R/H. Qualitative: all lenses surface
  blackmail/leverage/secret/threat as "dissociated latent" firings; R/H give more of them. Hypothesis-generating only
  (`artifacts/010`).
- **E6**: calibrated-z rescoring. Pooled first-half: logit 0.145, J 0.047, R 0.093, H 0.119. H−R not CI-clear.

## Never built [REPO appendix A.3]
Swap/clamp interventions (ablation only); the M4 guardrail (later built in E2); the D3 concept-vector delta family;
continuous α/β/γ/λ interior sweeps; AttnLRP softmax rule; tiny-model brute-force test; sparse-frame readout;
fp32/second-model replication.

## Open list O1–O8 [REPO]
- O1 H-lens → done, falsified.
- O2 Richardson/fp32 for SmoothGrad.
- O3 protocol-matched replication of the R-lens ablation.
- O4 probe-fold fix + A4 rerun.
- O5 input-adaptive blend.
- O6 sparse-frame readout.
- O7 depth-vs-width test of F11.
- O8 radial-annihilation check.
- O11 Koopman workspace-autonomy: parked, with priors and kill rules (`artifacts/011`).

## [MINE] takeaways
- The project's proxy (finite-effect transport cosine) is not the readout objective. F0.2 vs F8 is a measured
  dissociation, and F11 explains it. A per-input conservation operator applied to the full h reproduces ≈ the output
  (skip-ahead). Context averaging is partly *what makes* a lens read dispositions rather than the answer.
- Most reusable assets:
  - the exact replay harness (`research/ekko/harness.py`);
  - the twin-fit floor methodology;
  - the M2 ablation;
  - the M4 skip-ahead guardrail;
  - the E2 equal-footing harness with calibrated-z scoring;
  - the unused 38-item entity-disjoint eval set.
- The environment section of `research/CLAUDE.md` is stale for this machine (/home/ubuntu paths, Qwen2.5-7B cache).
