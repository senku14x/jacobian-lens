# 000 — Matrix-free J-lens validation: report

**Date** 2026-09-29 · **Design** `designs/000-matrix-free-lens.md` (committed 04c69ba, before the run) · **Code**
`scripts/000_matrix_free_lens.py`, `scripts/lib/{engine,mfl}.py` · **Raw** `results.json` · **Log**
`outputs/logs/000.log` (gitignored) · **Runtime** 305 s on one H100.

## Question
Does T̄_full(v), computed purely from forward passes (no fitted matrix), reproduce the released J_CB estimand? Which
fit convention does J_CB use? How noisy is the estimate on fresh host documents?

## Setup
- Qwen3.6-27B @6a9e13bd. Block-ℓ outputs are captured with the bf16 model. Blocks ℓ+1..62 are replayed as true fp32
  copies (TF32 off, fp32 rotary).
- The `(1+γ)` final-norm fold was verified on the loaded module: max |diff| 0.0.
- ε is set per vector so that ‖εv‖ = 1e-3·median‖x‖ over valid host positions.
- Estimand as in `jlens/fitting.py`: sources and targets share P = [4, T−2]; mean over sources of the sum over later
  targets; uniform mean over prompts.
- Test vectors, 12 per layer: 4 random (norm-matched), 4 block-ℓ activations at the final token of bundled multihop
  prompts, and 4 differences between such activations.
- Layers 40 / 48 / 56.
- Hosts: the first 25 pile-10k docs (both BOS conventions), and fresh docs 100–131 (31 usable).

## Observations

| layer | G0 ε vs ε/2 min cos (max relnorm) | G1 no-BOS: median cos (min) / norm ratio / vocab cos | G1 with `<\|endoftext\|>` prepended | G2 fresh: cos to J_CB / norm ratio | G2 split-half cos (16 vs 15 hosts) | G3 bf16 FD cos at ε = 1e-3 / 1e-2 / 5e-2 (L48, one length group) |
|---|---|---|---|---|---|---|
| L40 | 0.9999996 (0.07%) | **0.9967** (0.9955) / 1.001 / 0.996 | 0.899 / 1.22 | 0.972 / 0.90 | 0.951 | — |
| L48 | 0.9999997 (0.04%) | **0.9981** (0.9973) / 0.996 / 0.998 | 0.951 / 1.15 | 0.983 / 0.94 | 0.972 | 0.803 / 0.997 / 0.999 |
| L56 | 0.9999998 (0.02%) | **0.9994** (0.9993) / 0.999 / 0.9995 | 0.989 / 1.02 | 0.995 / 0.99 | 0.991 | — |

- Agreement is the same for random, activation and difference vectors, so it does not depend on the direction class.
- The fit docs' lengths range from 79 to 128 tokens. Several of the first 25 docs are shorter than 128, and the
  estimate handles them per length group.

## Gates (pre-registered)
- **G0 PASS** at every layer (criterion: cos ≥ 0.9999, relnorm ≤ 1%).
- **G1 PASS** at every layer on the no-BOS convention (criterion: cos ≥ 0.99, norm ratio within 3%). The BOS convention
  fails at L40/L48. **J_CB was fit on the first 25 docs of `NeelNanda/pile-10k` (train), with no BOS.** The residual
  0.3% at L40 is consistent with the bf16 batch-shape floor of the original backward fit (phrase_J 002: cos ≈ 0.998 at
  L36).
- G2 (report): fresh-host estimates agree with J_CB at 0.97 / 0.98 / 0.995. The magnitude is 10% lower at L40, so host
  choice shifts magnitude early. Split-half agreement between two ~16-host halves is 0.95 / 0.97 / 0.99. This is the
  sampling floor for any matrix-free comparison at K ≈ 30.
- G3 (report): bf16 finite differences at ε = 1e-3 give cos 0.80. The spec's ε under the RHUT bf16-matmul regime would
  have produced ~20% direction error. At ε = 5e-2 bf16 recovers 0.999, which is usable but not a tangent.

## What this supports
- **Supported, at these three layers:** the matrix-free lens computes J_CB's exact estimand (up to the original fit's
  bf16 floor) with no fitted matrix. Custom estimands (horizon windows, lag buckets, other host distributions) can
  therefore be computed from forward passes and compared with J_CB on equal terms.
- **Throughput** (fp32, 128-token hosts): ≈13 ms per host sequence from L56 and ≈45 ms from L40.

## What this does not establish
- Layers below 40 are untested. Agreement will degrade toward the bf16 fit floor there (early-layer shape noise), and
  the fp32 stack would need more memory.
- G1 validates the estimand on vectors of these three types. It does not validate finite-displacement quantities (S̄),
  which 000 did not test.
- The fresh-host magnitude shift (0.90 at L40) means matrix-free *magnitudes* on non-fit hosts should be compared
  within the same hosts, not against J_CB.

## Decision
Proceed to 001 (lag buckets, designed with fresh hosts and a within-host FULL_U comparator, which G2 justifies). Use
the fit-doc hosts whenever an exact J_CB match matters.
