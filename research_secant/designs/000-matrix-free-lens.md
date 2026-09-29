# 000 — Matrix-free J-lens: validation (design, written before any run)

**Date** 2026-09-29 · **Model** Qwen/Qwen3.6-27B (rev `6a9e13bd`) · **Status** pre-registered

## Why
Every Part-I gap and the II.4b/II.5 readers need J̄ applied to arbitrary vectors under custom estimands (horizon
windows, lag buckets, other host distributions). Refitting J for each estimand is a fit costing GPU-hours. The
matrix-free lens computes J̄v directly from forward passes on host contexts, with no fitting.

Before anything is built on it, we must show two things:
1. it reproduces the released J_CB estimand (definition, target layer, fold, positions, aggregation);
2. its finite differences are numerically clean.

## What we want to find out
1. Does T̄_full(v), computed on the hypothesised J_CB fit documents, reproduce J_CB[ℓ]·v?
2. Which tokenization convention (BOS prepended or not) and target range reproduces it?
3. How large is the host-sampling noise on fresh hosts? That noise is the floor under every later gap.
4. At which ε are the fp32 centred differences stable?

## Definitions
- Layer ℓ = output of decoder block ℓ (0-indexed). y = output of block 62 (J_CB `target_layer`).
- Host doc k is tokenized to `t_max` = 128. Source positions p ∈ [4, T−2] (skip_first 4; final position excluded).
  Targets t' ≥ p.
- **T̄_full(v)** = (1/K) Σ_k [ Σ_{t'} y_{t'}(x_k + εv·𝟙_P) − Σ_{t'} y_{t'}(x_k − εv·𝟙_P) ] / (2ε·N_p). Here 𝟙_P adds εv
  at every valid source position simultaneously. First-order equal to mean_p Σ_{t'≥p} ∂y_{t'}/∂h_p · v, which is
  J̄'s estimand.
- **Precision:** the block-ℓ outputs come from the bf16 model. Blocks ℓ+1..62 run in **true fp32** (weights converted,
  TF32 disabled, fp32 rotary tables). The perturbation v is added in fp32.
- **Readout comparison.** In residual space, cos(T̄_full(v), J_CB[ℓ]v) and ‖T̄_full(v)‖/‖J_CB[ℓ]v‖. In vocabulary
  space, cos over U·(·), where U = (1+γ)⊙W_U.

## Test vectors
For each ℓ ∈ {40, 48, 56}:
- 4 random isotropic unit directions;
- 4 real activations h (block-ℓ output at the final token of 4 WSB multihop prompts);
- 4 differences h − h′ (sibling multihop prompts).

## Conventions tested, the hypothesis space
- (a) docs = the first 25 of `NeelNanda/pile-10k` train, no BOS;
- (b) the same with BOS.

Qwen's tokenizer has no default BOS; jlens `from_hf(force_bos=True)` may add one. Target range: all t' ≥ p up to T−1.
If neither convention reaches the gate, the variant "targets exclude the final position" is tried next, and any
further variant is recorded as post hoc.

## Gates and pre-registered expectations
| gate | pass criterion | my prior |
|---|---|---|
| G0 Richardson | T̄_full at ε and ε/2 agree: residual-space cos ≥ 0.9999, relnorm ≤ 1% (ε chosen so ‖εv‖ ≈ 1e-3·median‖x‖) | 90% |
| G1 fit-doc reproduction | on the convention that wins, residual cos ≥ 0.99 and norm ratio in [0.97, 1.03] at every ℓ ∈ {40, 48, 56}, median over the 12 test vectors | 60% (doc identity unknown) |
| G2 fresh-host noise (report, not gate) | 32 fresh hosts (pile-10k docs 100–131): cos to J_CB·v, and split-half cos between two 16-host halves | expect ≈0.9–0.97 at L40, higher at L56 |
| G3 fp32 vs bf16 (report) | the same T̄_full computed with bf16 blocks at the same ε | expect badly degraded (ekko F5) |

## Decision rules
- **G0 fails:** fix the numerics before anything else.
- **G1 passes:** the matrix-free lens is validated as J_CB's estimand. Proceed to 001 (II.5 lag buckets) and 002 (Part I
  Stage 1).
- **G1 fails but fresh-host agreement is at the level G2 predicts:** the doc identity is not recoverable. Proceed, with
  every later matrix-free vs lens comparison carrying the G2 floor, and report that J_CB's exact estimand could not be
  confirmed.
- **G1 fails and cos is far below G2's level on every convention:** a definitional mismatch (target layer, fold,
  aggregation). Stop and diagnose.

## Cost
K=25 docs × 2 forwards × 3 layers × 12 vectors, batched. Minutes.
