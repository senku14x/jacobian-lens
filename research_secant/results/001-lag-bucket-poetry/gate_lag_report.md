# 001 gate G_lag — report (2026-09-30)

**Design** `designs/001-lag-bucket-poetry.md` Amendment 2 §5 (committed ea4406f before the run) · **Code**
`scripts/001_gate_lag.py` · **Raw** `gate_lag.json` · **Log** `outputs/logs/001_gate_lag.log` · L48, 2 poetry items
(`couplet-ahead-head`, `couplet-blame-game`), 32 fresh pile hosts (docs from index 100, full 128 tokens).

| check | item 0 | item 1 | pre-registered criterion | verdict |
|---|---|---|---|---|
| A: spaced vs exact single-source, identical samples (8 hosts × 2 draws × 3 sources = 48): B0 / B1 / B2 | 1.000 / 0.999 / 0.989 | 1.000 / 0.996 / 0.981 | ≥ 0.95 | pass |
| A: B3 (lags 9–16) | 0.967 | 0.945 | ≥ 0.95 | **fail (item 1)** |
| A: B4 (lags 17–32) | 0.927 | 0.851 | ≥ 0.95 | **fail** |
| A: FULL32 | 0.9985 | 0.996 | ≥ 0.95 | pass |
| B: v1 all-positions estimator, host split-half vocab cos (K=32, R=2): B1 / B2 / B3 / B4 | 0.88 / 0.69 / **0.61** / 0.03 | 0.90 / 0.57 / **0.11** / 0.06 | documentation | v1's B3 and B4 were mostly noise |
| B/C: spaced estimator, split-half (K=32, R=4): B1 / B2 / B3 / B4 / FULL32 | 0.93 / 0.85 / 0.87 / 0.72 / 0.97 | 0.91 / 0.88 / 0.78 / 0.69 / 0.96 | C: B3 ≥ 0.7 | pass (R stays 4) |

## Reading
- **The estimator is implemented correctly.** At short lags, where the signal is strong, it agrees with the exact
  reference at 0.98–1.00 on identical samples. So the shift direction, masking and normalisation are right.
- **The residual disagreement grows with lag.** It is cross-source leakage: a target in source p's window also
  receives source (p−40)'s effect at lags 40–72, with a random sign product. Real lag-δ effects shrink with δ, so the
  zero-mean leakage matters most for B3 and B4, exactly where the hypothesis lives.
- The disagreement would shrink with more samples (the main run has 384 per lag vs the gate's 48). But that's an
  extrapolation after a failed gate, and the pre-registered rule says the gate failed. **The main v2 run was not
  launched.**
- **v1 is confirmed invalid for B3/B4:** split-half 0.61 and 0.11, against 0.87 and 0.78 for the spaced estimator.
  The v1 cells are kept as a record and must not be analysed.

## Proposed fix (Amendment 3, pending the user's go)
Use the exact single-source estimator for the main run. It was the gate's reference: one perturbed source per
forward pair, so there is no cross-source leakage by construction, and every lag 0–32 comes from the same forward.
- Budget: 32 hosts × 8 source positions (stratified over [4, 94], so every lag up to 32 stays inside the window) =
  256 samples per cell, plus FULL_U.
- The gate then reduces to C (reliability), re-measured on the same 2 items before the main run: B3 split-half ≥ 0.7,
  or double the positions.
- Cost ≈ 17 s per cell (≈ 29 ms per fp32 host sequence averaged over layers 40–56).
  - 5 layers × 100 items ≈ 2.3 h.
  - 3 layers {40, 48, 56} ≈ 1.4 h.
