# 008 — Four proposed lens improvements, tested

**Date** 2026-08-12 · **Model** Qwen/Qwen3.6-27B · **Status** three directions tested and negative;
five deferred with reasons. Follows the external review logged in
[007](../007-2026-08-12-transport-operators/report.md).

---

## Summary

| direction | verdict | headline |
|---|---|---|
| Reference-anchored readout `μ_F + J(h−μ_h)` | **negative** | β=1 destroys early-layer readout; best β improves aggregate pass@10 but halves first-half pass@10 — and β was tuned on the eval |
| Spectral shrinkage `cI + U diag(αᵢsᵢ)Vᵀ` | **negative** | never beats J; (J−I) holds only 7.5–14.4% of its energy in the top 64 of 5120 modes, so there is no low-rank unreliable structure to shrink |
| Mode-wise J/R fusion `J + U diag(αᵢ)ΣVᵀ` | **negative** | converges to ≈ R; the fit sets ~65–70% of R−J modes to α=1 and ~15–33% to α=0, but the selection does not beat taking all of R |
| Lag-separated J⁽⁰⁾/J⁽¹⁾/J⁽ᶠᵃʳ⁾ | **dropped** | the paper already compared present-only/future-only aggregations and reports results robust to the choice |
| Sparse-frame readout, conditional atlas, curvature lens, content-vs-routing, depth-consistent fitting, Fisher directions, phrase verbalizers | **deferred** | reasons and costs in §5 |

Two of the three tested directions were the review's "best chance of a better fixed global lens."
Both fail, and both fail for a reason that is now measurable rather than speculative.

---

## 1. Why these were the right things to try

The secant failed with **d² ≈ 26.2M free parameters estimated from ~1700 deltas**, and collapsed to
~0 under a template-disjoint split. Every direction here keeps a single fixed d×d operator —
complete outside any calibration span, still a real global lens — while cutting free parameters to
**d or fewer** by reweighting an existing basis rather than learning a new operator.

The fit is closed form. For `T = A + Σᵢ aᵢsᵢuᵢvᵢᵀ`, the prediction is `Tδ = Aδ + Σᵢ aᵢfᵢ` with
`fᵢ = sᵢ(vᵢ·δ)uᵢ`, and since **U is orthonormal**,

```
G_ij = Σ_n ⟨fᵢⁿ, fⱼⁿ⟩ = sᵢsⱼ Σ_n (vᵢ·δₙ)(vⱼ·δₙ) ⟨uᵢ,uⱼ⟩   →   diagonal
```

so every mode has a scalar ridge solution `aᵢ = bᵢ/(Gᵢᵢ+λ)`, and clipping to [0,1] afterwards is the
**exact** solution of the box-constrained problem, not an approximation. No d×d solve, no
conditioning problem, ~85 s for all three layers × three split levels.

Everything is evaluated under the leak-repaired splits from 007 — an operator that only wins on the
base-prompt split has learned templates — and on relative residual error as well as cosine.

---

## 2. Reference-anchored readout — negative

The released lens is `softmax(W_U norm(J h))`, confirmed against the paper: applied to the
**absolute** activation, no reference subtraction, no intercept. But a Jacobian transports
differences, and Taylor licenses `F(h) ≈ F(r) + J(h−r)`; only `r=0, F(0)=0` reduces to `Jh`.

This cannot change any transport number — effects are differences, so `F(h+δ)−F(h) = Jδ` for any
anchor. It is purely a readout question, evaluated only on the permanent 59-item calibration slice
(93 intermediates), reproducing 002's exact draw so the frozen sets stay untouched.

The motivation is real: **‖μ_h‖ = 59.0 at L31 against a median ‖h‖ of 74.4**, so ~79% of a typical
activation's norm is the corpus mean, and `J·h` is dominated by the prompt-independent term `J·μ_h`.

| variant | pass@1 | pass@10 | pass@10 first half | mean first layer top-10 |
|---|---|---|---|---|
| logit lens | .1879 | .4110 | .1624 | 40.0 |
| **J (released)** | .1836 | **.4788** | **.0890** | 43.5 |
| J centred, no anchor | .1412 | .3588 | .0636 | 44.2 |
| J anchored β=0.1 | .1879 | .4407 | .0847 | 41.8 |
| J anchored β=0.25 | .2048 | **.5297** | .0381 | 43.9 |
| J anchored β=0.5 | .2175 | .4647 | .0000 | 47.9 |
| J anchored β=1 | .1158 | .4096 | .0000 | 52.7 |
| J anchored, norm-matched | .1540 | .4576 | .0720 | 45.5 |
| J anchored, position-conditioned μ_h | .1497 | .4096 | .0000 | 53.0 |
| **R (released)** | .2345 | **.5339** | .1497 | 39.7 |
| R anchored β=1 | .1328 | .4096 | .0000 | 51.1 |
| **INTERCEPT — `unembed(μ_F)` alone** | **.0000** | **.0000** | **.0000** | n/a |

**The intercept control is clean.** A constant vector, identical for every prompt, never reaches
top-10 for any intermediate at any layer. So nothing here is the literal tuned-lens pathology of an
affine bias decoding the answer while ignoring `h`.

**But anchoring still fails, and the failure has the same shape.** At β=1 the anchor swamps the
signal (‖μ_F‖=170.5 against a much smaller ‖J(h−μ_h)‖) — three different operators all collapse to
an identical .4096, which is the signature of the operator no longer mattering. At the best β the
aggregate metric improves (+0.051 pass@10 over J) **while first-half pass@10 more than halves**
(.0890 → .0381) and mean first-layer-to-top-10 moves later. The gain is bought entirely in late
layers.

That trade is the wrong direction for the North Star. Aggregate pass@10 is a proxy; what the
instrument is for is surfacing intermediates *early*, and first-half pass@10 tracks that. A change
that improves the proxy while degrading the thing the proxy stands for is a divergence, not a win.

**Caveat that weakens this further, stated because it is my error:** β was swept on the same 59
items it is evaluated on. The review explicitly warned against selecting on the readability
benchmark. The +0.051 is therefore optimistic and a proper test needs an inner split. It does not
change the verdict — the first-half collapse is visible at every β — but the aggregate number
should not be quoted.

Also worth recording: **centring alone is worse than doing nothing** (.3588 vs .4788). A plausible
mechanism is that the readout pipeline `RMSNorm(·)` then `W_U` is calibrated on inputs that contain
the mean, so removing it moves the input off-distribution for the model's own final norm.

---

## 3. Spectral shrinkage and mode-wise J/R fusion — negative

Native one-sided target, D4, three split levels. `randbasis512` is a control with the same number
of free coefficients in a meaningless orthonormal basis.

| L16 | J | R | shrinkJ | fuseJR | randbasis512 | aJ+bR+cI |
|---|---|---|---|---|---|---|
| base | .1040 | .1264 | .1046 | .1227 | .1018 | .1275 |
| template | .1155 | .1381 | .1121 | .1346 | .1123 | .1377 |
| category | .1049 | .1287 | .1016 | .1238 | .1015 | .1310 |

| L46 | J | R | shrinkJ | fuseJR | randbasis512 | aJ+bR+cI |
|---|---|---|---|---|---|---|
| base | .3727 | .3777 | .3735 | .3807 | .3724 | .3799 |
| template | .3851 | .3943 | .3847 | .3951 | .3845 | .3955 |
| category | .3925 | .3980 | .3904 | .3998 | .3917 | .4001 |

**Spectral shrinkage of J never beats J** — and at L16/category it is *worse* (.1016 vs .1049). The
reason is measurable: **(J−I) carries only 7.5–14.4% of its energy in the top 64 of 5120 modes.**
The deviation from identity is extremely high-rank, so there is no concentrated set of unreliable
modes for shrinkage to collapse toward `cI`. The premise of the method is not satisfied by this
operator.

**Mode-wise J/R fusion converges to ≈ R and never beats it meaningfully.** The fitted coefficients
do give the mechanistic account the review wanted: **~65–70% of R−J modes are driven to α=1 and
~15–33% to α=0** — R's correction is genuinely not uniformly useful. But applying that selection
scores at or below simply taking all of R (L16/category: .1238 vs R .1287). The modes the fit drops
were not the useless ones.

Two properties are worth keeping even though the methods failed:

- **They generalise.** Unlike `T_sec` (which fell from .43 to .01 base→category), every operator
  here is nearly *flat* across split levels, and several score slightly higher under the stricter
  split. Constraining to d parameters did fix the leakage-driven overfitting; it just did not buy
  any accuracy.
- **The random-basis control works.** 512 free coefficients in a random basis gives no improvement
  over J (.1018 vs .1040), so the small fusion gain is basis-specific, not free parameters.

**Nothing here changes relative error.** Every fixed operator stays at relerr 0.95–1.01, i.e. no
better in magnitude than predicting zero effect. That limitation of the fixed-operator class
survived every method tried in this artifact.

---

## 4. Lag separation — dropped, not run

The review's first message proposed separating `J⁽⁰⁾` (t′=t), `J⁽¹⁾` (t′=t+1) and `J⁽ᶠᵃʳ⁾`, and its
second message downranked it. Checking the paper directly: it states it examined variants
"computing only present and not future token effects" among others and that "qualitative results
are robust to these choices." Running it would be a replication of a published robustness check,
not a new result. Dropped in favour of the untested directions.

---

## 5. Deferred, with reasons

- **Sparse-frame readout** (solve `min_{c≥0} ‖h−Dc‖²+λ‖c‖₁` over the J-lens token dictionary and
  rank by `c*`). The most promising untested item: the paper *already* formalises J-space as a
  union of cones and solves for sparse nonnegative combinations by gradient pursuit, but uses that
  for decomposition and interventions rather than for the readout — so this is a real gap. Cost is
  the reason it is not here: the dictionary is 248,320 × 5,120. Tractable via a candidate set (top-k
  by raw score, then nonnegative lasso on that subset), which is the right next implementation.
- **Conditional Jacobian atlas + oracle upper bound.** The decisive feasibility test is cheap and
  should be run. Prior evidence is discouraging but not decisive: 007's C2 found the *continuous*
  low-rank deviation model fails (oracle rank-16 recovers 23%/40% of the gap at L16/L31). Discrete
  regimes could exist without low-rank structure, so the oracle-cluster test is genuinely
  independent of that result.
- **Low-rank curvature lens.** Correctly gated by the review on first establishing reproducible
  low-rank structure in `J(x) − J_own`, which 007's C2 already failed to find. Should stay gated.
- **Content-vs-routing decomposition.** Architecturally the most interesting, and it needs custom
  backward rules on GatedDeltaNet blocks (48 of 64 layers), where the γ/stop-gradient family is
  undefined. Real implementation risk, deferred deliberately rather than skipped.
- **Depth-consistent joint fitting**, **Fisher-normalised token directions**, **phrase verbalizers**
  — each needs machinery that does not exist yet (per-block Jacobians `A_ℓ`; the categorical Fisher
  `G = E[J_zᵀF_zJ_z]`; a phrase-gradient pipeline). All still open.

---

## 6. What this adds up to

The review's two strongest bets for a better *fixed* operator — spectral shrinkage and mode-wise
J/R fusion — are both negative, and the reference-anchoring "cheapest possible win" is negative on
the metric that matters. Combined with 007, the fixed-operator class now looks genuinely exhausted
on this model: **`R̄` remains the best fixed operator tested, the 3-scalar blend `aJ̄+bR̄+cI` matches
it, and nothing with more parameters beats it under a category-disjoint split.**

That sharpens rather than weakens the review's framing. The remaining live hypotheses are the ones
that leave the single-fixed-matrix class — the conditional atlas and the readout-side changes — and
the outstanding baseline gap is still `J_own`, without which "context averaging" is not isolated
from corpus and convention mismatch.

## Reproduce

```
research/experiments/C1_reference_anchor.py     readout variants + intercept control
research/experiments/C3_spectral_fusion.py      spectral shrinkage + J/R fusion  [no model]
research/experiments/B7_splits_targets_anchors.py   the leak-repaired splits these use
```
