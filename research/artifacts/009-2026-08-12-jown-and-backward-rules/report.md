# 009 — A fitted lens, an estimator noise floor, and the backward rules R-lens omits

**Date** 2026-08-12 · **Model** Qwen/Qwen3.6-27B · **Status** complete. First artifact in this
project that fits a lens rather than consuming released ones.

---

## Summary

| finding | verdict |
|---|---|
| A lens fit costs 183 s/prompt; n=25 ≈ 1.3 h on one H100 | it was never expensive; the earlier "infeasible" reading was a contaminated measurement |
| `J_own` (convention-matched fit) vs released `J` | **indistinguishable** in effect prediction: −0.0011 [−0.0025,+0.0001], not CI-clear |
| **Twin-fit noise floor** — two fits of the same estimator on disjoint halves | **0.0113 cos (native) / 0.0088 (antithetic)**; 0.56 relative Frobenius |
| Released `R` vs released `J` advantage | **+0.0155 / +0.0059** — at or *below* the noise floor |
| `J_loc` vs `J_own` (context averaging, isolated) | **+0.174 / +0.412**, 15–47× the noise floor — the one large, robust effect |
| **`R` itself vs `J` at block level** | **worse at ε≤0.2, better at ε=1.0 by +0.020–0.023 in all 4 conditions** — R is a finite-ε operator |
| Q/K-norm LN-rule, composed on R | **negative at every ε and both delta families** (−0.009 to −0.012 on natural deltas) |
| GatedDeltaNet gate / Q-K-L2-norm rules (48 of 64 layers) | **nil** — ±0.0012 |
| Routing-damping rules (value-only, tempered softmax, gate/output half) | +0.005…+0.010 at ε=1.0 on natural deltas, but **do not replicate on isotropic deltas** |
| bf16 finite difference at ε_ref=0.01 vs true JVP | **cos ≈ 0.93** — `J_loc` carries ~7% direction error by construction |

---

## 1. The fit, and the excuse that was wrong

Every `J` and `R` in artifacts 007 and 008 was the *released* lens. `J_own` was listed as
outstanding three times and deferred each time on cost. The cost was never measured.

D1 measured it. A first attempt reported OOM at every configuration and nearly became a
"fitting is infeasible on this hardware" conclusion — but a stale process was holding 72 GiB.
On a clean GPU:

```
after load                       50.1 GiB
seq=128 dim_batch=8  1 layer     185.9 s/prompt   peak 72.7 GiB   -> n=25 = 1.29 h
seq=128 dim_batch=32 1 layer     OOM
```

Three source layers at seq=128 also OOM, because the one-layer fit already peaks at 92% of
memory. So layers are fitted one at a time, and the two disjoint halves are fitted and **merged**
(`jlens.merge` is an n-weighted mean of disjoint-subset fits) — the same total cost as one n=25
fit, and it yields the noise floor for free.

Fitted at exactly the released convention, confirmed against the released lens's own embedded
provenance (`pile-10k`, `n_prompts=25`, `t_max=128`, `skip_first=4`, `target_layer=62`):
halfA 13/13 prompts in 39.6 min, halfB 12/12 in 36.5 min, both 183 s/prompt, no prompts silently
dropped.

---

## 2. The noise floor, and what it does to the rest of the study

halfA and halfB are two fits of the **same estimator** on disjoint halves of the **same corpus**.
Any difference between them is pure estimation noise.

| quantity, L31 | relative Frobenius | cos (flattened) |
|---|---|---|
| `J_own` vs released `J` | 0.328 | 0.949 |
| released `R` vs released `J` | 0.794 | 0.867 |
| **halfA vs halfB (noise)** | **0.562** | **0.855** |

Two independent estimates of the *same* operator disagree by 0.56, while `R` differs from `J` by
0.79. Scaling the floor to n=25 (noise ∝ 1/√n, so ÷√2) gives ≈0.40 — so **`R−J` is only about
twice the estimation noise of the lens it modifies**, and `J_own − J` (0.33) sits *below* it, i.e.
my fit reproduces the released lens to within noise. That is the positive control for the whole
fitting path.

In the units the study actually reports — cos against the true finite effect, template-disjoint
split, n=1680 directions over 20 templates:

| L31 | native one-sided | normalised antithetic |
|---|---|---|
| `I` | .1491 | .1358 |
| released `J` | .2124 | .1966 |
| **`J_own`** | **.2113** | **.1967** |
| released `R` | .2279 | .2025 |
| `J_loc` | **.3854** | **.6086** |
| `J_own − J_rel` | −0.0011 [−0.0025,+0.0001] | +0.0001 [−0.0011,+0.0015] |
| `R − J` | +0.0155 [+0.0122,+0.0183] * | +0.0059 [+0.0030,+0.0086] * |
| **halfA − halfB (noise floor)** | **+0.0113** | **+0.0088** |
| `J_loc − J_own` | **+0.1741** * | **+0.4119** * |

Two things follow, and the second is uncomfortable.

**The uncontrolled-comparison worry was real in principle and empirically nil.** `J_own` matches
the released `J` to −0.0011 / +0.0001, neither CI-clear. Corpus and convention mismatch contributed
nothing, so every "vs released J" comparison in 007 and 008 stands as written. I flagged this as a
load-bearing gap through three artifacts; measuring it changed nothing, which is worth saying
plainly rather than quietly dropping.

**The R-lens advantage in effect prediction does not clear the noise floor.** `R − J` is +0.0059 on
the antithetic target against a floor of +0.0088, and +0.0155 against +0.0113 on the native target
— i.e. below it in one case and 1.4× it in the other, despite bootstrap CIs that exclude zero in
both. The bootstrap resamples *evaluation prompts*; it does not resample the *fit*, so it cannot
see this source of variance. Any margin under ≈0.01 in this project — which includes most of the
spectral and fusion margins in artifact 008 and several in 007 — is inside the noise of the
estimator itself, however tight its CI.

**Context averaging survives, and it is the only large effect.** `J_loc − J_own` is +0.174 and
+0.412 — 15× and 47× the floor. It is also now properly isolated: `J_loc − J_own` and
`J_loc − J_rel` agree to 0.001, confirming the gap is context dependence and not corpus mismatch.
One correction to 007: on the **native one-sided** target the ratio is `.3854/.2113 = 1.82×`, not
the 3.17× reported on the normalised antithetic target. The ecologically valid target gives a
markedly smaller context-averaging cost.

---

## 3. Backward rules at block level

Per the review's own first experiment: test each rule on a single block before propagating anything
through the stack. One block of each type, 24 sites, **both delta families** (native counterfactual
D4 and small isotropic D1), all three metrics stored.

**Design point that decides the reading.** The exact Jacobian is by definition the best linear
predictor of the true effect as ε→0, so every LRP-style rule -- which deliberately deviates from the
true gradient -- *must* lose at small ε; a ranking there is near-tautological. Rules can only win at
finite ε, where the truth is a secant.

### 3a. The headline: R-lens is a finite-ε operator

The first version of this test omitted the R baseline entirely, so every rule was scored as
"J + rule" when the question was "does this complete R". With `R` implemented (LN-rule on both
residual RMSNorms, identity-rule on the MLP SiLU, half-rule on the gated MLP product):

| cos vs true effect | ε=0.01 | ε=0.05 | ε=0.2 | **ε=1.0** |
|---|---|---|---|---|
| full attn, D4 native — `J` | .9250 | .9949 | .9929 | .9264 |
| full attn, D4 native — **`R`** | .9137 | .9833 | .9853 | **.9468** |
| full attn, D1 isotropic — `J` | .9051 | .9936 | .9947 | .9428 |
| full attn, D1 isotropic — **`R`** | .8868 | .9737 | .9776 | **.9634** |
| GatedDeltaNet, D4 — `J` | .9282 | .9942 | .9942 | .9365 |
| GatedDeltaNet, D4 — **`R`** | .9161 | .9801 | .9843 | **.9585** |
| GatedDeltaNet, D1 — `J` | .9222 | .9933 | .9942 | .9403 |
| GatedDeltaNet, D1 — **`R`** | .9037 | .9731 | .9773 | **.9629** |

**R is worse than J at every ε ≤ 0.2 and better at ε = 1.0, by +0.020 to +0.023, in all four
block × delta-family conditions.** This is the first mechanistic account in this project of what the
R-lens recipe actually does: it trades tangent accuracy for secant accuracy. It is a finite-
displacement operator, which is exactly the regime a lens reading a whole activation operates in --
and it explains why R can help readability while (per §2) its effect-prediction advantage over J
sits at the noise floor.

### 3b. The new rules, composed on R

Δ against the R baseline. Only rows that move by >2e-4 are shown.

| full attention | D4 ε=0.05 | D4 **ε=1.0** | D1 ε=0.05 | D1 **ε=1.0** |
|---|---|---|---|---|
| `R+qk_norm` | −0.0104 | **−0.0088** | −0.0006 | −0.0005 |
| `R+attn_gate_half` | −0.0027 | **+0.0090** | −0.0034 | +0.0020 |
| `R+cp_value_only` | −0.0109 | **+0.0096** | −0.0067 | −0.0012 |
| `R+softmax_temper2` | −0.0051 | **+0.0093** | −0.0030 | +0.0012 |
| `R+softmax_temper4` | −0.0091 | **+0.0100** | −0.0054 | −0.0000 |

| GatedDeltaNet (48 of 64 layers) | D4 ε=0.05 | D4 **ε=1.0** | D1 ε=0.05 | D1 **ε=1.0** |
|---|---|---|---|---|
| `R+gdn_out_half` | −0.0026 | **+0.0051** | −0.0069 | **+0.0054** |
| `R+gdn_gate_frozen` | −0.0001 | −0.0001 | −0.0005 | −0.0000 |
| `R+gdn_qk_l2norm` | −0.0012 | −0.0000 | −0.0004 | −0.0000 |

Three conclusions, and two of them are negative for the highest-priority proposals.

**The Q/K-norm LN-rule is negative at every ε and both families** — the cheapest and most obvious
extension of R's own logic (the same operation R already repairs in the residual stream) does not
work. On natural deltas it costs −0.009 to −0.012.

**The GatedDeltaNet gate and Q/K-L2-norm rules do nothing** (±0.0012), despite governing 48 of 64
layers, which was the predicted highest-upside direction. Only `gdn_out_half` moves, and it is a
branch-gradient scaling rather than a structural repair.

**The rules that do help at ε=1 all damp routing sensitivity, and they do not replicate on
isotropic deltas.** `attn_gate_half`, `cp_value_only`, `softmax_temper` and `gdn_out_half` add
+0.005 to +0.010 at ε=1 on natural deltas, but on isotropic deltas the same rules give +0.002 to
−0.001. So the gain is specific to natural counterfactual directions, not a general property of the
operator — which is exactly the distinction the two delta families were included to expose.

Reading it together: **what pays is damping the routing/branch gradient, not repairing any
particular normalisation.** Every rule that helps is a variation on that (halve a branch, detach the
attention matrix, desaturate the softmax); every rule that targets a specific normalisation
(`qk_norm`, `gdn_qk_l2norm`) or a specific gate (`gdn_gate_frozen`) does nothing or hurts.

**Scale caveat.** The largest effect is ~0.011 on a cos of ~0.98, comparable to the §2 noise floor
for fitted operators. D3 has no CIs (24 sites, deterministic per site). These are suggestive, not
established, and no rule earns a full-stack lens fit on this evidence. `R` itself is the exception:
+0.020 replicated across four independent conditions.

**Not implemented:** AttnLRP's bilinear rules on `AV` and `QK^T` with the Taylor softmax rule. It is
the one item from the proposed list still missing; the composition trick (half-rule JVP on a product
= ½ the ordinary JVP, so it can be assembled from one-sided detach runs) makes it cheap to add.

### Three bugs, all of which silently produced "no effect"

Worth recording because each returned a plausible null:

1. **SDPA/flash has no forward-mode derivative** — every full-attention rule failed with
   `derivative for aten::_scaled_dot_product_flash_attention_backward is not implemented`, and the
   whole block reported NaN. Eager attention is required, and it is also what routes through
   `nn.functional.softmax`, which the value-only and tempered rules patch.
2. **`Qwen3_5RMSNorm` has no `variance_epsilon`** — it is `.eps`, and the weight enters as
   `(1.0 + weight)`. The q/k-norm rule silently raised and was recorded as unavailable.
3. **`beta = b.sigmoid()` is a tensor *method*.** Patching `torch.sigmoid` does not intercept it, so
   the first GDN run scored every gate rule *identically* to autograd. Fixed by patching
   `torch.Tensor.sigmoid` as well. The rules genuinely do nothing now, but the first run's null was
   an artifact.

### A numerical result that bears on artifact 007

Forward-mode AD works on this architecture (first time autograd of any kind has been verified here;
the whole project was built autograd-free because it was untested). That allows comparing the
finite-difference estimator against the true JVP:

| ε (× median‖h‖) | 0.001 | 0.01 | 0.05 | 0.2 | 1.0 |
|---|---|---|---|---|---|
| cos(finite difference, true JVP) — full attn | 0.262 | **0.925** | 0.995 | 0.993 | 0.926 |
| — GatedDeltaNet | 0.293 | **0.928** | 0.994 | 0.994 | 0.937 |

**At ε_ref = 0.01 — exactly how `J_loc` is defined throughout this project — the bf16 central
difference has cos ≈ 0.93 to the true Jacobian-vector product**, i.e. ~7% direction error from
truncation and rounding. Below that it degrades catastrophically (0.26 at ε=0.001).

This gives direct support to the numerical hypothesis raised in 007's corrections: `J_loc` is not an
exact Jacobian but a noisy estimate, and **SmoothGrad-J averages 8 such estimates**, so part of its
+0.17…+0.27 advantage may be variance reduction of the *measurement* rather than a property of the
model. The decisive follow-up is a Richardson-style control: average `J_loc` over ε_ref ∈
{0.005, 0.01, 0.02} at the *same* point, which reduces truncation error without any neighbourhood
smoothing. If that recovers much of the SmoothGrad gain, the gain is numerical.

---

## 4. What this changes

- **Fitting is cheap and works.** The remaining fitted-lens experiments (smoothed operating points,
  rule-modified lenses) are ~1.3 h each, not a different cost class.
- **The study needs a noise floor on every fixed-operator comparison.** ≈0.01 in cos at n=25. Most
  small margins in 007 and 008 do not clear it. This does not change any *verdict* — the secant
  failures were −0.1 to −0.9, and context averaging is +0.17 to +0.41 — but it retires several
  minor claims.
- **No backward rule earns a full-stack lens fit yet.** The two cheapest bets are negative; the
  routing-desaturation rules show the right qualitative crossover but at a magnitude comparable to
  the floor. The honest next step is more sites and CIs on D3, plus the Richardson control, before
  spending a fit.

## Reproduce

```
research/experiments/D1_time_fit.py        fit cost / memory sweep
research/experiments/D2_fit_jown.py        the fit (halves + merge)
research/experiments/D4_jown_analysis.py   noise floor + isolated context averaging  [no model]
research/experiments/D3_block_rules.py     block-level backward rules
research/ekko/rules.py                     the rules, as forward-preserving surrogates
```
