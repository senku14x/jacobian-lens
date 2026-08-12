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
| Q/K-norm LN-rule, attention output-gate half-rule | **negative or nil** at every ε |
| Value-only / tempered-softmax rules | lose at ε≤0.2, **win at ε=1.0** — the predicted crossover, but small |
| GatedDeltaNet gate/state rules (48 of 64 layers) | **nil** — ±0.0004 |
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

Per the review's own first experiment: test each rule on a single block before propagating
anything through the stack. One block of each type, 24 sites, cos against the true finite effect.

**Design point that decides the reading.** The exact Jacobian is by definition the best linear
predictor of the true effect as ε→0, so every LRP-style rule — which deliberately deviates from the
true gradient — *must* lose at small ε; a ranking there is near-tautological. Rules can only win at
finite ε, where the truth is a secant. So the table below is Δ against plain autograd, swept over ε.

**Full attention (block 31, 16 of 64 layers):**

| rule | ε=0.01 | ε=0.05 | ε=0.2 | **ε=1.0** |
|---|---|---|---|---|
| `cp_value_only` | −0.0114 | −0.0124 | −0.0083 | **+0.0093** |
| `softmax_temper4` | −0.0096 | −0.0104 | −0.0064 | **+0.0096** |
| `softmax_temper2` | −0.0056 | −0.0060 | −0.0028 | **+0.0090** |
| `qk_norm` (LN-rule on q/k) | −0.0021 | −0.0016 | −0.0021 | −0.0039 |
| `attn_gate_half` | −0.0006 | −0.0005 | −0.0004 | +0.0006 |

**GatedDeltaNet (block 30, 48 of 64 layers):**

| rule | ε=0.01 | ε=0.05 | ε=0.2 | ε=1.0 |
|---|---|---|---|---|
| `gdn_out_half` | −0.0051 | −0.0044 | −0.0037 | **+0.0041** |
| `gdn_gate_frozen` | −0.0001 | −0.0002 | −0.0001 | −0.0001 |
| `gdn_qk_l2norm` | −0.0003 | −0.0003 | −0.0002 | −0.0001 |

**The predicted crossover is real.** Every rule that detaches or desaturates *routing* — value-only,
tempered softmax, and the GDN gated-output half-rule — loses below ε=0.2 and wins at ε=1.0. That is
the signature of a modified backward implicitly behaving like a path average, and it is the same
direction as the SmoothGrad result.

**But the two cheapest bets are negative.** The Q/K-norm LN-rule — the most direct extension of
R-lens's existing logic — is negative at *every* ε including ε=1.0. The attention output-gate
half-rule is nil (±0.0007). And the GatedDeltaNet gate/state rules, predicted to have the highest
upside because they govern 75% of the model, do **nothing** (±0.0004).

**Scale caveat.** The largest effect here is 0.012 on a cos of ~0.99, and the ε=1.0 wins are
+0.009 — the same order as the §2 noise floor for fitted operators. D3 has no CIs (24 sites,
deterministic per site), so these should be treated as suggestive, not established, and a rule
should not be propagated through the stack on this evidence alone.

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
