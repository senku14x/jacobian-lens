# Research log — ekko-lens

Append-only. Newest entries at the bottom. The **Current state** block at the top is the only part
that gets rewritten. Null results and dead ends get entries too — they are what stops us repeating
work.

---

## Current state (2026-08-12, all runs complete)

**Full write-up:** [`artifacts/007-2026-08-12-transport-operators/report.md`](artifacts/007-2026-08-12-transport-operators/report.md)
· running numbers: [`STATUS_2026-08-12.md`](STATUS_2026-08-12.md)

- **G-SECANT: FAIL.** Broadening the bank 4.2× (D4 effective rank 117 → 555) tripled the secant's
  *in-family* margin over `J̄` and moved none of the three span-immune checks. Stein −0.31/−0.47,
  in-span −0.12/−0.20, behavioural M2 −0.86 (at the random-direction noise floor). Robust to the
  identifiability objection that made the first pass provisional.
- **G-CONTEXT: PASS 2/3.** `J_loc/J̄` = 4.34 / 3.17 / 1.95 at L16/L31/L46, monotone in depth;
  context-averaging is the dominant failure of the fixed lens, and worst early.
- **G-CONDVIABLE: FAIL.** An *oracle* rank-16 conditional model recovers 23% / 40% / 85% of the
  `J̄ → J_loc` gap — it works only where the gap is smallest.
- **The finding with headroom is positive.** SmoothGrad-J beats the *exact* local Jacobian by
  +0.17…+0.27, and the gain is **denoising, not path-averaging**: orthogonal-only smoothing
  recovers 99–109% of it in 9 of 9 conditions. So it is *not* structurally barred from a fixed
  operator. Validated ceiling: `T_IG` ≈ 0.95.
- **Next:** fit the released estimator at smoothed operating points and test whether the per-input
  gain survives averaging into one matrix.

---

## Superseded state (2026-08-11)

**Question.** Which transport operator best predicts the causal fate of an intermediate
representation? Competing hypotheses: `J̄` (J-lens), `R̄` (R-lens), `J_ℓ(x)` (input-specific
Jacobian), `J+λI`, `T_sec` (finite/secant), `T_RC` (R-anchored), `T_smooth` (SmoothGrad-J).

**Model.** Qwen3.6-27B, the released-lens convention throughout: `target_layer=62` (= n_layers−2),
`skip_first=4`, corpus `NeelNanda/pile-10k`, n=25. Released J/R pair from
`camilablank/workspace-lenses`. 4B is used only to debug harness code, never for science numbers.

**Where we are.** Stage 0 gate passed. §4.4 harness validation passed. Perturbation bank built at
layers 16/31/46. Next: Stage 2 secant fit + span diagnostic (004), then behavioral M2 (005), which
is the primary gate signal.

**Nothing is committed yet** — holding until the user says so.

---

## 2026-08-11 — Setup and Stage 0

Cloned `senku14x/jacobian-lens`, branch `ekko-lens`, upstream `anthropics/jacobian-lens` added
read-only. Environment: 1×H100 80GB, torch 2.13.0+cu126, transformers 5.15.0. `pip install -e '.[dev]'`,
upstream `pytest tests -q` → **32 passed**.

**Facts established that change the plan doc** (`ekko_lens.md`):

- Released lenses use **`skip_first=4`** and **`NeelNanda/pile-10k`**, not the doc's `skip_first=16`
  / WikiText-103. Verified from `provenance` inside `lens.pt`. Adopted the release convention as the
  project default — otherwise the `R = θ_R` anchor and any matched refit regularize toward an
  operator fit under a different position mask.
- Released format: `dict[int → fp16 [d,d]]`, layers `0..target` inclusive, **target row is exactly
  `I`** (‖J₆₂−I‖_F = 0.0). Loads with `weights_only=True` — no arbitrary pickle needed.
- **Every Qwen3.5/3.6 model in the release is a GatedDeltaNet hybrid**: `layer_types` = 48
  `linear_attention` + 16 `full_attention` (27B). The γ family in Stage 5 is therefore undefined on
  ¾ of the blocks and must be re-derived, not reused. Does not affect Stages 0–2, which live at the
  residual interface. `gemma-3-27b-it` is the only standard-attention dense release and it is gated.
- `fla` / `causal_conv1d` / `kernels` / `flash_attn` are **not installed**, so the GatedDeltaNet
  blocks run pure-PyTorch fallbacks.
- Eval sets are larger than the doc states: association 102, multihop 93, multilingual 107
  (428 intermediates), order-ops 55, poetry 98, typo 96.
- `tokenizer.add_bos_token` is `False` and `force_bos` does not change it on this tokenizer.
  Recorded because it is a tokenization-level confound if any later comparison assumes otherwise.

**Design decisions** (user-approved):

1. Science runs on 27B directly. The "smallest model with a visible R>J gap" rule was motivated by
   *fitting* cost, and Stages 0–2 do not fit anything — released lenses plus closed-form secant. It
   only bites at Stage 5.
2. §4.4's "R>J replicates" check was going to consume frozen corpus C. Resolved by carving a
   permanent **10-items-per-category calibration slice** (seed 0), never reported as a headline; the
   remaining ~85% of each eval set stays frozen.
3. Stages 0–2 run with **zero autograd**. `J_ℓ(x)δ` comes from the bank's own ε=0.01 antithetic
   measurement (the central difference *is* the JVP up to O(ε²)), and `T_sec`/`T_RC` are closed
   form. This keeps `torch.func` and double-backward — untested on GatedDeltaNet — off the critical
   path entirely.

### `000_probe_arch.py` — architecture probe

Blocks receive `(hidden_states,)` + `position_embeddings`, `attention_mask=None`, `position_ids`,
`past_key_values=None`, `use_cache=False`. **None of those depend on the hidden state**, and the
mask is `None` for both block types with no padding. So `forward_from` can replay the upper stack
with captured kwargs rather than reconstructing masks/RoPE by hand — decided against the doc's
hand-rolled hook-free function, which would have had to replicate the per-layer
`causal_mask_mapping` dispatch.

### `001_stage0_checks.py` — **GATE PASS** on both 4B and 27B

| check | 4B | 27B |
|---|---|---|
| A wrapper identity `unembed(h_L)` vs logits | max diff **0.0** | max diff **0.0** |
| B `forward_from` round-trip at 25/50/75% | **0.0** at L8/15/22 | **0.0** at L16/31/46 |
| B batch-of-8 self-consistency | exact | exact |
| C released-lens format / anchor `I` / J≠R | pass | pass |

Known-answer reads on 27B: boot-country → *Italy* rank 1 @L36, *euro* rank 1 @L57; web-spinner →
*spider* rank 4 @L39. Sensible, so the readout plumbing is right.

**Negative/inconclusive:** the R-lens post's four published early-layer examples did **not**
reproduce as stated (e.g. sushi→Japan first top-10 at L23 for both J and R, vs the published "R L2
vs J L14"). Only the `aganst`→`against` item is verbatim from the eval set; the other three prompts
are my reconstructions, so this is **uninformative about the published claim** — different prompts.
Not pursued further; superseded by the quantitative check below.

### `002_calib_passk.py` — §4.4 harness validation, **R>J replicates**

Calibration slice, 59 items over 6 categories, released J vs R, pass@10 (min-over-layers rank ≤ 10,
per `data/evaluations/README.md`):

| span | J | R | Δ |
|---|---|---|---|
| all layers | 0.479 | 0.534 | **+0.055** |
| first half | 0.089 | 0.150 | **+0.061** |

Mean first layer reaching top-10: J 43.5 → R **39.7**. Per-category Δ ≥ 0 in all six (typo +0.30 in
the first half, order-ops/poetry +0.10 all-layers, multihop/association 0.00).

*Observation, n=59, no CIs yet.* Direction matches the published claim and is unanimous across
categories; magnitude is small and the sample is a calibration slice, so this validates the
instrument — it is not a result. Practical consequence: **R is a meaningful anchor for `T_RC`** on
this model, which was not guaranteed.

### `003_bank.py` — perturbation bank

Layers 16/31/46 (25/50/75% of target 62). Minimal pairs from
`data/experiments/flexible-generalization.json`: 186 token-aligned (template, arg→arg′) pairs over
4 categories, 63 base prompts, 169 sites.

Design point worth keeping: each template is embedded in a **96-token pile-10k prefix** so the
perturbed position sits deep in a ~104-token context. Perturbing a bare 10-token prompt would put
every site off-distribution for operators fitted with `skip_first=4` on 128-token pile text, and
`J̄` would look bad for a reason unrelated to averaging.

Families: **D4** natural activation deltas (primary), **D1** isotropic (identification, ε-sweep,
span-filling), **D2** lens-token directions `v_t = J^T(γ⊙u_t)` (held out for transfer). Every
direction measured at ε ∈ {0.01, 0.05, 0.2, 1.0} × median‖h‖, plus native magnitude for D4.
Targets stored: `Δ_one` and `Δ_odd`, each summed-over-current-and-future (primary, matches the
J̄/R̄ estimator) and self.

---

## 2026-08-12 — Phase B: the broadened bank reopens G-SECANT

Full numbers in [`STATUS_2026-08-12.md`](STATUS_2026-08-12.md). What this session established, and
what it does *not*:

**B1 — broadened bank built** (`B1_bank_broad.py`, 3 358 s). 10 categories × 4 templates × 6 args,
240 base prompts, 30 096 deltas/layer (was 7 222). The pre-registered sanity gate passes: D4
effective rank at ε=0.2 rose **117 → 555**, participation ratio 57.6 → 169.2, distinct calibrate D4
directions 246 → 1 695. Still 11% of d_model, so off-family transfer stays identifiability-limited.

**The headline reversal.** On the broadened bank, held-out D4 at ε=0.2, `T_sec − J̄` = **+0.294 /
+0.180 / +0.142** at L16/L31/L46, CI-clear at every layer. On the narrow bank the same quantity was
+0.075 / +0.050 with CIs overlapping J̄. So the Phase-A conclusion "the secant is not a real new
object" was, on this axis, **a statement about 117 identified directions, not about the secant.**

Held at observation level: the secant's apparent quality scales with the breadth of the family it
is fitted on. That is equally the signature of (a) a rank-starved fit finally getting identified and
(b) a fit that is memorising a wider distribution. This table cannot separate them, which is exactly
why the gate was placed on the span-immune checks instead. Those are re-running.

**A6 / G-CONTEXT re-read.** `J_loc/J̄` = 4.34 [4.14,4.53] / 3.17 [3.05,3.29] / **1.95 [1.90,2.00]**.
Point estimates essentially unmoved from the narrow bank; CIs ~4× tighter on 3.75× the prompts.
**L46 crossed below 2×** — its CI was entirely above 2.0, it is now entirely below. G-CONTEXT passes
2 of 3, and the failing layer is the deepest. The depth gradient is monotone.

**A4 / M2 — behavioural gate (narrow-bank operators).** Ablation on 40 probe-swap items, judge-free
Δ log-prob, same band for every operator, matched-norm random controls. Band {16,31,46}: J̄ **+0.856**
[+0.44,+1.35] > R̄ +0.587 > logit +0.022 > T_RC +0.006 > T_sec^D4+D1 −0.002 > random −0.005 >
T_sec^D4 −0.029. `T_sec − J̄` = −0.858 CI-clear; `T_sec^D4 − random` = −0.025 CI-clear. Top-1
retention on pile text .96–.98 for every operator, so nobody wins by breaking the model. Ablation
never asks an operator to predict a held-out δ, so identifiability cannot explain this — **but the
operators are the narrow-bank ones, so the re-run is required before it counts against the refit.**

Also here: `J̄ − R̄` = +0.269 [+0.073,+0.554] CI-clear, i.e. **J beats R on this metric.** Scope —
40 single-token items, 3-layer band, Δ log-prob; not the post's protocol (30 multihop, first-half
band, autorater). Not a replication attempt, no contradiction claimed; it does mean the R anchor's
value is unestablished on this metric.

**C2 — conditional low-rank model: negative where it matters.** Deviation PCA over 56 sites explains
top-16 = .606/.525/.495 of variance (L16/L31/L46) — 2/3 clear the pre-registered >50% bar, but 0.525
and 0.495 straddle it, so the proxy is doing no work. The direct measurement: with **oracle**
coefficients, rank-16 recovers **23.2% / 40.0% / 85.2%** of the `J̄ → J_loc` gap, and rank-32 adds
+2.4 / +0.4 pts. Recovery is anti-correlated with need — it works at L46 where the gap is smallest.
Called FAIL on the direct measurement rather than the proxy. Predicted-coefficient variant not run:
a failing oracle bounds it.

### Bugs found and fixed (verification layer — no science number affected)

1. **A0's `counts` assert was hard-coded to the original bank** (`total=7222`), so the broadened
   bank reported FAIL on all three layers. Now manifest-driven: the real question is whether the
   file contains what the builder recorded writing.
2. **A0's duplicate check used an 80-dim rounded signature** and flagged 1–4 collisions/layer.
   Every one is a **D2 row sharing a `tag`** — and D2 is `v_t = Jᵀ(γ⊙u_t)`, a function of token id
   and layer only, so identical directions at different sites are the family's *definition*. The
   assert now permits exactly that, rejects any duplicate inside D4/D1, and adds a full-precision
   exact-duplicate check. **Both banks PASS** — including the original, the regression check that
   the fix did not merely loosen the test.

### Bug found and fixed (analysis layer — this one could have produced quiet nonsense)

3. **`A2_A5_model.py` rebuilt base prompts by re-running the original builder's template code.**
   Against the broadened bank its base indices addressed the wrong prompts; it crashed on
   `index 101 out of bounds`. A crash was the lucky outcome. Fixed at the root rather than patched:
   `B2_patch_bank_bases.py` writes `base_ids` into every bank pack and **verifies by
   re-measurement** — recompute `h_ℓ(x′)[p] − h_ℓ(x)[p]` with the model and require cos > 0.999
   against the stored delta. Observed **1.000000** on every checked row, both banks, all layers.
   Consumers now read `base_ids` from the pack and assert its presence. General lesson kept: an
   artifact that requires re-running its producer to interpret is not an artifact.

### What this changes about the plan

The routing decision made yesterday (G-SECANT FAIL → Branch C2) was made on narrow-bank numbers and
**C2 has now itself failed**. G-SECANT is reopened and is the live question again. Priority order:
(i) A2/A5 on B1 — Stein control and in-span transfer decide it; (ii) A4 M2 on B2 operators;
(iii) a **learning curve** — fit `T_sec` on 1/8, 1/4, 1/2, all calibrate bases and plot held-out
cos, which is the cheapest test that separates "rank-starved but converging" from "converged and
still worse", and which the reversal above makes the obvious next measurement.

---

## 2026-08-12 (later) — Phase B closes: G-SECANT fails on all three span-immune axes

Detailed write-up with figures: [`artifacts/007-2026-08-12-transport-operators/report.md`](artifacts/007-2026-08-12-transport-operators/report.md).

**The reopening closed.** Broadening the bank raised `T_sec`'s in-family accuracy a lot and its
span-immune accuracy not at all:

| L16 | narrow bank | broadened bank |
|---|---|---|
| in-family `T_sec − J̄` | +0.075 | **+0.294** |
| Stein `T_sec − SG-J` | −0.302 | **−0.306** |
| in-span `T_sec − J̄` (D1) | −0.108 | **−0.119** |
| in-span `T_sec − J̄` (D2) | −0.127 | **−0.199** |
| behavioural `T_sec − J̄` (M2) | −0.858 | **−0.861** |

**G-SECANT: FAIL**, now robust to the identifiability objection that made the first pass
provisional. Correction to yesterday's entry: on the refit operators `T_sec^D4 − random` is −0.005
and no longer CI-clear, so "worse than a random direction" was specific to the rank-starved D4-only
fit. "Indistinguishable from random" survives.

**B3 learning curves** put a number on the negative instead of leaving it categorical. Fitting on
15→120 calibrate prompts, nothing has plateaued: in-family D4 +16–26% over the last doubling,
cross-family D2 +37–64%, captured D2 energy +24–27%, effective rank +71%. At 240 base prompts
`T_sec` reaches 0.092 cross-family where `J̄` reaches 0.311, closing at ~+0.03 per doubling —
parity would need ~6 more doublings (~64× corpus) if the trend held, which it must not. So: **a
bounded, quantified negative at this data scale, not a proof of impossibility.** Worth stating that
way; "the secant doesn't work" would have been an overclaim.

**B4 — the real finding, and it is positive.** SmoothGrad-J beats the *exact* local Jacobian by
+0.17…+0.27 CI-clear. Two mechanisms were possible and they differ in whether a lens can ever
capture the gain: path-averaging (δ-dependent, barred from any fixed matrix) or denoising
(δ-independent, fittable). Decomposing the smoothing direction settles it — 60 sites × 300
directions × 51 held-out prompts per layer:

**Orthogonal-only smoothing recovers 99–109% of the isotropic gain in 9 of 9 (layer × ε)
conditions**, while doing no path averaging at all. Parallel-only — which is *nothing but* path
averaging — recovers far less and at L16/ε=0.05 is catastrophically worse than no smoothing
(.217 vs J_loc .649). The σ×ε grid separately shows σ* tracking ε (0.05→0.05, 0.2→0.2, 1.0→0.5) at
all three layers, which initially looked like the opposite conclusion. The two facts coexist:
smoothing is an **isotropic regulariser whose optimal strength scales with the effect being
predicted but whose direction is irrelevant** — variance reduction, which a 2-point average along
one line cannot deliver and any (d−1)-dimensional average can.

**Positive control caught a bug in itself.** `T_IG` must read ~1.0 by the fundamental theorem of
calculus; it read 0.81/0.84/0.62. Cause: ∫₀¹J(h+tδ)dt·δ = `Δ_one`, but the target is the antithetic
`Δ_odd`, which equals the integral over the *symmetric* interval [−1,1]. Sampling t symmetrically →
**0.921/0.947/0.879**. Lesson kept: a control reading 0.84 where theory says 1.0 is not "close
enough" — it was pointing at a genuine estimand/target mismatch, and chasing it bought a validated
ceiling (T_IG ≈ 0.95 at ε ≤ 0.2) that every other operator is now scored against.

### Where this leaves the plan

Both fitted-operator branches are closed (G-SECANT fail, G-CONDVIABLE fail). What replaces them is
better supported than either: the largest available gain over `J̄` is **not** structurally barred
from a lens, because it is about *where* the Jacobian is evaluated, not about which δ is applied.
Next experiment is one change to the fitting loop — fit the released estimator at **smoothed
operating points**, `E_x E_u[∂h_final/∂h_ℓ|_{h+u}]` with isotropic u at σ ≈ 0.05–0.2×median‖h‖ —
and test whether the per-input +0.2 survives averaging into a single matrix. It has a measured
ceiling (T_IG ≈ 0.95) and a measured irreducible floor (G-CONTEXT says 2–4.3× of the gap is
context-averaging no fixed matrix can recover).
