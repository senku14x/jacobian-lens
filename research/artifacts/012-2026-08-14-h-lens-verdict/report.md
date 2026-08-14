# 012 — The H-lens test, the Patchscopes baseline, and project close-out

**Date** 2026-08-14 · **Model** Qwen/Qwen3.6-27B · **Status** complete; project closed by
user decision after the pre-registered prediction failed. Raw metrics in
`research/artifacts/data/metrics/{E1_fit_config,E2_Honly_result,E2_result,E3_result}.json`.
Fitted H matrices (fp16, layers {8..60}, released convention) archived off-repo (too large
for git); fit config and twin halves documented in `E1_fit_config.json`.

---

## What was tested

**H-lens** = the R-lens recipe + the two remaining output-reproducing ("conservation")
backward rules: frozen attention pattern (exact for the value path) and the half-rule on
the GatedDeltaNet gated output (48/64 layers). Derived from the F11 account of *why* the
R-lens works (each LRP rule is the linear operator reproducing its component's output on
the full activation). Pre-registered prediction: **H > R on first-half pass@10, no
skip-ahead regression.**

Fit: released convention (target 62, skip_first 4, pile-10k, n=25, same 25 documents for
both halves), 13 source layers L8–L60, dim_batch 4, 593 s/prompt, ~4.1 h. Gates passed:
forward invariance top-1 agreement 1.0000 / max|Δlogit| 0.0000 on the full 27B (both
variants); probe Jacobians finite.

## Verdict: prediction falsified — branch closed

59-item calibration slice, matched 13-layer grid, identical readout plumbing:

| operator | pass@10 all | pass@10 first-half | mean first@10 layer | M4 answer-early |
|---|---|---|---|---|
| logit lens | 0.377 | 0.129 | 42.5 | 0.000 |
| J (released) | 0.390 | 0.068 | 46.0 | 0.118 |
| **R (released)** | **0.483** | **0.140** | 42.3 | 0.059 |
| H (ours) | 0.477 | 0.102 | 43.3 | **0.000** |

`H − R` first-half = **−0.038 [−0.089, +0.000]** — sign against the prediction, ~9× H's
twin-fit pass@10 floor (0.004). All-layers `H − R` = −0.006 (null). `H − J` all-layers
= +0.088 CI-clear (H is R-like; inherited, not new). Matrix level: H sits 0.25–0.45 relF
from released R (~2× twin floor), so the rules *did* move the operator — the movement
just didn't buy readability.

**What the falsification teaches.** Per-component conservation is exact mathematics
(the identities in FINDINGS F11 stand, and still explain R's existing rules), but it
does **not compound into better early readability through composition**. Most plausible
mechanism: the *fit averages the frozen coefficients across contexts* — the per-input
exactness argument licenses `T(x)·h(x)`, not `E_x[T(x)]·h(x*)`, and the averaging step
is exactly where the workspace project's largest measured error (context dependence,
1.8–4×, worst early) lives. The R_own control fit was skipped by user decision after
the preliminary table (gap ≈ 0 → nothing to attribute); consequence: our full-stack R
implementation remains unvalidated against the released R, which is acceptable only
because no positive claim rests on it.

Two survivals worth one line each: H never surfaces final answers early (M4 answer-pass
0.000 vs J's 0.118) while matching R overall — the least skip-ahead-prone operator
tested; and the trash-token column is script-biased (the heuristic counts non-Latin
tokens as trash, penalizing exactly the CJK meta-tokens shown meaningful elsewhere) —
do not quote it.

## The Patchscopes baseline: the actually load-bearing result of the evening

Same 60-item slice and grid; same-layer identity patchscope ("cat -> cat\n1135 ->
1135\nhello -> hello\n? ->"), 8-token greedy generation, substring scoring (metric
analogous to, not identical with, pass@10):

| set | surface rate all / first-half |
|---|---|
| typo | **1.000 / 0.900** |
| order-ops | 0.600 / 0.300 |
| multihop | 0.100 / 0.100 |
| multilingual, poetry, association | 0.000 / 0.000 |
| ALL | 0.283 / **0.217** |

Per-layer rates are **depth-flat** (0.07–0.20 from L8 to L60, no early collapse). On the
first-half aggregate the identity patchscope beats every linear lens (0.217 vs R's
0.140) despite losing overall; its zeros on semantic sets are template artifacts (an
echo-prompt cannot elicit "name the evoked concept" — patchscopes' known template
sensitivity), so the wins are strong evidence and the zeros weak.

## The concept the whole arc points to (close-out diagnosis)

Assembled from this project's own measurements:

1. Linear probes identify early-layer concepts at ~0.99 where every lens reads ~0.1
   (R-lens post) — the content is linearly *present* early, per concept.
2. No single linear map surfaces it: every fixed operator clusters at R̄ ± noise
   (007–009), the input-specific operator beats the averaged one 1.8–4× worst early
   (F3), the deviation is not low-rank-shared (F4), fitted maps memorize rather than
   generalize (F1/B7), and a principled per-component fix (H) does not help.
3. A decoder that *runs the model* (patchscope) is depth-flat.

The consistent explanation: **a probe picks its direction per concept; a lens is one
global map that must align every early concept direction with its vocabulary direction
simultaneously — and no such global linear correspondence exists, because early and
late layers use different codes, and the early→vocabulary translation is itself the
computation the intervening layers perform.** The early readout gap is *computational,
not representational*. Asking a matrix to close it is asking a matrix to be layers
8–30. This is why every matrix-class improvement (J→R→H, secant, anchored, blend,
conditional) moves within a narrow band, and why the two instrument families that cross
the gap are those that either train per-concept (probes, causally-calibrated
dictionaries) or execute computation (patchscopes, NLA/AO).

**Implication for the North Star** (causally-validated surfacing of personas/anomalies):
the tripwire+describe+certify stack stands, but the describe layer should be
decode-side (NLA/AO/patchscope-with-learned-templates), with lenses as the cheap
always-on flagging layer at mid depth where they demonstrably work, and interchange-
style interventions as the certifier. The three-instrument comparison on Qwen2.5-7B
(released NLA at L20 + cheap J/R fits) remains the natural next experiment — in the
NLA project, where it now belongs.

## Ledger of what this project leaves behind

Validated instruments: exact autograd-free effect measurement (`forward_from`,
T_IG ≈ 0.95 ceiling); leak-proof nested splits; twin-fit noise floors (transport and
pass@10); the M4 skip-ahead guardrail; a 38-item entity-disjoint independent eval set
(`research/data/e8_independent/`, unused — built for a win that didn't materialize);
the E2 equal-footing harness with calibrated-z scoring mode; the blackmail qualitative
sweep (E5). Closed branches: fitted transport (secant/anchored/low-rank/spectral/
fusion), continuous conditional lenses, backward-rule composition (H). Parked with
priors: Koopman/workspace-autonomy (011). In flight at close: calibrated-z rescoring
(E6) and the E5 blackmail sweep, both auto-committing on completion.
