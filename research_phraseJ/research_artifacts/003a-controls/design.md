# 003a-controls — Three cheap uncertainties before choosing 003a2 vs 004

Status: **agreed 2026-09-17** (execution order handed over by Vishesh after the 003a report). Steps 1, 3, 6 are
re-analyses of existing 003a/001 data (CPU); steps 2, 4, 5 are small new GPU runs (< 20 min).

## Questions

1. **Is the 003a readout comparison symmetric?** J-sum was scored as a pairwise difference S_J(A) − S_J(B) while
   Phrase-J "own" was scored as hᵀv_A. Produce the full 2×2 {own, difference} × {Phrase-J lin, J-sum} on the same
   held-out latent activations, pairs, and AUC code. Reading: contrastive scoring must help Phrase-J *more than it
   helps J* to count; parity means the post hoc difference result was restoring symmetry.
2. **Is the mid-band probe-vs-J gap multi-token-specific?** Matched single-token countries (France, Japan, Brazil,
   Egypt, Australia, Italy) in the same two-hop harness (001's four routes × four frames, rotating cues), full
   probe (leave-one-cue-out) vs released J (own-token difference score) vs logit lens across L8–L62.
   Δ_single(L) = AUC_probe − AUC_J vs Δ_multi(L) from 001 (probe cue-out − J-sum). Reading: Δ_single ≈ Δ_multi →
   the early gap is a generic early-J problem, stop treating it as a phrase target; Δ_multi ≫ Δ_single → mid-band
   remains a Phrase-J target.
3. **How many San-like families exist?** For every 001 family with latent prompts (New, South, North, United, San),
   every layer: Δ_residual = AUC_full-probe − AUC_constituent-ceiling (both leave-one-cue-out), bootstrap CI over
   cue groups. Triage only, not selection. Reading: San alone → no general readout claim; several → they are the
   benchmark class.
4. **Does `lin` at L52 converge?** 40 more natural contexts for New Zealand and South Korea (n = 20, 40, 60), same
   objective/reduction/layers/eval set; split-half reliability and latent AUC (own and difference) vs n.
5. **Positive-control decomposition.** For the first tokens " New" and " South": single-lag g₁^lin at t′ and
   source-mean; all-target v_lin on the same natural contexts; all-target v_lin on 20 generic pile sequences; each vs
   the released J row and vs each other, with split-half cosines. Attribute the low 003a cosine to target lag,
   context distribution, or sample size before calling it variance.
6. **Constituent-span geometry (wording).** The ceiling features are s_t(h) = ⟨h, Jᵀq_t⟩ / rms(Jh) (a per-sample
   scalar), so the linear span of the raw covectors {Jᵀq_t} is exactly the feature-defining span; membership is
   unaffected by the per-sample scale. The report will say "raw causal-covector span, identical to the feature span
   up to per-sample scaling" and not recompute.

## Decision after 1–6

Branch A (several San-like families and matched Phrase-J difference > matched J difference) → pre-registered 003a2
with family-centering. Branch B (San unique, or Δ_single ≈ Δ_multi) → drop the generic readout claim; pull 004
forward for intervention geometry. Branch C (`lin` does not stabilize with n; matched comparisons do not help) →
corpus-averaged Phrase-J is a negative on this model; pivot to template / subspace / binding approaches.

## Cost

CPU: minutes. GPU: 96 single-token prompts captured (1 min); 80 new natural contexts × 2 `lin` backwards (≈ 5 min);
≈ 80 all-target backwards + 20 pile forwards (≈ 4 min).
