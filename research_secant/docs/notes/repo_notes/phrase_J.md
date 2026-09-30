# phrase_J — multi-token extensions of the J-lens (senku14x/jacobian-lens, branch `phrase_J`)

Commit 87af681, last activity 2026-09-17. Clone: `/workspace/repos/phrase_J`. Everything is under `research_phraseJ/`:
`README.md` (state + reproduction order), `STATUS.md`, `docs/06_plan_decisions.md` (decision log),
`docs/05_ideas_review.md` (scored ideas ledger, ~60 ideas), and `research_artifacts/NNN/{design,report}.md` (design
written before the run).

**Setup [REPO].** Qwen3.6-27B bf16 on one A100. Released J/R/template lenses. Baselines: released J, R, logit lens, and
the released template lens (`templates+phrases_v3`). Prefix-collision families with a shared first token: New {York,
Zealand, Delhi, Jersey}, South {Africa, Korea, Dakota, Carolina}, North {…}, United {…}, San {Francisco, Diego,
Antonio, Jose}. A 349-prompt latent two-hop harness (5 routes × 4 frames, phrase/answer absent from the prompt).

## Experiments and results [REPO]

| # | question | result |
|---|---|---|
| 001 template geometry | Is within-family phrase identity linearly present, emission vs latent? Does the template lens recover it? | Latent identity decodable by a grouped cue-out probe: AUC 0.94–1.00 at L52–62; 0.72–0.83 at L8–48, where J-sum reads 0.55–0.83. Emission-trained directions transfer to latent. J rows ⟂ template rows (cos 0.01–0.15). Covariance-duals unstable (split-half 0.78–0.92). |
| 002 exact compat | Does a scalar backward reproduce the released J estimator? | Yes, cos 0.9999 at the fit's graph shape. Early-layer discrepancy is bf16 batch shape only (gone by ~L44). log-prob phrase objectives saturate in emission contexts. |
| 003a objective screen | Which phrase objective (lin/logit/odds/logp) is reliable, and does it beat J-sum? | `lin` (J-compatible linear target) is reliable at L56–60. **No readout gain over J-sum.** Constituent-J ceiling = full probe, except San. |
| 003a-controls | Symmetric comparison; is the mid-band gap phrase-specific; how many San-like families? | Diff-vs-diff: phrase-J at parity or below J-sum (except San L52). The mid-band gap is **generic early-J** (single tokens show the same +0.24 at L8). Only San has a late residual (+0.10…+0.17). `lin` is a lag-specific covector, not a J row. |
| 004 causal geometry | At matched norm/damage, one pinv swap operator: does phrase-J (PJ) edit a latent intermediate more selectively than constituent J (Jc)? | Gate PASS (first-order slope 1.01, R² 0.97). **Pooled: Jc > PJ** (−0.33…−0.61 nats, CIs < 0 at ρ≤0.2), also at matched damage. **No method flips** (≤3%; clean margin ≈4.9 nats; two-layer edits L52+56 move 0.4–1.6). **San: PJ > Jc at every dose and position set** (up to +4.6 nats; Jc ≈ 0). n=115 items (San 11). |

**Verdict [REPO].** Phrase-J is closed as a general method (read and write). The surviving object is San-type concepts,
where phrase information lies outside constituent-token J coordinates. Next (not run): a CPU Δ_residual screen over many
candidate families, then a pre-registered 004 replication on hits. If none are found, close with "late multi-token
identity is compositional in J coordinates, one exception".

## Other learned facts [REPO, decision log]
- Only United States / United Nations is a two-member family in the released template vocabulary. The vocabulary is
  lowercase-normalized (1,298 lowercase-artifact rows).
- `blackmail` is a single Qwen token. Los Angeles / Las Vegas share no token.
- The Qwen3.6-27B checkpoint ships an MTP module (one block, conditioned on the t+1 embedding) that HF doesn't load.
  It's a nonlinear t+2 readout.
- Same-token J rows across prompts: cos 0.03 at L8 → 0.95 at L60.
- The ideas review's highest-scored items not done anywhere:
  - V (swap/clamp battery, prerequisite);
  - R (sparse-frame readout);
  - JJJ (silence score);
  - AA2 (reflection training on open weights, "highest field value", out of scope then);
  - P (per-prompt J ceiling with an LRP arm).

## [MINE] takeaways
- This directly answers "does a multi-token causal lens beat composition of token lenses?" At late layers on these
  families: no. The WorkspaceBench multi-token families may still reward phrase readers, but part of that reward could
  be recovered by *summarizing constituent-token J bags*, which is exactly what WSB's summarizer does for token lenses.
- 004's "no flips at two layers" matters for eval design. Swap success rates depend strongly on band width
  (the paper clamps a wide band; phrase-lens uses L24–47). Report band with every swap number.
- Reusable code: `scripts/lib/edit_harness.py` (single pinv swap operator, norm-targeted doses, matched-damage
  interpolation) and `phrase_objective.py`.
