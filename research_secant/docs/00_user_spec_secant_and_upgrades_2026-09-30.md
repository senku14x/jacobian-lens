# Spec: the secant decomposition and J-lens method upgrades

*(The user's method spec, pasted in the working session and copied here verbatim for reference; only this note is added. Review: `01_spec_review_2026-09-29.md`. What was run from it: `../STATUS.md`.)*

*Method branch, 2026-09-30.* The companion branch on what the workspace is lives in `claude/workspace-identity-branch-2026-09-30.md`. Both supersede the relevant parts of `claude/idea-review-2026-09-30.md`.

**Contents**
- **Part I** is the implementation spec for the secant idea.
- **Part II** covers the other method upgrades I would try, ranked, each with a mini-spec.
- **Part III** lists what not to do.

All priors are my guesses, recorded before anything is run.

---

# Part I — The secant decomposition: why readable content is or isn't used

## I.0 What it answers, in one paragraph

RHUT showed that a concept can be readable at a position without that position driving behaviour. This spec explains each readable-but-unused cell with an exact identity, not a correlation.

The idea is to take one content difference Δ at one cell and read its effect four ways. The four ways differ on two axes:
- where the effect is measured: averaged over generic contexts, or in this context;
- how it is measured: first order (a tangent), or finite (a secant).

|  | tangent (first order) | secant (finite) |
|---|---|---|
| **averaged over generic contexts** | **T̄** — the J-lens quantity, horizon-matched | **S̄** — the "secant lens" |
| **this context** | **A** — in-context tangent (= AtP) | **P** — patching effect (= leverage) |

The four readouts are tied together by exact identities:

```
T̄ − P = (T̄ − A) + (A − P)         = context gap (first order) + in-context curvature
T̄ − P = (T̄ − S̄) + (S̄ − P)        = host curvature            + context gap (finite)
R_lens − T̄ = horizon/weighting gap  (released lens vs this cell's actual future)
```

Single-state readouts (Δ = h − h̄ rather than a difference of two states) also carry an offset term, O = ⟨J̄ᵀu, h̄⟩. This is the frequency term.

**Why the theorem doesn't kill this.** The secant theorem (idea review §3.1) shows that no *fixed matrix* beats J̄ at transporting perturbations that don't depend on context. None of the four quantities here is a fitted matrix:
- **P** and **A** are exact per-context quantities.
- **S̄** is a nonlinear average over contexts. To first order it equals T̄, so any difference between them is curvature. That is exactly what we want to measure.

## I.1 Definitions

**Model and harness.** Qwen3.6-27B, run on the RHUT harness:
- float32 from block 35;
- writes at block outputs;
- the realized-write gate: ρ ∈ [0.9, 1.1], κ ≥ 0.99.

**Lens.** Use **J_CB** (`camilablank/workspace-lenses`), because its construction is recorded in the checkpoint: target is the block-62 output, t_max 128, skip_first 4, 25 pile-10k documents, uniform weighting. J_NP's target layer and skip settings are not stored, so report it only as a secondary, rank-level comparison.

The estimator, from `jlens/fitting.py`, is

```
J̄_ℓ = mean over source positions p ∈ [4, 126] of  Σ_{t'=p}^{126} ∂y_{t'} / ∂h_{ℓ,p}
```

where y_{t'} is the block-62 output. Because each source position p only sees targets after it, lag δ gets weight (123 − δ)/123. In other words, **J_CB forecasts future logits over up to 122 positions, with triangular weights.** Section I.4 explains why this matters.

**Readout covector.** u_w = (1+γ) ⊙ W_U[w], i.e. the RMSNorm gain folded in (the O4 fix). Stack the rows as U.

**The lens-matched functional.** For token w, cell (ℓ, t), context c and target window W:

```
φ^w_{c,W}(h) = ⟨u_w, Σ_{t' ∈ W} y_{t'}(h)⟩
```

Here y is recomputed with h_{ℓ,t} := h and everything else in c held fixed (teacher-forced). Default W = [t, end of sequence]. Causality makes y_{t'<t} unchanged, so you can sum over the whole sequence and the restriction to t' ≥ t is automatic.

Always work with **contrasts**, φ^{a−b} = φ^a − φ^b, for example clean intermediate minus donor intermediate. Contrasts cancel shared offsets and most of the frequency effect.

**The behavioural functional ψ.** Use RHUT's sequence endpoint: the log-prob margin (donor answer − own answer), computed through the real model (block 63 and the final norm). Where an organism has two consumers, compute ψ for each, e.g. report and parity on the sum organism.

**Content difference Δ.**
- *Primary:* Δ = h_{ℓ,t}(c) − h_{ℓ,t}(c_d), where c_d is the token-aligned donor (the counterfactual item).
- *Secondary, for single-state readability:* Δ = h − h̄_role, where h̄_role is the mean of the same role over other items.

**The six quantities, per cell and functional** (all vocabulary-wide unless noted):

| Quantity | Definition | Cost |
|---|---|---|
| **P** | f(h) − f(h − Δ): a single-cell donor patch | 1 forward from ℓ (clean run cached) |
| **A** | ⟨∇f(h), Δ⟩ | 1 backward per functional per item covers every cell; spot-check with finite differences |
| **R_lens** | ⟨J̄_ℓᵀu, Δ⟩ = U·(J̄_ℓΔ) | 1 mat-vec |
| **T̄_H** | E_k [φ_k(x_k + εΔ at p_k) − φ_k(x_k − εΔ at p_k)] / 2ε, with window [p_k, p_k + H_c] | 2 forwards per host |
| **T̄_full** | the same, but perturbing *every* valid host position at once, summing y over all valid targets, and dividing by N_p | 2 forwards per host |
| **S̄(α)** | E_k [φ_k(x_k + αΔ) − φ_k(x_k)] / α for α ∈ {0.5, 1, 2}, plus −1 for the odd/even split | 4 forwards per host |

Notes on the host quantities:
- **H_c** = end − t is this cell's actual horizon in the item.
- **T̄_H** is horizon-matched: each host k gets one position p_k chosen so that p_k + H_c ≤ 126.
- **T̄_full** needs only one forward pair per host because a simultaneous perturbation is linear to first order. It therefore reproduces J̄'s estimand exactly in expectation, and the lens is never needed as a matrix. Call this the **matrix-free lens.**
- **The odd/even split at α = 1:**
  - S̄_odd = ½E[φ(x+Δ) − φ(x−Δ)] = T̄ + O(Δ³);
  - S̄_even = ½E[φ(x+Δ) + φ(x−Δ) − 2φ(x)] ≈ ½E[∇²φ][Δ,Δ].
  
  S̄_even is a curvature readout: it shows which tokens respond quadratically to the content.

**Gaps**, all computed per cell:

| Gap | Definition | Meaning |
|---|---|---|
| G_ctx | T̄_H − A | would a typical context carry this forward, when this one does not? |
| G_curv | A − P | finite effect ≠ first-order effect, in context |
| G_hcurv | T̄_H − S̄(1) | nonlinear amplification in generic contexts (α-profile) |
| G_hz | T̄_full − T̄_H | the part of the lens reading that forecasts beyond this sequence's end |
| O | ⟨J̄ᵀu, h̄_corpus⟩ | frequency prior (single-state only) |

**Hosts.** Use 32 pile-10k documents that are not in J_CB's fit set (take documents 100–131), truncated to 128 tokens, with positions in [4, 126]. Cache each host's block-ℓ outputs once. Every host forward then starts at ℓ and is batched.

## I.2 What each gap would mean

- **G_ctx large:** in generic text this content would push w's future logits, but this context does not carry it forward. Either attention goes elsewhere, or another copy makes this one redundant. This is the "J marks format, routing decides use" reading.
- **G_curv large:** the content *is* used, but nonlinearly (a threshold or saturation). First-order tools are then wrong at this site: AtP, the attribution lens, and P2's gradient covectors. This is exactly AtP's known failure mode.
- **G_hcurv large, i.e. S̄ changes with α:** the lens's linearization misses nonlinear amplification. That would be consistent with swaps needing α ≈ 2, and it would make the secant lens a candidate reader in its own right (II.4).
- **G_hz large at late positions:** "readable" at the question turn partly means "will be said after the sequence ends". No in-context consumer can use that part. This is a confound for every readability-vs-leverage comparison at late positions, RHUT's included, so it is worth knowing about even if nothing else here pans out.
- **O large:** the lens rank reflects the frequency prior, not content.

**Two extras that come free from the same forwards.**

1. **The P-vs-ψ 2×2.** P^{int} asks whether this position pushes the intermediate's future logits in this context (is it read at all?). P^ψ asks whether it moves the answer. Crossing them per cell:

| | P^ψ high | P^ψ low |
|---|---|---|
| **P^{int} high** | used | **read, but this consumer ignores it** (consumer-specific) |
| **P^{int} low** | used through non-J content | **not routed at all** |

   The two off-diagonal cells are different stories that RHUT currently cannot tell apart. On the sum organism, having report and parity consumers gives a direct consumer-specific test.

2. **The leverage map.** P over the whole vocabulary gives tokens with high leverage but a low lens rank ("used, not readable") at every cell. Aggregated by position class (content, delimiter, template), this *is* P9 (delimiter write slots), at no extra cost.

## I.3 Stages

### Stage 0 — plumbing (half a day, one item). Do not proceed until all five pass.

1. **The realized-write gate** passes for single-cell donor patches in fp32. It should, since these are whole-residual writes; check anyway.
2. **The matrix-free lens reproduces J_CB.**
   - Run T̄_full on J_CB's *own* 25 fit documents, if you can recover which ones they were ("docs_consumed 25"; probably the first 25 of pile-10k after filtering). Then cos(U·T̄_full, U·J̄Δ) across the vocabulary should be ≥ 0.999, and the magnitudes should agree to within 1%.
   - On fresh hosts, check agreement within host-sampling noise instead (report the SE across hosts).

   **This is the single most important check.** It validates the target layer, the fold, the positions and the aggregation in one go. If it fails, every gap below includes a definitional mismatch.
3. **ε for the finite differences.** Richardson test: ε and ε/2 must agree to within 1% (centered differences, fp32). Pick ε so that ‖εΔ‖ is about 10⁻³·‖x‖, then adjust as the test requires.
4. **Completeness.** A path integral of in-context tangents along h − Δ → h, with 8 midpoints, must equal P to within 5%. This checks the tangent machinery against an exact quantity.
5. **Reproduction.** Patching the donor at *all* positions from the clue onward must reproduce RHUT's full-residual number for that item.

### Stage 1 — in-context only (1 day)

- **Scope:** 12 two-hop items (the released set).
- **Roles:** source (clue span, last token); carrier (3 positions); question turn (3 positions, including the scoring position).
- **Layers:** ℓ ∈ {40, 48, 56}.
- **Functionals:** φ^{int} contrast, φ^{ans} contrast, ψ.
- **Compute:** P, A, R_lens (value and rank), plus vocabulary-wide P.
- **Outputs:**
  - G_curv by role;
  - the P^{int} × P^ψ 2×2;
  - the rank correlation between R_lens and P across the vocabulary, per cell.
- **Gate to Stage 2:** R_lens and P disagree materially in some role. Given RHUT, carriers should. If P ≈ R_lens everywhere and only P^ψ is small, skip Stage 2 and go straight to the consumer question (P2). The non-use would then be consumer-specific rather than a routing failure.

### Stage 2 — the host quantities (1 day)

Add T̄_H, T̄_full, S̄(α) and the odd/even split. This completes the exact decomposition. Everything is forward-only and vocabulary-wide.

**Cost:** about 130 host forwards per cell from ℓ on 128-token hosts. Batched, 250 cells is under an hour.

### Stage 3 — redundancy and absent source (half a day)

- **Joint vs single ablation.** Compute P_joint (source and carrier patched together) and compare it with P_source + P_carrier, for both ψ and φ^{int}.
  - **Superadditive** means OR-like backup: the carrier is readable and unused because the source suffices.
  - **Additive with P_carrier ≈ 0** means the carrier is not used, and the consumer works from the source.
- **Absent source.** Repeat on RHUT's absent-source variants (hidden operands, unanswerable clue).

### Stage 4 — scale (2–3 days)

- All 35 admitted two-hop items.
- The sum organism, with report and parity as the two ψ consumers.
- Brew from WorkspaceBench: 100 items, token-aligned donors, the `regions` field gives roles, and J reads it at .86.
- A dense replication on Qwen3-32B (J_NP, n = 80).
- **Optional:** a chat-host arm, where the hosts are generic assistant transcripts instead of pile text. It splits G_ctx into "distribution shift" versus "this specific context".

## I.4 Pseudocode (the core loop)

```python
# item c, donor c_d, layer ℓ, role positions T; U = folded unembedding [V, d]
Hc  = block_outputs(c,   upto=ℓ)          # fp32 from block 35
Hd  = block_outputs(c_d, upto=ℓ)
Y0  = y62(run_from(ℓ, Hc))                 # [seq, d]
psi0 = psi(run_from(ℓ, Hc))
g_int, g_psi = grads_wrt_block_out(c, ℓ, functionals=[phi_int_contrast, psi])  # 1 backward each

for t in T:
    D   = Hc[t] - Hd[t]
    Hp  = Hc.clone(); Hp[t] = Hd[t]
    out = run_from(ℓ, Hp)
    P_vocab = U @ (Y0[t:] - y62(out)[t:]).sum(0)      # every token at once
    P_psi   = psi0 - psi(out)
    A_int   = g_int[t] @ D;  A_psi = g_psi[t] @ D
    R_lens  = U @ (J_CB[ℓ] @ D)
    Hc_t    = seq_len(c) - t                          # this cell's horizon
    # hosts (batched over k): horizon-matched tangent and secants at p_k
    for k in hosts:
        p = sample_pos(k, max=126 - Hc_t)
        for s in (+eps, -eps, 0.5, 1, 2, -1):          # finite-difference pair plus secant doses
            Yk[s] = window_sum(run_from_host(k, ℓ, add=s*D, at=p), p, p + Hc_t)
        T_H  += U @ (Yk[+eps] - Yk[-eps]) / (2*eps) / K
        S[a] += U @ (Yk[a] - Yk0) / a / K              # for a in (0.5, 1, 2)
        ...
    # matrix-free lens (perturb all valid positions, full target sum) -> T_full
```

## I.5 Registered predictions (mine, written before any run) and what each outcome changes

| Role | Prediction |
|---|---|
| Source | T̄ ≈ S̄ ≈ A ≈ P; all gaps under 20% of P |
| Carrier, source visible | G_ctx dominant; P^{int} and P^ψ both small ("not routed"); joint ablation superadditive for ψ |
| Question turn | G_hz large (short horizon); G_curv moderate |
| Carrier, source hidden | G_ctx shrinks; P^ψ rises |

What each outcome changes:
- **G_ctx dominates the readable-unused cells:** routing account. The leverage map becomes a standard companion display to any lens readout. P2 should be framed as "which consumers route from where".
- **G_curv dominates:** do not use first-order attribution (AtP, U3, gradient covectors) at those sites. P2 needs finite transfer matrices.
- **G_hz dominates at the question turn:** amend RHUT's late-position readability numbers to horizon-matched readouts. This finding would matter beyond this project.
- **α-profile not flat:** evaluate the secant lens as a reader (II.4).
- **The "read, but not by this consumer" cell is populated:** that is the cleanest evidence yet for an interface with consumer-specific uptake. It feeds directly into the workspace branch.

**Stats.** The unit is the item. Use cluster bootstraps over items (2,000 resamples), break results down by relation type, and report medians alongside means because gaps are heavy-tailed. Register the roles, layers and windows before running Stage 1.

**Total cost.** About 4–5 days to finish Stage 4. Stages 0–2 fit in about 2.5 days.

---

# Part II — Other upgrades, ranked

## II.1 Write directions: filter vs pattern vs Fisher (the swap upgrade). Prior 50%.

**The problem.**
- J rows detect but often are not causal: the "gcd" row is readable but its J vector is not causal, while the class centroid is.
- Constituent-J writes flipped nothing in 004.
- G&W's held states swap at 58–64%.

**The math (Haufe et al. 2014).**
- A covector g defines a readout r = gᵀh. That makes g a *filter*.
- The activation direction that actually co-varies with r is the *pattern*: a = Cov(h, r)/Var(r) = Σg / (gᵀΣg).
- When Σ is anisotropic, the two differ. Writing along the filter pushes energy into directions the model rarely varies along. Writing along the pattern stays on the activation manifold.
- Chen's ridge sweep is this in disguise: (Σ + ρI)⁻¹Δμ runs from filter (small ρ) to pattern (large ρ), and causal success rose from 25% to 40% along it.
- **Caveat:** Σg also carries whatever co-varies with the concept in the data. That is answer smuggling in another form, so the best point may sit in between, as Chen's optimum did.

**Candidate directions.** For a contrast w → w′, with g = J̄ᵀ(u_{w′} − u_w):

1. **The filter itself:** g.
2. **Pattern family:** Σ̂^β g for β ∈ {0.25, 0.5, 1}.
   - Σ̂ is a Ledoit–Wolf covariance of h_ℓ at matched position types.
   - Handle the massive-activation dimensions (exclude or clip them), and check that Σ̂^β g is not dominated by them.
3. **Ridge path:** (J̄ᵀJ̄ + λI)⁻¹J̄ᵀ(u_{w′} − u_w), with λ at singular-value quantiles.
   - As λ → ∞ this tends to g (maximum target change per unit norm).
   - As λ → 0 it tends to J̄⁺u, the minimum-norm write of exactly that output change.
4. **Fisher-budgeted combination.** Maximizing gᵀδ subject to a damage budget ½δᵀF̄δ ≤ b gives δ ∝ F̄⁻¹g. Restricted to a candidate span S = [the directions above, held-state, template], this is δ* = S(SᵀF̄S + λI)⁻¹Sᵀg.
   - You can get SᵀF̄S forward-only by polarization of small-dose KLs on pile text, because KL(v) ≈ ½vᵀF̄v:

     ```
     (SᵀF̄S)_ij = KL(s_i + s_j) − KL(s_i) − KL(s_j)
     ```

   - For ten candidates that is 55 KL measurements.
5. **Empirical pattern ceiling:** G&W held-state vectors.
6. **Template baseline:** Chen's ρ = 1 template.
7. **Controls:** a norm-matched random direction, and the answer-row control (write the answer's own contrast row).

**Protocol.**
- **Operator:** add-only, α·d̂, with the same operator for every candidate.
- **Doses:** chosen to span damage levels.
- **Damage:** pile KL under the same write.
- **Primary metric:** effect at matched damage (0.01, 0.03 and 0.1 nats), interpolated from each dose curve.
- **Endpoint:** sequence margin; flips are secondary.
- **Harness:** fp32, with the realized-write gate. Apply the O4 fold fix first.

**Items, in order.**
1. The ekko M2 set (40 single-token items).
2. RHUT two-hop (35 items).
3. The 004 multi-token set (115 items), with the constituent-J contrast and a band sweep over {L36–47, L48–59, L24–59}.

**Diagnostic.** Compute the dissociation angle θ = ∠(g, Σ̂g) per contrast. The prediction is that the filter write's deficit relative to the pattern write grows with θ.

**Decision.**
- **Pattern, ridge or Fisher beats the filter at matched damage (CI-clear):** J's write failure is a metric problem. Adopt the winner as J's write rule, and rerun 004 with it.
- **Held-state beats every J-derived direction by a wide margin:** J is a reader, not a writer. Use held states for swaps.

**Cost.** About a day on M2; 2 days including 004.

## II.2 Check the premise: is the Oracle Lens swappable? (half a day)

The Oracle Lens ships a reconstructor that maps a phrase to an activation (a template-style vector). The 09-24 memo says the release (`agu18dec/local-workspace`) includes it; verify that, along with the reconstructor's layer and output convention (whitened or raw, centred or not).

**Test, on brew:**
1. Take the Oracle readout at the read cell.
2. Reconstruct vectors for the clean colour phrase and the donor colour phrase.
3. Swap: h − a·r_clean + a·r_donor, with a set by NNLS, as the Oracle's own decode does.
4. Score with the sequence margin toward the donor's answer.
5. Compare against a J pinv swap, a held-state swap, the full-residual ceiling and a random direction.

**Decision.**
- **Oracle swaps ≥ J swaps:** J's intervenability advantage is moot. The multi-token program should use the Oracle's reconstructor as its writer.
- **Oracle swaps ≪ J swaps:** this is the concrete argument for building a J-based reader.

## II.3 One refit, many uses (a few GPU hours)

Refit J at n = 25 with J_CB's settings (target 62, t_max 128, skip_first 4, pile-10k) on 16 layers, with per-prompt saving turned on. That is about 21 GB in bf16. The one refit enables four things:
- II.4's context-agreement lens;
- ŝ and the name-free directions in the workspace branch (B1);
- a prompt-jackknife noise floor, which fixes ekko F8's single-draw floor;
- per-hit rank confidence intervals.

## II.4 Two new readers to score (a day each after II.3 and Part I)

### (a) The context-agreement lens: the lens's own error bars. Prior 35%.

For a cell h, compute per-context scores s_{c,w} = u_wᵀ(J_c h). That is n mat-vecs followed by one projection onto U. Then:
- m_w = mean_c s_{c,w}, which is the ordinary lens score;
- se_w = sd_c(s_{c,w}) / √n.

Rank tokens three ways:
1. by t_w = m_w / se_w;
2. by per-token positive-part James–Stein, m_w·max(0, 1 − se_w²/m_w²);
3. by the median over contexts.

**Why it could help.** Heavy-tailed per-prompt Jacobians are the reason the paper drops high-Frobenius prompts. Tokens with a high mean but low agreement are the few-contexts-dominate cases, and these may be the trash tokens.

**Why it could hurt.** It favours context-robust tokens by design, so it may drop legitimately context-specific content.

**Evaluation:** a dev/test split on the ekko 59-item slice plus WorkspaceBench's single-token families, per layer. Early layers are where I would expect any gain.

**How it differs from R2.** R2 shrinks the whole matrix toward the identity to correct *estimation* noise. That noise is tiny at n = 1000, so R2 cannot matter much. This lens shrinks per token and per cell, using the *context disagreement* that is actually large.

### (b) The secant lens: S̄ at α = 1 as a reader. Prior 30%.

This comes from Part I's host machinery:

```
S̄(h) = E_k[φ_k(x_k + α(h − h̄)) − φ_k(x_k)] / α  =  J̄-lens(h − h̄) + ½·E[curvature] + …
```

It costs 2K forwards per cell and reads the whole vocabulary.

- Run it only if Part I finds a non-flat α-profile.
- Compare it against the *centred* J-lens, because centring changes pass@10 by itself (.479 → .359 in ekko 008). Add back an offset term, with its weight chosen on a dev split.
- Use α < 1 as the default. Adding a whole deviation to a host state is a large, off-manifold perturbation.

## II.5 Custom estimands without refitting: the matrix-free lens

Part I's T̄_full generalizes. For any cell, J̄h costs 2K forwards at host contexts: perturb along h and difference the target sums. That gives the whole vocabulary at once. Changing the estimand then means changing the hosts or the target window; nothing gets refit. Two cheap kill tests follow directly:

- **Lag buckets, for poetry.**
  - Perturb a single host position and sum targets over [p + a, p + b], for (a, b) ∈ {(0, 0), (1, 3), (4, 8), (9, 16)}.
  - Read at the newline of the WorkspaceBench poetry items. The rhyme word sits 8–12 tokens ahead (median 10).
  - *Registered:* bucket 9–16 ranks the rhyme word well above J̄ (J scores .07 on this family).
  - *Kill:* no bucket improves the rhyme word's rank.
- **The question-indexed / report lens (J^Q).**
  - The hosts are contexts followed by a fixed question. Perturb all context positions and read only the answer slot.
  - *Kill:* J^Q rows have cosine > .95 with J̄ rows for most tokens, or the readout is dominated by generic "thinking/AI" tokens even after subtracting the mean over contexts.
  - If it survives, test it on the directed-modulation and user-modeling families and on RHUT carriers.

**Precision.** Centred finite differences in fp32, with the Richardson check from Stage 0. Forward-mode AD will probably fail on the GDN kernels.

## II.6 Multi-token readout: a labelling problem (cheap, scored for free)

Constituent J already reaches the probe ceiling on 4 of 5 families (004), so the content is in the bag of tokens. What's missing is assembling fragments into words.

- **MTP-lens positive control** (hours).
  - At the last layer on teacher-forced text, check that the MTP module predicts x_{t+2} from the true final hidden state plus emb(x_{t+1}).
  - Only then substitute J̄h for the final hidden state, run on the real prefix (the module attends over the sequence), and beam-search to word ends.
- **Baselines:** a tokenizer trie; CJK single-token rows.
- **Summarizer variants:** one that may abstain; self-summarization by Qwen of its own bag.
- **Scoring:** the six multi-token families are regex-scored, so there are no judge costs. Use a dev/test split.

## II.7 Ceilings before building (hours to a day)

Per family and read cell, on brew and chain:
- **C1:** a linear probe, split by template, with an input-only floor and learning curves.
- **C2:** DAS at increasing rank, with the rowspace/nullspace check.

Together they set the headroom for any J++ reader.
- **Probe ≈ J:** nothing to gain at that cell.
- **Probe ≫ DAS:** the content is decodable at that cell but not causally sufficient there. A reader could score on the benchmark without reading what drives the answer.

## Where each upgrade is measured

WorkspaceBench is the readout leg only. It has no causal mode, and at about 100 items per family its noise is roughly ±0.1–0.15 per family.

| Upgrade | Measured on | Primary metric | Bench role |
|---|---|---|---|
| Part I secant decomposition | RHUT two-hop and sum organisms; bench brew items (Stage 4) | Gaps per role | Not a reader score; brew items only |
| II.1 write directions | ekko M2 set, then RHUT two-hop, then 004 items; then bench brew / chain / relational through a causal scoring mode you build (M1-C) | Effect at matched damage | Items only; the bench has no causal leg |
| II.2 Oracle swappability | Bench brew, through M1-C | Margin share vs a J swap | Items only |
| II.4a context-agreement lens | Dev: ekko 59-item slice. Test: bench single-token families (basic, multilingual, multihop, brew), per layer | Gold concept's rank in a closed candidate set, no judge. Bench pass rate secondary | Held-out test |
| II.4b secant lens | Same as II.4a, only if the α-profile is non-flat | Same | Held-out test |
| II.5 lag buckets | Bench poetry family | Rank of the rhyme word at the newline | Primary |
| II.5 J^Q | Pretest cosines; then bench user modeling, jailbreak, and directed modulation (its 10 single-token concepts or your own set); RHUT carriers | Pretest cosine, then pass rate | Primary after the pretest. The DM family is confounded: 55 of 65 concepts are multi-token, and it pools think with don't-think |
| II.6 labelling | Bench's six multi-token families (regex-scored, free) | Pass rate on the test split | Primary |
| II.7 ceilings | Bench brew, chain, arithmetic and relational items | Probe / DAS accuracy vs J | Sets the headroom on the bench itself |

Rules for every bench comparison:
- **Reproduce the bench's own J numbers first.** Brew should come out near .86 and basic multi-token near .24.
- **Use matched n.** J_CB against R_CB (both n = 25), or the II.3 refit, never the n = 1000 J arm.
- **Report per layer, at matched cells.** Do not use the any-cell rule.
- **Compare readers paired on the same items.** Use McNemar on pass/fail, or a paired bootstrap on ranks. Two unpaired pass rates 0.1 apart are within noise.
- **Tune on a dev split.** Split by family or by item, tune there, report on the rest.
- **Report the increment over prompt-only** for every family.
- **Add a summarizer variant that may abstain** for the token-bag families. Part of a token lens's hallucination rate belongs to the summarizer prompt, not the lens.

## Evaluation standards for everything in Part II

- A dev/test split, whether by family or by item.
- Matched n across lenses.
- Per-layer reporting, not the any-cell rule.
- The twin-fit or jackknife noise floor.
- The increment over the prompt-only baseline.
- For writes: matched damage, a norm-matched random control, an answer-row control, and the realized-write gate.

---

# Part III — Not doing, and why

| Idea | Reason |
|---|---|
| Fitted secant, anchored or low-rank transports; H-lens; smoothed operating point | The theorem, plus ekko F1–F13 |
| R2 noise-calibrated shrinkage | Corrects estimation noise, which is tiny at n = 1000; it cannot fix context variation |
| γ-lens | At most an hour, as one line in 012 |
| Phrase-J and its variants | Closed by 004 and Chen |
| Typed sparse dictionary (N2), as written | Phrase atoms sit almost in the span of their constituent rows, so NNLS picks between them arbitrarily. Needs a hierarchical or MDL selection rule first |
| Fair tournament (idea 12) | Only if a multi-token lens becomes a dependency |
| Causal NLA; trained verbalizers | Compute cost, plus the confabulation evidence |
| E6 calibrated-z as a default | It cut pass@10 across the board; revisit only through II.4 with κ tuned on dev |

**Order:** Part I Stages 0–1 → II.1 and II.2 in parallel → Part I Stage 2 → II.3 → II.4–II.5 → the rest.
