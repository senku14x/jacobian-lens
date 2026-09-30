# Repo notes — index and cross-repo synthesis

Written 2026-09-29 from reading four repositories that precede today's work. One file per repo, plus this
synthesis. Companions: `../workspace_paper_notes.md` (the paper), `../workspace_followups_notes.md` (LW posts),
`../wsb_subsets/` (WorkspaceBench screening subsets).

Tags: **[REPO]** = what the repo reports. **[MINE]** = my reading or assessment.

| file | repo @ commit (last commit date) | local clone | one line |
|---|---|---|---|
| [ekko_lens.md](ekko_lens.md) | senku14x/jacobian-lens `ekko-lens` @ 89ae076 (2026-08-14) | `/workspace/ekko-lens` | Which fixed linear transport operator is best? None beats R̄ beyond noise; context dependence is the one big effect |
| [phrase_J.md](phrase_J.md) | senku14x/jacobian-lens `phrase_J` @ 87af681 (2026-09-17) | `/workspace/repos/phrase_J` | Phrase-level J gradients: closed as a general method; constituent-token J reads and writes phrases as well or better, except "San"-type concepts |
| [phrase_lens.md](phrase_lens.md) | chen-jan/phrase-lens @ ea85583 (2026-09-05) | `/workspace/repos/phrase-lens` | Portable methods + frozen readout AND causal (swap/ablation/broadcast) benchmark banks for ST and MT concepts; no results shipped |
| [rhut.md](rhut.md) | senku14x/Readable-Here-Used-There `extension` @ d4197b0 (2026-09-24) | `/workspace/repos/rhut` | Where J-readable content is vs where it has causal leverage: they come apart; a large route survives outside the tested J span |

The reading depth differed by repo. ekko-lens: the full ledger, all reports and the log. phrase_J: README, STATUS, the
decision log, the ideas review, and reports 001 / 003a-controls / 004. phrase-lens: all of its docs. RHUT: the
README, STATUS, the H1 stopping point, and the full evidence ledger. I did **not** read RHUT's ~40 per-stage reports or
its 871-line write-up line by line, so cite RHUT numbers from its ledger and STATUS, not from these notes, when it matters.

---

## Cross-repo synthesis [MINE]

All four repos use **Qwen3.6-27B** (64 layers, d 5120, hybrid GatedDeltaNet), the released J/R lenses from
`camilablank/workspace-lenses` (plus Neuronpedia's J in RHUT), and the released-lens convention (target 62, skip_first 4,
pile-10k, n=25).

### 1. What is settled (do not re-run without new structure)

- **Fixed d×d transport is exhausted at the R-lens level** (ekko F1–F8, F13). This covers secant, anchored, low-rank
  conditional, spectral, fusion and H-lens. The only fitted operator that generalises is the 3-scalar blend aJ̄+bR̄+cI.
  The twin-fit noise floor is ≈0.01 cos, so R vs J effect-prediction margins sit at that floor.
- **Generic phrase-J is closed** (phrase_J 003a/004):
  - Late-layer (L52–62) multi-token identity is compositional in constituent-token J coordinates, for both reading and
    writing. Matched-norm and matched-damage swaps favour J-const over phrase-J by 0.3–0.6 nats.
  - One exception survives: "San" (San Francisco / Diego / Antonio / Jose). There phrase information lies outside the
    constituent span, and only the phrase gradient reads and writes it (n=11 items, one route).
- **The early probe-over-J gap is generic early-J, not phrase-specific** (phrase_J 003a-controls: a +0.24 AUC gap at L8
  for single tokens too). This agrees with ekko's F3/F15: the input-specific Jacobian beats the average 1.8–4×, worst
  early; same-token J rows across prompts have cos 0.03 at L8 → 0.95 at L60.
- **Readability ≠ leverage** (RHUT H3):
  - Where an intermediate is J-readable (the carrier) and where it moves the answer (source span, question turn) come
    apart. A carrier swap installs the intermediate at the readout as strongly as the full swap, but moves the answer
    only 0.07 of the ceiling.
  - Removing the best 25–64 J directions at the question turn keeps 0.61 of the effect, vs 0.95–0.96 for random
    directions. So the J span is causally privileged, but it is not the dominant bottleneck.
  - ~0.41 survives with the tested J-derived span clamped at every answer position (a late route via the last layer's
    read of the question-turn template tokens).

### 2. Traps these repos already paid for (instrument hygiene)

| trap | where learned | rule |
|---|---|---|
| Qwen3.5/3.6 RMSNorm multiplies by `(1+w)`; folds must use `(1+γ)⊙W_U[t]` | ekko F10.1, RHUT §C, phrase_J 002 | verify on the loaded module before any fold |
| bf16 coordinate writes are rounding-limited: realized clamp ρ≈0.55, κ≈0.45 for small clamps | RHUT (all CoordClamp/CoordSwap rows) | hook in float32 for coordinate interventions; report realized ρ/κ |
| bf16 backward depends on batch shape early (relerr 0.41 @L8, gone by ~L44) | phrase_J 002 | fix graph shape when comparing gradients; don't read early-layer gradient differences naively |
| bf16 central finite difference at ε=0.01 has cos 0.93 to the true JVP | ekko F5 | use forward-mode JVP or Richardson; fp32 |
| SDPA has no forward-mode derivative (NaN) | ekko F7 | force eager attention for JVPs |
| Split leakage: (a→b)/(b→a) reversed pairs, same template across splits | ekko F1 (B7) | template/category-disjoint splits |
| Frame-matched latent designs: random k-fold CV invalid | phrase_J 001 | grouped leave-one-cue/frame/route-out |
| Bootstrap over eval prompts can't see fit variance | ekko F8 | twin-fit (disjoint half-corpus) noise floor for any lens-vs-lens margin |
| Released template passages leak the phrase's first word in 12–16% | phrase_J 001 | filter |
| Qwen3.6 emits `<think>` after `Q: … A:` frames | phrase_J 001 | avoid that frame / disable thinking |
| Coordinate swaps along answer-naming directions = answer steering | RHUT (sum organism; bridgeswap: answer swap 0.54 > intermediate 0.41) | intermediate ≠ answer; check the target set (phrase-lens MT multihop targets include answers) |
| The jlens fit silently drops prompts ≤17 tokens | ekko CLAUDE.md | check `lens.n_prompts` |
| The WorkspaceBench producer sends byte-level BPE mojibake to the judge; its J arm is a cosine readout | this session | see memory `workspace-bench-setup` |
| An attention mask on consumer→source is not a complete cut | RHUT replication | account for source→carrier→consumer paths |

### 3. Evaluation assets that already exist (the "we lack an eval" problem is mostly solved by assembly)

| axis | asset | notes |
|---|---|---|
| readout / recall | WorkspaceBench (installed, `/workspace/workspace-bench`); ekko 59-item calibration slice (E2 harness, pass@10, calibrated-z); ekko 38-item entity-disjoint independent set `research/data/e8_independent/` (never used) | prompt-only floor problem on many WSB families |
| causal (swap / ablation / broadcast) | **phrase-lens frozen causal banks**: multihop swap+ablation 96 ST / 67 MT, broadcast 192 ST / 192 MT; Qwen3.6-27B gated; random-direction controls | MT multihop targets can include answers; 16 `gate_scrutiny` flags |
| causal (behavioural ablation) | ekko M2 (`A4_m2_ablation.py`): folded-direction ablation, Δlog-prob, matched random, pile damage | fix the `(1+γ)` fold first (O4) |
| skip-ahead / specificity | ekko M4 guardrail (built in E2) | |
| leverage vs readability | RHUT two-hop organism: 35 admitted items, fp32 regime, all gates | heavy; use when asking "is the readout causally used" |
| noise floor | ekko twin-fit (0.01 cos transport; 0.004 pass@10 first-half for H) | needed for any lens-vs-lens claim |

### 4. Open threads worth picking up (ranked by my estimate of information per hour)

1. **Sparse-frame / nonneg-cone readout** (ekko O6; paper's machinery; phrase_J ideas ledger "R", scored 7).
   Proposed in three places, never run in any repo. Zero training, input-adaptive, cheap.
2. **Evaluation scorecard assembly**: WSB 40-item panel + phrase-lens causal banks + M2 + M4 + twin floor, in one
   harness. Prerequisite for claiming any "better lens".
3. **San-type concept screen** (phrase_J next step): a CPU Δ_residual screen over many families after one capture. Is
   the "outside constituent-J" phenomenon common?
4. **Causal handle vs readout** (meta-tokens / RHUT): per-token directions that are causally calibrated. Tempered by
   phrase_J 004: J-const already writes better than phrase gradients for compositional phrases.
5. **SmoothGrad mechanism** (ekko O2): Richardson ε sweep + fp32. Decides whether a smoothed-operating-point lens branch
   exists.
6. **RHUT H2** (maintenance mechanism) and the float32 hook repair: not lens work, but the leverage question the whole
   programme depends on.

### 5. Correction to my own earlier proposal (2026-09-29)

Earlier today I proposed a "C-lens": per-concept causal directions, multi-token, gradient-defined. **phrase_J already
built and tested its first-order version and closed it as a general method** (003a, 004). The finite-displacement
variant is untested, but 004's result (constituent J is the better writer at matched norm *and* damage) lowers my
prior on it considerably. The idea survives only as a targeted tool for San-type concepts.
