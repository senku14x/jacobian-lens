# Review of the J-lens ideas ledger (batches 1–3), 2026-09-16

Independent scores were assigned before reading the ledger's; disagreements of ≥3 are marked ★. Scores are information per hour toward a better instrument for reading and intervening on workspace content on open models, on Qwen3.6-27B with the released J/R/template artifacts.

## 1. Updates from the R-lens release that affect the plan

- **Template lens for Qwen3.6-27B is public** (13,174 word rows; 13,731 with mined phrases; all 64 layers; 2.1M generated passages with the 80/20 ridge split). This replaces the planned "template replica from ~200 natural occurrences" baseline. It also lets B (gradient template 2×2) run on the exact passages, and lets SS's dissociation angle be computed for every concept at zero GPU cost.
- **Matched J/R pairs** (same forward pass, n=25, skip_first=4, penultimate target) make R a free baseline arm for P, JJ, and A.
- **R-lens result adds a third explanation of early-layer failure**: backward-pass error accumulation that LRP suppresses. Any experiment whose interpretation depends on "why do early layers fail" (P, R′, Z3) must include an LRP arm.
- II (CJK single-token phrase vocabulary) loses most of its value now that English phrase rows exist.

## 2. Corrections that change designs

1. **A positive control, lag matching.** The released estimator sums cotangents over all targets t′ ≥ t. A single insertion point yields a lag-specific row (idea M), not J̄. Average over insertion lags, or compare against a lag-matched J fit, before declaring the control failed.
2. **A estimand.** First-token term via logit at the penultimate residual → W_U (paper convention). Conditional terms (i ≥ 2) via log p(w_i | c, w_<i): the log-prob gradient subtracts Σ_j p_j ∂logit_j, i.e. the model's expected-token direction, which is the disambiguation signal. Raw logit gradients for later tokens will be dominated by the generic "say something after black" direction.
3. **A zero-cost pre-test.** From the released template lens, compute per-layer cos(t_w, t_w′) for the minimal pairs you intend to disambiguate (blackmail/blackboard, photosynthesis/photograph, etc.). If ≈1 at workspace layers, the whitened representation does not separate them and no linear lens will; skip to the "labelling problem" branch.
4. **A swap leg needs supernode handling.** v_blackmail and v_black are strongly correlated; a two-row pseudoinverse swap leaves the fragment loaded and will produce the "moves in the right direction but not far enough" pattern. X moves from 4 to a prerequisite.
5. **P must run with raw and LRP backward.** Without the R arm, a persistent early gap is ambiguous between "computational" and "gradient-path error".
6. **Fair universe for lens comparisons.** Rank within the 13,731-row template universe for all lenses (map each row to its first-token J row); report the paper's any-layer pass@k AUC, not only per-layer mean pass@10 (the R-lens post's metric rewards breadth across depth).
7. **FF word head.** Trained at word-start positions on the final residual, the head is a motor instrument by construction; for multi-token words it can only recover what the model has planned about token 2+ at that position, which the paper found to be fragment-shaped. Keep as a one-hour cheap bet, not a lead.
8. **DDD MTP head.** DeepSeek-style MTP modules condition on the t+1 token embedding through a transformer block, so W^MTP J h is not a linear composition. Check config.json for the field; if the module is nonlinear, DDD collapses into phrase-J under teacher forcing.
9. **BB/CC report and question-indexed lenses.** The answer-slot residual is a function of what it attends to; the resulting rows may be the future-only J up to a shared rotation. Include a rotation-invariant comparison (CKA of row sets against future-only J) alongside the cosine kill criterion.
10. **JJ local re-rank.** Per-prompt Jacobians are heavy-tailed (paper A.7); re-ranking 50 candidates by local rows may reintroduce trash. Add the trash-token rate to JJ's metric, and gate on M4 at the last position.
11. **7.6 norm hygiene** is correct and should be applied before V is built, since every swap number depends on it.

## 3. Scores

| # | Idea | Ledger | Mine | Note |
|---|---|---|---|---|
| A | Phrase-J | 9 | 8 | Core bet; run regime (iii) on released passages first, then (ii), then (i). Corrections 1–4 apply. |
| B | Gradient template 2×2 | 8 | 8 | Now nearly free with released passages. Run before A's multi-token pilot. |
| P | Per-prompt J readout ceiling | 8 | 8 | Add R arm (correction 5). |
| FF | Word-head lens | 8 | 5 ★ | Motor by construction; cheap, run in parallel, expect kill. |
| R | Sparse-frame readout | 7 | 7 | Mandatory once phrase rows exist. |
| V | Swap/clamp battery | 7 | 8 | Prerequisite for everything; fold WW and X in. |
| D | Gradient phrase dictionary | 7 | 6 | Only after A validates. |
| SS | Metric-split + dissociation angle | 7 | 7 | Zero cost now; run K first as its diagnostic. |
| WW | Swap-validity certificate | 7 | 8 | Fold into V. |
| AAA | Three-way aggregation | 7 | 6 | Good, but single-token; not this sprint. |
| BB | Report lens | 7 | 6 | Correction 9. |
| CC | Question-indexed lenses | 7 | 6 | Correction 9; OOO is its eval. |
| JJ | Local re-rank | 7 | 6 | Correction 10. |
| DDD | MTP head | 7* | 4 ★ | Correction 8. |
| M | Lag-resolved J | 6 | 6 | Also the right positive control for A. |
| C | Natural-bigram lens | 6 | 5 | Coverage limited to corpus bigrams. |
| G | Lemma-merged readout | 6 | 6 | Do with R. |
| L | Frozen-QK J for interventions | 6 | 6 | One fit; run when GPU idle. |
| O | Prefix-conditional directions | 6 | 5 | Attribution drift risk is real. |
| Z3 | Dense-model replication | 6 | 6 | Scoping. |
| AA2 | Reflection training on open weights | 6 | 8 ★ | Out of today's scope; highest field value on the board. |
| AA3 | Lens-equipped auditing agent | 6 | 5 | After a readout wins. |
| GG | Read-side lens | 6 | 5 | Echo detection is doable with simpler probes. |
| II | CJK phrase vocabulary | 6 | 4 | Superseded by released phrase rows. |
| QQ | Lens diffing across fine-tunes | 6 | 6 | Cheap on a 7B. |
| TT | Causal inner product | 6 | 5 | Test on multilingual only. |
| UU | Hierarchical readout | 6 | 6 | Pairs with the list-capacity result. |
| YY | Subspace swaps for circular features | 6 | 6 | Specific prediction on a known failure; good. |
| EEE | Erasure-aligned layer prediction | 6 | 5 | Validation. |
| FFF | Pre-cache vs breadcrumb | 6 | 6 | Needed to interpret M. |
| GGG | Attribution graphs with phrase nodes | 6 | 6 | Best benchmark-free validation of A/FF. |
| JJJ | Silence score | 6 | 7 | The metric the programme assumes; compute it. |
| MMM | NLA-proposed, lens-verified | 6 | 6 | Clean, cheap. |
| MM | Cross-context consistency | 6 | 5 | Fold into AAA. |
| E | Dictionary-only oracle | 5 | 5 | |
| J | Meta-token hunt | 5 | 5 | Needs U/Y5 base rates. |
| K | Stein lens | 3 | 5 ★ | Diagnostic for SS and the whitening reading. |
| S, T, U, W, N, X | | 5,5,5,5,4,4 | 4,5,6,4,4,6 | X becomes a prerequisite for phrase swaps. |
| H, I, F, Q, R′, Z1, Z2 | | 4,3,4/7,3,3,3,5 | 3,3,6,2,4,3,5 | |
| KK, LL, HHH, NN, OO, PP | | 6,5,5,5,5,4 | 5,6,4,5,5,4 | LL is good tooling. |
| DD, EE, HH | | 5,4,5 | 4,3,4 | |
| VV, XX, ZZ, BBB, CCC, III, KKK, LLL, NNN, OOO, PPP, QQQ, RRR | | 5,5,4,5,4,4,5,5,5,5,4,4,4/6 | 4,5,5,4,4,4,5,5,5,5,4,4,6 | |
| AA1, AA4, AA5 | | 4/7,5,— | 5,6,— | AA4 sharpens my earlier "sparse remainder" concern. |

## 4. Sprint 1 as I would run it

1. Load released J, R, template on Qwen3.6-27B. Minimal-pair cosines (correction 3), K (Stein cosine per token), SS dissociation angle. No GPU beyond one forward-pass corpus. One hour.
2. V + WW + X + 7.6 norm fix. One to two days of engineering; everything downstream depends on it.
3. A single-token positive control, lag-matched (correction 1), on the 50-item multihop set. Afternoon. Stop if it fails.
4. B 2×2 on ~40 words from the released passages. Small.
5. Build the multi-token set with Y1 admission; run A regimes (iii) → (ii) → (i) against the released template lens in the shared universe, with R and G applied to all readouts. Pre-registered decision rules from the ledger.
6. FF in parallel on CPU (cached final residuals, logistic regression). Expect kill; cheap enough to confirm.
7. P overnight with raw and LRP backward.
8. JJJ (silence score) on whatever readout wins, before any auditing number is reported.

## 5. Open items I could not resolve

- Whether Qwen3.6-27B ships an MTP module (check config for `num_nextn_predict_layers` or similar).
- Whether the paper's 126-item multi-token set is released; ledger says no. Build with Y1 and release.
- Whether the released template lens's whitening covariance Σ is stored or must be recomputed from the passages (the README says cosine scoring, which suggests the stored rows are already whitened directions; verify before computing SS's angle).
