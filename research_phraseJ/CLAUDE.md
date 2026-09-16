# research_phraseJ — working agreement

Branch `phrase_J` of `senku14x/jacobian-lens`, a clean branch off upstream `main` (Anthropic's
reference `jlens`). Everything we add lives under `research_phraseJ/`. `jlens/` itself is the
measurement instrument: changes there are minimal, isolated, and logged here.

`ekko-lens` is a separate earlier project on transport operators. It is never merged. It may be read,
and small pieces may be copied into `scripts/lib/` with an attribution header (see below).

## Goal

Multi-token extensions of the Jacobian lens on **Qwen3.6-27B only**, compared head-to-head against
the released template lens for the same model (`camilablank/workspace-lenses`). If a method
outperforms the template lens on readout and swap for multi-token intermediates, we scale to other
models; not before.

## Layout

```
research_phraseJ/
  CLAUDE.md              this file
  STATUS.md              current state: what is decided, running, next. Mutable.
  docs/                  notes and learnings, numbered in reading order
  scripts/               runnable code: NNN_slug.py; shared code in scripts/lib/
  plots/NNN-slug/        figures for experiment NNN
  results/NNN-slug/      analysis.md + the raw json/pt it was computed from
  research_artifacts/NNN-slug/
                         design.md  — written BEFORE the run, agreed with Vishesh
                         report.md  — written AFTER the run
  outputs/               gitignored: lenses, caches, anything large or regenerable
```

One experiment number `NNN` means the same thing in `scripts/`, `plots/`, `results/`, and
`research_artifacts/`. Numbers are permanent; a superseded analysis is annotated, never renumbered.

## Design before run (non-negotiable)

No experiment code is written until `research_artifacts/NNN-slug/design.md` exists and Vishesh has
agreed to it. The design states:

1. **Question** — one sentence.
2. **Why now** — what decision it unblocks.
3. **What we run** — model, lens, data, positions, layers, n, metrics, baselines, controls.
4. **What we will learn** — for each possible outcome, what it changes about the plan.
5. **Cost** — GPU-hours and wall-clock.
6. **Kill criterion** — the observation that closes the idea.
7. **Pre-registered decision rule** where the experiment is meant to validate a claim.

## Evidence discipline

- Claims carry their evidence level: observation / recurring pattern / supported claim / causal
  claim / interpretation / speculation. Scope every result: model, prompts, layers, positions, n.
- Baselines are not optional: released J, released R, logit lens (`use_jacobian=False`), released
  template lens, matched-norm random directions. Same tuning effort for baselines as for the method.
- A null needs a positive control. A margin needs a noise floor (twin-fit or bootstrap).
- Read the data: random examples alongside chosen ones; per-item distributions, not only means.
- Post-hoc explanations are labelled as such and need held-out confirmation.
- A readout margin without an item-admission criterion and a benign base rate is an observation,
  not a finding.

## Environment (verified 2026-09-16)

| | |
|---|---|
| GPU | 1× NVIDIA A100-SXM4 80 GB |
| Python / torch / transformers | 3.13.15 / 2.11.0+cu128 / 5.16.1 |
| model | `/content/models/Qwen3.6-27B` (bf16 load: 53.8 GB, 24 s; arch `Qwen3_5ForCausalLM`, 64 layers, d_model 5120; hybrid GatedDeltaNet: 48 linear-attention + 16 full-attention blocks) |
| released lenses | `/content/lenses/qwen3.6-27b/{j-lens,r-lens}/lens.pt`; `template-lens/{templates,templates+phrases_v3}.safetensors` + passages |
| install | `pip install -e .` from repo root; `python -m pytest tests -q` → 32 passed |
| smoke test | released J on "…animal that spins webs is": `spider` rank 4 @L40, rank 8 @L46; motor by L55 |

## jlens API facts that bite (verified against source)

- `jlens.from_hf` **mutates** the HF model: `eval()` + `requires_grad_(False)` on all parameters.
  Fine for activation-gradients (phrase-J), wrong for anything needing parameter grads.
- `force_bos=True` by default sets `tokenizer.add_bos_token = True`. Any external pipeline that did
  not do this is token-level confounded.
- Reference `fit` uses `SKIP_FIRST_N_POSITIONS = 16`; the **released lenses used `skip_first = 4`**,
  `target_layer = 62` (penultimate), `n = 25` prompts of 128 tokens from `NeelNanda/pile-10k`.
- The estimator sums cotangents over all targets `t' ≥ t` and averages over sources `t`. A single
  insertion point gives a lag-specific row, not J̄. Phrase-J positive controls must match lags.
- `apply(use_jacobian=False)` is the logit-lens baseline. One flag; always report it.
- Released `lens.pt` keys: `J` (stacked per-layer, target row = I), `n_prompts`, `source_layers`,
  `d_model`, `provenance`. Saved fp16.
- **Qwen3_5 RMSNorm is `(1 + w)`.** The ranking-exact fold of a vocabulary direction into residual
  space is `Jᵀ((1+γ)⊙W_U[t])`, not `γ⊙W_U[t]` (ekko-lens F10.1 found this post hoc).
- The final norm divides by `‖J h‖`; the effective read direction is position-dependent while the
  paper's intervention vector ignores it. For swaps use `Jᵀ(I − ĥĥᵀ)(1+γ)⊙W_U[t]` or verify the
  mismatch is negligible.
- Template rows are stored **pre-whitened** (`space = raw`, `ridge_c = 0.001`): dot them with raw
  residuals. The whitening covariance Σ is not shipped.

## Borrowed code (`scripts/lib/`)

| file | from | what |
|---|---|---|
| `ekko_harness.py` | ekko-lens `research/ekko/harness.py` @89ae076 | `load_model`, `load_released_lens`, `capture`, exact `forward_from(h, layer)` replay (0 error verified there), `lens_ranks`, `lens_topk`, `single_token_id` |
| `ekko_rules.py` | ekko-lens `research/ekko/rules.py` @89ae076 | LRP stop-gradient surrogates for Qwen3_5 blocks (R-lens arm only) |
| `ekko_002_calib_passk_reference.py` | ekko-lens `002_calib_passk.py` | reference for the 59-item calibration-slice carve (seed 0, 10/category) and pass@10 |
| `ekko_E2_eval_equal_reference.py` | ekko-lens `E2_eval_equal.py` | reference for the matched-grid evaluator and the M4 skip-ahead guardrail |

Reference files are not imported; they are ported into numbered scripts when needed and the port is
noted in the design doc.

## Data discipline

- `data/evaluations/` frozen sets are touched only for final claims. Development uses the 59-item
  calibration slice (seed 0, 10 items per category), same carve as ekko-lens, so the two projects'
  numbers are comparable.
- Multi-token item sets are built by us, tokenizer-verified for Qwen3.6-27B, entity-disjoint from
  any fit or passage data, and released under `results/`.

## Commits and pushes

- Author **Vishesh Gupta <visheshguptaw14x@gmail.com>**. Trailer
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` is allowed.
- Commit messages name the experiment number and what changed: `001: design doc`, `001: run + report`.
- Push to `origin phrase_J`. Never to upstream.
- No secrets, tokens, or `.env` in commits. Large outputs stay in `outputs/` (gitignored).
- Every session that runs anything updates `STATUS.md` and the relevant `research_artifacts/NNN`.
