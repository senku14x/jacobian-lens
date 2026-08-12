# research/ — working agreement

Operating context for research work on this fork. Read before touching anything in `research/`.

## What this is

`senku14x/jacobian-lens` is a fork of `anthropics/jacobian-lens` — the companion code for
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html).
Upstream is a **reference implementation, unmaintained, not accepting contributions**, so we do not
send changes back; `upstream` is a remote for pulling only.

Our work lives on branch **`ekko-lens`**. Everything we add goes under `research/`. Changes to
`jlens/` itself are allowed but should be minimal, isolated, and logged — the package is our
measurement instrument, and silently editing an instrument invalidates comparisons against
anything measured before the edit.

## Layout

```
research/
  CLAUDE.md          this file
  RESEARCH_LOG.md    append-only log; "Current state" block at the top is the only mutable part
  artifacts/         one directory per analysis: report.md + figures/
    _TEMPLATE.md     copy this to start a report
    README.md        index of all artifacts
  experiments/       runnable code that produced the artifacts
  outputs/           gitignored — .pt lenses, raw dumps, anything large or regenerable
```

Naming: `artifacts/NNN-YYYY-MM-DD-slug/` and `experiments/NNN_slug.py`, sharing the same `NNN`.
Sequence numbers are permanent; a superseded analysis is annotated, never renumbered or deleted.

## Environment (verified 2026-08-11)

| | |
|---|---|
| GPU | 1× NVIDIA H100 80GB HBM3 |
| Python | 3.12.3 in `/home/ubuntu/.venvs/mechinterp` (already active) |
| torch | 2.13.0+cu126, CUDA available |
| transformers | 5.15.0 (jlens needs ≥5.5) |
| also installed | nnsight 0.7.0, transformer-lens 3.7.1, datasets, matplotlib, einops |
| HF cache | `/home/ubuntu/cot-oracle/hf_home` — **not exported**; set `HF_HOME` to it or re-download |
| cached weights | `Qwen/Qwen2.5-7B-Instruct` (15G) — 28 layers, d_model 3584 |
| install | `pip install -e '.[dev]'` (done); `python -m pytest tests -q` → 32 passed |

Set `HF_HOME=/home/ubuntu/cot-oracle/hf_home` in any script that loads models, or the 15G Qwen
download repeats.

## jlens API, and the parts that will bite

```python
model = jlens.from_hf(hf_model, tok)                       # -> HFLensModel
lens  = jlens.fit(model, prompts, source_layers=[...], checkpoint_path="...")
lens_logits, model_logits, ids = lens.apply(model, prompt, layers=[...], positions=[-2])
h_final = lens.transport(residual, layer)                  # bare J_l @ h
JacobianLens.save / load / from_pretrained / merge
```

Read before trusting a result:

- **`from_hf` mutates the model you hand it** (`jlens/hf.py:112`): `eval()`, and
  `requires_grad_(False)` on every parameter. Do not reuse that object for anything needing grads.
- **`force_bos=True` by default** (`jlens/hf.py:105`) sets `tokenizer.add_bos_token = True`. This
  changes tokenization. Any comparison against a pipeline that did not do this is confounded at the
  token level before anything else.
- **Fitting silently drops short prompts.** `SKIP_FIRST_N_POSITIONS = 16` (`jlens/fitting.py:42`)
  and the final position is excluded, so prompts of ≤17 tokens raise; `fit` catches that and only
  logs a warning (`jlens/fitting.py:344`). **Always check `lens.n_prompts` against
  `len(prompts)`** — a corpus filtered by length is a silent selection effect on the fit.
- **The estimator is not a per-position Jacobian** (`jlens/fitting.py:11-17`). Cotangents are
  injected at *every valid target position at once*, so the gradient at source position `p` is
  `sum_{p' >= p} ∂h_final[p'] / ∂h_l[p]`, then averaged over `p`. Any claim phrased as "the
  Jacobian" should say which estimator it means.
- **`use_jacobian=False` in `apply()` is the vanilla logit-lens baseline** (`jlens/lens.py:168`).
  It is one flag. There is no excuse for reporting a lens result without it.
- **`save()` writes fp16 by default** (`jlens/lens.py:52`). A fitted lens and its saved-reloaded
  copy are not bit-identical; don't attribute the difference to anything else.
- **Checkpoints are huge.** `n_source_layers · d_model² · 4` bytes, written every prompt by default
  (`checkpoint_every=1`, `jlens/fitting.py:234`). For Qwen2.5-7B fitting all layers that is ≈1.4 GB
  per prompt written to disk. Raise `checkpoint_every`.
- Cost per prompt is one forward plus `ceil(d_model / dim_batch)` backward passes. `dim_batch`
  trades memory for nothing else — total backward FLOPs are constant.
- Third-party lenses on HF Hub (`andyx10/…`, `anicka/…` for Qwen2.5-7B) are **untrusted
  instruments**. If we use one, it gets validated against a positive control before it supports any
  claim, and the fit corpus it came from is unknown — treat that as an uncontrolled variable.

## How we work

Condensed from the standing collaboration spec; these are the rules that change what gets written
down, not a philosophy statement.

- **Claims carry their evidence level.** Observation / recurring pattern / supported claim / causal
  claim / mechanistic explanation / hypothesis / speculation. Never upgrade a claim because it has
  been repeated or because the project has moved on. Every report states scope: model, prompts,
  layers, token positions, seeds, n.
- **Baselines are not optional and are not sandbagged.** Logit lens (`use_jacobian=False`), random
  and norm-matched transports, prompt-only and behavioural baselines, shuffled controls. Give the
  alternative the same tuning effort as the thing being proposed.
- **A null result needs a positive control.** Failure to detect is evidence of absence only with
  demonstrated sensitivity on a comparable positive case.
- **Read the data, not just the metric.** Randomly sampled examples alongside chosen ones; check
  for outliers, position effects, prompt-template clusters, and subgroups before interpreting a mean.
- **Cheapest decisive experiment first.** Prefer the test whose outcome changes what we do next.
  De-risk before scaling: does the phenomenon exist, does the instrument fire on a positive control,
  what does the trivial baseline get.
- **Failures get diagnosed, not just recorded.** Distinguish: phenomenon absent / data lacked
  variation / instrument insensitive / implementation wrong / this method failed / the approach is
  wrong / underpowered. One failed implementation does not condemn an approach.
- Track which hypotheses were formed before vs. after seeing the result. Post-hoc explanations need
  held-out confirmation before they are reported as anything else.

## Logging and artifact discipline

- **Every session that runs anything appends to `RESEARCH_LOG.md`.** Including sessions that
  produced nothing — dead ends and negative results are the entries that stop us repeating work.
  Entries record what was actually run (script, commit, command), what was observed, and what it
  changed about the plan.
- **Every analysis produces an artifact directory** with a `report.md` from `_TEMPLATE.md`, and is
  added to `artifacts/README.md`. The report separates observations from interpretation and states
  what the result does *not* establish.
- **Plots when they carry information a sentence cannot** — distributions, layer sweeps,
  dose-response, per-example scatter. Not for two numbers. Save as PNG (≥150 dpi) under
  `figures/`, and commit the generating code. Show the distribution, not only the mean.
- Numbers in a report must be reproducible from committed code plus a named commit hash.

## Commits, pushes, secrets

- Author is **Vishesh Gupta <visheshguptaw14x@gmail.com>** (set locally in this clone), with
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>` as a trailer.
- Push to `origin ekko-lens`. Never push to `upstream`.
- **Nothing secret goes in a commit.** No API keys, tokens, `.env` files, or credentials — these
  are covered by `.gitignore` and by the `pre-commit` hook in `.githooks/`, enabled with
  `git config core.hooksPath .githooks`. The hook is a backstop, not permission to stop checking.
- Large or regenerable outputs (`*.pt`, raw dumps, HTML slice pages) stay out of git; they go in
  `research/outputs/`.
