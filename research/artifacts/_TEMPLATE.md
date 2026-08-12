# NNN — <title>

**Date:** YYYY-MM-DD
**Code:** `research/experiments/NNN_slug.py` @ `<commit>`
**Phase:** exploration | validation | execution
**Status:** current | superseded by NNN | retracted

## Question

What uncertainty this was meant to resolve, and what outcomes would have pushed us which way.
State the hypothesis and its strongest alternative if this is a validation experiment.

## Setup

Model, checkpoint revision, dtype, tokenizer settings (incl. `add_bos_token`), prompts and where
they came from, layers, token positions, seeds, n, hardware. Enough that someone can rerun it.

Anything that was chosen after seeing data goes here, flagged as such.

## Observations

What was measured. Numbers, plots, and randomly sampled examples — not only the illustrative ones.
No interpretation in this section.

![caption](figures/name.png)

## Controls and baselines

What was compared against, and what it got. Logit lens (`use_jacobian=False`), random/norm-matched
transports, shuffled labels, prompt-only baselines, positive control for any null result.

## What this supports

The strongest claim the evidence permits, at its evidence level, scoped to the tested conditions.

## What this does not establish

The alternatives still standing, the confounds not ruled out, the scope not tested. Be specific —
"needs more work" is not a limitation.

## Decision

What we do next as a result, and what this rules out. If the answer is "nothing changes", say so —
that is information about the experiment, not a failure of the write-up.
