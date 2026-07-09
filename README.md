# jlens — Jacobian lens

> **Reference implementation.** Not maintained and not accepting contributions.

Companion code for [**Verbalizable Representations Form a Global Workspace in
Language Models**](https://transformer-circuits.pub/2026/workspace/index.html).

The Jacobian lens reads out what an internal activation is disposed to make the
model say. It linearly transports a residual-stream vector at any layer and
position into the final-layer basis, then decodes it with the model's own
unembedding into a ranked list of vocabulary tokens.

The transport is the average input–output Jacobian over a text corpus:

```
lens_l(h) = unembed( J_l @ h ), J_l = E[∂h_final / ∂h_l]
```

The expectation is over prompts, source positions, and all current-and-future
target positions in a generic web-text corpus; the precise estimator
(cotangents summed over target positions, then averaged over source positions)
is documented in the [`jlens.fitting`](jlens/fitting.py) module docstring.

This repo fits the lens on open-weights decoder transformers, applies it, and
renders the interactive layer × position view shown below. Examples use Qwen;
other HuggingFace decoders adapt cleanly.

![Slice visualisation: ASCII-face example](assets/slice_vis.png)

*The ASCII-face example: selecting the `^` (nose) position shows the lens
reading out "nose" at mid layers, although the word never appears in the
prompt.*

## Install

```bash
pip install -e .
```

## Usage

### Apply

To apply a pre-fitted lens:

```python
import transformers, jlens

hf = transformers.AutoModelForCausalLM.from_pretrained("org/model").cuda()
tok = transformers.AutoTokenizer.from_pretrained("org/model")
model = jlens.from_hf(hf, tok)

lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")
lens_logits, model_logits, _ = lens.apply(
    model, "Fact: The currency used in the country shaped like a boot is",
    positions=[-2])
for layer, logits in sorted(lens_logits.items()):
    print(layer, [tok.decode([t]) for t in logits[0].topk(5).indices])
```

### Fit

To fit a lens on your own model:

```python
lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
lens.save("out/jacobian_lens.pt")
```

The paper's lenses use 1000 sequences of 128 tokens from a pretraining-like
corpus. Quality saturates quickly (§9.3); ~100 prompts is usable. This is a
reference implementation and is not optimized; fitting time is dominated by
the model's own backward pass. Parallelize by running `fit()` on disjoint
slices and combining with `JacobianLens.merge()`.

## Walkthrough

[`walkthrough.ipynb`](walkthrough.ipynb) is the end-to-end notebook: load a
model, load (or fit) a lens, apply it at a few layers, and render a slice page
like the one above.

## Qwen3-8B chain of thought

This branch adds first-class support for reading (and intervening on) the
lens over **Qwen3-8B thinking-mode rollouts** — the `<think>…</think>` chain
of thought the model emits before its answer.

[`qwen3_cot_walkthrough.ipynb`](qwen3_cot_walkthrough.ipynb) is the
end-to-end notebook. The pieces:

- **`jlens.cot`** — `chat_prompt` renders the chat template with
  `enable_thinking`; `generate_cot` samples a rollout (Qwen3's recommended
  thinking-mode sampling) and returns a `CoTTrace` with the exact token ids
  and the located thinking/answer spans. Sampled text does not reliably
  re-encode to the tokens the model produced, so `JacobianLens.apply` and
  `compute_slice` now accept `input_ids=` and the trace is consumed that way:

  ```python
  trace = jlens.generate_cot(model, "Is 977 prime? Answer yes or no.", seed=0)
  slice_data = compute_slice(model, lens, input_ids=trace.input_ids,
                             last_n_tokens=trace.total_len - trace.prompt_len)
  ```

- **`jlens.interventions`** — the paper's §5 primitives over the token
  directions `D_l = W_U J_l`: `steer` (additive thought injection),
  `coordinate_swap` (projection-based swap preserving the orthogonal
  complement), `project_out` (span ablation), applied over
  `jlens.workspace_band(n_layers)` (the paper's normalized L38–92 mid-layer
  band; L14–L32 on Qwen3-8B's 36 layers):

  ```python
  with jlens.coordinate_swap(model, lens, source_token_id=italy,
                             target_token_id=japan, layers=band):
      swapped = jlens.generate_cot(model, question, seed=0)
  ```

- **`scripts/`** —
  [`fit_lens.py`](scripts/fit_lens.py) fits the paper-default 1000×128
  WikiText lens on `Qwen/Qwen3-8B` (checkpointed; `--shard i/N` +
  `JacobianLens.merge` to spread across GPUs; `--dim-batch` is the GPU-memory
  knob: 8 for 40 GB, 16–32 for 80 GB).
  [`cot_slice.py`](scripts/cot_slice.py) samples a rollout and renders the
  slice page over the reasoning tokens.
  [`lens_eval.py`](scripts/lens_eval.py) scores J-lens vs logit lens
  (pass@k) on the six bundled [`data/evaluations/`](data/evaluations/)
  distributions.
  [`swap_eval.py`](scripts/swap_eval.py) runs the causal coordinate-swap
  test on the 90 bundled two-hop prompts, with an unrelated-pair control.

The fit is calibrated on ordinary web text and applied to chat/thinking
traces, matching the paper's protocol. A full 1000-prompt fit on Qwen3-8B is
`ceil(4096/dim_batch)` backward passes per prompt — hours-to-days on one
GPU; ~100 prompts is already usable (§Fit above).

Reading a slice page:

- Each cell shows the lens top-1 word at that (position, layer); the
  superscript is its rank over the full vocabulary.
- Click a cell to select a (position, layer) and pin its top-1 token; pinned
  tokens get rank-tracking charts and a rank heatmap.
- The bottom row (`L = n_layers − 1`) is the model's actual output.

## License and data

Code is released under the Apache License 2.0 — see [LICENSE](LICENSE).

The replication and lens-eval prompt sets in [`data/`](data/) are synthetic,
authored by Anthropic, and released under the same Apache License 2.0 as the
code. See the READMEs in [`data/experiments/`](data/experiments/) and
[`data/evaluations/`](data/evaluations/) for what each set contains.

The slice-vis pages use [d3](https://github.com/d3/d3) (ISC license), loaded
from the jsDelivr CDN with subresource integrity or inlined into
self-contained pages.

No model weights or text corpora are bundled; models and datasets downloaded
at run time are subject to their own licenses.
