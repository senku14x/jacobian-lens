# WorkspaceBench — local setup notes and patches (for our runs and as feedback to its authors)

WorkspaceBench: github.com/camilablank/workspace-bench, cloned at commit `92d763e` to `/workspace/workspace-bench`.
`local_patches.diff` holds our local changes to its optional readout producer (`src/wsbench/produce/`). The benchmark
itself (item banks, judges, scoring) is untouched.

Apply with `git apply local_patches.diff` in a clone at `92d763e`.

## What the patches fix (all verified on one H100 80GB, Qwen3.6-27B)

1. **CJK tokens sent to the judge as garbled bytes (`render.py`, `display_tokens`).** The producer mapped `Ġ` to a
   space but never decoded the byte-level BPE alphabet. So `的名称` reached the judge as `çļĦåĲįç§°`, and `铁` ("iron")
   was unreadable.
   - This penalises token lenses on Qwen, where many workspace readouts are Chinese tokens.
   - Fix: `tokenizer.decode([id])`, keeping the raw form only for partial-UTF-8 fragments.
   - Verified effect: a February item passes via `二月` only after the fix.
2. **OOM on one H100 (`methods.py`)**, from three causes:
   - `backend.unembed.float()` keeps `requires_grad=True`, so every lens computation built an autograd graph. This was
     the main cause; fixed with `.detach()`.
   - The J-lens normaliser materialised the full (vocab × d) product at once. Now chunked.
   - The full 63-layer fp32 J stack sat on the GPU. It is now kept on CPU, with a one-layer GPU cache.
   - Peak memory after the fixes: ≈ 59 GB.

## Findings worth passing on (not patched)
- **The WSB "J-lens" arm is a cosine readout** of the Neuronpedia n=1000 lens: (W_U·J·h)/‖Jᵀ W_U[t]‖ with raw W_U. It is
  not the paper's softmax(W_U·norm(J·h)). We checked on two prompts: the cosine readout is *less* dominated by `____`
  tokens than the standard one, so it is a defensible choice, but it should be stated.
- **The default oracle-lens LoRA path `agu18dec/local-workspace` is private** (HTTP 401). The public copy is
  `agu18dec/olens_and_ar` (`olens_s3d_rl600/`).
- **The lockfile pins torch cu130**, which does not run on CUDA-12.5 drivers. We used `torch==2.14.0+cu126` with
  `uv run --no-sync`.
- **On many families, the prompt-only baseline matches or beats every reader** (multihop .90, typo 1.0, poetry .71,
  brew .96, role-bound 1.0, relational .82). Those families cannot show that a reader extracts information from
  activations rather than from the prompt.

## Our uses of WSB in `research_secant`
- 001: the poetry bank, the WSB judge as a reporting metric (Amendment 5).
- 002: the hallucination bank's verbatim sampled token ids (`capture_rows.json`) as on-policy text.
- `../notes/wsb_subsets/`: the seeded 40-item screening subsets (seed 20260929) from the earlier screening protocol.
