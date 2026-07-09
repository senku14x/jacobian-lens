# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end on a tiny randomly-initialized Qwen3 (the architecture the
Qwen3-8B chain-of-thought workflow targets): layout detection, lens fitting
through GQA attention, readout, CoT generation, and interventions — all on
CPU with no downloads. transformers is a dev dependency; skip without it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

import jlens  # noqa: E402
from jlens.cot import generate_cot  # noqa: E402
from jlens.vis import compute_slice  # noqa: E402

VOCAB = 128
THINK, END_THINK = 5, 6


class TinyQwenTokenizer:
    """Just enough tokenizer surface for HFLensModel + jlens.cot, with
    Qwen3-style single-token thinking markers."""

    bos_token_id = None
    eos_token_id = 2
    pad_token_id = None
    unk_token_id = 0

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
        ids = [1] + [10 + (b % (VOCAB - 10)) for b in text.encode()][: max_length - 1]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, ids, skip_special_tokens=False, **_kw):
        keep = [i for i in ids if not (skip_special_tokens and int(i) < 10)]
        return " ".join(f"t{int(i)}" for i in keep)

    def convert_tokens_to_ids(self, token):
        return {"<think>": THINK, "</think>": END_THINK}.get(token, self.unk_token_id)

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        body = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return body + ("\nassistant:" if add_generation_prompt else "")


@pytest.fixture(scope="module")
def model() -> jlens.HFLensModel:
    config = transformers.Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=VOCAB,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    hf_model = transformers.Qwen3ForCausalLM(config)
    return jlens.from_hf(hf_model, TinyQwenTokenizer())


@pytest.fixture(scope="module")
def lens(model) -> jlens.JacobianLens:
    prompts = ["the quick brown fox jumps " * 3, "colorless green ideas sleep " * 3]
    return jlens.fit(model, prompts, source_layers=[0, 1, 2], dim_batch=8)


def test_layout_autodetected(model):
    assert model.layout == jlens.Layout("model")
    assert model.n_layers == 4
    assert model.d_model == 32


def test_unembed_rows_match_unembed_up_to_rms(model):
    """unembed_rows folds the RMSNorm gain: for any residual h,
    logits == (rows @ h) / rms(h). Check proportionality across tokens."""
    residual = torch.randn(32)
    logits = model.unembed(residual).float()
    rows = model.unembed_rows(torch.arange(VOCAB))
    raw = rows @ residual
    scale = logits / raw
    finite = raw.abs() > 1e-4
    assert scale[finite].std() / scale[finite].mean().abs() < 1e-3


def test_fit_and_readout(model, lens):
    assert lens.source_layers == [0, 1, 2]
    assert lens.d_model == 32
    lens_logits, model_logits, input_ids = lens.apply(
        model, "a two hop factual prompt about countries", layers=[1], positions=[-1]
    )
    assert lens_logits[1].shape == (1, VOCAB)
    assert model_logits.shape == (1, VOCAB)


def test_generate_cot_and_slice_over_rollout(model, lens):
    trace = generate_cot(
        model, "why is the sky blue?", seed=0, max_new_tokens=24, greedy=True
    )
    assert trace.prompt_len > 16  # long enough to fit/slice
    assert trace.total_len > trace.prompt_len
    # Random weights: a think block may or may not appear; the spans must be
    # consistent either way.
    if trace.think_span is not None:
        start, end = trace.think_span
        assert trace.prompt_len < start <= end <= trace.total_len
    assert trace.answer_span[1] == trace.total_len

    completion_len = trace.total_len - trace.prompt_len
    slice_data = compute_slice(
        model, lens, input_ids=trace.input_ids, last_n_tokens=completion_len
    )
    assert slice_data.seq_len == completion_len
    assert slice_data.ctx_offset == trace.prompt_len
    assert slice_data.layers == [0, 1, 2, 3]  # fitted + final
    assert len(slice_data.context_token_ids) == trace.total_len


def test_interventions_run_through_qwen3(model, lens):
    input_ids = model.encode("swap the bridge entity in this prompt")
    band = jlens.workspace_band(model.n_layers)  # [2] for 4 layers
    band = [l for l in band if l in lens.jacobians]
    _, base_logits, _ = lens.apply(model, input_ids=input_ids, layers=band)

    with jlens.coordinate_swap(
        model,
        lens,
        source_token_id=20,
        target_token_id=40,
        layers=band,
        positions=list(range(input_ids.shape[1])),
    ):
        _, swapped_logits, _ = lens.apply(model, input_ids=input_ids, layers=band)
    assert not torch.allclose(base_logits, swapped_logits)

    # Steering during (cache-less chunked) generation also goes through.
    with jlens.steer(model, lens, token_id=30, layers=band, alpha=4.0):
        steered = model.hf_model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=4,
            do_sample=False,
            use_cache=False,
            pad_token_id=2,
        )
    assert steered.shape[1] == input_ids.shape[1] + 4
