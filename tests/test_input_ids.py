# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Pre-tokenized ``input_ids=`` paths through apply() and compute_slice():
must match the prompt path exactly and reject ambiguous calls."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from jlens.fitting import fit
from jlens.vis import compute_slice

from .tiny import TinyDecoder

PROMPT = "the quick brown fox jumps over the lazy dog"


@pytest.fixture(scope="module")
def model() -> TinyDecoder:
    return TinyDecoder(n_layers=4, d_model=8)


@pytest.fixture(scope="module")
def lens(model):
    return fit(
        model,
        ["abcdefghij " * 5, "klmnopqrst " * 5],
        source_layers=[0, 1, 2],
        dim_batch=4,
        max_seq_len=64,
    )


def test_apply_input_ids_matches_prompt_path(model, lens):
    ids = model.encode(PROMPT)
    from_prompt, model_logits_prompt, _ = lens.apply(model, PROMPT, layers=[0, 2])
    from_ids, model_logits_ids, returned_ids = lens.apply(
        model, input_ids=ids, layers=[0, 2]
    )
    assert torch.equal(returned_ids, ids)
    torch.testing.assert_close(model_logits_ids, model_logits_prompt)
    for layer in (0, 2):
        torch.testing.assert_close(from_ids[layer], from_prompt[layer])


def test_apply_input_ids_bypasses_truncation(model, lens):
    ids = model.encode(PROMPT)  # 45 tokens
    _, model_logits, returned = lens.apply(
        model, input_ids=ids, layers=[0], max_seq_len=8
    )
    assert returned.shape[1] == ids.shape[1]  # max_seq_len only guards prompts
    assert model_logits.shape[0] == ids.shape[1]


def test_apply_rejects_ambiguous_inputs(model, lens):
    ids = model.encode(PROMPT)
    with pytest.raises(ValueError, match="exactly one"):
        lens.apply(model, PROMPT, input_ids=ids)
    with pytest.raises(ValueError, match="exactly one"):
        lens.apply(model)


def test_compute_slice_input_ids_matches_prompt_path(model, lens):
    ids = model.encode(PROMPT)
    from_prompt = compute_slice(model, lens, PROMPT)
    from_ids = compute_slice(model, lens, input_ids=ids)
    assert from_ids.context_token_ids == from_prompt.context_token_ids
    np.testing.assert_array_equal(from_ids.top_ids, from_prompt.top_ids)
    np.testing.assert_array_equal(from_ids.rank_tensor, from_prompt.rank_tensor)


def test_compute_slice_rejects_ambiguous_inputs(model, lens):
    ids = model.encode(PROMPT)
    with pytest.raises(ValueError, match="exactly one"):
        compute_slice(model, lens, PROMPT, input_ids=ids)
    with pytest.raises(ValueError, match="exactly one"):
        compute_slice(model, lens)
