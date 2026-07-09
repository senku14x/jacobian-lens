# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""J-space interventions on the tiny CPU model: steering geometry,
coordinate-swap invariants, projection ablation, hook plumbing."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from jlens.fitting import fit
from jlens.hooks import ActivationRecorder
from jlens.interventions import (
    coordinate_swap,
    project_out,
    residual_edits,
    steer,
    token_directions,
    workspace_band,
)

from .tiny import TinyDecoder

PROMPT = "the quick brown fox jumps over the lazy dog near the river bank"


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


def _residual_at(model, layer: int, edit_context=None) -> torch.Tensor:
    """Residual stream at ``layer`` for PROMPT, optionally under an edit.

    The edit context is entered *before* the recorder: forward hooks fire in
    registration order, so the recorder then observes the edited stream (the
    same ordering interventions rely on inside ``apply``/``compute_slice``).
    """
    from contextlib import nullcontext

    input_ids = model.encode(PROMPT)
    with edit_context or nullcontext():
        with ActivationRecorder(model.layers, at=[layer]) as recorder:
            model.forward(input_ids)
            return recorder.activations[layer][0].detach().float()


# --------------------------------------------------------------------------- #
# workspace_band
# --------------------------------------------------------------------------- #


def test_workspace_band_fractions():
    assert workspace_band(36) == list(range(14, 33))  # Qwen3-8B: L14..L32
    assert workspace_band(4) == [2]
    assert workspace_band(10, start=0.0, end=1.0) == list(range(10))
    with pytest.raises(ValueError, match="start"):
        workspace_band(10, start=0.9, end=0.3)


# --------------------------------------------------------------------------- #
# token_directions
# --------------------------------------------------------------------------- #


def test_token_directions_shape_norm_and_orientation(model, lens):
    directions = token_directions(model, lens, layer=1, token_ids=[4, 7])
    assert directions.shape == (2, 8)
    torch.testing.assert_close(
        directions.norm(dim=-1), torch.ones(2), rtol=0, atol=1e-5
    )
    # Unnormalized rows are (W_U * norm_gain) @ J_l: check against the direct
    # product for this exactly-known tiny model.
    raw = token_directions(model, lens, layer=1, token_ids=[4], normalize=False)
    expected = (
        model.lm_head.weight[4].float() * model.norm.weight.float()
    ) @ lens.jacobians[1]
    torch.testing.assert_close(raw[0], expected, rtol=0, atol=1e-5)


def test_token_directions_unfitted_layer_raises(model, lens):
    with pytest.raises(ValueError, match="not fitted"):
        token_directions(model, lens, layer=3, token_ids=[4])


# --------------------------------------------------------------------------- #
# steer
# --------------------------------------------------------------------------- #


def test_steer_adds_direction_at_selected_positions(model, lens):
    base = _residual_at(model, 1)
    target = 5
    steered = _residual_at(
        model,
        1,
        steer(model, lens, token_id=target, layers=[1], positions=[3, 6], alpha=2.0),
    )
    delta = steered - base
    untouched = [p for p in range(base.shape[0]) if p not in (3, 6)]
    torch.testing.assert_close(delta[untouched], torch.zeros_like(delta[untouched]))

    v = token_directions(model, lens, 1, [target])[0]
    for position in (3, 6):
        cosine = torch.nn.functional.cosine_similarity(delta[position], v, dim=0)
        assert cosine > 0.999
    # alpha is in units of the mean residual norm over the edited positions.
    expected_scale = 2.0 * base[[3, 6]].norm(dim=-1).mean()
    torch.testing.assert_close(delta[3].norm(), expected_scale, rtol=1e-4, atol=1e-4)


def test_steer_raises_target_token_logit(model, lens):
    target = 5
    input_ids = model.encode(PROMPT)
    with ActivationRecorder(model.layers, at=[3]) as recorder:
        model.forward(input_ids)
        base_logits = model.unembed(recorder.activations[3][0].detach())
    with steer(model, lens, token_id=target, layers=[0, 1, 2], alpha=8.0):
        with ActivationRecorder(model.layers, at=[3]) as recorder:
            model.forward(input_ids)
            steered_logits = model.unembed(recorder.activations[3][0].detach())
    # Steering along the transported unembedding row must raise that token's
    # logit relative to the field (mean-centered, since LayerNorm rescales).
    base_edge = base_logits[:, target] - base_logits.mean(-1)
    steered_edge = steered_logits[:, target] - steered_logits.mean(-1)
    assert (steered_edge > base_edge).all()


def test_steer_out_of_range_positions_skipped_negative_raises(model, lens):
    base = _residual_at(model, 1)
    seq_len = base.shape[0]
    unchanged = _residual_at(
        model,
        1,
        steer(model, lens, token_id=5, layers=[1], positions=[seq_len + 50]),
    )
    torch.testing.assert_close(unchanged, base)

    with pytest.raises(ValueError, match="non-negative"):
        _residual_at(
            model, 1, steer(model, lens, token_id=5, layers=[1], positions=[-1])
        )


# --------------------------------------------------------------------------- #
# coordinate_swap
# --------------------------------------------------------------------------- #


def test_coordinate_swap_exchanges_coords_preserves_complement(model, lens):
    source, target = 4, 7
    base = _residual_at(model, 1)
    swapped = _residual_at(
        model,
        1,
        coordinate_swap(
            model,
            lens,
            source_token_id=source,
            target_token_id=target,
            layers=[1],
            positions=None,
        ),
    )
    V = token_directions(model, lens, 1, [source, target]).T  # [d, 2]
    pinv = torch.linalg.pinv(V)
    c_base = base @ pinv.T
    c_swapped = swapped @ pinv.T
    torch.testing.assert_close(c_swapped, c_base[:, [1, 0]], rtol=1e-3, atol=1e-4)
    # Orthogonal complement untouched: residual minus span reconstruction.
    torch.testing.assert_close(
        base - c_base @ V.T, swapped - c_swapped @ V.T, rtol=1e-3, atol=1e-4
    )


def test_coordinate_swap_position_scoping(model, lens):
    base = _residual_at(model, 1)
    swapped = _residual_at(
        model,
        1,
        coordinate_swap(
            model,
            lens,
            source_token_id=4,
            target_token_id=7,
            layers=[1],
            positions=[0, 1],
        ),
    )
    torch.testing.assert_close(swapped[2:], base[2:])
    assert not torch.allclose(swapped[:2], base[:2])


# --------------------------------------------------------------------------- #
# project_out
# --------------------------------------------------------------------------- #


def test_project_out_zeroes_span_coordinates(model, lens):
    token_ids = [4, 7, 9]
    base = _residual_at(model, 1)
    ablated = _residual_at(
        model, 1, project_out(model, lens, token_ids=token_ids, layers=[1])
    )
    V = token_directions(model, lens, 1, token_ids).T
    pinv = torch.linalg.pinv(V)
    # Least-squares coordinates on the ablated stream vanish...
    torch.testing.assert_close(
        ablated @ pinv.T, torch.zeros(base.shape[0], 3), rtol=0, atol=1e-4
    )
    # ...and the removal is exactly the span reconstruction.
    torch.testing.assert_close(
        base - ablated, (base @ pinv.T) @ V.T, rtol=1e-3, atol=1e-4
    )


# --------------------------------------------------------------------------- #
# residual_edits plumbing
# --------------------------------------------------------------------------- #


def test_residual_edits_out_of_range_layer_raises(model):
    with pytest.raises(ValueError, match="out of range"):
        with residual_edits(model, {17: lambda h: h}):
            pass


def test_residual_edits_handles_tuple_outputs():
    """HF blocks return (hidden, ...) tuples; the hook must rewrap them.

    The tuple-returning block is the *last* layer so TinyDecoder's forward
    never feeds a tuple into a following block; the recorder (which also
    unwraps tuples) observes the edit, and non-tensor tuple tail survives.
    """

    class TupleBlock(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden):
            return (self.inner(hidden), "kv-cache-stand-in")

    wrapped = TinyDecoder(n_layers=4, d_model=8)
    wrapped.layers[3] = TupleBlock(wrapped.layers[3])
    marker = torch.zeros(8)
    marker[0] = 1.0

    input_ids = wrapped.encode(PROMPT)
    with ActivationRecorder(wrapped.layers, at=[3]) as recorder:
        wrapped.forward(input_ids)
        base = recorder.activations[3].detach().clone()
    with residual_edits(wrapped, {3: lambda h: h + marker}):
        with ActivationRecorder(wrapped.layers, at=[3]) as recorder:
            wrapped.forward(input_ids)
            edited = recorder.activations[3].detach()
    torch.testing.assert_close(edited, base + marker)


def test_edits_do_not_leak_after_context(model, lens):
    base = _residual_at(model, 1)
    _ = _residual_at(model, 1, steer(model, lens, token_id=5, layers=[1], alpha=4.0))
    after = _residual_at(model, 1)
    torch.testing.assert_close(after, base)
