# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Residual-stream interventions in J-space (paper §5).

The J-lens induces, at each fitted layer, a token-aligned input-space
direction per vocabulary entry: row ``w`` of the effective dictionary
``D_l = W_U @ J_l`` (:func:`token_directions`). This module applies the
paper's three intervention primitives along those directions, as forward-hook
context managers over a model's residual blocks:

- :func:`steer` — additive steering ``h' = h + alpha * ||h|| * v_hat``
  (§5.2, "thought injection");
- :func:`coordinate_swap` — the projection-based coordinate swap
  ``h' = h + V (sigma(c) - c)`` with ``c = V^+ h``, which exchanges the
  least-squares coordinates of a source/target direction pair while leaving
  the orthogonal complement untouched (§5.4);
- :func:`project_out` — remove the span of selected directions,
  ``h' = h - V V^+ h`` (the simple form of §5.3's ablation; the paper's
  top-k gradient-pursuit ablation is not implemented here).

Interventions are typically applied over a mid-network **workspace band**
(:func:`workspace_band`) and a set of token positions, following the paper's
band-wide protocol (§5.5).

Position semantics: ``positions`` are absolute indices into the sequence and
must be non-negative; positions outside a forward pass's sequence length are
skipped for that pass. Under KV-cached generation the model only sees the new
token(s) each step, so explicit positions effectively patch the prompt/prefill
(the cache then carries the edit forward), while ``positions=None`` edits
every position of every chunk — including each newly generated token. For
full-sequence semantics on generated positions, generate with
``use_cache=False``.

Everything runs at inference; edits clone the block output rather than
mutating it in place.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import torch

from jlens.lens import JacobianLens
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)

#: An edit takes the residual ``[batch, seq, d_model]`` and returns its
#: replacement (same shape/dtype).
EditFn = Callable[[torch.Tensor], torch.Tensor]

#: Paper band endpoints: the hypothesized workspace is roughly L38–L92 on the
#: normalized 0–100 layer axis (§2.2).
WORKSPACE_START_FRAC = 0.38
WORKSPACE_END_FRAC = 0.92


def workspace_band(
    n_layers: int,
    *,
    start: float = WORKSPACE_START_FRAC,
    end: float = WORKSPACE_END_FRAC,
) -> list[int]:
    """Layer indices whose normalized depth ``l / (n_layers - 1)`` lies in
    ``[start, end]`` — the paper's mid-network workspace band mapped onto this
    model's layer count. For Qwen3-8B (36 layers) the default is L14–L32.
    The exact effective band varies by experiment; treat this as a starting
    point, not a discovered boundary."""
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError(f"need 0 <= start <= end <= 1, got [{start}, {end}]")
    depth = n_layers - 1
    return [l for l in range(n_layers) if start <= l / depth <= end]


def token_directions(
    model: LensModel,
    lens: JacobianLens,
    layer: int,
    token_ids: Sequence[int],
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Rows of the effective dictionary ``D_l = W_U J_l`` as input-space
    vectors: ``[n_tokens, d_model]`` fp32 on CPU.

    Row ``i`` is the layer-``l`` residual direction that, transported through
    the average Jacobian and decoded, scores token ``token_ids[i]`` — the
    lens's steering/swap direction for that token (§3.3). ``normalize`` (the
    default) rescales each row to unit norm, the convention the paper's
    steering experiments use.
    """
    if layer not in lens.jacobians:
        raise ValueError(f"layer {layer} not fitted; lens has {lens.source_layers}")
    rows = model.unembed_rows(torch.as_tensor(list(token_ids), dtype=torch.long))
    directions = rows.detach().float().cpu() @ lens.jacobians[layer]
    if normalize:
        norms = directions.norm(dim=-1, keepdim=True)
        if (norms < 1e-8).any():
            raise ValueError("degenerate (near-zero) token direction")
        directions = directions / norms
    return directions


def _position_index(
    positions: Sequence[int] | None, seq_len: int, device: torch.device
) -> torch.Tensor | None:
    """Resolve ``positions`` against a forward pass of length ``seq_len``:
    ``None`` means every position; out-of-range positions are dropped (see the
    module docstring for the KV-cache semantics). Returns ``None`` for "all"
    and an empty tensor when nothing is in range."""
    if positions is None:
        return None
    kept = [p for p in positions if p < seq_len]
    if any(p < 0 for p in kept):
        raise ValueError("positions must be non-negative absolute indices")
    return torch.tensor(kept, dtype=torch.long, device=device)


@contextmanager
def residual_edits(model: LensModel, edits: dict[int, EditFn]) -> Iterator[None]:
    """Apply ``edits[layer]`` to the residual emitted by ``model.layers[layer]``
    on every forward pass inside the context.

    The hook replaces the block output (handling HF blocks that return
    ``(hidden, ...)`` tuples), so downstream layers — and any
    :class:`~jlens.hooks.ActivationRecorder` entered *after* this context —
    see the edited stream.
    """
    out_of_range = sorted(l for l in edits if not 0 <= l < model.n_layers)
    if out_of_range:
        raise ValueError(
            f"edit layers {out_of_range} out of range for {model.n_layers} layers"
        )

    def make_hook(edit: EditFn):
        def hook(module, inputs, output):
            if torch.is_tensor(output):
                return edit(output)
            return (edit(output[0]), *output[1:])

        return hook

    handles = []
    try:
        for layer, edit in edits.items():
            handles.append(model.layers[layer].register_forward_hook(make_hook(edit)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _apply_at_positions(
    hidden: torch.Tensor,
    positions: Sequence[int] | None,
    update: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Clone ``hidden`` and replace the selected positions with
    ``update(selected)`` (computed in fp32, cast back)."""
    index = _position_index(positions, hidden.shape[1], hidden.device)
    if index is not None and index.numel() == 0:
        return hidden
    if index is None:
        return update(hidden.float()).to(hidden.dtype)
    edited = hidden.clone()
    edited[:, index, :] = update(hidden[:, index, :].float()).to(hidden.dtype)
    return edited


@contextmanager
def steer(
    model: LensModel,
    lens: JacobianLens,
    *,
    token_id: int,
    layers: Sequence[int],
    positions: Sequence[int] | None = None,
    alpha: float = 4.0,
) -> Iterator[None]:
    """Additive steering (§5.2): at each layer in ``layers``, add
    ``alpha * scale * v_hat`` to the residual at ``positions``, where
    ``v_hat`` is the unit J-direction for ``token_id`` at that layer and
    ``scale`` is the mean residual norm over the edited positions of the
    current forward pass (so ``alpha`` is in units of typical residual norm;
    the paper sweeps strength — try 1–16). Negative ``alpha`` suppresses.
    """
    per_layer = {
        layer: token_directions(model, lens, layer, [token_id])[0] for layer in layers
    }

    def make_edit(direction: torch.Tensor) -> EditFn:
        def edit(hidden: torch.Tensor) -> torch.Tensor:
            v = direction.to(hidden.device)

            def update(selected: torch.Tensor) -> torch.Tensor:
                scale = selected.norm(dim=-1).mean()
                return selected + alpha * scale * v

            return _apply_at_positions(hidden, positions, update)

        return edit

    with residual_edits(model, {l: make_edit(v) for l, v in per_layer.items()}):
        yield


def _swap_basis(
    model: LensModel,
    lens: JacobianLens,
    layer: int,
    source_token_id: int,
    target_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(V, V^+)`` for the two-column span ``[v_source, v_target]`` at
    ``layer``: ``V`` is ``[d_model, 2]``, ``V^+`` its pseudoinverse ``[2,
    d_model]``, both fp32 CPU. Warns when the pair is nearly collinear (the
    swap is then poorly conditioned)."""
    directions = token_directions(
        model, lens, layer, [source_token_id, target_token_id]
    )
    V = directions.T.contiguous()  # [d_model, 2]
    singular_values = torch.linalg.svdvals(V)
    if singular_values[-1] / singular_values[0] < 1e-3:
        logger.warning(
            "coordinate_swap: source/target directions nearly collinear at "
            "layer %d (condition %.1e); the swap is ill-conditioned",
            layer,
            (singular_values[0] / singular_values[-1]).item(),
        )
    return V, torch.linalg.pinv(V)


@contextmanager
def coordinate_swap(
    model: LensModel,
    lens: JacobianLens,
    *,
    source_token_id: int,
    target_token_id: int,
    layers: Sequence[int],
    positions: Sequence[int] | None = None,
) -> Iterator[None]:
    """Projection-based coordinate swap (§5.4): with ``V = [v_s, v_t]`` and
    least-squares coordinates ``c = V^+ h``, replace
    ``h <- h + V (sigma(c) - c)`` where ``sigma`` exchanges the two
    coordinates. The component of ``h`` orthogonal to ``span(V)`` is
    preserved exactly; this is the paper's preferred alternative to adding
    ``v_t - v_s`` when the directions are correlated.
    """
    bases = {
        layer: _swap_basis(model, lens, layer, source_token_id, target_token_id)
        for layer in layers
    }

    def make_edit(V: torch.Tensor, pinv: torch.Tensor) -> EditFn:
        def edit(hidden: torch.Tensor) -> torch.Tensor:
            V_dev = V.to(hidden.device)
            pinv_dev = pinv.to(hidden.device)

            def update(selected: torch.Tensor) -> torch.Tensor:
                coords = selected @ pinv_dev.T  # [..., 2]
                swapped = coords[..., [1, 0]]
                return selected + (swapped - coords) @ V_dev.T

            return _apply_at_positions(hidden, positions, update)

        return edit

    with residual_edits(
        model, {l: make_edit(V, pinv) for l, (V, pinv) in bases.items()}
    ):
        yield


@contextmanager
def project_out(
    model: LensModel,
    lens: JacobianLens,
    *,
    token_ids: Sequence[int],
    layers: Sequence[int],
    positions: Sequence[int] | None = None,
) -> Iterator[None]:
    """Ablate the span of the given token directions (§5.3, simple form):
    ``h <- h - V V^+ h``, the orthogonal projection of ``h`` off
    ``span(v_w for w in token_ids)`` at each layer in ``layers``."""
    bases: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in layers:
        V = token_directions(model, lens, layer, token_ids).T.contiguous()
        bases[layer] = (V, torch.linalg.pinv(V))

    def make_edit(V: torch.Tensor, pinv: torch.Tensor) -> EditFn:
        def edit(hidden: torch.Tensor) -> torch.Tensor:
            V_dev = V.to(hidden.device)
            pinv_dev = pinv.to(hidden.device)

            def update(selected: torch.Tensor) -> torch.Tensor:
                return selected - (selected @ pinv_dev.T) @ V_dev.T

            return _apply_at_positions(hidden, positions, update)

        return edit

    with residual_edits(
        model, {l: make_edit(V, pinv) for l, (V, pinv) in bases.items()}
    ):
        yield
