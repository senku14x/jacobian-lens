# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Chain-of-thought (thinking-mode) rollouts for the Jacobian lens.

Reasoning models in the Qwen3 family emit an explicit chain of thought between
``<think>`` and ``</think>`` marker tokens before the visible answer. To read
the lens out *over the reasoning trace*, the trace has to be sampled first and
then re-run as a fixed token sequence:

1. :func:`chat_prompt` renders a user message through the tokenizer's chat
   template with thinking mode switched on or off;
2. :func:`generate_cot` samples a completion and returns a :class:`CoTTrace`
   holding the exact token ids plus the located thinking/answer spans;
3. the trace's ``input_ids`` go straight to
   :meth:`jlens.lens.JacobianLens.apply` or
   :func:`jlens.vis.compute_slice` via their ``input_ids=`` argument, and its
   spans give the positions to read out (or intervene on, see
   :mod:`jlens.interventions`).

Token ids are passed around rather than text because sampled text does not
reliably re-encode to the tokens the model actually produced.

Generation needs a real HuggingFace model (:class:`jlens.hf.HFLensModel`);
everything else in this module is model-library-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

#: Marker-token surfaces probed by :func:`think_marker_ids`, in order. Qwen3
#: (and several other open reasoning models) use ``<think>``/``</think>``.
THINK_MARKERS: tuple[tuple[str, str], ...] = (("<think>", "</think>"),)


def think_marker_ids(tokenizer: Any) -> tuple[int, int] | None:
    """Return ``(open_id, close_id)`` for the tokenizer's thinking markers, or
    ``None`` if no marker pair from :data:`THINK_MARKERS` is a single token."""
    unk = getattr(tokenizer, "unk_token_id", None)
    for open_str, close_str in THINK_MARKERS:
        try:
            open_id = tokenizer.convert_tokens_to_ids(open_str)
            close_id = tokenizer.convert_tokens_to_ids(close_str)
        except Exception:
            continue
        if open_id is None or close_id is None:
            continue
        if unk is not None and (open_id == unk or close_id == unk):
            continue
        return int(open_id), int(close_id)
    return None


def chat_prompt(
    tokenizer: Any,
    user: str,
    *,
    system: str | None = None,
    assistant_prefill: str = "",
    enable_thinking: bool | None = None,
) -> str:
    """Render a single-turn chat prompt through the tokenizer's chat template.

    Args:
        tokenizer: HF tokenizer with ``apply_chat_template``.
        user: User-turn content.
        system: Optional system-turn content.
        assistant_prefill: If non-empty, start the assistant turn with this
            text and leave it open (``continue_final_message=True``).
        enable_thinking: Passed through to the chat template when not ``None``.
            On Qwen3, ``True`` (the model default) leaves the assistant turn
            open for a ``<think>`` block; ``False`` inserts an empty
            ``<think>\\n\\n</think>`` so the model answers directly. Templates
            without the variable ignore it.
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    kwargs: dict[str, Any] = {"tokenize": False}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    if assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill})
        return tokenizer.apply_chat_template(
            messages, continue_final_message=True, **kwargs
        )
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)


@dataclass(frozen=True)
class CoTTrace:
    """A sampled rollout with its thinking/answer spans located.

    All spans are half-open ``[start, end)`` token-index ranges into
    ``input_ids[0]``. ``think_span`` covers the tokens *between* the
    ``<think>`` markers (markers excluded); it is ``None`` when the completion
    contains no thinking block. An unclosed block (generation hit the token
    limit) yields a ``think_span`` that runs to the end of the sequence and an
    empty ``answer_span``.

    Attributes:
        input_ids: ``[1, total_len]`` — prompt plus sampled completion.
        prompt_len: Number of prompt tokens.
        think_span: Span of the chain-of-thought content, or ``None``.
        answer_span: Span of the post-thinking answer (marker tokens and the
            leading whitespace token the template puts after ``</think>``
            included; decode with ``skip_special_tokens`` to drop markers).
        tokenizer: Kept for the text properties; not serialized anywhere.
    """

    input_ids: torch.Tensor
    prompt_len: int
    think_span: tuple[int, int] | None
    answer_span: tuple[int, int]
    tokenizer: Any = field(repr=False)

    @property
    def total_len(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def completion_ids(self) -> torch.Tensor:
        """``[n_completion]`` — the sampled tokens only."""
        return self.input_ids[0, self.prompt_len :]

    @property
    def think_positions(self) -> list[int]:
        """Positions of the chain-of-thought tokens (empty if no block)."""
        return [] if self.think_span is None else list(range(*self.think_span))

    def _decode(self, span: tuple[int, int]) -> str:
        ids = self.input_ids[0, span[0] : span[1]].tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def thinking_text(self) -> str:
        return "" if self.think_span is None else self._decode(self.think_span)

    @property
    def answer_text(self) -> str:
        return self._decode(self.answer_span)

    @property
    def completion_text(self) -> str:
        """Full completion (thinking markers included, other specials skipped
        by the tokenizer's own decode rules)."""
        return self.tokenizer.decode(self.completion_ids.tolist())


def locate_spans(
    token_ids: list[int], prompt_len: int, tokenizer: Any
) -> tuple[tuple[int, int] | None, tuple[int, int]]:
    """Locate ``(think_span, answer_span)`` in a full token-id sequence.

    Searches the completion region ``token_ids[prompt_len:]`` for the first
    thinking-marker pair. A dangling ``<think>`` in the *prompt* (some chat
    templates or assistant prefills open the block themselves rather than
    letting the model emit the marker) means the block is already open when
    generation starts, so the thinking content begins at ``prompt_len``. A
    closed pair in the prompt — e.g. the empty block Qwen3 inserts with
    ``enable_thinking=False`` — is ignored. See :class:`CoTTrace` for the
    span conventions.
    """
    total_len = len(token_ids)
    markers = think_marker_ids(tokenizer)
    if markers is None:
        return None, (prompt_len, total_len)
    open_id, close_id = markers
    completion = token_ids[prompt_len:]

    if open_id in completion:
        think_start = prompt_len + completion.index(open_id) + 1
    else:
        prompt_ids = token_ids[:prompt_len]
        last_close = max(
            (i for i, t in enumerate(prompt_ids) if t == close_id), default=-1
        )
        if open_id not in prompt_ids[last_close + 1 :]:
            return None, (prompt_len, total_len)
        think_start = prompt_len  # block opened by the prompt, still open

    tail = token_ids[think_start:]
    if close_id not in tail:  # unclosed: ran out of tokens mid-thought
        return (think_start, total_len), (total_len, total_len)
    close_pos = think_start + tail.index(close_id)
    return (think_start, close_pos), (close_pos + 1, total_len)


def generate_cot(
    model: Any,
    user: str | None = None,
    *,
    prompt: str | None = None,
    system: str | None = None,
    assistant_prefill: str = "",
    enable_thinking: bool | None = True,
    max_new_tokens: int = 2048,
    greedy: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int | None = None,
    max_prompt_len: int = 4096,
) -> CoTTrace:
    """Sample a chain-of-thought rollout and locate its spans.

    Args:
        model: An :class:`~jlens.hf.HFLensModel` (anything with ``encode``,
            ``tokenizer``, and an ``hf_model`` exposing ``generate``).
        user: User message, rendered via :func:`chat_prompt`. Pass exactly one
            of ``user`` / ``prompt``.
        prompt: Alternatively, a fully formatted prompt string used verbatim.
        system: Optional system message (with ``user``).
        assistant_prefill: Optional open assistant prefix (with ``user``).
        enable_thinking: Chat-template thinking switch; see :func:`chat_prompt`.
        max_new_tokens: Sampling budget. Thinking traces are long; if the
            block comes back unclosed, raise this.
        greedy: Use greedy decoding. Off by default — Qwen3's model card
            recommends sampling (T=0.6, top-p=0.95, top-k=20) in thinking mode
            and warns that greedy decoding can loop.
        temperature / top_p / top_k: Sampling parameters when ``greedy`` is
            ``False``.
        seed: If set, seeds torch's global RNG so the rollout is reproducible.
        max_prompt_len: Truncation limit for the rendered prompt.

    Returns:
        A :class:`CoTTrace`. Downstream lens readouts should consume
        ``trace.input_ids`` (via ``input_ids=``), never re-encoded text.
    """
    if (user is None) == (prompt is None):
        raise ValueError("pass exactly one of user= / prompt=")
    tokenizer = model.tokenizer
    if user is not None:
        prompt = chat_prompt(
            tokenizer,
            user,
            system=system,
            assistant_prefill=assistant_prefill,
            enable_thinking=enable_thinking,
        )

    input_ids = model.encode(prompt, max_length=max_prompt_len)
    prompt_len = int(input_ids.shape[1])
    if seed is not None:
        torch.manual_seed(seed)

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not greedy,
    }
    if not greedy:
        generate_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None and eos_id is not None:
        generate_kwargs["pad_token_id"] = eos_id

    with torch.no_grad():
        full_ids = model.hf_model.generate(
            input_ids, attention_mask=torch.ones_like(input_ids), **generate_kwargs
        )
    full_ids = full_ids[:1].to("cpu")

    think_span, answer_span = locate_spans(full_ids[0].tolist(), prompt_len, tokenizer)
    return CoTTrace(
        input_ids=full_ids,
        prompt_len=prompt_len,
        think_span=think_span,
        answer_span=answer_span,
        tokenizer=tokenizer,
    )
