# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Chain-of-thought plumbing: chat templating, span location, generation.

Uses a fake Qwen3-shaped tokenizer (single-token ``<think>`` / ``</think>``
markers) and a stub ``generate`` so everything runs on CPU with no downloads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from jlens.cot import (
    CoTTrace,
    chat_prompt,
    generate_cot,
    locate_spans,
    think_marker_ids,
)

BOS, THINK, END_THINK, EOS = 0, 1, 2, 3


class FakeThinkTokenizer:
    """Vocab: 0=bos 1=<think> 2=</think> 3=eos; 10+ are words 'w10', 'w11'…"""

    unk_token_id = 9
    eos_token_id = EOS
    pad_token_id = None

    def __init__(self) -> None:
        self.chat_calls: list[dict] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<think>": THINK, "</think>": END_THINK}.get(token, self.unk_token_id)

    def decode(self, ids, skip_special_tokens: bool = False, **_kw) -> str:
        parts = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id < 10:
                if not skip_special_tokens:
                    parts.append(f"<{token_id}>")
            else:
                parts.append(f"w{token_id} ")
        return "".join(parts).strip()

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
        # 3 fixed prompt tokens regardless of text (the stub model ignores text).
        return SimpleNamespace(input_ids=torch.tensor([[BOS, 10, 11]]))

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
        **kwargs,
    ) -> str:
        self.chat_calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "continue_final_message": continue_final_message,
                **kwargs,
            }
        )
        rendered = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        return rendered + ("[assistant]" if add_generation_prompt else "")


class FakeLensModel:
    """Duck-typed stand-in for HFLensModel: encode + tokenizer + hf_model."""

    def __init__(self, completion: list[int]) -> None:
        self.tokenizer = FakeThinkTokenizer()
        self.generate_kwargs: dict = {}
        outer = self

        class _Gen:
            def generate(self, input_ids, attention_mask=None, **kwargs):
                outer.generate_kwargs = kwargs
                assert attention_mask.shape == input_ids.shape
                tail = torch.tensor([completion], dtype=input_ids.dtype)
                return torch.cat([input_ids, tail], dim=1)

        self.hf_model = _Gen()

    def encode(self, text: str, *, max_length: int = 128) -> torch.Tensor:
        return self.tokenizer(text, max_length=max_length).input_ids


def test_think_marker_ids_found_and_missing():
    assert think_marker_ids(FakeThinkTokenizer()) == (THINK, END_THINK)

    class NoMarkers:
        unk_token_id = 9

        def convert_tokens_to_ids(self, token):
            return 9  # everything is unk

    assert think_marker_ids(NoMarkers()) is None


def test_chat_prompt_passes_enable_thinking():
    tok = FakeThinkTokenizer()
    rendered = chat_prompt(tok, "hi", system="sys", enable_thinking=True)
    assert rendered == "[system]sys[user]hi[assistant]"
    call = tok.chat_calls[-1]
    assert call["enable_thinking"] is True
    assert call["add_generation_prompt"] is True

    chat_prompt(tok, "hi")  # None -> kwarg omitted entirely
    assert "enable_thinking" not in tok.chat_calls[-1]

    chat_prompt(tok, "hi", assistant_prefill="I think")
    call = tok.chat_calls[-1]
    assert call["continue_final_message"] is True
    assert call["messages"][-1] == {"role": "assistant", "content": "I think"}


def test_locate_spans_closed_block():
    tok = FakeThinkTokenizer()
    #        prompt....      <think> 20 21  </think>   30   eos
    ids = [BOS, 10, 11, THINK, 20, 21, END_THINK, 30, EOS]
    think, answer = locate_spans(ids, prompt_len=3, tokenizer=tok)
    assert think == (4, 6)
    assert answer == (7, 9)


def test_locate_spans_no_block_and_unclosed():
    tok = FakeThinkTokenizer()
    ids = [BOS, 10, 11, 30, 31, EOS]
    think, answer = locate_spans(ids, prompt_len=3, tokenizer=tok)
    assert think is None
    assert answer == (3, 6)

    ids = [BOS, 10, 11, THINK, 20, 21]  # ran out of budget mid-thought
    think, answer = locate_spans(ids, prompt_len=3, tokenizer=tok)
    assert think == (4, 6)
    assert answer == (6, 6)

    # A *closed* <think> pair in the prompt (Qwen3 enable_thinking=False
    # inserts an empty one) must not count as completion thinking.
    ids = [THINK, 10, END_THINK, 30, 31]
    think, answer = locate_spans(ids, prompt_len=3, tokenizer=tok)
    assert think is None
    assert answer == (3, 5)


def test_locate_spans_prompt_prefilled_open_marker():
    """QwQ-style templates (and manual assistant prefills) open the <think>
    block in the prompt; the completion then starts mid-thought."""
    tok = FakeThinkTokenizer()
    # prompt ends with <think>; completion: 20 21 </think> 30
    ids = [BOS, 10, THINK, 20, 21, END_THINK, 30]
    think, answer = locate_spans(ids, prompt_len=3, tokenizer=tok)
    assert think == (3, 5)
    assert answer == (6, 7)

    # Dangling open marker and the completion never closes it.
    ids = [BOS, 10, THINK, 20, 21]
    think, answer = locate_spans(ids, prompt_len=3, tokenizer=tok)
    assert think == (3, 5)
    assert answer == (5, 5)

    # A closed pair earlier in the prompt plus a dangling open still counts.
    ids = [THINK, END_THINK, 10, THINK, 20, END_THINK, 30]
    think, answer = locate_spans(ids, prompt_len=4, tokenizer=tok)
    assert think == (4, 5)
    assert answer == (6, 7)


def test_generate_cot_end_to_end_spans_and_text():
    model = FakeLensModel(completion=[THINK, 20, 21, END_THINK, 30, EOS])
    trace = generate_cot(model, "why?", seed=0, max_new_tokens=16)

    assert trace.prompt_len == 3
    assert trace.total_len == 9
    assert trace.think_span == (4, 6)
    assert trace.answer_span == (7, 9)
    assert trace.think_positions == [4, 5]
    assert trace.thinking_text == "w20 w21"
    assert trace.answer_text == "w30"
    assert torch.equal(
        trace.completion_ids, torch.tensor([THINK, 20, 21, END_THINK, 30, EOS])
    )
    # Sampling defaults flow through; pad falls back to eos.
    assert model.generate_kwargs["do_sample"] is True
    assert model.generate_kwargs["temperature"] == pytest.approx(0.6)
    assert model.generate_kwargs["pad_token_id"] == EOS


def test_generate_cot_greedy_and_arg_validation():
    model = FakeLensModel(completion=[30, EOS])
    trace = generate_cot(model, prompt="raw text", greedy=True)
    assert model.generate_kwargs["do_sample"] is False
    assert "temperature" not in model.generate_kwargs
    assert trace.think_span is None
    assert trace.answer_text == "w30"

    with pytest.raises(ValueError, match="exactly one"):
        generate_cot(model)
    with pytest.raises(ValueError, match="exactly one"):
        generate_cot(model, "user text", prompt="also a prompt")


def test_cot_trace_is_plain_data():
    trace = CoTTrace(
        input_ids=torch.tensor([[BOS, 10, 11, 30]]),
        prompt_len=3,
        think_span=None,
        answer_span=(3, 4),
        tokenizer=FakeThinkTokenizer(),
    )
    assert trace.completion_text == "w30"
    assert trace.thinking_text == ""
    assert trace.think_positions == []
