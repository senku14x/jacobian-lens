# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Sample a Qwen3 chain-of-thought rollout and render the lens slice over it.

Generates a thinking-mode completion for a question (or a bundled
``jlens.examples`` slug), locates the ``<think>`` span, reads the Jacobian
lens out over the exact generated tokens, and writes the interactive
layer x position page. The console gets the trace text, the span indices,
and a J-lens vs logit-lens readout at the answer boundary.

Examples::

    python scripts/cot_slice.py --lens lenses/qwen3-8b_n1000.pt \\
        --example cot-two-hop --out slices/cot-two-hop

    python scripts/cot_slice.py --lens lenses/qwen3-8b_n1000.pt \\
        --question "Is 977 prime? Answer yes or no." \\
        --pin "prime,977,31" --out slices/prime
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    add_lens_args,
    add_model_args,
    load_lens,
    load_model,
    single_token_variants,
)

import jlens  # noqa: E402
from jlens.cot import generate_cot  # noqa: E402
from jlens.examples import EXAMPLES  # noqa: E402
from jlens.vis import build_page, compute_slice  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser, default_model="Qwen/Qwen3-8B")
    add_lens_args(parser)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--question", help="User message to ask")
    what.add_argument(
        "--example",
        choices=[e.slug for e in EXAMPLES if e.user is not None],
        help="A bundled chat example slug (the cot-* ones are thinking-mode)",
    )
    parser.add_argument("--system", default=None)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable the <think> block (Qwen3 enable_thinking=False)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--greedy", action="store_true", help="Greedy decoding (see jlens.cot docs)"
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output directory for the page")
    parser.add_argument(
        "--pin",
        default="",
        help="Comma-separated words to pin rank-tracking charts for",
    )
    parser.add_argument("--layer-stride", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--max-tracked", type=int, default=512, help="Cap on tracked tokens"
    )
    parser.add_argument(
        "--full-prompt",
        action="store_true",
        help="Slice the whole sequence (default: completion tokens only)",
    )
    parser.add_argument(
        "--no-mask-display",
        action="store_true",
        help="Show raw top-K (default masks the display to word-like tokens, "
        "which reads better on Qwen vocabularies)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jlens.configure_logging()
    model = load_model(args)
    lens = load_lens(args)

    if args.example:
        example = next(e for e in EXAMPLES if e.slug == args.example)
        user, system = example.user, example.system or args.system
        enable_thinking = (
            False
            if args.no_thinking
            else (True if example.enable_thinking is None else example.enable_thinking)
        )
        title, description = example.section, example.description
    else:
        user, system = args.question, args.system
        enable_thinking = not args.no_thinking
        title, description = "CoT slice", user

    trace = generate_cot(
        model,
        user,
        system=system,
        enable_thinking=enable_thinking,
        max_new_tokens=args.max_new_tokens,
        greedy=args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(
        f"prompt: {trace.prompt_len} tokens; completion: "
        f"{trace.total_len - trace.prompt_len} tokens"
    )
    print(f"think_span={trace.think_span} answer_span={trace.answer_span}")
    if trace.think_span:
        print(f"\n--- thinking ---\n{trace.thinking_text}")
    print(f"\n--- answer ---\n{trace.answer_text}\n")
    if trace.think_span and trace.answer_span[0] == trace.answer_span[1]:
        print("NOTE: <think> block unclosed; raise --max-new-tokens for an answer\n")

    pinned: set[int] = set()
    for word in filter(None, (w.strip() for w in args.pin.split(","))):
        pinned.update(single_token_variants(model.tokenizer, word))

    slice_data = compute_slice(
        model,
        lens,
        input_ids=trace.input_ids,
        top_n=args.top_n,
        max_tracked=args.max_tracked,
        pinned_token_ids=pinned,
        layer_stride=args.layer_stride,
        last_n_tokens=None if args.full_prompt else trace.total_len - trace.prompt_len,
        mask_display=not args.no_mask_display,
    )

    out_dir = Path(args.out)
    page, _, _ = build_page(
        slice_data,
        trace.tokenizer.decode(trace.input_ids[0].tolist()),
        title=title,
        description=description,
        mode="fetch",
        out_dir=out_dir,
    )
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"slice page -> {out_dir}/index.html  (serve the directory over HTTP)")

    # Quick console readout at the answer boundary: what is the model
    # "about to say" according to each lens, right before it says it?
    boundary = max(trace.answer_span[0] - 1, trace.prompt_len)
    mid_layers = [
        l for l in lens.source_layers if l in set(jlens.workspace_band(model.n_layers))
    ][:: max(1, args.layer_stride)]
    if mid_layers:
        j_logits, model_logits, _ = lens.apply(
            model, input_ids=trace.input_ids, layers=mid_layers, positions=[boundary]
        )
        base_logits, _, _ = lens.apply(
            model,
            input_ids=trace.input_ids,
            layers=mid_layers,
            positions=[boundary],
            use_jacobian=False,
        )

        def top5(logits) -> list[str]:
            return [model.tokenizer.decode([t]) for t in logits.topk(5).indices]

        print(f"\nreadout at position {boundary} (last token before the answer):")
        for layer in mid_layers:
            print(f"  L{layer:>3} logit-lens: {top5(base_logits[layer][0])}")
            print(f"  L{layer:>3} J-lens:     {top5(j_logits[layer][0])}")
        print(f"  model output:    {top5(model_logits[0])}")


if __name__ == "__main__":
    main()
