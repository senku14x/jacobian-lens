# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Causal coordinate-swap evaluation on the bundled two-hop factual set.

For each of the 90 ``data/experiments/probe-swap.json`` items: check the
model answers the two-hop prompt correctly, then swap the intermediate
("bridge") concept's J-space coordinate for the counterfactual bridge across
the workspace band at every prompt position, and score whether the greedy
next token becomes the counterfactual answer (paper §7.5; the J-direction
variant of the probe-swap run, ~60% on Claude Sonnet 4.5).

``--control unrelated`` reruns each item swapping a fixed unrelated pair
(piano -> violin) instead; success there should be near zero (context README
§14.4's unrelated-swap control).

Example::

    python scripts/swap_eval.py --lens lenses/qwen3-8b_n1000.pt \\
        --out results/swap_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    add_lens_args,
    add_model_args,
    load_lens,
    load_model,
    pick_token_id,
    single_token_variants,
)

import jlens  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "experiments"


def greedy_next(model: jlens.LensModel, lens, input_ids) -> int:
    _, model_logits, _ = lens.apply(
        model, input_ids=input_ids, layers=[lens.source_layers[0]], positions=[-1]
    )
    return int(model_logits[0].argmax())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser, default_model="Qwen/Qwen3-8B")
    add_lens_args(parser)
    parser.add_argument(
        "--band",
        default=None,
        help='Swap layers: "l0,l1,..." or "START:END" fractions like "0.38:0.92" '
        "(default: the paper workspace band)",
    )
    parser.add_argument(
        "--control",
        choices=["none", "unrelated"],
        default="none",
        help="Also run the unrelated-pair control swap",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="Write per-item results here")
    args = parser.parse_args()

    jlens.configure_logging()
    model = load_model(args)
    lens = load_lens(args)

    if args.band is None:
        band = jlens.workspace_band(model.n_layers)
    elif ":" in args.band:
        start, end = (float(x) for x in args.band.split(":"))
        band = jlens.workspace_band(model.n_layers, start=start, end=end)
    else:
        band = [int(x) for x in args.band.split(",")]
    band = [l for l in band if l in lens.jacobians]
    if not band:
        raise SystemExit("no fitted layers in the requested band")
    print(f"swap band: L{band[0]}..L{band[-1]} ({len(band)} layers)")

    items = json.loads((DATA / "probe-swap.json").read_text())["items"]
    if args.limit:
        items = items[: args.limit]

    tokenizer = model.tokenizer
    results = []
    n_baseline = n_swap = n_swap_given_baseline = n_control = 0
    for item in items:
        input_ids = model.encode(item["prompt"], max_length=512)
        positions = list(range(input_ids.shape[1]))

        answer_ids = set(single_token_variants(tokenizer, item["answer"]))
        swap_answer_ids = set(single_token_variants(tokenizer, item["swap_answer"]))
        baseline_token = greedy_next(model, lens, input_ids)
        baseline_ok = baseline_token in answer_ids

        with jlens.coordinate_swap(
            model,
            lens,
            source_token_id=pick_token_id(tokenizer, item["intermediate"]),
            target_token_id=pick_token_id(tokenizer, item["swap_to"]),
            layers=band,
            positions=positions,
        ):
            swap_token = greedy_next(model, lens, input_ids)
        swap_ok = swap_token in swap_answer_ids

        control_ok = None
        if args.control == "unrelated":
            with jlens.coordinate_swap(
                model,
                lens,
                source_token_id=pick_token_id(tokenizer, "piano"),
                target_token_id=pick_token_id(tokenizer, "violin"),
                layers=band,
                positions=positions,
            ):
                control_token = greedy_next(model, lens, input_ids)
            control_ok = control_token in swap_answer_ids
            n_control += control_ok

        n_baseline += baseline_ok
        n_swap += swap_ok
        n_swap_given_baseline += swap_ok and baseline_ok
        results.append(
            {
                "name": item["name"],
                "category": item["category"],
                "baseline_ok": baseline_ok,
                "baseline_token": tokenizer.decode([baseline_token]),
                "swap_ok": swap_ok,
                "swap_token": tokenizer.decode([swap_token]),
                **({"control_ok": control_ok} if control_ok is not None else {}),
            }
        )
        print(
            f"  {item['name']:>28}  baseline={'Y' if baseline_ok else 'n'}"
            f"({results[-1]['baseline_token']!r})  "
            f"swap={'Y' if swap_ok else 'n'}({results[-1]['swap_token']!r})"
        )

    total = len(items)
    print(
        f"\nbaseline accuracy:        {n_baseline}/{total} = {n_baseline / total:.0%}"
    )
    print(f"swap success (all items): {n_swap}/{total} = {n_swap / total:.0%}")
    if n_baseline:
        print(
            f"swap success | baseline:  {n_swap_given_baseline}/{n_baseline} "
            f"= {n_swap_given_baseline / n_baseline:.0%}"
        )
    if args.control == "unrelated":
        print(
            f"unrelated-swap control:   {n_control}/{total} = {n_control / total:.0%}"
        )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"band": band, "items": results}, indent=2)
        )
        print(f"per-item results -> {args.out}")


if __name__ == "__main__":
    main()
