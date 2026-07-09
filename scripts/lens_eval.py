# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Lens-quality evaluation: J-lens vs logit lens on the bundled distributions.

Runs the six ``data/evaluations/lens-eval-*.json`` prompt sets and scores
pass@k per the conventions in ``data/evaluations/README.md``: read out at a
single position (the final prompt token; for poetry, the last newline token),
take each intermediate's best (min) rank over layers, and report the mean
fraction of intermediates with rank <= k. Order-of-operations intermediates
expand to digit/word and symbol/word synonym sets.

This is the paper's "minimum viable reproduction" readout comparison
(context README §14.1): the J-lens should beat the logit lens at exposing
intermediate concepts (the bridge entity, the language, the planned rhyme)
rather than the final answer.

Example::

    python scripts/lens_eval.py --lens lenses/qwen3-8b_n1000.pt \\
        --out results/lens_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    add_lens_args,
    add_model_args,
    expand_intermediate,
    load_lens,
    load_model,
    single_token_variants,
)

import jlens  # noqa: E402
from jlens.vis import _ranks_of  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluations"
EVAL_SLUGS = ("multihop", "multilingual", "poetry", "order-ops", "association", "typo")


def readout_position(slug: str, model: jlens.LensModel, input_ids: torch.Tensor) -> int:
    """Final prompt token, except poetry: the last newline token (the end of
    the couplet's first line, where the rhyme is planned)."""
    if slug != "poetry":
        return input_ids.shape[1] - 1
    tokenizer = model.tokenizer
    for position in range(input_ids.shape[1] - 1, -1, -1):
        if "\n" in tokenizer.decode([int(input_ids[0, position])]):
            return position
    return input_ids.shape[1] - 1


def eval_one(
    slug: str,
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    ks: list[int],
    limit: int | None,
) -> dict:
    items = json.loads((EVAL_DIR / f"lens-eval-{slug}.json").read_text())["items"]
    if limit:
        items = items[:limit]
    layers = lens.source_layers
    per_item: list[dict] = []

    for item in items:
        input_ids = model.encode(item["prompt"], max_length=512)
        position = readout_position(slug, model, input_ids)
        results: dict[str, list[int]] = {}
        for lens_name, use_jacobian in (("jlens", True), ("logit", False)):
            lens_logits, _, _ = lens.apply(
                model,
                input_ids=input_ids,
                layers=layers,
                positions=[position],
                use_jacobian=use_jacobian,
            )
            best_ranks: list[int] = []
            for intermediate in item["intermediates"]:
                variant_ids = sorted(
                    {
                        token_id
                        for synonym in (
                            expand_intermediate(intermediate)
                            if slug == "order-ops"
                            else [intermediate]
                        )
                        for token_id in single_token_variants(model.tokenizer, synonym)
                    }
                )
                targets = torch.tensor(variant_ids, dtype=torch.long)
                best = min(
                    int(_ranks_of(lens_logits[layer], targets).min()) + 1  # 1-based
                    for layer in layers
                )
                best_ranks.append(best)
            results[lens_name] = best_ranks
        per_item.append(
            {"name": item["name"], "position": position, "best_ranks": results}
        )

    summary = {}
    for lens_name in ("jlens", "logit"):
        summary[lens_name] = {
            f"pass@{k}": sum(
                sum(rank <= k for rank in entry["best_ranks"][lens_name])
                / len(entry["best_ranks"][lens_name])
                for entry in per_item
            )
            / len(per_item)
            for k in ks
        }
    return {"n_items": len(per_item), "summary": summary, "items": per_item}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser, default_model="Qwen/Qwen3-8B")
    add_lens_args(parser)
    parser.add_argument(
        "--evals",
        default=",".join(EVAL_SLUGS),
        help=f"Comma-separated subset of {EVAL_SLUGS}",
    )
    parser.add_argument("--k", default="1,5,10", help="pass@k thresholds")
    parser.add_argument("--limit", type=int, default=None, help="Cap items per eval")
    parser.add_argument("--out", default=None, help="Write full results JSON here")
    args = parser.parse_args()

    jlens.configure_logging()
    model = load_model(args)
    lens = load_lens(args)
    ks = [int(k) for k in args.k.split(",")]

    all_results: dict[str, dict] = {}
    for slug in args.evals.split(","):
        slug = slug.strip()
        print(f"== lens-eval-{slug}")
        result = eval_one(slug, model, lens, ks, args.limit)
        all_results[slug] = result
        for lens_name in ("jlens", "logit"):
            scores = "  ".join(
                f"{key}={value:.3f}"
                for key, value in result["summary"][lens_name].items()
            )
            print(f"   {lens_name:>5}: {scores}   (n={result['n_items']})")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"full results -> {args.out}")


if __name__ == "__main__":
    main()
