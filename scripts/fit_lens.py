# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Fit a Jacobian lens on an open-weights model (default: Qwen/Qwen3-8B).

Paper-default calibration is 1000 WikiText sequences at 128 tokens
(quality saturates early; ~100 prompts is usable — README §"Fit"). For
Qwen3-8B (d_model=4096) each prompt costs one forward plus
``ceil(4096 / dim_batch)`` backward passes, so a full fit is hours-to-days of
single-GPU compute; shard it across GPUs with ``--shard i/N`` and combine with
``JacobianLens.merge`` (a merge snippet is printed on completion).

Examples::

    # single GPU (A100/H100-80GB: --dim-batch 16; 40GB: 8)
    python scripts/fit_lens.py --out lenses/qwen3-8b_n1000.pt

    # smoke-test scale
    python scripts/fit_lens.py --n-prompts 20 --source-layers 8,14,20,26,32 \\
        --out lenses/qwen3-8b_smoke.pt

    # 4-way shard (run one per GPU), then merge
    python scripts/fit_lens.py --shard 0/4 --out lenses/qwen3-8b_n1000.pt ...
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import add_model_args, load_model  # noqa: E402

import jlens  # noqa: E402
from jlens.examples import load_wikitext_prompts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser, default_model="Qwen/Qwen3-8B")
    parser.add_argument("--out", required=True, help="Where to save the fitted lens")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Resumable running-sum checkpoint path (default: <out>.ckpt)",
    )
    parser.add_argument(
        "--corpus",
        default="wikitext",
        help='"wikitext" (default; streams WikiText-103) or a .jsonl path with a '
        '"text" field per line',
    )
    parser.add_argument("--n-prompts", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument(
        "--dim-batch",
        type=int,
        default=8,
        help="Output dims per backward pass; the GPU-memory knob (default: 8)",
    )
    parser.add_argument(
        "--source-layers",
        default="all",
        help='"all" (default) or a comma-separated list, e.g. "8,14,20,26,32"',
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=None,
        help="Layer to take gradients to (default: final layer)",
    )
    parser.add_argument(
        "--shard",
        default=None,
        metavar="I/N",
        help="Fit only prompts i::N (0-based) for multi-GPU sharding",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Prompts between checkpoint writes; each write is "
        "n_layers * d_model^2 * 4 bytes (~2.2 GB for Qwen3-8B), so keep this "
        "high (default: 25)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile each block (faster backward; not with --device-map)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jlens.configure_logging()

    if args.corpus == "wikitext":
        prompts = load_wikitext_prompts(args.n_prompts)
    else:
        import json

        with open(args.corpus, encoding="utf-8") as f:
            prompts = [json.loads(line)["text"] for line in f if line.strip()]
        prompts = prompts[: args.n_prompts]
    if args.shard is not None:
        index, count = (int(x) for x in args.shard.split("/"))
        if not 0 <= index < count:
            raise SystemExit(f"--shard {args.shard}: need 0 <= i < N")
        prompts = prompts[index::count]
        print(f"shard {index}/{count}: {len(prompts)} prompts")

    model = load_model(args)
    if args.compile:
        for i in range(len(model.layers)):
            model.layers[i] = torch.compile(model.layers[i], dynamic=False)

    source_layers = (
        None
        if args.source_layers == "all"
        else [int(x) for x in args.source_layers.split(",")]
    )
    checkpoint = args.checkpoint or f"{args.out}.ckpt"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    lens = jlens.fit(
        model,
        prompts,
        source_layers=source_layers,
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=checkpoint,
        checkpoint_every=args.checkpoint_every,
    )

    out = args.out
    if args.shard is not None:
        index, count = (int(x) for x in args.shard.split("/"))
        root, ext = os.path.splitext(args.out)
        out = f"{root}.shard{index}of{count}{ext}"
    lens.save(out)
    print(f"saved {lens!r} -> {out}")
    if args.shard is not None:
        root, ext = os.path.splitext(args.out)
        print(
            "merge when all shards are done:\n"
            '  python -c "import glob, jlens; jlens.JacobianLens.merge('
            f"[jlens.JacobianLens.load(p) for p in sorted(glob.glob('{root}.shard*{ext}'))]"
            f").save('{args.out}')\""
        )


if __name__ == "__main__":
    main()
