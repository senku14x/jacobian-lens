# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for the example scripts: model/lens loading and
single-token handling for scoring words against a tokenizer's vocabulary."""

from __future__ import annotations

import argparse

import torch

import jlens


def add_model_args(parser: argparse.ArgumentParser, *, default_model: str) -> None:
    parser.add_argument(
        "--model",
        default=default_model,
        help=f"HF model id or local path (default: {default_model})",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load the model on (default: cuda if available)",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help='HF device_map (e.g. "auto" to shard a big model); overrides --device',
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)",
    )


def add_lens_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lens",
        required=True,
        help="Fitted lens: a local .pt path, a local directory, or a HF Hub repo id",
    )
    parser.add_argument(
        "--lens-file",
        default="lens.pt",
        help="Filename inside the lens directory/repo (default: lens.pt)",
    )
    parser.add_argument("--lens-revision", default=None, help="Hub branch/tag/commit")


def load_model(args: argparse.Namespace) -> jlens.HFLensModel:
    import transformers

    kwargs: dict = {"dtype": getattr(torch, args.dtype)}
    if args.device_map is not None:
        kwargs["device_map"] = args.device_map
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.device_map is None:
        hf_model = hf_model.to(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    return jlens.from_hf(hf_model, tokenizer)


def load_lens(args: argparse.Namespace) -> jlens.JacobianLens:
    return jlens.JacobianLens.from_pretrained(
        args.lens, filename=args.lens_file, revision=args.lens_revision
    )


def encode_single(tokenizer, text: str) -> list[int]:
    """Token ids of ``text`` with no special tokens added."""
    return tokenizer(text, add_special_tokens=False).input_ids


def single_token_variants(tokenizer, word: str) -> list[int]:
    """Token ids scoring ``word``: every casing/leading-space surface variant
    that encodes to a single token, falling back to the first token of
    ``" word"`` when no variant is a single token (an approximation for
    multi-token words; noted in script output)."""
    surfaces = {word, " " + word, word.lower(), " " + word.lower()}
    if word[:1].islower():
        surfaces |= {word.capitalize(), " " + word.capitalize()}
    ids = []
    for surface in surfaces:
        encoded = encode_single(tokenizer, surface)
        if len(encoded) == 1:
            ids.append(encoded[0])
    if not ids:
        ids = [encode_single(tokenizer, " " + word)[0]]
    return sorted(set(ids))


def pick_token_id(tokenizer, word: str) -> int:
    """One token id to build a steering/swap direction for ``word``:
    the ``" word"`` form if it is a single token, else ``word``, else the
    first token of ``" word"``."""
    for surface in (" " + word, word):
        encoded = encode_single(tokenizer, surface)
        if len(encoded) == 1:
            return encoded[0]
    return encode_single(tokenizer, " " + word)[0]


#: number words for the order-ops synonym expansion (paper convention:
#: numbers score as digit and word forms; operations as symbol and word forms).
_NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
    12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
    70: "seventy", 80: "eighty", 90: "ninety", 100: "hundred",
}  # fmt: skip

_OPERATION_WORDS = {
    "addition": ["+", "plus", "add", "sum"],
    "subtraction": ["-", "minus", "subtract"],
    "multiplication": ["*", "×", "times", "multiply"],
    "division": ["/", "÷", "divided", "divide"],
    "exponentiation": ["^", "**", "power", "exponent"],
}


def expand_intermediate(word: str) -> list[str]:
    """Synonym set for an order-ops intermediate; other evals score the word
    itself."""
    expansions = [word]
    if word.isdigit() and int(word) in _NUMBER_WORDS:
        expansions.append(_NUMBER_WORDS[int(word)])
    reverse_numbers = {v: k for k, v in _NUMBER_WORDS.items()}
    if word.lower() in reverse_numbers:
        expansions.append(str(reverse_numbers[word.lower()]))
    if word.lower() in _OPERATION_WORDS:
        expansions.extend(_OPERATION_WORDS[word.lower()])
    return expansions
