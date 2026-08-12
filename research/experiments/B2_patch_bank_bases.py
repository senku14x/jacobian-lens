"""Make both banks self-describing: store base_ids (token-id list per base index)
and the minimal pair map inside each layer pack, so no consumer ever re-derives
prompts from builder code.

The bug this kills: A2_A5_model.py reconstructed base prompts from the ORIGINAL
builder's template file; run against the broadened bank its base indices pointed
at the wrong (shorter) prompts and it crashed on a position index. Silent wrong
answers were one tokenizer quirk away.

Verification is by re-measurement, not by construction: for N_VERIFY native-eps
D4 rows per pack, recompute h_l(x')[p] - h_l(x)[p] from the reconstructed pair
with the real model and require cos > 0.999 against the stored dir*eps. Runs the
model; needs the GPU.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ekko import harness as H  # noqa: E402

MODEL = os.environ.get("EKKO_MODEL", "Qwen/Qwen3.6-27B")
LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
N_VERIFY = int(os.environ.get("EKKO_N_VERIFY", "8"))
DEV = "cuda:0"

BANKS = {"research/outputs/003_bank": "orig", "research/outputs/B1_bank": "broad"}


def orig_pairs(tok, prefix_ids):
    """Reproduce 003_bank's pair construction (flexible-generalization.json)."""
    cats = json.load(open("data/experiments/flexible-generalization.json"))["categories"]
    pairs = []
    for cat in cats:
        for fn in cat["funcs"]:
            enc = {a: prefix_ids + tok.encode(fn["template"].replace("{arg}", a),
                                              add_special_tokens=False) for a in cat["args"]}
            ln = {a: len(v) for a, v in enc.items()}
            for a in cat["args"]:
                for b in cat["args"]:
                    if a == b or ln[a] != ln[b]:
                        continue
                    ia, ib = enc[a], enc[b]
                    if any(x != y for x, y in zip(ia, ib)):
                        pairs.append({"ids_a": ia, "ids_b": ib, "alt": b})
    return pairs, None


def broad_pairs(tok, prefix_ids):
    """Reproduce B1_bank_broad's pair construction (imports the builder)."""
    from B1_bank_broad import build_pairs, MAX_BASES
    pairs, _ = build_pairs(tok, prefix_ids)
    return pairs, MAX_BASES


@torch.no_grad()
def main() -> None:
    t0 = time.time()
    model, hf, tok = H.load_model(MODEL)
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    doc = next(r["text"] for r in ds if len(r["text"]) > 2000)
    prefix_ids = tok.encode(doc, add_special_tokens=False)[:96]
    print(f"model loaded ({time.time()-t0:.0f}s)", flush=True)

    for bank, kind in BANKS.items():
        pairs, cap = (orig_pairs if kind == "orig" else broad_pairs)(tok, prefix_ids)
        bykey: dict[tuple, list] = {}
        for p in pairs:
            bykey.setdefault(tuple(p["ids_a"]), []).append(p)
        keys = list(bykey)[: cap or len(bykey)]
        base_ids = [list(k) for k in keys]
        print(f"\n== {bank} ({kind}): {len(base_ids)} bases, {len(pairs)} pairs", flush=True)

        for layer in (16, 31, 46):
            path = f"{bank}/{LENS_DIR}_L{layer}.pt"
            pack = torch.load(path, weights_only=False)
            meta = pack["meta"]
            n_base = max(m["base"] for m in meta) + 1
            assert n_base <= len(base_ids), f"{path}: {n_base} bases > {len(base_ids)}"

            nat = [i for i, m in enumerate(meta) if m["family"] == "D4" and m["eps"] < 0]
            step = max(1, len(nat) // N_VERIFY)
            nchk, wcos = 0, 1.0
            for i in nat[::step][:N_VERIFY]:
                m = meta[i]
                cand = [p for p in bykey[keys[m["base"]]] if p["alt"] == m["tag"]]
                if not cand:
                    continue
                p = cand[0]
                _, aa = H.capture(model, torch.tensor([p["ids_a"]], device=model.input_device))
                _, ab = H.capture(model, torch.tensor([p["ids_b"]], device=model.input_device))
                truth = (ab[layer][0, m["pos"]] - aa[layer][0, m["pos"]]).float().cpu()
                got = pack["dir"][i].float() * pack["eps"][i].float()
                c = torch.nn.functional.cosine_similarity(truth, got, dim=0).item()
                wcos, nchk = min(wcos, c), nchk + 1
            print(f"  L{layer}: re-measured {nchk} native D4 deltas, worst cos={wcos:.6f}",
                  flush=True)
            assert nchk >= max(3, N_VERIFY // 2), f"{path}: too few verifiable rows ({nchk})"
            assert wcos > 0.999, f"{path}: stored deltas do not match re-measurement"

            pack["base_ids"] = base_ids
            torch.save(pack, path)
            print(f"  L{layer}: base_ids[{len(base_ids)}] written ({time.time()-t0:.0f}s)",
                  flush=True)

    print(f"\ndone ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
