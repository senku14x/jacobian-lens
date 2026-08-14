"""E3: Patchscopes baseline on the same items and layer grid as E2.

The strongest published competitor for early-layer readout is not a linear
lens at all: patch the activation into a few-shot identity prompt and let the
MODEL decode it (Patchscopes, Ghandeharioun et al. 2024). Nonlinear, needs no
training, and is not bound by this project's fixed-operator negatives. We have
zero numbers on it; this fills that hole.

Protocol (same-layer identity patchscope):
    decode prompt   "cat -> cat\n1135 -> 1135\nhello -> hello\n? ->"
    for each eval item and grid layer l: take h_l at the item's readout
    position (same capture convention as E2), overwrite the residual at the
    "?" position at layer l of the decode prompt, greedy-generate 8 tokens.
    Score: any intermediate form appears (case-insensitive substring) in the
    generation.

Metric note (stated, not hidden): "surfaced in an 8-token generation" is not
identical to "rank<=10 in a vocabulary readout". For comparability the E2
lenses' analogue is pass@10; treat cross-instrument gaps < ~10 points as
inconclusive. What IS decisive: if patchscopes surfaces intermediates in the
first half at several times the rate of every linear lens, the better-lens
question changes shape (decode-side, not transport-side).
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
OUT = os.environ.get("EKKO_OUT", "/home/ubuntu/ekko/outputs/E3")
GRID = [int(x) for x in os.environ.get(
    "EKKO_GRID", "8,12,16,20,24,28,31,36,40,46,52,58,60").split(",")]
N_CALIB = int(os.environ.get("EKKO_N_CALIB", "10"))
N_NEW = 8
SEED = 0
SETS = ["multihop", "multilingual", "order-ops", "poetry", "typo", "association"]
HALF = 31


def readout_position(tok, prompt: str, slug: str) -> int:
    ids = tok.encode(prompt, add_special_tokens=False)
    if slug != "poetry":
        return len(ids) - 1
    nl = [i for i, t in enumerate(ids) if "\n" in tok.decode([t])]
    return nl[-1] if nl else len(ids) - 1


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    model, hf, tok = H.load_model(MODEL)
    dev = model.input_device

    prefix = "cat -> cat\n1135 -> 1135\nhello -> hello\n"
    ids_prefix = tok.encode(prefix, add_special_tokens=False)
    ids_q = tok.encode("?", add_special_tokens=False)
    ids_arrow = tok.encode(" ->", add_special_tokens=False)
    patch_pos = len(ids_prefix)
    decode_ids = torch.tensor([ids_prefix + ids_q + ids_arrow], device=dev)
    print(f"decode prompt: {len(decode_ids[0])} toks, patch at {patch_pos} "
          f"('?' = {ids_q})", flush=True)

    # same slice draw as E2
    g = torch.Generator().manual_seed(SEED)
    work = []
    for slug in SETS:
        items = json.load(open(f"data/evaluations/lens-eval-{slug}.json"))["items"]
        perm = torch.randperm(len(items), generator=g)[:N_CALIB].tolist()
        for i in perm:
            it = items[i]
            forms = []
            for x in it["intermediates"]:
                forms.extend(x if isinstance(x, list) else [x])
            forms = [f.strip().lower() for f in forms if isinstance(f, str) and f.strip()]
            if not forms:
                continue
            work.append({"slug": slug, "name": it["name"], "prompt": it["prompt"],
                         "pos": readout_position(tok, it["prompt"], slug),
                         "forms": forms})
    print(f"{len(work)} items  ({time.time()-t0:.0f}s)", flush=True)

    class Patch:
        """Overwrite block `layer`'s output at patch_pos during prefill."""

        def __init__(self, layer):
            self.layer, self.h, self.handle = layer, None, None

        def __enter__(self):
            def fn(mod, args, out):
                t = out if torch.is_tensor(out) else out[0]
                if t.shape[1] > patch_pos:          # prefill only
                    t[:, patch_pos] = self.h.to(t.dtype)
                return out
            self.handle = model.layers[self.layer].register_forward_hook(fn)
            return self

        def __exit__(self, *e):
            self.handle.remove()

    hits = torch.zeros(len(work), len(GRID))
    gens_sample = []
    for n, w in enumerate(work):
        _, acts = H.capture(model, w["prompt"], max_length=512)
        for li, l in enumerate(GRID):
            patch = Patch(l)
            patch.h = acts[l][0, w["pos"]]
            with patch, torch.no_grad():
                out = hf.generate(decode_ids, max_new_tokens=N_NEW,
                                  do_sample=False,
                                  pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0, decode_ids.shape[1]:]).lower()
            hits[n, li] = float(any(f in text for f in w["forms"]))
            if n < 3 and li in (1, 6):
                gens_sample.append(f"{w['name']}@L{l}: {text[:60]!r}")
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(work)}  ({time.time()-t0:.0f}s)", flush=True)

    first_idx = [i for i, l in enumerate(GRID) if l < HALF]
    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "grid": GRID, "n_items": len(work), "n_new_tokens": N_NEW,
           "metric": "intermediate substring in 8-token greedy generation",
           "results": {}, "samples": gens_sample}
    print(f"\n{'set':<14s} {'n':>3s} {'all':>7s} {'first½':>7s}", flush=True)
    for slug in SETS + ["ALL"]:
        sel = [n for n, w in enumerate(work) if slug == "ALL" or w["slug"] == slug]
        if not sel:
            continue
        s = hits[sel]
        row = {"n": len(sel),
               "all": float(s.max(1).values.mean()),
               "first_half": float(s[:, first_idx].max(1).values.mean()),
               "per_layer": {str(GRID[i]): float(s[:, i].mean())
                             for i in range(len(GRID))}}
        rep["results"][slug] = row
        print(f"{slug:<14s} {len(sel):>3d} {row['all']:>7.3f} "
              f"{row['first_half']:>7.3f}", flush=True)
    print("\nper-layer surface rate (ALL): " + "  ".join(
        f"L{GRID[i]}={hits[:, i].mean():.2f}" for i in range(len(GRID))), flush=True)

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
