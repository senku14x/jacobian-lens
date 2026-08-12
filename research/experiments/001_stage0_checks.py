"""Stage 0 gate: instrument checks. No science claims here.

A  wrapper identity        unembed(h_final) == model logits
B  forward_from round-trip replaying blocks l+1..target on the true h_l
                           reproduces the true h_target
C  released-lens format    dict[int]->fp16 [d,d], layers 0..target, J[target] == I
D  known-answer reads      boot-country -> Italy/lira, web-spinner -> spider
E  R-vs-J qualitative      reproduce the R-lens post's published early-layer
                           examples on the released pair (replaces the §4.4
                           check that would have consumed frozen corpus C)

Gate: A, B, C must pass exactly. D and E are diagnostic — on a model with no
published R>J gap, E is a code check only.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ekko import harness as H  # noqa: E402

MODEL = os.environ.get("EKKO_MODEL", "Qwen/Qwen3.5-4B")
LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.5-4b")
OUT = os.environ.get("EKKO_OUT", "research/outputs/001_stage0")

# The R-lens post's published early-layer examples. "aganst" is verbatim from
# data/evaluations/lens-eval-typo.json (one item, used as an instrument check,
# never reported as a result). The other three are our reconstructions in the
# paper's multihop style — the post gave the concepts, not the exact prompts.
QUALITATIVE = [
    {"name": "typo/aganst", "prompt": "The final vote was nine to two aganst",
     "concept": "against", "published": "R rank-1 at L4; J never at this position"},
    {"name": "multihop/sushi", "prompt": "Fact: The capital of France is Paris.\n"
     "Fact: The currency used in the country where sushi originated is",
     "concept": "Japan", "published": "R L2 vs J L14"},
    {"name": "multihop/verona", "prompt": "Fact: The capital of France is Paris.\n"
     "Fact: The currency used in the country containing the city of Verona is",
     "concept": "Italy", "published": "R rank-1 ~L5 vs J rank >1000"},
    {"name": "assoc/jordan", "prompt": "Michael Jordan is best known for playing the sport of",
     "concept": "basketball", "published": "R L4 vs J L20"},
]

KNOWN_ANSWER = [
    ("Fact: The capital of Japan is Tokyo.\n"
     "Fact: The currency used in the country shaped like a boot is", ["Italy", "lira", "euro"]),
    ("The number of legs on the animal that spins webs is", ["spider", "eight"]),
]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    results: dict = {"model": MODEL, "lens_dir": LENS_DIR}
    print(f"=== Stage 0 checks: {MODEL} / {LENS_DIR} ===", flush=True)

    model, hf, tok = H.load_model(MODEL)
    print(f"loaded {model!r}  ({time.time()-t0:.0f}s)", flush=True)
    results["n_layers"], results["d_model"] = model.n_layers, model.d_model
    results["add_bos_token"] = getattr(tok, "add_bos_token", None)

    # ---- C: released lenses -------------------------------------------
    jl = H.load_released_lens(LENS_DIR, "j")
    rl = H.load_released_lens(LENS_DIR, "r")
    tgt = jl.target_layer
    checks = {}
    eye = torch.eye(jl.d_model)
    checks["C_format"] = (
        isinstance(jl.jacobians, dict)
        and jl.jacobians[0].dtype == torch.float16
        and jl.source_layers == list(range(tgt + 1))
        and jl.d_model == model.d_model
        and tgt == model.n_layers - 2
    )
    checks["C_anchor_is_I"] = bool(
        torch.allclose(jl.jacobians[tgt].float(), eye, atol=1e-3)
        and torch.allclose(rl.jacobians[tgt].float(), eye, atol=1e-3)
    )
    checks["C_j_r_differ"] = bool(
        (jl.jacobians[2].float() - rl.jacobians[2].float()).norm() > 1e-3
    )
    results["provenance"] = {k: jl.provenance.get(k) for k in
                             ("model_id", "dataset_id", "target_layer", "skip_first", "n_prompts")}
    print(f"C: format={checks['C_format']} anchor_I={checks['C_anchor_is_I']} "
          f"J!=R={checks['C_j_r_differ']}  target_layer={tgt}", flush=True)
    print(f"   provenance: {results['provenance']}", flush=True)

    # ---- A: wrapper identity -------------------------------------------
    ids = model.encode("The capital of France is the city of", max_length=128)
    ctx, acts = H.capture(model, ids)
    with torch.no_grad():
        ours = model.unembed(acts[model.n_layers - 1][0]).float()
        ref = hf(input_ids=ids, use_cache=False).logits[0].float()
    a_max = (ours - ref).abs().max().item()
    checks["A_wrapper_identity"] = a_max == 0.0
    print(f"A: max|unembed(h_L) - logits| = {a_max:.3e}", flush=True)

    # ---- B: forward_from round-trip ------------------------------------
    b_errs = {}
    for frac in (0.25, 0.50, 0.75):
        L = int(round(frac * tgt))
        got = H.forward_from(model, acts[L], L, ctx, target=tgt)
        b_errs[L] = (got.float() - acts[tgt].float()).abs().max().item()
    checks["B_roundtrip_exact"] = all(v == 0.0 for v in b_errs.values())
    print(f"B: round-trip max|err| per layer {b_errs}", flush=True)

    # batched round-trip: replaying B copies must give B identical answers
    hb = acts[int(round(0.5 * tgt))].expand(8, -1, -1).contiguous()
    gb = H.forward_from(model, hb, int(round(0.5 * tgt)), ctx, target=tgt)
    checks["B_batch_consistent"] = bool((gb - gb[:1]).abs().max().item() == 0.0)
    print(f"B: batch-of-8 self-consistency = {checks['B_batch_consistent']}", flush=True)

    # ---- D: known-answer reads -----------------------------------------
    d_out = []
    for prompt, wanted in KNOWN_ANSWER:
        _, a = H.capture(model, prompt)
        best = {}
        for w in wanted:
            tid = H.single_token_id(tok, w)
            if tid is None:
                best[w] = None
                continue
            ranks = [int(H.lens_ranks(model, jl, a[l][0, -1], l,
                                      torch.tensor([tid]))[0]) for l in range(tgt + 1)]
            best[w] = {"min_rank": min(ranks), "argmin_layer": int(torch.tensor(ranks).argmin())}
        d_out.append({"prompt": prompt[:60], "j_lens": best})
        print(f"D: {prompt[-45:]!r} -> {best}", flush=True)
    results["known_answer"] = d_out

    # ---- E: R vs J on the post's published examples --------------------
    e_out = []
    for case in QUALITATIVE:
        tid = H.single_token_id(tok, case["concept"])
        if tid is None:
            e_out.append({**case, "skipped": "concept not single-token"})
            continue
        _, a = H.capture(model, case["prompt"])
        row = {"name": case["name"], "concept": case["concept"], "published": case["published"]}
        for tag, lens in (("J", jl), ("R", rl)):
            ranks = [int(H.lens_ranks(model, lens, a[l][0, -1], l,
                                      torch.tensor([tid]))[0]) for l in range(tgt + 1)]
            top1 = [i for i, r in enumerate(ranks) if r == 0]
            top10 = [i for i, r in enumerate(ranks) if r < 10]
            row[tag] = {"min_rank": min(ranks), "argmin_layer": int(torch.tensor(ranks).argmin()),
                        "first_top1_layer": top1[0] if top1 else None,
                        "first_top10_layer": top10[0] if top10 else None,
                        "ranks": ranks}
        e_out.append(row)
        print(f"E: {row['name']:18s} [{case['published']}]", flush=True)
        for tag in ("J", "R"):
            s = row[tag]
            print(f"     {tag}: min_rank={s['min_rank']:<7d} @L{s['argmin_layer']:<3d} "
                  f"first_top10=L{s['first_top10_layer']} first_top1=L{s['first_top1_layer']}",
                  flush=True)
    results["qualitative"] = e_out

    results["checks"] = checks
    results["gate_pass"] = all(checks[k] for k in
                               ("A_wrapper_identity", "B_roundtrip_exact",
                                "B_batch_consistent", "C_format", "C_anchor_is_I", "C_j_r_differ"))
    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGATE {'PASS' if results['gate_pass'] else 'FAIL'}  "
          f"({time.time()-t0:.0f}s)  -> {OUT}/{LENS_DIR}.json", flush=True)


if __name__ == "__main__":
    main()
