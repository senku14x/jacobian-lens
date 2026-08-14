"""E2: equal-footing evaluation of every operator on the matched layer grid.

Operators (all under IDENTICAL conditions -- same items, same layer grid, same
readout plumbing through the model's own final norm + unembedding):
    logit   identity transport
    J_rel   released J-lens          (restricted to the grid)
    R_rel   released R-lens          (restricted to the grid)
    R_own   E1's R refit             (control: rules vs fit-instance noise)
    H       E1's homogenized lens    (the F11 prediction under test)
plus H_halfA/H_halfB and R_own halves, whose pass@10 gap is the first
READABILITY twin-fit noise floor in this line of work -- an H vs R_own margin
only counts if it clears it.

Items: the permanent 59-item calibration slice, reproduced from 002 exactly
(seed 0, 10/category; the frozen ~85% of each eval set stays untouched until a
final one-shot report).

Metrics per operator:
    pass@10 all-grid / first-half-grid (l < target//2), per category and ALL
    mean earliest grid layer reaching top-10
    trash-token rate in the top-10 (fixed heuristic, first-half and all)
    M4 skip-ahead guardrail (multihop + order-ops, single-token targets):
        (a) rate at which the ANSWER outranks the best intermediate at grid
            layers BELOW the intermediate's first top-10 layer
        (b) answer pass@10 on the first-half grid
        A readability "win" with an M4 regression is the tuned-lens pathology
        and is rejected regardless of pass@10.
Paired bootstrap (2000, over items) for the pre-registered contrasts:
    H - R_own, H - R_rel, R_own - R_rel  (first-half pass@10 and all-grid).

PRE-REGISTERED CALL (FINDINGS.md O1): H > R on first-half pass@10 with no M4
regression => conservation account supported. H ~= R_own => composition breaks
the account; report that.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ekko import harness as H  # noqa: E402

MODEL = os.environ.get("EKKO_MODEL", "Qwen/Qwen3.6-27B")
LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
E1 = os.environ.get("EKKO_E1", "/home/ubuntu/ekko/outputs/E1")
OUT = os.environ.get("EKKO_OUT", "/home/ubuntu/ekko/outputs/E2")
N_CALIB = int(os.environ.get("EKKO_N_CALIB", "10"))
N_BOOT = 2000
SEED = 0
SETS = ["multihop", "multilingual", "order-ops", "poetry", "typo", "association"]
M4_SETS = {"multihop", "order-ops"}

_ALNUM = re.compile(r"[A-Za-z0-9]")


def is_trash(s: str) -> bool:
    """Fixed heuristic, defined once: no alphanumeric content, or replacement
    chars, after stripping whitespace."""
    t = s.strip()
    return (not t) or ("�" in t) or (_ALNUM.search(t) is None)


def readout_position(tok, prompt: str, slug: str) -> int:
    ids = tok.encode(prompt, add_special_tokens=False)
    if slug != "poetry":
        return len(ids) - 1
    nl = [i for i, t in enumerate(ids) if "\n" in tok.decode([t])]
    return nl[-1] if nl else len(ids) - 1


def target_ids(tok, intermediate) -> list[int]:
    forms = intermediate if isinstance(intermediate, list) else [intermediate]
    out = []
    for f in forms:
        if isinstance(f, str):
            tid = H.single_token_id(tok, f)
            if tid is not None:
                out.append(tid)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    model, hf, tok = H.load_model(MODEL)
    jl = H.load_released_lens(LENS_DIR, "j")
    rl = H.load_released_lens(LENS_DIR, "r")
    tgt = jl.target_layer
    half = tgt // 2

    if os.environ.get("EKKO_GRID"):
        grid = [int(x) for x in os.environ["EKKO_GRID"].split(",")]
    else:
        e1 = json.load(open(f"{E1}/{LENS_DIR}.json"))
        grid = e1["variants"]["H"]["source_layers"]
    ops: dict[str, dict[int, torch.Tensor]] = {
        "logit": {l: None for l in grid},
        "J_rel": {l: jl.jacobians[l] for l in grid},
        "R_rel": {l: rl.jacobians[l] for l in grid},
    }
    for var, name in (("H", "H"), ("R", "R_own")):
        for suffix in ("all", "halfA", "halfB"):
            p = f"{E1}/{LENS_DIR}_{var}_{suffix}.pt"
            if os.path.exists(p):
                d = torch.load(p, map_location="cpu", weights_only=True)
                key = name if suffix == "all" else f"{name}_{suffix}"
                ops[key] = {l: d[l] for l in grid if l in d}
    ops = {k: v for k, v in ops.items() if len(v) == len(grid)}
    if "R_own" not in ops and "R_own_halfA" in ops:
        # half-A-only R fit: same 13 prompts as H_halfA, so the paired
        # comparison is H_halfA vs R_own (matched corpus and n).
        ops["R_own"] = ops["R_own_halfA"]
    print(f"grid={grid}  operators={list(ops)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- calibration slice, exactly 002's draw --------------------------
    g = torch.Generator().manual_seed(SEED)
    work = []
    for slug in SETS:
        items = json.load(open(f"data/evaluations/lens-eval-{slug}.json"))["items"]
        perm = torch.randperm(len(items), generator=g)[:N_CALIB].tolist()
        for i in perm:
            it = items[i]
            groups = [target_ids(tok, x) for x in it["intermediates"]]
            groups = [g_ for g_ in groups if g_]
            if not groups:
                continue
            ans = target_ids(tok, it.get("target")) if slug in M4_SETS else []
            work.append({"slug": slug, "name": it["name"], "prompt": it["prompt"],
                         "pos": readout_position(tok, it["prompt"], slug),
                         "groups": groups, "answer": ans})
    print(f"calibration slice: {len(work)} items", flush=True)

    # ---- one clean forward per item -------------------------------------
    resid = torch.zeros(len(work), len(grid), model.d_model, dtype=torch.float32)
    for n, w in enumerate(work):
        _, acts = H.capture(model, w["prompt"], max_length=512)
        for li, l in enumerate(grid):
            resid[n, li] = acts[l][0, w["pos"]].float().cpu()
    print(f"captured {len(work)} items  ({time.time()-t0:.0f}s)", flush=True)

    # ---- ranks + trash rate per operator --------------------------------
    dev = model.input_device
    maxg = max(len(w["groups"]) for w in work)
    R = {}          # op -> [n_items, n_grid, maxg] intermediate ranks
    RA = {}         # op -> [n_items, n_grid] answer rank (or big)
    trash = {}      # op -> [n_grid] mean trash fraction of top-10
    for name, mats in ops.items():
        r = torch.full((len(work), len(grid), maxg), 10**9, dtype=torch.long)
        ra = torch.full((len(work), len(grid)), 10**9, dtype=torch.long)
        tr = torch.zeros(len(grid))
        for li, l in enumerate(grid):
            T = None if mats[l] is None else mats[l].to(dev, torch.float32)
            with torch.no_grad():
                x = resid[:, li].to(dev)
                logits = model.unembed(x if T is None else x @ T.T).float()
                top10 = logits.topk(10, dim=-1).indices
                tr[li] = float(torch.tensor(
                    [sum(is_trash(tok.decode([int(t)])) for t in row) / 10.0
                     for row in top10]).mean())
                for n, w in enumerate(work):
                    for gi, tids in enumerate(w["groups"]):
                        t = torch.tensor(tids, device=dev)
                        r[n, li, gi] = int((logits[n] > logits[n, t].max()).sum())
                    if w["answer"]:
                        t = torch.tensor(w["answer"], device=dev)
                        ra[n, li] = int((logits[n] > logits[n, t].max()).sum())
            del T
        R[name], RA[name], trash[name] = r, ra, tr
        print(f"  ranked {name}  ({time.time()-t0:.0f}s)", flush=True)

    first_idx = [li for li, l in enumerate(grid) if l < half]

    def per_item_passk(r, w, idx, k=10):
        m = r[idx].min(0).values[: len(w["groups"])]
        return (m < k).float().mean().item()

    # ---- assemble report -------------------------------------------------
    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "grid": grid, "first_half_grid": [grid[i] for i in first_idx],
           "n_items": len(work), "operators": list(ops), "results": {}, "m4": {},
           "contrasts": {}, "twin_floor_pass10": {}}

    item_vals = {}   # (op, span) -> [n_items] pass@10
    for name in ops:
        for span, idx in (("all", list(range(len(grid)))), ("first_half", first_idx)):
            item_vals[(name, span)] = [per_item_passk(R[name][n], w, idx)
                                       for n, w in enumerate(work)]
    for slug in SETS + ["ALL"]:
        sel = [n for n, w in enumerate(work) if slug == "ALL" or w["slug"] == slug]
        rep["results"][slug] = {"n": len(sel)}
        for name in ops:
            rep["results"][slug][name] = {
                span: sum(item_vals[(name, span)][i] for i in sel) / len(sel)
                for span in ("all", "first_half")}

    for name in ops:
        firsts = []
        for n, w in enumerate(work):
            for gi in range(len(w["groups"])):
                hit = (R[name][n, :, gi] < 10).nonzero()
                if len(hit):
                    firsts.append(grid[int(hit[0])])
        rep["results"]["ALL"][name]["mean_first_top10_layer"] = (
            sum(firsts) / max(len(firsts), 1))
        rep["results"]["ALL"][name]["n_ever_top10"] = len(firsts)
        rep["results"]["ALL"][name]["trash_first_half"] = float(
            trash[name][first_idx].mean())
        rep["results"]["ALL"][name]["trash_all"] = float(trash[name].mean())

    # ---- M4 skip-ahead ---------------------------------------------------
    m4_items = [n for n, w in enumerate(work) if w["answer"]]
    for name in ops:
        outrank, apass = [], []
        for n in m4_items:
            w = work[n]
            best_i = R[name][n, :, : len(w["groups"])].min(-1).values  # per layer
            hit = (best_i < 10).nonzero()
            below = list(range(int(hit[0]))) if len(hit) else list(range(len(grid)))
            if below:
                outrank.append(float((RA[name][n, below] <
                                      best_i[below]).float().mean()))
            apass.append(float((RA[name][n, first_idx].min() < 10)))
        rep["m4"][name] = {
            "answer_outranks_below_first_intermediate": sum(outrank) / max(len(outrank), 1),
            "answer_pass10_first_half": sum(apass) / max(len(apass), 1),
            "n_items": len(m4_items)}

    # ---- pre-registered contrasts with paired bootstrap ------------------
    gboot = torch.Generator().manual_seed(3)
    def contrast(a, b, span):
        va = torch.tensor(item_vals[(a, span)])
        vb = torch.tensor(item_vals[(b, span)])
        d = []
        for _ in range(N_BOOT):
            pick = torch.randint(len(va), (len(va),), generator=gboot)
            d.append((va[pick] - vb[pick]).mean().item())
        d = torch.tensor(d)
        return {"delta": float((va - vb).mean()),
                "lo": float(torch.quantile(d, .025)),
                "hi": float(torch.quantile(d, .975)),
                "ci_clear": bool(torch.quantile(d, .025) > 0
                                 or torch.quantile(d, .975) < 0)}
    for a, b in (("H", "R_own"), ("H_halfA", "R_own"), ("H", "R_rel"),
                 ("R_own", "R_rel"), ("H", "J_rel"), ("R_rel", "J_rel")):
        if a in ops and b in ops:
            for span in ("first_half", "all"):
                rep["contrasts"][f"{a}-{b}|{span}"] = contrast(a, b, span)

    for name in ("H", "R_own"):
        ha, hb = f"{name}_halfA", f"{name}_halfB"
        if ha in ops and hb in ops:
            rep["twin_floor_pass10"][name] = {
                span: abs(sum(item_vals[(ha, span)]) / len(work)
                          - sum(item_vals[(hb, span)]) / len(work))
                for span in ("first_half", "all")}

    # ---- print -----------------------------------------------------------
    main_ops = [o for o in ("logit", "J_rel", "R_rel", "R_own", "H") if o in ops]
    print(f"\n{'operator':<10s} {'all':>7s} {'1st½':>7s} {'first@10':>9s} "
          f"{'trash½':>7s} {'M4-out':>7s} {'M4-a½':>7s}", flush=True)
    for name in main_ops:
        r_ = rep["results"]["ALL"][name]
        m4 = rep["m4"].get(name, {})
        print(f"{name:<10s} {r_['all']:>7.3f} {r_['first_half']:>7.3f} "
              f"{r_['mean_first_top10_layer']:>9.1f} {r_['trash_first_half']:>7.3f} "
              f"{m4.get('answer_outranks_below_first_intermediate', float('nan')):>7.3f} "
              f"{m4.get('answer_pass10_first_half', float('nan')):>7.3f}", flush=True)
    for k, v in rep["contrasts"].items():
        star = " *" if v["ci_clear"] else ""
        print(f"  {k:<24s} {v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]{star}",
              flush=True)
    for name, v in rep["twin_floor_pass10"].items():
        print(f"  twin pass@10 floor {name}: first_half {v['first_half']:.4f}  "
              f"all {v['all']:.4f}", flush=True)

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
