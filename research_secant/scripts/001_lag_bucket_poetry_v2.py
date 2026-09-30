"""001 v2 — lag-bucket matrix-free lens on WSB poetry, with Amendment 1-3 controls.
Design: designs/001-lag-bucket-poetry.md (Amendments 1-3). Estimator: exact single-source (Amendment 3), preceded by
the pre-committed reliability preflight (gate C: B3 host split-half vocab cos >= 0.7 at L48 on items 0-1, else
positions per host are doubled).

Per (item, layer) records: ranks of the own rhyme token, of all items' rhyme tokens (decoy null), of line-two tokens
with their lags (positive control), host-half ranks (reliability), top-10 tokens; saves readout vectors (fp16).
Usage: python 001_lag_bucket_poetry_v2.py [--positions 8] [--K 32] [--layers 40,48,56] [--limit N]
"""
import argparse, json, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
from engine import Engine
from mfl import (lag_profile_exact_halves, buckets_from_profile, tbar_uniform_halves, valid_positions, cos)

ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "results", "001-lag-bucket-poetry", "v2"); os.makedirs(OUT, exist_ok=True)
VEC = os.path.join(ROOT, "outputs", "001_v2_vecs"); os.makedirs(VEC, exist_ok=True)
POETRY = "/workspace/workspace-bench/evals/poetry/items.json"
BUCKETS = {"B0": (0, 0), "B1": (1, 3), "B2": (4, 8), "B3": (9, 16), "B4": (17, 32)}
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def load_lens(repo, fn, layers):
    from huggingface_hub import hf_hub_download
    o = torch.load(hf_hub_download(repo, fn), map_location="cpu", weights_only=False)
    J = o["J"] if "J" in o else o["jacobians"]
    if isinstance(J, dict):
        return {l: (J[l] if l in J else J[str(l)]).float() for l in layers}
    return {l: J[l].float() for l in layers}


def ranks_of(sc, ids):
    """0-based ranks of token ids in score vector sc [V]."""
    t = torch.tensor(ids, device=sc.device)
    return (sc[None, :] > sc[t][:, None]).sum(1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=8)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--layers", default="40,48,56")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--items_file", default=None, help="JSON with 'names': process only these (Amendment 4); "
                    "seeds stay tied to the bank index so reused and new cells are consistent")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    items = json.load(open(POETRY))["items"][: a.limit]
    eng = Engine(l0=min(layers)); tok = eng.tok
    JCB = load_lens("camilablank/workspace-lenses", "qwen3.6-27b/j-lens/lens.pt", layers)
    JNP = load_lens("neuronpedia/jacobian-lens",
                    "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt", layers)

    eng.to_gpu_bf16()
    cap = []
    for it in items:
        ids = tok(it["prompt"], add_special_tokens=False)["input_ids"]
        nl = [i for i, t in enumerate(ids) if "\n" in tok.decode([t])][-1]
        wid = tok(" " + it["intermediates"][0], add_special_tokens=False)["input_ids"]
        X, _ = eng.capture(torch.tensor([ids]), layers)
        pc = [(ids[q], q - 1 - nl) for q in range(nl + 1, len(ids))]           # line-two tokens and their lags
        cap.append({"name": it["name"], "word": it["intermediates"][0], "tid": wid[0], "single": len(wid) == 1,
                    "source": it.get("source"), "nl": nl, "delta_item": len(ids) - 1 - nl, "pc": pc,
                    "h": {l: X[l][0, nl].clone() for l in layers}})
    decoy_ids = sorted({c["tid"] for c in cap})
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    host_ids, i = [], 100
    while len(host_ids) < a.K:
        x = tok(ds[i]["text"], truncation=True, max_length=128, add_special_tokens=False)["input_ids"]
        if len(x) == 128:
            host_ids.append(x)
        i += 1
    HX, hkw = eng.capture(torch.tensor(host_ids), layers)
    dist = {}
    for c in cap:
        dist[c["delta_item"]] = dist.get(c["delta_item"], 0) + 1
    log(f"{len(cap)} items, {len(host_ids)} hosts (docs 100..{i-1}), delta_item distribution {dict(sorted(dist.items()))}")
    eng.to_cpu_bf16(); eng.build_fp32_upper()
    U = eng.U
    Np = len(valid_positions(128))

    # ---- preflight (gate C, pre-committed): B3 host split-half reliability at L48 on items 0-1
    npos = a.positions
    pre = {"layer": 48, "items": [c["name"] for c in cap[:2]], "positions": npos, "b3_split_half_vocab_cos": []}
    if 48 in layers:
        for c in cap[:2]:
            S, Cn = lag_profile_exact_halves(eng, HX[48], hkw, 48, c["h"][48], n_pos=npos, seed=7)
            Cd = Cn.to(S.device).clamp_min(1)
            bh = [buckets_from_profile(S[k] / Cd[k][:, None], BUCKETS, Np) for k in (0, 1)]
            pre["b3_split_half_vocab_cos"].append(float(cos(U @ bh[0]["B3"], U @ bh[1]["B3"])))
        if min(pre["b3_split_half_vocab_cos"]) < 0.7:
            npos = 2 * npos
        pre["positions_used"] = npos
        json.dump(pre, open(os.path.join(OUT, "preflight.json"), "w"), indent=1)
        log("preflight", pre)

    fp = os.path.join(OUT, "cells.jsonl")
    done = set()
    if os.path.exists(fp):
        for x in open(fp):
            r = json.loads(x); done.add((r["name"], r["layer"]))
    f = open(fp, "a")
    only = set(json.load(open(a.items_file))["names"]) if a.items_file else None
    for ci, c in enumerate(cap):
        if only is not None and c["name"] not in only:
            continue
        row = None
        for l in layers:
            if (c["name"], l) in done:
                continue
            h = c["h"][l]
            S, Cn = lag_profile_exact_halves(eng, HX[l], hkw, l, h, n_pos=npos, seed=100000 + 100 * ci + l)
            Cd = Cn.to(S.device).clamp_min(1)
            m = S.sum(0) / Cd.sum(0)[:, None]
            mh = [S[k] / Cd[k][:, None] for k in (0, 1)]
            di = c["delta_item"]
            extra = {"WITEM": (di - 1, di + 1)}
            vecs = buckets_from_profile(m, BUCKETS, Np, extra)
            halves = [buckets_from_profile(x, BUCKETS, Np, extra) for x in mh]
            fu, (fu0, fu1) = tbar_uniform_halves(eng, HX[l], hkw, l, h)
            vecs["FULL_U"] = fu.to(U.device); halves[0]["FULL_U"] = fu0.to(U.device); halves[1]["FULL_U"] = fu1.to(U.device)
            vecs["JCB"] = (JCB[l] @ h).to(U.device)
            vecs["JNP"] = (JNP[l] @ h).to(U.device)
            vecs["LOGIT"] = h.to(U.device)
            row = {"name": c["name"], "word": c["word"], "single": c["single"], "source": c["source"], "layer": l,
                   "delta_item": di, "rank": {}, "decoy_ranks": {}, "pc": {}, "half_rank": {}, "top10": {}}
            pc_ids = [t for t, _ in c["pc"]]
            for n, v in vecs.items():
                sc = U @ v.float()
                row["rank"][n] = ranks_of(sc, [c["tid"]])[0]
                row["decoy_ranks"][n] = ranks_of(sc, decoy_ids)
                row["pc"][n] = [[lag, r] for (t, lag), r in zip(c["pc"], ranks_of(sc, pc_ids))]
                row["top10"][n] = [tok.decode([t]) for t in sc.topk(10).indices.tolist()]
                if n in halves[0]:
                    row["half_rank"][n] = [ranks_of(U @ halves[k][n].float(), [c["tid"]])[0] for k in (0, 1)]
            row["decoy_ids"] = decoy_ids
            torch.save({n: v.half().cpu() for n, v in vecs.items()}, os.path.join(VEC, f"{c['name']}_L{l}.pt"))
            f.write(json.dumps(row) + "\n"); f.flush()
        if row is None:
            continue
        r = row["rank"]
        log(f"item {ci+1}/{len(cap)} {c['name']} ({c['word']}, delta {c['delta_item']}) L{l}: "
            + " ".join(f"{k}={r[k]}" for k in ("B2", "B3", "WITEM", "FULL32", "FULL_U", "JCB", "JNP")))
    f.close()
    log("done")


if __name__ == "__main__":
    main()
