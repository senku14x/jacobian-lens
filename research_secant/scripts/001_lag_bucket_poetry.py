"""001 — lag-bucket matrix-free lens on WSB poetry. Design: designs/001-lag-bucket-poetry.md (pre-registered).

Usage: python 001_lag_bucket_poetry.py [--limit N] [--layers 40,44,48,52,56] [--R 2] [--K 32]
Writes results/001-lag-bucket-poetry/{ranks.jsonl, summary.json, qualitative.md}. Resumable (skips done items).
"""
import argparse, json, os, random, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
from engine import Engine
from mfl import lag_buckets, token_rank

ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "results", "001-lag-bucket-poetry"); os.makedirs(OUT, exist_ok=True)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--layers", default="40,44,48,52,56")
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--K", type=int, default=32)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    items = json.load(open(POETRY))["items"][: a.limit]

    eng = Engine(l0=min(layers))
    tok = eng.tok
    JCB = load_lens("camilablank/workspace-lenses", "qwen3.6-27b/j-lens/lens.pt", layers)
    JNP = load_lens("neuronpedia/jacobian-lens",
                    "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt", layers)

    # ---- item captures (plain render, read at the last token containing a newline = WSB line_one_newline)
    eng.to_gpu_bf16()
    cap = []
    for it in items:
        ids = tok(it["prompt"], add_special_tokens=False)["input_ids"]
        nls = [i for i, t in enumerate(ids) if "\n" in tok.decode([t])]
        pos = nls[-1]
        word = it["intermediates"][0]
        wid = tok(" " + word, add_special_tokens=False)["input_ids"]
        X, _ = eng.capture(torch.tensor([ids]), layers)
        cap.append({"name": it["name"], "word": word, "tid": wid[0], "single": len(wid) == 1, "pos": pos,
                    "h": {l: X[l][0, pos].clone() for l in layers}})
    # ---- hosts: fresh pile docs 100-131 at full length
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    host_ids = []
    i = 100
    while len(host_ids) < a.K:
        x = tok(ds[i]["text"], truncation=True, max_length=128, add_special_tokens=False)["input_ids"]
        if len(x) == 128:
            host_ids.append(x)
        i += 1
    HX, hkw = eng.capture(torch.tensor(host_ids), layers)
    log(f"captured {len(cap)} items, {len(host_ids)} hosts (docs 100..{i-1}); single-token rhymes "
        f"{sum(c['single'] for c in cap)}/{len(cap)}")
    eng.to_cpu_bf16(); eng.build_fp32_upper()
    U = eng.U

    done = set()
    fp = os.path.join(OUT, "ranks.jsonl")
    if os.path.exists(fp):
        done = {(json.loads(x)["name"], json.loads(x)["layer"]) for x in open(fp)}
    f = open(fp, "a")
    for ci, c in enumerate(cap):
        for l in layers:
            if (c["name"], l) in done:
                continue
            h = c["h"][l]
            est, hv = lag_buckets(eng, HX[l], hkw, l, h, BUCKETS, R=a.R, seed=1000 * ci + l)
            vecs = {n: v for n, v in est.items()}
            vecs["JCB"] = JCB[l] @ h
            vecs["JNP"] = JNP[l] @ h
            vecs["LOGIT"] = h
            row = {"name": c["name"], "word": c["word"], "single": c["single"], "layer": l, "rank": {}, "top10": {},
                   "half_rank": {}}
            for n, v in vecs.items():
                sc = U @ v.to(U.device).float()
                row["rank"][n] = token_rank(sc, c["tid"])
                row["top10"][n] = [tok.decode([t]) for t in sc.topk(10).indices.tolist()]
            for n, (v1, v2) in hv.items():
                row["half_rank"][n] = [token_rank(U @ v1.to(U.device), c["tid"]), token_rank(U @ v2.to(U.device), c["tid"])]
            f.write(json.dumps(row) + "\n"); f.flush()
        log(f"item {ci+1}/{len(cap)} {c['name']} ({c['word']}): L48 ranks "
            + (" ".join(f"{k}={v}" for k, v in row["rank"].items()) if row["layer"] else ""))
    f.close()
    log("done")


if __name__ == "__main__":
    main()
