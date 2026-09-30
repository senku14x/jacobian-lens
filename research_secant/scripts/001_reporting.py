"""001 reporting metrics (designs/001-lag-bucket-poetry.md Amendment 5). The decision rule lives in 001_analysis.py.

(1) Anthropic §A.6: pass@k (any of layers 40/48/56), normalized AUC over log10 k in [0, 3], paired bootstrap vs FULL_U.
(2) WorkspaceBench's own poetry judge on each arm's top-10 tokens (layers 40/48/56 subset, subset50 items), incl. the
    official WSB J arm (J_NP_cos: cosine readout, raw W_U), recomputed from the saved readout vectors (CPU).
Usage: python 001_reporting.py [--no-judge]   (needs OPENROUTER_API_KEY for the judge step)
"""
import argparse, glob, json, math, os, subprocess, sys
import numpy as np
import torch
from scipy.stats import binomtest

torch.set_num_threads(48)
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
V2 = os.path.join(ROOT, "results", "001-lag-bucket-poetry", "v2")
VEC = os.path.join(ROOT, "outputs", "001_v2_vecs")
WSB = "/workspace/workspace-bench"
POETRY = f"{WSB}/evals/poetry/items.json"
LAYERS = [40, 48, 56]
ARMS_STD = ["B2", "B3", "WITEM", "FULL_U", "JCB", "JNP", "LOGIT"]
KS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def snapshot():
    return glob.glob("/workspace/.hf_home/hub/models--Qwen--Qwen3.6-27B/snapshots/*/")[0]


def load_unembed():
    from safetensors import safe_open
    snap = snapshot(); wm = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
    def get(k):
        with safe_open(snap + wm[k], "pt") as f:
            return f.get_tensor(k).float()
    return get("lm_head.weight"), get("model.language_model.norm.weight")


def passk_auc(best_ranks):
    r = np.array(best_ranks)
    pk = [float(np.mean(r < k)) for k in KS]
    xs = np.log10(KS)
    auc = float(getattr(np, "trapezoid", getattr(np, "trapz", None))(pk, xs) / (xs[-1] - xs[0]))
    return pk, auc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-judge", action="store_true"); a = ap.parse_args()
    names = json.load(open(os.path.join(V2, "subset50.json")))["names"]
    rows = [json.loads(x) for x in open(os.path.join(V2, "cells.jsonl"))]
    by = {(r["name"], r["layer"]): r for r in rows if r["name"] in names}
    names = [n for n in names if all((n, l) in by for l in LAYERS) and by[(n, LAYERS[0])]["single"]]
    R = {"n_items": len(names), "passk": {}, "wsb_judge": {}}

    # ---------------- (1) paper metric from stored ranks
    arms_rank = ["B0", "B1", "B2", "B3", "B4", "WITEM", "FULL32", "FULL_U", "JCB", "JNP", "LOGIT"]
    best = {n: [min(by[(x, l)]["rank"][n] for l in LAYERS) for x in names] for n in arms_rank}
    for n in arms_rank:
        pk, auc = passk_auc(best[n])
        R["passk"][n] = {"pass_at_k": dict(zip(map(str, KS), pk)), "auc_norm": auc}
    rng = np.random.default_rng(0)
    for n in arms_rank:
        diffs = []
        for _ in range(2000):
            idx = rng.integers(0, len(names), len(names))
            diffs.append(passk_auc([best[n][i] for i in idx])[1] - passk_auc([best["FULL_U"][i] for i in idx])[1])
        R["passk"][n]["auc_minus_FULL_U_ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]

    # ---------------- (2) WSB judge: rebuild top-10 (+ scores) per arm from saved vectors, incl. J_NP_cos
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    W, g = load_unembed()
    U = W * (1.0 + g)[None, :]
    jnp = torch.load(hf_hub_download("neuronpedia/jacobian-lens",
                                     "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"),
                     map_location="cpu", weights_only=False)
    JN = jnp["J"] if "J" in jnp else jnp["jacobians"]
    denom = {}
    for l in LAYERS:
        Jl = (JN[l] if l in JN else JN[str(l)]).float() if isinstance(JN, dict) else JN[l].float()
        denom[l] = torch.cat([(c @ Jl).norm(dim=1) for c in W.split(16384)]).clamp_min(1e-9)   # WSB JLens denominator
    items = {it["name"]: it for it in json.load(open(POETRY))["items"]}
    out_dir = os.path.join(V2, "wsb_readouts"); os.makedirs(out_dir, exist_ok=True)
    files = {n: open(os.path.join(out_dir, f"{n}.jsonl"), "w") for n in ARMS_STD + ["J_NP_cos"]}
    for x in names:
        ids = tok(items[x]["prompt"], add_special_tokens=False)["input_ids"]
        pos = [i for i, t in enumerate(ids) if "\n" in tok.decode([t])][-1]
        for l in LAYERS:
            v = torch.load(os.path.join(VEC, f"{x}_L{l}.pt"))
            for n in ARMS_STD + ["J_NP_cos"]:
                sc = (W @ v["JNP"].float()) / denom[l] if n == "J_NP_cos" else U @ v[n].float()
                top = sc.topk(10)
                files[n].write(json.dumps({"id": x, "layer": l, "pos": pos, "token": tok.decode([ids[pos]]),
                                           "tokens": [tok.decode([t]) for t in top.indices.tolist()],
                                           "scores": [round(float(s), 4) for s in top.values.tolist()]}) + "\n")
    for f in files.values():
        f.close()
    if a.no_judge:
        json.dump(R, open(os.path.join(V2, "reporting.json"), "w"), indent=1); print("readouts written; judge skipped"); return

    passed = {}
    for n in ARMS_STD + ["J_NP_cos"]:
        jd = os.path.join(V2, "wsb_judged", n)
        subprocess.run(["uv", "run", "--no-sync", "wsbench", "judge", "family=poetry",
                        f"readouts={os.path.join(out_dir, n + '.jsonl')}", f"out={jd}",
                        "items=" + ",".join(names), "layers=" + ",".join(map(str, LAYERS))],
                       cwd=WSB, check=True, capture_output=True, text=True)
        res = json.load(open(os.path.join(jd, "results.json")))
        passed[n] = {r["id"]: bool(r.get("pass")) for r in res["rows"]}
        R["wsb_judge"][n] = {"pass_rate": res["numbers"]["value"], "n": res["n_items"],
                             "spend_usd": res["counts"].get("spend_usd"), "complete": res.get("complete"),
                             "pinned_instrument": res.get("pinned_instrument")}
    for ref in ("FULL_U", "J_NP_cos"):
        for n in passed:
            if n == ref:
                continue
            b = sum(passed[n].get(x, False) and not passed[ref].get(x, False) for x in names)
            c = sum(not passed[n].get(x, False) and passed[ref].get(x, False) for x in names)
            R["wsb_judge"][n][f"vs_{ref}"] = {"only_arm": b, "only_ref": c,
                                             "mcnemar_p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}
    json.dump(R, open(os.path.join(V2, "reporting.json"), "w"), indent=1)
    print(json.dumps({n: R["wsb_judge"][n]["pass_rate"] for n in R["wsb_judge"]}))


if __name__ == "__main__":
    main()
