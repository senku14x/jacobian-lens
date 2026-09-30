"""002 — do distance-matched lenses read the words the model will write later? Design:
designs/002-future-words-own-text.md (pre-registered). Nothing is fitted.

Usage: python 002_future_words.py [--n 50] [--layers 40,48,56] [--limit N] [--positions 8] [--K 32]
Writes results/002-future-words-own-text/{items.json, cells.jsonl}; vectors to outputs/002_vecs/. Resumable.
"""
import argparse, json, os, random, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
from engine import Engine
from mfl import lag_profile_exact_halves, buckets_from_profile, tbar_uniform_halves, valid_positions

ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "results", "002-future-words-own-text"); os.makedirs(OUT, exist_ok=True)
VEC = os.path.join(ROOT, "outputs", "002_vecs"); os.makedirs(VEC, exist_ok=True)
BANK = "/workspace/workspace-bench/evals/hallucination/capture_rows.json"   # verbatim sampled token ids
BUCKETS = {"B0": (0, 0), "B1": (1, 3), "B2": (4, 8), "B3": (9, 16), "B4": (17, 32)}
MAX_LAG = 32
SEED = 20260930
# fixed English stopword list (pre-registered with the design)
STOP = set("""a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each else ever every few for from further get got had has
have having he her here hers herself him himself his how however i if in into is it its itself just let like made make
many may me might more most much must my myself never no nor not now of off often on once one only or other our ours
ourselves out over own per please put quite rather really said same say says see she should since so some still such
take than that the their theirs them themselves then there these they this those though through thus to too under
until up upon us use used using very via was way we well were what when where whether which while who whom whose why
will with within without would yes yet you your yours yourself yourselves""".split())
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def load_lens(repo, fn, layers):
    from huggingface_hub import hf_hub_download
    o = torch.load(hf_hub_download(repo, fn), map_location="cpu", weights_only=False)
    J = o["J"] if "J" in o else o["jacobians"]
    if isinstance(J, dict):
        return {l: (J[l] if l in J else J[str(l)]).float() for l in layers}
    return {l: J[l].float() for l in layers}


def future_targets(tok, ids, t, content_only=True):
    """(token_id, lag) for lags 0..MAX_LAG: the token at t+lag+1, first occurrence only.
    content_only: word-initial, alphabetic, >=3 letters, not a stopword, id not present in ids[0..t]."""
    ctx, seen, out = set(ids[:t + 1]), set(), []
    for d in range(MAX_LAG + 1):
        tid = ids[t + d + 1]
        if tid in seen:
            continue
        if content_only:
            s = tok.decode([tid]); w = s.strip()
            if not s.startswith(" ") or not w.isalpha() or len(w) < 3 or w.lower() in STOP or tid in ctx:
                continue
        seen.add(tid); out.append((tid, d))
    return out


def ranks_of(sc, ids):
    if not ids:
        return []
    t = torch.tensor(ids, device=sc.device)
    return (sc[None, :] > sc[t][:, None]).sum(1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--layers", default="40,48,56")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--positions", type=int, default=8)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="write to smoke_cells.jsonl (code check only, never analysed)")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    bank = json.load(open(BANK))
    eng = Engine(l0=min(layers)); tok = eng.tok

    # ---- item selection and read positions (seeded; recorded). Verbatim sampled ids from capture_rows.json.
    rng = random.Random(SEED)
    cand = []
    for bi, it in enumerate(bank):
        ids = json.loads(it["input_ids"]) if isinstance(it["input_ids"], str) else list(it["input_ids"])
        rs = int(it["prompt_len"])
        if len(ids) - rs >= 48:
            cand.append((bi, ids, rs))
    pick = sorted(rng.sample(range(len(cand)), a.n))
    items = []
    for j in pick:
        bi, ids, rs = cand[j]
        t = rng.randint(rs + 8, len(ids) - 34)
        items.append({"bank_index": bi, "id": bank[bi]["id"], "rs": rs, "t": t, "ids": ids,
                      "targets": future_targets(tok, ids, t, True), "all_targets": future_targets(tok, ids, t, False)})
    perm = list(range(len(items)))
    while any(i == p for i, p in enumerate(perm)):
        rng.shuffle(perm)
    for i, it in enumerate(items):
        it["decoy_of"] = items[perm[i]]["id"]
        it["decoy_targets"] = items[perm[i]]["targets"]
    if not a.smoke:
        json.dump([{k: v for k, v in it.items() if k != "ids"} for it in items],
                  open(os.path.join(OUT, "items.json"), "w"), indent=1)
    log(f"{len(cand)} candidates with >=48 response tokens; picked {len(items)};"
        f" content targets per item: median {sorted(len(x['targets']) for x in items)[len(items)//2]}")
    items = items[: a.limit]

    JCB = load_lens("camilablank/workspace-lenses", "qwen3.6-27b/j-lens/lens.pt", layers)
    JNP = load_lens("neuronpedia/jacobian-lens",
                    "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt", layers)
    eng.to_gpu_bf16()
    for it in items:                                   # causal: the read activation depends only on ids[0..t]
        X, _ = eng.capture(torch.tensor([it["ids"][: it["t"] + 1]]), layers)
        it["h"] = {l: X[l][0, it["t"]].clone() for l in layers}
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    host_ids, i = [], 100
    while len(host_ids) < a.K:
        x = tok(ds[i]["text"], truncation=True, max_length=128, add_special_tokens=False)["input_ids"]
        if len(x) == 128:
            host_ids.append(x)
        i += 1
    HX, hkw = eng.capture(torch.tensor(host_ids), layers)
    eng.to_cpu_bf16(); eng.build_fp32_upper()
    U = eng.U; Np = len(valid_positions(128))

    fp = os.path.join(OUT, "smoke_cells.jsonl" if a.smoke else "cells.jsonl")
    done = {(json.loads(x)["id"], json.loads(x)["layer"]) for x in open(fp)} if os.path.exists(fp) else set()
    f = open(fp, "a")
    for ii, it in enumerate(items):
        for l in layers:
            if (it["id"], l) in done:
                continue
            h = it["h"][l]
            S, Cn = lag_profile_exact_halves(eng, HX[l], hkw, l, h, n_pos=a.positions,
                                             seed=200000 + 100 * it["bank_index"] + l)
            Cd = Cn.to(S.device).clamp_min(1)
            vecs = buckets_from_profile(S.sum(0) / Cd.sum(0)[:, None], BUCKETS, Np)
            halves = [buckets_from_profile(S[k] / Cd[k][:, None], BUCKETS, Np) for k in (0, 1)]
            fu, (fu0, fu1) = tbar_uniform_halves(eng, HX[l], hkw, l, h)
            vecs["FULL_U"] = fu.to(U.device); halves[0]["FULL_U"] = fu0.to(U.device); halves[1]["FULL_U"] = fu1.to(U.device)
            vecs["JCB"] = (JCB[l] @ h).to(U.device); vecs["JNP"] = (JNP[l] @ h).to(U.device); vecs["LOGIT"] = h.to(U.device)
            tg = [x for x, _ in it["targets"]]; ag = [x for x, _ in it["all_targets"]]; dg = [x for x, _ in it["decoy_targets"]]
            row = {"id": it["id"], "layer": l, "t": it["t"], "targets": it["targets"], "all_targets": it["all_targets"],
                   "decoy_of": it["decoy_of"], "decoy_targets": it["decoy_targets"],
                   "ranks": {}, "all_ranks": {}, "decoy_ranks": {}, "half_ranks": {}, "top10": {}}
            for n, v in vecs.items():
                sc = U @ v.float()
                row["ranks"][n] = ranks_of(sc, tg); row["all_ranks"][n] = ranks_of(sc, ag)
                row["decoy_ranks"][n] = ranks_of(sc, dg)
                row["top10"][n] = [tok.decode([x]) for x in sc.topk(10).indices.tolist()]
                if n in halves[0]:
                    row["half_ranks"][n] = [ranks_of(U @ halves[k][n].float(), tg) for k in (0, 1)]
            torch.save({n: v.half().cpu() for n, v in vecs.items()}, os.path.join(VEC, f"{it['id']}_L{l}.pt"))
            f.write(json.dumps(row) + "\n"); f.flush()
        log(f"item {ii+1}/{len(items)} {it['id']} (t={it['t']}, {len(it['targets'])} content targets)")
    f.close()
    log("done")


if __name__ == "__main__":
    main()
