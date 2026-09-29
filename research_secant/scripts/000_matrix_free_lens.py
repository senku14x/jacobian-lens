"""000 — matrix-free J-lens validation. Design: designs/000-matrix-free-lens.md (pre-registered).

Gates: G0 Richardson (eps vs eps/2), G1 reproduction of J_CB on the hypothesised fit docs (two BOS conventions),
G2 fresh-host noise (report), G3 bf16-vs-fp32 finite differences (report). Writes results/000-matrix-free-lens/.
"""
import json, os, sys, time, collections
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
from engine import Engine, TARGET
from mfl import tbar_full, cos

ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "results", "000-matrix-free-lens"); os.makedirs(OUT, exist_ok=True)
LAYERS = [40, 48, 56]
REL_EPS = 1e-3
SEED = 0
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def load_jcb():
    from huggingface_hub import hf_hub_download
    o = torch.load(hf_hub_download("camilablank/workspace-lenses", "qwen3.6-27b/j-lens/lens.pt"),
                   map_location="cpu", weights_only=False)
    return {l: o["J"][l].float() for l in LAYERS}, o["provenance"]


def pile_docs(idx):
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    return [ds[i]["text"] for i in idx]


def tokenize(tok, texts, bos):
    out = []
    for t in texts:
        ids = tok(t, truncation=True, max_length=128 - (1 if bos else 0), add_special_tokens=False)["input_ids"]
        if bos:
            b = tok.bos_token_id if tok.bos_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
            ids = [b] + ids
        out.append(ids)
    return out


def capture_groups(eng, id_lists, layers):
    """Group docs by length (no padding); returns list of (X{l:[k,T,d]}, kw, doc_indices)."""
    groups = collections.defaultdict(list)
    for i, ids in enumerate(id_lists):
        if len(ids) > 17:
            groups[len(ids)].append(i)
    res = []
    for T, idx in sorted(groups.items()):
        X, kw = eng.capture(torch.tensor([id_lists[i] for i in idx]), layers)
        res.append((X, kw, idx))
    return res


def tbar_groups(eng, groups, l, V, rel_eps=REL_EPS):
    """Uniform mean over prompts of per-prompt estimates (matches jlens aggregation)."""
    per = []
    for X, kw, idx in groups:
        _, ph = tbar_full(eng, X[l], kw, l, V, rel_eps=rel_eps, return_per_host=True, chunk=13)
        per.append(ph)                         # [n, k, d]
    ph = torch.cat(per, 1)
    return ph.mean(1), ph


def main():
    torch.manual_seed(SEED)
    J, prov = load_jcb()
    log("J_CB provenance", prov)
    eng = Engine(l0=min(LAYERS))
    tok = eng.tok
    log("bos_token", tok.bos_token, tok.bos_token_id)

    # ---------------- test vectors (block-l outputs at the final token of bundled multihop prompts)
    items = json.load(open(os.path.join(ROOT, "..", "data", "evaluations", "lens-eval-multihop.json")))["items"]
    prompts = [it["prompt"] for it in items[:8]]
    eng.to_gpu_bf16()
    acts = {l: [] for l in LAYERS}
    for p in prompts:
        ids = torch.tensor([tok(p, add_special_tokens=False)["input_ids"]])
        X, _ = eng.capture(ids, LAYERS)
        for l in LAYERS:
            acts[l].append(X[l][0, -1])
    V = {}
    for l in LAYERS:
        a = torch.stack(acts[l])
        rnd = torch.randn(4, a.shape[1]); rnd = rnd / rnd.norm(dim=1, keepdim=True) * a.norm(dim=1).median()
        V[l] = {"random": rnd, "activation": a[:4], "difference": a[:4] - a[4:8]}

    # ---------------- host captures
    fit_idx, fresh_idx = list(range(25)), list(range(100, 132))
    texts_fit, texts_fresh = pile_docs(fit_idx), pile_docs(fresh_idx)
    conv = {}
    for bos in (False, True):
        ids = tokenize(tok, texts_fit, bos)
        conv[bos] = capture_groups(eng, ids, LAYERS)
        log(f"captured fit docs bos={bos}: lengths", sorted(len(x) for x in ids)[:5], "...", "n", len(ids))
    fresh = capture_groups(eng, tokenize(tok, texts_fresh, False), LAYERS)

    # ---------------- G3: bf16 finite differences (report). Temporarily point the replay stack at bf16 GPU blocks.
    g3 = {}
    l = 48
    eng.up = {i: eng.inner.layers[i] for i in range(eng.l0 + 1, eng.n_layers)}
    Vt = torch.cat([V[l]["activation"], V[l]["difference"]])
    X, kw, _ = conv[False][-1]
    _orig = eng.run_from
    def run_bf16(h, ll, kw_, to=TARGET):
        return _orig(h.to(torch.bfloat16), ll, kw_, to).float()
    eng.run_from = run_bf16
    for re_ in (1e-3, 1e-2, 5e-2):
        est = tbar_full(eng, X[l], kw, l, Vt, rel_eps=re_, chunk=13)
        ref = (J[l] @ Vt.T).T
        g3[str(re_)] = {"cos_to_JCB_median": float(cos(est, ref).median())}
    eng.run_from = _orig
    eng.up = {}
    log("G3 bf16 (single length-group, L48)", g3)

    # ---------------- fp32 stack
    eng.to_cpu_bf16()
    eng.build_fp32_upper()
    fold_err = eng.check_norm_convention()
    log("(1+gamma) fold check max|diff|", fold_err)

    results = {"provenance": prov, "fold_err": fold_err, "G3_bf16": g3, "layers": {}}
    for l in LAYERS:
        R = {}
        Vall = torch.cat([V[l][k] for k in ("random", "activation", "difference")])
        kinds = ["random"] * 4 + ["activation"] * 4 + ["difference"] * 4
        ref = (J[l] @ Vall.T).T                                   # [12, d]
        # G0 Richardson on the no-BOS fit docs
        e1, _ = tbar_groups(eng, conv[False], l, Vall, REL_EPS)
        e2, _ = tbar_groups(eng, conv[False], l, Vall, REL_EPS / 2)
        R["G0"] = {"cos_eps_vs_half_min": float(cos(e1, e2).min()),
                   "relnorm_max": float(((e1 - e2).norm(dim=1) / e2.norm(dim=1)).max())}
        # G1 both conventions
        R["G1"] = {}
        for bos in (False, True):
            est, ph = (e1, None) if not bos else tbar_groups(eng, conv[True], l, Vall, REL_EPS)
            c = cos(est, ref)
            nr = est.norm(dim=1) / ref.norm(dim=1)
            Ug = eng.U
            cu = cos((est.to(Ug.device) @ Ug.T), (ref.to(Ug.device) @ Ug.T)).cpu()
            R["G1"]["bos" if bos else "nobos"] = {
                "cos_resid_median": float(c.median()), "cos_resid_min": float(c.min()),
                "norm_ratio_median": float(nr.median()), "cos_vocab_median": float(cu.median()),
                "by_kind": {k: float(c[[i for i, kk in enumerate(kinds) if kk == k]].median()) for k in set(kinds)}}
        # G2 fresh hosts
        ef, phf = tbar_groups(eng, fresh, l, Vall, REL_EPS)
        K = phf.shape[1]; perm = torch.randperm(K)
        h1, h2 = phf[:, perm[:K // 2]].mean(1), phf[:, perm[K // 2:]].mean(1)
        R["G2"] = {"n_hosts": K, "cos_fresh_to_JCB_median": float(cos(ef, ref).median()),
                   "norm_ratio_median": float((ef.norm(dim=1) / ref.norm(dim=1)).median()),
                   "split_half_cos_median": float(cos(h1, h2).median()),
                   "cos_fresh_to_fitdocs_median": float(cos(ef, e1).median())}
        results["layers"][l] = R
        log(f"L{l}", json.dumps(R))
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
