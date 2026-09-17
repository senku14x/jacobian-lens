"""004b — secondary arm (design §4 'secondary = all prompt positions'): the same norm-targeted swap applied at EVERY prompt
position at L52 and L56 (the paper's convention), for PJ, Jc and random, rho in {0.1, 0.2, 0.4} (per-position norm target).
Same items and operators as 004_swaps.py. Writes results/004-causal-geometry/swaps_allpos.json.
"""
import json, os, sys, re, time, itertools
import numpy as np, torch, jlens
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ekko_harness import load_model
from lib.edit_harness import swap_delta, forward_logits_and_resid, Edits
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "004-causal-geometry"); GR = os.path.join(ROOT, "outputs", "003a", "grads"); GRX = os.path.join(ROOT, "outputs", "003a_controls"); RES1 = os.path.join(ROOT, "results", "001-template-geometry")
MODEL = "/content/models/Qwen3.6-27B"; LENS = "/content/lenses/qwen3.6-27b"; LAYERS = [52, 56]; RHOS = [0.1, 0.2, 0.4]; N_RAND = 2; dev = "cuda"
items = json.load(open(f"{RES}/items.json")); ITEMS = items["items"]; r_grid = {l: [rho * items["median_hnorm"][str(l)] for rho in RHOS] for l in LAYERS}
model, hf, tok = load_model(MODEL)
gamma = model._final_norm.weight.detach().float().cpu(); W_U = hf.lm_head.weight.detach().float().cpu()
def q_vec(t): return (1 + gamma) * W_U[t]
Jl = jlens.JacobianLens.load(f"{LENS}/j-lens/lens.pt"); Jrow = {l: (lambda t, l=l: Jl.jacobians[l].float().T @ q_vec(t)) for l in LAYERS}
def safe(s): return re.sub(r"[^A-Za-z0-9]+", "_", s)
fam = json.load(open(f"{RES1}/families.json")); members = {m["phrase"]: m for F in fam["families"] for m in F["members"]}
vec = {}
for ph in members:
    p = f"{GR}/{safe(ph)}__natural.pt"
    if not os.path.exists(p): continue
    d = torch.load(p, weights_only=False); k = d["prefix_len"]; recs = [r["obj"]["lin"] for r in d["recs"]]
    px = f"{GRX}/{safe(ph)}_extra_lin.pt"
    if os.path.exists(px): recs += torch.load(px, weights_only=False)
    vec[ph] = {"k": k, "v": {l: torch.stack([r[l]["at"].float()[k:].sum(0) for r in recs]).mean(0) for l in LAYERS}}
def build(A, B, method, l):
    kA = vec[A]["k"]
    if method == "PJ": V = torch.stack([vec[A]["v"][l], vec[B]["v"][l]], 1); perm = [1, 0]
    else:
        p_t = members[A]["ids"][:kA]; V = torch.stack([Jrow[l](t) for t in p_t] + [Jrow[l](members[A]["ids"][kA]), Jrow[l](members[B]["ids"][kA])], 1); m = V.shape[1]; perm = list(range(m - 2)) + [m - 1, m - 2]
    V = V.to(dev); return V, torch.linalg.pinv(V), perm
def allpos_swap_fn(V, Vp, perm, r):
    def fn(h):
        h = h.clone()
        for pos in range(h.shape[0]):
            d = swap_delta(h[pos], V, Vp, perm); h[pos] = h[pos] + (d * (r / (d.float().norm() + 1e-8))).to(h.dtype)
        return h
    return fn
def allpos_rand_fn(r, seed, d_model):
    g = torch.Generator().manual_seed(seed)
    def fn(h):
        h = h.clone(); d = torch.randn(h.shape[0], d_model, generator=g).to(h.device); d = d / d.norm(dim=1, keepdim=True) * r; return h + d.to(h.dtype)
    return fn
def metrics(logits, c):
    lp = torch.log_softmax(logits[0], -1); p = lp.exp(); return {"lpA": float(lp[c["tA"]]), "lpB": float(lp[c["tB"]]), "top1": int(logits[0].argmax()), "entropy": float(-(p * lp).sum())}
def kl(lc_, le_):
    lc = torch.log_softmax(lc_[0], -1); le = torch.log_softmax(le_[0], -1); return float((lc.exp() * (lc - le)).sum())
ops = {}; out = []; t0 = time.time()
for n, c in enumerate(ITEMS):
    key = (c["A"], c["B"])
    if key not in ops: ops[key] = {m: {l: build(c["A"], c["B"], m, l) for l in LAYERS} for m in ("PJ", "Jc")}
    ids = model.encode(c["prompt"], max_length=256); clean, _ = forward_logits_and_resid(model, ids, None)
    rec = {"item": n, "A": c["A"], "B": c["B"], "family": c["family"], "clean": metrics(clean, c), "runs": []}
    for ri, rho in enumerate(RHOS):
        for m in ("PJ", "Jc"):
            lg, _ = forward_logits_and_resid(model, ids, {l: allpos_swap_fn(*ops[key][m][l], r_grid[l][ri]) for l in LAYERS}); rec["runs"].append({"method": m, "rho": rho, **metrics(lg, c), "kl": kl(clean, lg)})
        for s in range(N_RAND):
            lg, _ = forward_logits_and_resid(model, ids, {l: allpos_rand_fn(r_grid[l][ri], 7000 + 10 * n + s, model.d_model) for l in LAYERS}); rec["runs"].append({"method": f"rand{s}", "rho": rho, **metrics(lg, c), "kl": kl(clean, lg)})
    out.append(rec)
    if (n + 1) % 20 == 0: print(f"[{n+1}/{len(ITEMS)}] allpos ({time.time()-t0:.0f}s)", flush=True)
json.dump({"rhos": RHOS, "swaps": out}, open(f"{RES}/swaps_allpos.json", "w"))
print("\nall-positions arm: selectivity / flip / item-KL")
for m in ("PJ", "Jc", "rand"):
    for rho in RHOS:
        sel, flip, klv = [], [], []
        for rec in out:
            rs = [r for r in rec["runs"] if r["rho"] == rho and (r["method"] == m or (m == "rand" and r["method"].startswith("rand")))]; tB = ITEMS[rec["item"]]["tB"]
            sel.append(np.mean([(r["lpB"] - rec["clean"]["lpB"]) - (r["lpA"] - rec["clean"]["lpA"]) for r in rs])); flip.append(np.mean([r["top1"] == tB for r in rs])); klv.append(np.mean([r["kl"] for r in rs]))
        print(f"{m:5s} rho={rho:<4} selectivity {np.mean(sel):+.2f}  flip {np.mean(flip):.2f}  KL {np.mean(klv):.2f}", flush=True)
print(f"done ({time.time()-t0:.0f}s)")
