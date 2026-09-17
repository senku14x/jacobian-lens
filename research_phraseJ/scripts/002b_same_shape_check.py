"""002b — Post-run implementation check at the fit's own graph shape (design 002, Amendment 1, item 3).

For the 3 floor tokens on the same 3 pile sequences:
  same_shape : cos / relerr of v_lin_batched(reps=dim_batch)  vs  J_freshᵀq   (fit and scalar backward share a graph shape)
  batch1     : cos / relerr of v_lin (batch 1)                 vs  J_freshᵀq   (what compat.json gated)
  repeat     : cos / relerr of two batch-1 v_lin runs           (pure run-to-run nondeterminism)
  reps1      : cos / relerr of v_lin_batched(reps=1) vs v_lin   (kernel path with batch dim, no replication)
  cross_prompt: cos between the same token's batch-1 rows on different sequences (scale of genuine per-prompt variation)
Writes results/002-phraseJ-exact-compat/same_shape_check.json. Runs after 002 (needs outputs/002/{compat_vectors,J_fresh_3seq}.pt).
"""
import json, os, sys, time
import numpy as np, torch, jlens
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ekko_harness import load_model
from lib.phrase_objective import PhraseJ

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "002-phraseJ-exact-compat"); OUTD = os.path.join(ROOT, "outputs", "002")
MODEL = "/content/models/Qwen3.6-27B"
LAYERS = list(range(8, 61, 4)); TARGET = 62; SKIP = 4; SEQ = 128
compat = json.load(open(f"{RES}/compat.json")); DB = int(compat["dim_batch"])
cv = torch.load(f"{OUTD}/compat_vectors.pt", weights_only=False); prompts = cv["prompts"]; toks = cv["tokens"][:3]
fresh = jlens.JacobianLens.load(f"{OUTD}/J_fresh_3seq.pt")
model, hf, tok = load_model(MODEL)
pj = PhraseJ(model, LAYERS, TARGET, SKIP)
gamma = model._final_norm.weight.detach().float().cpu(); W_U = hf.lm_head.weight.detach().float().cpu()
def q_vec(t): return (1 + gamma) * W_U[t]
def cos(a, b): return float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))
def relerr(a, b): return float((a - b).norm() / b.norm())
ids = [model.encode(p, max_length=SEQ) for p in prompts]

t0 = time.time()
acc = {k: {l: torch.zeros(model.d_model) for l in LAYERS} for k in ("b1", "b1_rep", "bDB", "b_reps1")}
acc = {t: {k: {l: torch.zeros(model.d_model) for l in LAYERS} for k in acc} for t in toks}
per_seq_b1 = {t: [] for t in toks}
for x in ids:
    for t in toks:
        q = q_vec(t)
        g1 = pj.v_lin(x, q); g1r = pj.v_lin(x, q); gD = pj.v_lin_batched(x, q, reps=DB); gR1 = pj.v_lin_batched(x, q, reps=1)
        per_seq_b1[t].append(g1)
        for l in LAYERS:
            acc[t]["b1"][l] += g1[l] / len(ids); acc[t]["b1_rep"][l] += g1r[l] / len(ids)
            acc[t]["bDB"][l] += gD[l] / len(ids); acc[t]["b_reps1"][l] += gR1[l] / len(ids)
print(f"backwards done in {time.time()-t0:.0f}s", flush=True)

out = {"dim_batch": DB, "tokens": {int(t): tok.decode([t]) for t in toks}, "per_layer": {}}
print(f"{'L':>3} | same_shape cos/relerr | batch1 cos/relerr | repeat cos/relerr | reps1 cos/relerr | cross_prompt cos")
for l in LAYERS:
    r = {k: [] for k in ("same_shape_cos", "same_shape_relerr", "batch1_cos", "batch1_relerr", "repeat_cos", "repeat_relerr", "reps1_cos", "reps1_relerr", "cross_prompt_cos")}
    for t in toks:
        vf = fresh.jacobians[l].float().T @ q_vec(t); a = acc[t]
        r["same_shape_cos"].append(cos(a["bDB"][l], vf)); r["same_shape_relerr"].append(relerr(a["bDB"][l], vf))
        r["batch1_cos"].append(cos(a["b1"][l], vf)); r["batch1_relerr"].append(relerr(a["b1"][l], vf))
        r["repeat_cos"].append(cos(a["b1"][l], a["b1_rep"][l])); r["repeat_relerr"].append(relerr(a["b1"][l], a["b1_rep"][l]))
        r["reps1_cos"].append(cos(a["b_reps1"][l], a["b1"][l])); r["reps1_relerr"].append(relerr(a["b_reps1"][l], a["b1"][l]))
        g = per_seq_b1[t]
        for i in range(len(g)):
            for j in range(i + 1, len(g)): r["cross_prompt_cos"].append(cos(g[i][l], g[j][l]))
    out["per_layer"][str(l)] = {k: {"min": float(min(v)), "median": float(np.median(v)), "max": float(max(v))} for k, v in r.items()} | {"raw": r}
    m = lambda k: np.median(r[k])
    print(f"{l:>3} | {m('same_shape_cos'):.4f}/{m('same_shape_relerr'):.4f} | {m('batch1_cos'):.4f}/{m('batch1_relerr'):.4f} | {m('repeat_cos'):.4f}/{m('repeat_relerr'):.4f} | {m('reps1_cos'):.4f}/{m('reps1_relerr'):.4f} | {m('cross_prompt_cos'):.3f}", flush=True)
# --- Amendment 2, step 1 identity: fitter-row contraction equals J_fresh^T q (fp32 sum order only)
out["rowwise_identity_max_relerr"] = float(max(relerr(sum(q_vec(t)[d] * fresh.jacobians[l].float()[d] for d in range(model.d_model)), fresh.jacobians[l].float().T @ q_vec(t)) for t in toks[:1] for l in (LAYERS[0], LAYERS[-1])))
print("rowwise identity max relerr:", out["rowwise_identity_max_relerr"])

# --- Amendment 2, step 3: single J rows (one-hot cotangent) at batch 1 vs batch dim_batch, twice each
def row_grad(x, d, reps):
    e = torch.zeros(model.d_model); e[d] = 1.0
    return pj.v_lin(x, e) if reps == 1 else pj.v_lin_batched(x, e, reps=reps)
rng = np.random.default_rng(0); dims = [int(d) for d in rng.choice(model.d_model, 4, replace=False)]
rows = {str(l): {"cross_shape_cos": [], "cross_shape_relerr": [], "repeat_b1_cos": [], "repeat_bDB_cos": []} for l in LAYERS}
for x in ids[:2]:
    for d in dims:
        r1a, r1b, rDa, rDb = row_grad(x, d, 1), row_grad(x, d, 1), row_grad(x, d, DB), row_grad(x, d, DB)
        for l in LAYERS:
            rows[str(l)]["cross_shape_cos"].append(cos(r1a[l], rDa[l])); rows[str(l)]["cross_shape_relerr"].append(relerr(r1a[l], rDa[l]))
            rows[str(l)]["repeat_b1_cos"].append(cos(r1a[l], r1b[l])); rows[str(l)]["repeat_bDB_cos"].append(cos(rDa[l], rDb[l]))
out["row_reproducibility"] = {l: {k: {"min": float(min(v)), "median": float(np.median(v))} for k, v in r.items()} for l, r in rows.items()}
print(f"{'L':>3} | row cross-shape cos (min/med) | relerr med | repeat B=1 cos min | repeat B=DB cos min")
for l in LAYERS:
    r = rows[str(l)]; print(f"{l:>3} | {min(r['cross_shape_cos']):.4f}/{np.median(r['cross_shape_cos']):.4f} | {np.median(r['cross_shape_relerr']):.4f} | {min(r['repeat_b1_cos']):.4f} | {min(r['repeat_bDB_cos']):.4f}", flush=True)
json.dump(out, open(f"{RES}/same_shape_check.json", "w"), indent=1)
print("done")
