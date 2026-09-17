"""002 — Phrase-J implementation validation.

A. exact compat: v_lin (scalar backward, released convention) vs fresh J fit on the same 3 sequences (gate),
   and vs released J (storage/corpus difference; reported).
B. v_logit vs v_lin.
C. multi-token smoke: 5 family members x 8 emission-natural contexts; per-token g_i, v_seq, v_cond, v_mean,
   v_PB; norm ratios; split-half cosines.
Outputs: results/002-phraseJ-exact-compat/{compat.json, smoke.json}; outputs/002/*.pt
"""
import json, os, sys, time, random
import numpy as np, torch, jlens
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ekko_harness import load_model
from lib.phrase_objective import PhraseJ
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "002-phraseJ-exact-compat"); OUTD = os.path.join(ROOT, "outputs", "002"); os.makedirs(RES, exist_ok=True); os.makedirs(OUTD, exist_ok=True)
RES1 = os.path.join(ROOT, "results", "001-template-geometry")
MODEL = "/content/models/Qwen3.6-27B"; LENS = "/content/lenses/qwen3.6-27b"
LAYERS = list(range(8, 61, 4)); TARGET = 62; SKIP = 4; SEQ = 128
model, hf, tok = load_model(MODEL)
pj = PhraseJ(model, LAYERS, TARGET, SKIP)
gamma = model._final_norm.weight.detach().float().cpu(); W_U = hf.lm_head.weight.detach().float().cpu()
def q_vec(t): return (1 + gamma) * W_U[t]
def cos(a, b): return float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))
def relerr(a, b): return float((a - b).norm() / b.norm())

# ---------------------------------------------------------------- A/B: compat on 3 pile sequences (first docs of pile-10k that are long enough)
ds = load_dataset("NeelNanda/pile-10k", split="train")
prompts = []
for d in ds:
    if len(tok.encode(d["text"], add_special_tokens=False)) >= SEQ + 8: prompts.append(d["text"])
    if len(prompts) == 3: break
ids = [model.encode(p, max_length=SEQ) for p in prompts]
random.seed(0)
vocab_words = [" the", " of", " New", " Zealand", " spider", " Paris", " Oceania", " dollar", " blackmail", " hemisphere", " continent", " Korea", " Diego", " twelve", " honest", " fake", " 大", " 是", " Zealand", " ünd", " ...", "\n", " ;", " purple", " Tokyo", " Nations", " Emirates", " Auckland", " noodles", " leverage"]
toks = []
for w in vocab_words:
    i = tok.encode(w, add_special_tokens=False)
    if len(i) == 1 and i[0] not in toks: toks.append(i[0])
toks = toks[:30]
print("compat tokens:", [tok.decode([t]) for t in toks], flush=True)

DIM_BATCH = int(os.environ.get("DIM_BATCH", "4"))   # 8 OOMs on an 80 GB A100: batch-1 graph already peaks at 59 GB
t0 = time.time(); vlin = {t: {l: torch.zeros(model.d_model) for l in LAYERS} for t in toks}; vlog = {t: {l: torch.zeros(model.d_model) for l in LAYERS} for t in toks}
for x in ids:
    for t in toks:
        g = pj.v_lin(x, q_vec(t))
        for l in LAYERS: vlin[t][l] += g[l] / len(ids)
        g = pj.v_logit(x, t)
        for l in LAYERS: vlog[t][l] += g[l] / len(ids)
print(f"scalar backwards done in {time.time()-t0:.0f}s", flush=True)

# bf16 batch-shape noise floor: v_lin on the prompt replicated DIM_BATCH times (jlens.fit's graph shape) vs batch 1,
# for the first 3 tokens on the same 3 sequences. Any fresh-vs-v_lin discrepancy below this floor is precision, not code.
floor = {l: {"cos": [], "relerr": []} for l in LAYERS}
for x in ids:
    for t in toks[:3]:
        gb = pj.v_lin_batched(x, q_vec(t), reps=DIM_BATCH); g1 = pj.v_lin(x, q_vec(t))
        for l in LAYERS: floor[l]["cos"].append(cos(gb[l], g1[l])); floor[l]["relerr"].append(relerr(gb[l], g1[l]))
floor_summary = {str(l): {"cos_min": float(min(v["cos"])), "relerr_max": float(max(v["relerr"]))} for l, v in floor.items()}
print("batch-shape floor (min cos / max relerr per layer):", {l: (round(v["cos_min"], 5), round(v["relerr_max"], 4)) for l, v in floor_summary.items()}, flush=True)
torch.save({"vlin": vlin, "vlog": vlog, "tokens": toks, "prompts": prompts, "floor": floor}, f"{OUTD}/compat_vectors.pt")

t0 = time.time()
def fit_fresh(db):
    return jlens.fit(model, prompts=prompts, source_layers=LAYERS, target_layer=TARGET, skip_first=SKIP, max_seq_len=SEQ, dim_batch=db, checkpoint_path=None)
try:
    fresh = fit_fresh(DIM_BATCH)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache(); DIM_BATCH = max(1, DIM_BATCH // 2); print(f"OOM; retrying fresh fit with dim_batch={DIM_BATCH}", flush=True)
    fresh = fit_fresh(DIM_BATCH)
print(f"fresh J fit ({fresh.n_prompts} prompts, dim_batch={DIM_BATCH}) in {time.time()-t0:.0f}s", flush=True)
fresh.save(f"{OUTD}/J_fresh_3seq.pt")
released = jlens.JacobianLens.load(f"{LENS}/j-lens/lens.pt")
compat = {"tokens": {int(t): tok.decode([t]) for t in toks}, "prompts_first80": [p[:80] for p in prompts], "dim_batch": DIM_BATCH, "floor": floor_summary, "per_layer": {}}
for l in LAYERS:
    Jf = fresh.jacobians[l].float(); Jr = released.jacobians[l].float(); rec = {"fresh_cos": [], "fresh_relerr": [], "released_cos": [], "released_relerr": [], "logit_vs_lin_cos": [], "logit_over_lin_norm": []}
    for t in toks:
        q = q_vec(t); vf = Jf.T @ q; vr = Jr.T @ q
        rec["fresh_cos"].append(cos(vlin[t][l], vf)); rec["fresh_relerr"].append(relerr(vlin[t][l], vf))
        rec["released_cos"].append(cos(vlin[t][l], vr)); rec["released_relerr"].append(relerr(vlin[t][l], vr))
        rec["logit_vs_lin_cos"].append(cos(vlog[t][l], vlin[t][l])); rec["logit_over_lin_norm"].append(float(vlog[t][l].norm() / vlin[t][l].norm()))
    compat["per_layer"][str(l)] = {k: {"min": float(min(v)), "median": float(np.median(v)), "max": float(max(v))} for k, v in rec.items()} | {"raw": rec}
    print(f"L{l}: fresh cos min/med {min(rec['fresh_cos']):.4f}/{np.median(rec['fresh_cos']):.4f}  relerr med/max {np.median(rec['fresh_relerr']):.4f}/{max(rec['fresh_relerr']):.4f} (floor {floor_summary[str(l)]['relerr_max']:.4f}) | released cos med {np.median(rec['released_cos']):.3f} | logit-vs-lin cos med {np.median(rec['logit_vs_lin_cos']):.3f} norm ratio med {np.median(rec['logit_over_lin_norm']):.2f}", flush=True)
# Gate: cosine is the implementation check (design: > 0.999). Relative norm error is judged against the measured
# bf16 batch-shape floor: pass if max relerr < max(1%, 3 x floor) at every layer.
gate_cos = all(compat["per_layer"][str(l)]["fresh_cos"]["min"] > 0.999 for l in LAYERS)
gate_norm = all(compat["per_layer"][str(l)]["fresh_relerr"]["max"] < max(0.01, 3 * floor_summary[str(l)]["relerr_max"]) for l in LAYERS)
gate_norm_strict = all(compat["per_layer"][str(l)]["fresh_relerr"]["max"] < 0.01 for l in LAYERS)
compat["gate_pass"] = gate_cos and gate_norm; compat["gate_cos"] = gate_cos; compat["gate_norm_vs_floor"] = gate_norm; compat["gate_norm_strict_1pct"] = gate_norm_strict
print(f"GATE {'PASS' if compat['gate_pass'] else 'FAIL'}  (cos>0.999: {gate_cos}; relerr<max(1%,3*floor): {gate_norm}; strict relerr<1%: {gate_norm_strict})", flush=True)
json.dump(compat, open(f"{RES}/compat.json", "w"), indent=1)

# ---------------------------------------------------------------- C: multi-token smoke on emission-natural contexts
fam = json.load(open(f"{RES1}/families.json")); ctx = json.load(open(f"{RES1}/contexts.json"))
members = {m["phrase"]: m for F in fam["families"] for m in F["members"]}
family_of = {m["phrase"]: F for F in fam["families"] for m in F["members"]}
SMOKE = ["New Zealand", "South Korea", "San Diego", "United States", "ice cream"]
# Null "phrases" teacher-forced after New Zealand's contexts: a frequent non-word bigram and fixed random junk tokens.
# They give the norm-ratio and split-half reference that the real phrases are read against.
NULLS = {"null:of the": tok.encode(" of the", add_special_tokens=False),
         "null:junk": [tok.encode(" spider", add_special_tokens=False)[0], tok.encode("ünd", add_special_tokens=False)[0]]}
NULLS = {n: v for n, v in NULLS.items() if len(v) == 2}
smoke = {}; t0 = time.time(); fast_check = []
for ph in SMOKE + list(NULLS):
    if ph in NULLS:
        pids = NULLS[ph]; k = 1; sibs = []; suffixes = [pids[k:]]; texts = [c["text"] for c in ctx["emission_natural"] if c["phrase"] == "New Zealand"][:8]
    else:
        m = members[ph]; F = family_of[ph]; k = F["prefix_len"]; pids = m["ids"]
        sibs = [s for s in F["members"] if s["phrase"] != ph and s["n_tokens"] > k][:3]
        suffixes = [pids[k:]] + [s["ids"][k:] for s in sibs]
        texts = [c["text"] for c in ctx["emission_natural"] if c["phrase"] == ph][:8]
    if len(texts) < 8: print(f"WARNING {ph}: only {len(texts)} emission-natural contexts", flush=True)
    per_ctx = []
    for ci, text in enumerate(texts):
        cids = model.encode(text, max_length=SEQ)
        if cids.shape[1] < SKIP + 4: continue
        g_i, lps, tprime = pj.per_token_fast(cids, pids)
        if ci == 0:   # fast (retained graph) vs slow (one forward per token) agreement, once per phrase
            g_slow, lps_slow, _ = pj.per_token(cids, pids)
            fast_check.append({"phrase": ph, "max_abs_logprob_diff": float(max(abs(a - b) for a, b in zip(lps, lps_slow))),
                               "min_cos_mean_grad": float(min(cos(g_i[i][l][1], g_slow[i][l][1]) for i in range(len(pids)) for l in LAYERS)),
                               "max_relerr_mean_grad": float(max(relerr(g_i[i][l][1], g_slow[i][l][1]) for i in range(len(pids)) for l in LAYERS))})
            print(f"  fast-vs-slow per_token [{ph}]: {fast_check[-1]}", flush=True)
        pb, pb_lps = pj.v_pb(cids, pids[:k], suffixes, 0) if len(suffixes) > 1 else ({l: (g_i[-1][l][0] * 0, g_i[-1][l][1] * 0) for l in LAYERS}, [])
        rec = {"logprobs": lps, "pb_logprobs": pb_lps, "tprime": int(tprime), "layers": {}}
        for l in LAYERS:
            g_at = [g[l][0] for g in g_i]; g_mn = [g[l][1] for g in g_i]
            v_seq = sum(g_mn); v_cond = sum(g_mn[k:]); v_mean = v_seq / len(g_mn)
            rec["layers"][l] = {"g_at": torch.stack(g_at), "g_mean": torch.stack(g_mn), "v_seq": v_seq, "v_cond": v_cond, "v_mean": v_mean, "v_pb": pb[l][1], "v_pb_at": pb[l][0],
                                "norm_ratio_cond_over_g1": float(v_cond.norm() / g_mn[0].norm()), "cos_seq_g1": cos(v_seq, g_mn[0]), "cos_cond_g1": cos(v_cond, g_mn[0]), "cos_pb_cond": cos(pb[l][1], v_cond)}
        per_ctx.append(rec)
    torch.save(per_ctx, f"{OUTD}/smoke_{ph.replace(' ', '_').replace(':', '_')}.pt")
    # split-half cosines and summaries
    summ = {"n_contexts": len(per_ctx), "prefix_len": k, "siblings": [s["phrase"] for s in sibs], "is_null": ph in NULLS, "tokens": [tok.decode([t]) for t in pids], "per_layer": {}}
    h1, h2 = per_ctx[::2], per_ctx[1::2]
    for l in LAYERS:
        d = {}
        for key in ["v_seq", "v_cond", "v_mean", "v_pb"]:
            a = sum(r["layers"][l][key] for r in h1) / len(h1); b = sum(r["layers"][l][key] for r in h2) / len(h2); d[f"splithalf_cos_{key}"] = cos(a, b)
        g1a = sum(r["layers"][l]["g_mean"][0] for r in h1) / len(h1); g1b = sum(r["layers"][l]["g_mean"][0] for r in h2) / len(h2); d["splithalf_cos_g1"] = cos(g1a, g1b)
        for key in ["norm_ratio_cond_over_g1", "cos_seq_g1", "cos_cond_g1", "cos_pb_cond"]: d[key + "_median"] = float(np.median([r["layers"][l][key] for r in per_ctx]))
        summ["per_layer"][str(l)] = d
    smoke[ph] = summ
    print(f"{ph}: n={len(per_ctx)} | L56 norm_ratio cond/g1 {summ['per_layer']['56']['norm_ratio_cond_over_g1_median']:.2f} cos(cond,g1) {summ['per_layer']['56']['cos_cond_g1_median']:.2f} | split-half cos seq/cond/pb @L56: {summ['per_layer']['56']['splithalf_cos_v_seq']:.2f}/{summ['per_layer']['56']['splithalf_cos_v_cond']:.2f}/{summ['per_layer']['56']['splithalf_cos_v_pb']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
json.dump({"phrases": smoke, "fast_vs_slow_per_token": fast_check, "nulls": list(NULLS)}, open(f"{RES}/smoke.json", "w"), indent=1)
print("done")
