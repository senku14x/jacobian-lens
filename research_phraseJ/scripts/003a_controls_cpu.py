"""003a-controls, steps 1 and 3 (CPU; no model forwards).
Step 1: symmetric 2x2 {own, diff} x {Phrase-J lin (at t'), J-sum} on the 003a latent pairs, layers 36-60.
Step 3: Delta_residual = full probe - constituent ceiling (both leave-one-cue-out) per 001 family and layer (all 15 layers),
        bootstrap CI over cue groups; plus J-diff for reference.
Outputs: results/003a-controls/{symmetric_2x2.json, residual_screen.json, analysis_cpu.md}
"""
import json, os, sys, re, itertools
import numpy as np, torch, jlens
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "003a-controls"); os.makedirs(RES, exist_ok=True)
R3 = os.path.join(ROOT, "results", "003a-phrase-objective"); GR = os.path.join(ROOT, "outputs", "003a", "grads")
RES1 = os.path.join(ROOT, "results", "001-template-geometry"); O1 = os.path.join(ROOT, "outputs", "001")
LENS = "/content/lenses/qwen3.6-27b"; dev = "cuda" if torch.cuda.is_available() else "cpu"; LAM = 0.1; NB = 1000
A = torch.load(f"{O1}/acts.pt", weights_only=False); index = A["index"]; ALL_L = A["layers"]; acts = {l: A["acts"][l].float() for l in ALL_L}
S = torch.load(f"{O1}/sigma.pt", weights_only=False); U = torch.load(f"{O1}/unembed.pt", weights_only=False)
W_U = U["W_U"].float(); gamma = U["final_norm_weight"]; eps = U["norm_eps"] or 1e-6
Jl = jlens.JacobianLens.load(f"{LENS}/j-lens/lens.pt")
fam = json.load(open(f"{RES1}/families.json")); members = {m["phrase"]: m for F in fam["families"] for m in F["members"]}
def q_vec(t): return (1 + gamma) * W_U[t]
def rmsnorm(x): return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps)
def lens_logits(h, l, toks): return rmsnorm(Jl.transport(h, l)) @ torch.stack([q_vec(t) for t in toks]).T
def auc(p, n):
    p = np.asarray(p, float); n = np.asarray(n, float)
    return float("nan") if len(p) == 0 or len(n) == 0 else float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())
chol = {}
for l in ALL_L:
    mu, Sg, _ = S["pooled"][l]; tau = Sg.diagonal().mean(); chol[l] = torch.linalg.cholesky(((1 - LAM) * Sg + LAM * tau * torch.eye(Sg.shape[0])).to(dev))
def solve(l, v): return torch.cholesky_solve(v.to(dev).T.contiguous(), chol[l]).T.cpu()
def rows_for(cond, ph, pos=-1): return [i for i, r in enumerate(index) if r["cond"] == cond and r["phrase"] == ph and r["pos"] == pos]
ROUTE_ORDER = ["continent", "language", "currency", "hemisphere", "second_word_letters"]
def cue_group(i): r = index[i]; return (r["frame"] + ROUTE_ORDER.index(r["route"])) % 5
def safe(s): return re.sub(r"[^A-Za-z0-9]+", "_", s)

# ---------------------------------------------------------------- step 1: symmetric 2x2
L5 = [36, 44, 52, 56, 60]; FAMS = {"New Zealand": "New", "South Korea": "South", "San Diego": "San", "North Carolina": "North"}
items = json.load(open(f"{R3}/items.json"))
def vcond(name, l):
    d = torch.load(f"{GR}/{safe(name)}__natural.pt", weights_only=False); k = d["prefix_len"]
    return torch.stack([r["obj"]["lin"][l]["at"].float()[k:].sum(0) for r in d["recs"]]).mean(0)
sym = {}
md = ["# 003a-controls — CPU steps", "", "## Step 1: symmetric 2×2, latent pairwise AUC (mean over sibling pairs), at t′, lin", "",
      "| phrase | L | PJ own | PJ diff | J-sum own | J-sum diff | ceiling | probe |", "|---|---|---|---|---|---|---|---|"]
rd = json.load(open(f"{R3}/readout.json"))
for ph, fname in FAMS.items():
    F = [f for f in fam["families"] if f["name"] == fname][0]
    mem = [m for m in F["members"] if os.path.exists(f"{GR}/{safe(m['phrase'])}__natural.pt") and len(rows_for("latent", m["phrase"])) >= 10]
    sibs = [m for m in mem if m["phrase"] != ph]; sym[ph] = {}
    for l in L5:
        H = acts[l]; ra = rows_for("latent", ph); va = vcond(ph, l); toks = sorted({t for m in mem for t in m["ids"]}); ia = [toks.index(t) for t in members[ph]["ids"]]
        rec = {"PJ_own": [], "PJ_diff": [], "J_own": [], "J_diff": []}
        for sb in sibs:
            rb = rows_for("latent", sb["phrase"]); vb = vcond(sb["phrase"], l); ib = [toks.index(t) for t in sb["ids"]]
            La, Lb = lens_logits(H[ra], l, toks), lens_logits(H[rb], l, toks)
            rec["PJ_own"].append(auc(H[ra] @ va, H[rb] @ va)); rec["PJ_diff"].append(auc(H[ra] @ (va - vb), H[rb] @ (va - vb)))
            rec["J_own"].append(auc(La[:, ia].sum(1), Lb[:, ia].sum(1))); rec["J_diff"].append(auc(La[:, ia].sum(1) - La[:, ib].sum(1), Lb[:, ia].sum(1) - Lb[:, ib].sum(1)))
        b = rd["baselines"][ph][str(l)]; ceil = np.nanmean([b[p]["constituent_ceiling"] for p in b]); prb = np.nanmean([b[p]["full_probe"] for p in b])
        sym[ph][str(l)] = {k: float(np.mean(v)) for k, v in rec.items()} | {"ceiling": float(ceil), "probe": float(prb), "pairs": {sb["phrase"]: {k: rec[k][i] for k in rec} for i, sb in enumerate(sibs)}}
        s = sym[ph][str(l)]; md.append(f"| {ph} | {l} | {s['PJ_own']:.2f} | {s['PJ_diff']:.2f} | {s['J_own']:.2f} | {s['J_diff']:.2f} | {ceil:.2f} | {prb:.2f} |")
json.dump(sym, open(f"{RES}/symmetric_2x2.json", "w"), indent=1)

# ---------------------------------------------------------------- step 3: residual screen over 001 families
md += ["", "## Step 3: Δ_residual = full probe − constituent ceiling (leave-one-cue-out; bootstrap 95% CI over cue groups), mean over pairs", "",
       "| family | L | probe | ceiling | Δ_residual [CI] | J-diff |", "|---|---|---|---|---|---|"]
screen = {}
for F in fam["families"]:
    mem = [m for m in F["members"] if len(rows_for("latent", m["phrase"])) >= 10]
    if len(mem) < 2: continue
    toks_all = sorted({t for m in mem for t in m["ids"]}); screen[F["name"]] = {}
    for l in ALL_L:
        H = acts[l]; per_group = {"probe": [], "ceil": []}; jd = []
        for a, b in itertools.combinations(mem, 2):
            ra, rb = rows_for("latent", a["phrase"]), rows_for("latent", b["phrase"]); ia, ib = [toks_all.index(t) for t in a["ids"]], [toks_all.index(t) for t in b["ids"]]
            La, Lb = lens_logits(H[ra], l, toks_all), lens_logits(H[rb], l, toks_all); jd.append(auc(La[:, ia].sum(1) - La[:, ib].sum(1), Lb[:, ia].sum(1) - Lb[:, ib].sum(1)))
            for g in sorted({cue_group(i) for i in ra + rb}):
                tra, trb = [i for i in ra if cue_group(i) != g], [i for i in rb if cue_group(i) != g]; tea, teb = [i for i in ra if cue_group(i) == g], [i for i in rb if cue_group(i) == g]
                if min(len(tra), len(trb), len(tea), len(teb)) < 2: continue
                d = solve(l, (H[tra].mean(0) - H[trb].mean(0))[None])[0]; per_group["probe"].append(auc(H[tea] @ d, H[teb] @ d))
                Xa, Xb = lens_logits(H[tra], l, toks_all), lens_logits(H[trb], l, toks_all); X = torch.cat([Xa, Xb]); C = torch.cov(X.T) + 1e-3 * torch.eye(X.shape[1])
                w = torch.linalg.solve(C, Xa.mean(0) - Xb.mean(0)); per_group["ceil"].append(auc(lens_logits(H[tea], l, toks_all) @ w, lens_logits(H[teb], l, toks_all) @ w))
        pr, ce = np.array(per_group["probe"]), np.array(per_group["ceil"]); rng = np.random.default_rng(0)
        boots = [float(np.mean(pr[idx]) - np.mean(ce[idx])) for idx in (rng.integers(0, len(pr), len(pr)) for _ in range(NB))]
        screen[F["name"]][str(l)] = {"probe": float(pr.mean()), "ceiling": float(ce.mean()), "delta": float(pr.mean() - ce.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "J_diff": float(np.mean(jd)), "n_groups": int(len(pr))}
        s = screen[F["name"]][str(l)]
        if l in (8, 20, 36, 44, 52, 56, 60, 62): md.append(f"| {F['name']} | {l} | {s['probe']:.2f} | {s['ceiling']:.2f} | {s['delta']:+.2f} [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}] | {s['J_diff']:.2f} |")
json.dump(screen, open(f"{RES}/residual_screen.json", "w"), indent=1)
md += ["", "Families with Δ_residual CI entirely above 0 at any layer ≥ L44: " + ", ".join(sorted({f for f, d in screen.items() for l, s in d.items() if int(l) >= 44 and s["ci"][0] > 0})) + "."]
open(f"{RES}/analysis_cpu.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
