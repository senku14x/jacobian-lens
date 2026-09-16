"""001e — Analysis for 001: template geometry, labelled discrimination with a probe ceiling, and
covariance infrastructure. CPU/GPU, no model forward passes.

Inputs: results/001-template-geometry/{families,contexts}.json; outputs/001/{acts,sigma,null_acts,unembed}.pt;
released template stack (v3), released J and R lenses, outputs/001/token_freq.json.
Outputs: results/001-template-geometry/{geometry,discrimination,covariance_stability,scoring_convention}.*,
plots/001-template-geometry/*.png
"""
import json, os, sys, math, random, itertools, collections
import numpy as np, torch
from safetensors import safe_open
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jlens, transformers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "001-template-geometry"); OUTD = os.path.join(ROOT, "outputs", "001")
PLT = os.path.join(ROOT, "plots", "001-template-geometry"); os.makedirs(PLT, exist_ok=True)
TL = "/content/lenses/qwen3.6-27b/template-lens"; LENS = "/content/lenses/qwen3.6-27b"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0); np.random.seed(0)
LAM = [0.01, 0.1, 0.3]; NBOOT = 2000
AUC_HI, AUC_LO, COS_COL = 0.85, 0.65, 0.95

fam = json.load(open(f"{RES}/families.json")); ctx = json.load(open(f"{RES}/contexts.json"))
A = torch.load(f"{OUTD}/acts.pt"); LAYERS = A["layers"]; index = A["index"]
acts = {l: A["acts"][l].float() for l in LAYERS}
S = torch.load(f"{OUTD}/sigma.pt"); U = torch.load(f"{OUTD}/unembed.pt")
W_U = U["W_U"].float(); gamma = U["final_norm_weight"]; eps = U["norm_eps"] or 1e-6
freq = {int(k): v for k, v in json.load(open(f"{OUTD}/token_freq.json")).items()}
tok = transformers.AutoTokenizer.from_pretrained("/content/models/Qwen3.6-27B")

# ------------------------------------------------------------------ template stack and lenses
with safe_open(f"{TL}/templates+phrases_v3.safetensors", framework="pt") as f:
    T_all = f.get_tensor("templates")  # [64, 13731, 5120] bf16
rows = [l.rstrip("\n").split("\t", 1)[1] for l in open(f"{TL}/template_words+phrases_v3.txt")]
row_of = {t: i for i, t in enumerate(rows)}
def trow(text, l): return T_all[l, row_of[text]].float()
Jlens = jlens.JacobianLens.load(f"{LENS}/j-lens/lens.pt"); Rlens = jlens.JacobianLens.load(f"{LENS}/r-lens/lens.pt")
def q_vec(t): return (1 + gamma) * W_U[t]          # (1+γ)⊙W_U[t]
def rmsnorm(x): return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps)
def lens_logits(lens, h, l, toks):                   # exact readout logits for chosen tokens
    z = lens.transport(h, l) if lens is not None else h
    return rmsnorm(z) @ torch.stack([q_vec(t) for t in toks]).T
def sigma(l, half="pooled", lam=0.1):
    mu, Sg, n = S[half][l]; tau = Sg.diagonal().mean()
    return mu, (1 - lam) * Sg + lam * tau * torch.eye(Sg.shape[0])

# scoring convention (locked): template score = cosine(t_w, h_raw); pairwise = (t_a - t_b)ᵀh
open(f"{RES}/scoring_convention.md", "w").write(
"""# Scoring convention for the released template lens (locked before ranks were computed)
- Rows are stored pre-whitened (`space=raw`, `ridge_c=0.001`); no Σ shipped.
- Full-universe score: s_w(h) = cos(t_w, h) on the raw residual (HF README: "scored by cosine of the per-layer residual").
- Pairwise score: (t_a - t_b)ᵀ h (intercept-free; AUC is invariant to scale).
- Residual convention: jlens hook = output of block l. Template layer alignment is verified empirically below
  (cos(t_w[l+δ], h[l]) for δ ∈ {-1,0,+1} on emission-template contexts; the δ with the highest mean cosine is used).
""")

# ------------------------------------------------------------------ helpers
def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
def auc_ci(pos, neg, nb=NBOOT):
    pos = np.asarray(pos); neg = np.asarray(neg); rng = np.random.default_rng(0)
    if len(pos) < 2 or len(neg) < 2: return [float("nan")] * 2
    v = [auc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(nb)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def cos(a, b): return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))

members = {m["phrase"]: m for F in fam["families"] for m in F["members"]}
members.update({m["phrase"]: m for P in fam["non_family_pairs"] for m in P["members"]})
def rows_for(cond, phrase, pos=-1):
    return [i for i, r in enumerate(index) if r["cond"] == cond and r["phrase"] == phrase and r["pos"] == pos]

# ------------------------------------------------------------------ template layer alignment check
align = {}
for delta in (-1, 0, 1):
    cs = []
    for ph, m in members.items():
        if not m["in_template_vocab"]: continue
        ri = rows_for("emission_template", ph)
        for l in LAYERS:
            if 0 <= l + delta < 64:
                t = trow(m["template_text"], l + delta); h = acts[l][ri]
                cs.append(torch.nn.functional.cosine_similarity(h, t[None], dim=1).mean().item())
    align[delta] = float(np.mean(cs))
DELTA = max(align, key=align.get)
open(f"{RES}/scoring_convention.md", "a").write(f"\nAlignment check (mean cosine template row vs residual on emission-template contexts): {align}; using δ={DELTA}.\n")
def tvec(text, l): return trow(text, min(max(l + DELTA, 0), 63))

PARTS = os.environ.get('PARTS', 'ABC')
# ------------------------------------------------------------------ Part A: vector geometry
geometry = {"families": {}, "controls": {}}
if 'A' not in PARTS: geometry = json.load(open(f"{RES}/geometry.json"))
Sig = {l: sigma(l)[1] for l in LAYERS}
unrel_D2 = {l: [] for l in LAYERS}
def pair_geom(ta, tb, l):
    d = {"cos": cos(ta, tb)}
    if l in Sig:
        Sa, Sb = Sig[l] @ ta, Sig[l] @ tb
        d["rho"] = float((ta @ Sb) / math.sqrt((ta @ Sa) * (tb @ Sb)))
        dd = ta - tb; d["D2"] = float(dd @ (Sig[l] @ dd))
    return d
for F in (fam["families"] if 'A' in PARTS else []):
    Tm = [m for m in F["members"] if m["in_template_vocab"]]
    if len(Tm) < 2: continue
    out = {}
    for a, b in itertools.combinations(Tm, 2):
        key = f"{a['phrase']}|{b['phrase']}"
        out[key] = {str(l): pair_geom(tvec(a["template_text"], l), tvec(b["template_text"], l), l) for l in range(64)}
    geometry["families"][F["name"]] = out
for name, pairs in (fam["controls"].items() if 'A' in PARTS else []):
    lst = []
    for P in pairs:
        a, b = P["members"]
        lst.append({str(l): pair_geom(T_all[min(max(l + DELTA, 0), 63), a["template_row"]].float(), T_all[min(max(l + DELTA, 0), 63), b["template_row"]].float(), l) for l in LAYERS})
        if name == "random_unrelated":
            for l in LAYERS: unrel_D2[l].append(lst[-1][str(l)]["D2"])
    geometry["controls"][name] = lst
if 'A' in PARTS: geometry["unrelated_D2_floor"] = {str(l): [float(np.percentile(v, 5)), float(np.median(v)), float(np.percentile(v, 95))] for l, v in unrel_D2.items()}
# template row vs first-token J direction, and vs prefix-word row where present
if 'A' in PARTS: geometry["first_token_alignment"] = {}
for ph, m in (members.items() if 'A' in PARTS else []):
    if not m["in_template_vocab"]: continue
    q = q_vec(m["ids"][0]); d = {}
    for l in LAYERS:
        v = Jlens.transport(q[None], l)[0] if False else (Jlens.jacobians[l].float().T @ q)  # Jᵀ q
        d[str(l)] = {"cos_Jt_q": cos(tvec(m["template_text"], l), v)}
        pre = tok.decode(m["ids"][:1]).strip().lower()
        if pre in row_of: d[str(l)]["cos_prefix_row"] = cos(tvec(m["template_text"], l), trow(pre, l))
    geometry["first_token_alignment"][ph] = d
if 'A' in PARTS: json.dump(geometry, open(f"{RES}/geometry.json", "w"))

# ------------------------------------------------------------------ Part B: labelled discrimination
chol = {(l, lam): torch.linalg.cholesky(sigma(l, "pooled", lam)[1].to(dev)) for l in LAYERS for lam in LAM}
def solve(l, lam, v): return torch.cholesky_solve(v.to(dev).T.contiguous(), chol[(l, lam)]).T.cpu()

def probe_pair_cv(ha, hb, l, k=5):
    """5-fold CV LDA direction (Σ_λ)^-1(μ_a-μ_b); λ picked on inner split. Returns out-of-fold scores."""
    na, nb = len(ha), len(hb); ia = np.arange(na); ib = np.arange(nb); rng = np.random.default_rng(0); rng.shuffle(ia); rng.shuffle(ib)
    fa, fb = np.array_split(ia, k), np.array_split(ib, k); pos, neg = [], []; lam_used = []; fold_auc = []
    for f in range(k):
        tra = np.concatenate([fa[j] for j in range(k) if j != f]); trb = np.concatenate([fb[j] for j in range(k) if j != f])
        # inner split for λ
        best, bestauc = LAM[1], -1
        cut_a, cut_b = int(0.8 * len(tra)), int(0.8 * len(trb))
        for lam in LAM:
            d = solve(l, lam, (ha[tra[:cut_a]].mean(0) - hb[trb[:cut_b]].mean(0))[None])[0]
            a_ = auc((ha[tra[cut_a:]] @ d).numpy(), (hb[trb[cut_b:]] @ d).numpy())
            if a_ > bestauc: best, bestauc = lam, a_
        d = solve(l, best, (ha[tra].mean(0) - hb[trb].mean(0))[None])[0]
        # standardize out-of-fold scores by the training-fold score distribution so folds are comparable
        tr_sc = torch.cat([ha[tra] @ d, hb[trb] @ d]); m_, s_ = tr_sc.mean(), tr_sc.std() + 1e-8
        sp, sn = ((ha[fa[f]] @ d - m_) / s_), ((hb[fb[f]] @ d - m_) / s_)
        pos += sp.tolist(); neg += sn.tolist(); lam_used.append(best); fold_auc.append(auc(sp.numpy(), sn.numpy()))
    return np.array(pos), np.array(neg), lam_used, float(np.mean(fold_auc))

def probe_multiclass_cv(H, y, l, k=5):
    """CV LDA argmax accuracy over members; H [N,d], y labels 0..C-1."""
    N = len(y); idx = np.arange(N); rng = np.random.default_rng(0); rng.shuffle(idx); folds = np.array_split(idx, k)
    correct = 0; C = int(y.max()) + 1
    for f in range(k):
        tr = np.concatenate([folds[j] for j in range(k) if j != f]); te = folds[f]
        mus = torch.stack([H[tr][y[tr] == c].mean(0) for c in range(C)])
        W = solve(l, LAM[1], mus); b = -0.5 * (W * mus).sum(1)
        pred = (H[te] @ W.T + b).argmax(1).numpy(); correct += (pred == y[te]).sum()
    return correct / N


def group_of(i, kind):
    r = index[i]
    if kind == "frame": return r["frame"]                     # true frame-out: hold a frame style out across all routes
    if kind == "rf":    return (r["route"], r["frame"])       # paired same-text comparison (was mislabelled "frame" before 2026-09-16 fix)
    if kind == "route": return r["route"]
    if kind == "cue":   return (r["frame"] + list(ROUTE_ORDER).index(r["route"])) % 5   # cue index shared across members
    return None
ROUTE_ORDER = ["continent", "language", "currency", "hemisphere", "second_word_letters"]
def probe_pair_grouped(ia, ib, l, kind, lam=0.1):
    """Leave-one-group-out LDA: per group, fit on all other items of both classes, score the held-out items.
    Returns per-group AUCs (paired when one item per class per group) and the mean."""
    groups = sorted({group_of(i, kind) for i in ia + ib}, key=str); out = []
    for g in groups:
        te_a = [i for i in ia if group_of(i, kind) == g]; te_b = [i for i in ib if group_of(i, kind) == g]
        tr_a = [i for i in ia if group_of(i, kind) != g]; tr_b = [i for i in ib if group_of(i, kind) != g]
        if not te_a or not te_b or len(tr_a) < 2 or len(tr_b) < 2: continue
        d = solve(l, lam, (acts[l][tr_a].mean(0) - acts[l][tr_b].mean(0))[None])[0]
        out.append(auc((acts[l][te_a] @ d).numpy(), (acts[l][te_b] @ d).numpy()))
    return out, (float(np.mean(out)) if out else float("nan"))
def probe_multiclass_grouped(mem, cond, pos, l, kind, lam=0.1):
    ids = {m["phrase"]: rows_for(cond, m["phrase"], pos) for m in mem}
    groups = sorted({group_of(i, kind) for v in ids.values() for i in v}, key=str); correct = total = 0
    for g in groups:
        tr = {ph: [i for i in v if group_of(i, kind) != g] for ph, v in ids.items()}; te = {ph: [i for i in v if group_of(i, kind) == g] for ph, v in ids.items()}
        if any(len(v) < 2 for v in tr.values()) or not any(te.values()): continue
        mus = torch.stack([acts[l][tr[m["phrase"]]].mean(0) for m in mem]); W = solve(l, lam, mus); b = -0.5 * (W * mus).sum(1)
        for ci, m in enumerate(mem):
            if te[m["phrase"]]:
                pred = (acts[l][te[m["phrase"]]] @ W.T + b).argmax(1).numpy(); correct += (pred == ci).sum(); total += len(pred)
    return correct / total if total else float("nan")

def answer_id(ans): return tok.encode(" " + ans, add_special_tokens=False)[0] if ans else None
disc = {}
if 'B' not in PARTS: disc = json.load(open(f"{RES}/discrimination.json"))
for F in (fam["families"] if 'B' in PARTS else []) + ([{"name": "nonfamily:" + "|".join(m["phrase"] for m in P["members"]), "members": P["members"], "prefix_len": P["prefix_len"]} for P in fam["non_family_pairs"]] if 'B' in PARTS else []):
    Fn = F["name"]; disc[Fn] = {}
    for cond in ["emission_template", "emission_natural", "latent"]:
        pos_list = [-1, -2] if cond == "latent" else [-1]
        for pos in pos_list:
            mem = [m for m in F["members"] if len(rows_for(cond, m["phrase"], pos)) >= (10 if cond != "emission_template" else 10)]
            if len(mem) < 2: continue
            key = cond if pos == -1 else cond + "@-2"; disc[Fn][key] = {"members": [m["phrase"] for m in mem], "n": {m["phrase"]: len(rows_for(cond, m["phrase"], pos)) for m in mem}, "layers": {}}
            for l in LAYERS:
                H = {m["phrase"]: acts[l][rows_for(cond, m["phrase"], pos)] for m in mem}
                R = {"pairs": {}, "within_family_acc": {}}
                # --- pairwise
                for a, b in itertools.combinations(mem, 2):
                    ha, hb = H[a["phrase"]], H[b["phrase"]]; pk = f"{a['phrase']}|{b['phrase']}"; pr = {}
                    if a["in_template_vocab"] and b["in_template_vocab"]:
                        d = tvec(a["template_text"], l) - tvec(b["template_text"], l)
                        sa, sb = (ha @ d).numpy(), (hb @ d).numpy(); pr["template"] = {"auc": auc(sa, sb), "ci": auc_ci(sa, sb)}
                    if cond == "latent":
                        ia_, ib_ = rows_for(cond, a["phrase"], pos), rows_for(cond, b["phrase"], pos); pr["probe"] = {}
                        for kind in ("cue", "frame", "route", "rf"):
                            per, mean_ = probe_pair_grouped(ia_, ib_, l, kind)
                            rng = np.random.default_rng(0); boots = [float(np.mean(rng.choice(per, len(per)))) for _ in range(NBOOT)] if per else []
                            pr["probe"][f"loo_{kind}"] = {"auc": mean_, "n_groups": len(per), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [float("nan")] * 2}
                        pr["probe"]["auc"] = pr["probe"]["loo_cue"]["auc"]; pr["probe"]["ci"] = pr["probe"]["loo_cue"]["ci"]
                    else:
                        sa, sb, lu, fauc = probe_pair_cv(ha, hb, l); pr["probe"] = {"auc": auc(sa, sb), "ci": auc_ci(sa, sb), "fold_mean_auc": fauc, "lambda": collections.Counter(lu).most_common(1)[0][0]}
                    # first-token J (identical for both): AUC on the shared token's J-logit
                    ta, tb = a["ids"], b["ids"]
                    for nm, lens in [("J", Jlens), ("R", Rlens), ("logit", None)]:
                        La = lens_logits(lens, ha, l, ta + tb); Lb = lens_logits(lens, hb, l, ta + tb)
                        # score for phrase a = sum/mean over its tokens; for b likewise; pairwise score = s_a - s_b
                        def sc(L, agg):
                            f_ = (lambda x: x.sum(1)) if agg == "sum" else (lambda x: x.mean(1))
                            return f_(L[:, :len(ta)]) - f_(L[:, len(ta):])
                        for agg in ("sum", "mean"):
                            pr[f"{nm}-{agg}"] = {"auc": auc(sc(La, agg).numpy(), sc(Lb, agg).numpy())}
                        if nm == "J":
                            pr["J-first"] = {"auc": auc(La[:, 0].numpy(), Lb[:, 0].numpy()) if ta[0] == tb[0] else float("nan")}
                    R["pairs"][pk] = pr
                # --- within-family argmax accuracy (probe, template where all T, J-sum/mean)
                Hs = torch.cat([H[m["phrase"]] for m in mem]); y = np.concatenate([[i] * len(H[m["phrase"]]) for i, m in enumerate(mem)])
                R["within_family_acc"]["chance"] = float(max(np.bincount(y)) / len(y))
                R["within_family_acc"]["probe"] = float(probe_multiclass_grouped(mem, cond, pos, l, "cue")) if cond == "latent" else float(probe_multiclass_cv(Hs, y, l))
                if cond == "latent":
                    R["within_family_acc"]["probe_loo_frame"] = float(probe_multiclass_grouped(mem, cond, pos, l, "frame")); R["within_family_acc"]["probe_loo_route"] = float(probe_multiclass_grouped(mem, cond, pos, l, "route"))
                if all(m["in_template_vocab"] for m in mem):
                    Tm = torch.stack([tvec(m["template_text"], l) for m in mem]); sc_ = torch.nn.functional.normalize(Hs, dim=1) @ torch.nn.functional.normalize(Tm, dim=1).T
                    R["within_family_acc"]["template_cos"] = float((sc_.argmax(1).numpy() == y).mean())
                for nm, lens in [("J", Jlens), ("R", Rlens), ("logit", None)]:
                    for agg in ("sum", "mean"):
                        scs = []
                        for m in mem:
                            L = lens_logits(lens, Hs, l, m["ids"]); scs.append(L.sum(1) if agg == "sum" else L.mean(1))
                        R["within_family_acc"][f"{nm}-{agg}"] = float((torch.stack(scs, 1).argmax(1).numpy() == y).mean())
                disc[Fn][key]["layers"][str(l)] = R
            # --- latent extras: leave-one-route-out probe AUC and correctness stratum (best layer by probe)
            if cond == "latent" and pos == -1:
                routes = sorted({index[i]["route"] for m in mem for i in rows_for(cond, m["phrase"], pos)})
                loro = {}
                for l in LAYERS:
                    per_route = []
                    for r in routes:
                        pa = []
                        for a, b in itertools.combinations(mem, 2):
                            tr_a = [i for i in rows_for(cond, a["phrase"], pos) if index[i]["route"] != r]; te_a = [i for i in rows_for(cond, a["phrase"], pos) if index[i]["route"] == r]
                            tr_b = [i for i in rows_for(cond, b["phrase"], pos) if index[i]["route"] != r]; te_b = [i for i in rows_for(cond, b["phrase"], pos) if index[i]["route"] == r]
                            if min(len(te_a), len(te_b), len(tr_a), len(tr_b)) < 2: continue
                            d = solve(l, LAM[1], (acts[l][tr_a].mean(0) - acts[l][tr_b].mean(0))[None])[0]
                            pa.append(auc((acts[l][te_a] @ d).numpy(), (acts[l][te_b] @ d).numpy()))
                        if pa: per_route.append(float(np.mean(pa)))
                    loro[str(l)] = {"mean": float(np.mean(per_route)) if per_route else float("nan"), "per_route": dict(zip(routes, per_route))}
                disc[Fn][key]["probe_leave_one_route_out"] = loro
                corr = {m["phrase"]: [index[i]["greedy_id"] == answer_id(index[i]["answer"]) for i in rows_for(cond, m["phrase"], pos)] for m in mem}
                disc[Fn][key]["first_token_match_rate"] = {k: float(np.mean(v)) for k, v in corr.items()}  # first predicted token == first token of answer; NOT full-answer correctness
if 'B' in PARTS: json.dump(disc, open(f"{RES}/discrimination.json", "w"))

# ------------------------------------------------------------------ full-universe template rank (secondary)
rank_out = {}
for ph, m in (members.items() if 'B' in PARTS else []):
    if not m["in_template_vocab"]: continue
    for cond in ["emission_template", "emission_natural", "latent"]:
        ri = rows_for(cond, ph)
        if len(ri) < 5: continue
        per_l = {}
        for l in LAYERS:
            Tn = torch.nn.functional.normalize(T_all[min(max(l + DELTA, 0), 63)].float().to(dev), dim=1)
            hn = torch.nn.functional.normalize(acts[l][ri].to(dev), dim=1)
            s = hn @ Tn.T; own = s[:, row_of[m["template_text"]]]
            per_l[str(l)] = {"median_rank": float((s > own[:, None]).sum(1).float().median()), "top10_rate": float(((s > own[:, None]).sum(1) < 10).float().mean())}
        rank_out[f"{ph}|{cond}"] = per_l
if 'B' in PARTS: json.dump(rank_out, open(f"{RES}/template_full_universe_rank.json", "w"))

# ------------------------------------------------------------------ Part C: covariance stability and instrument geometry
single = [(t, tok.encode(" " + t, add_special_tokens=False)) for t in rows]
single = [(t, i[0]) for t, i in single if len(i) == 1]
fq = np.array([freq.get(i, 0) for _, i in single]); dec = np.digitize(fq, np.percentile(fq[fq > 0], np.arange(10, 100, 10)))
strat = []
for dcl in range(10):
    cand = [k for k in range(len(single)) if dec[k] == dcl]; strat += random.sample(cand, min(20, len(cand)))
stab = {"members": {}, "single_tokens": {}, "instrument": {}}
if 'C' not in PARTS:
    stab = json.load(open(f"{RES}/covariance_stability.json")); strat = []
def stab_rec(t_by_l):
    d = {}
    for l in LAYERS:
        t = t_by_l(l); d[str(l)] = {}
        for lam in LAM:
            ua, ub = sigma(l, "A", lam)[1] @ t, sigma(l, "B", lam)[1] @ t
            d[str(l)][str(lam)] = {"cos_AB": cos(ua, ub), "norm_ratio": float(ua.norm() / ub.norm())}
    return d
for ph, m in (members.items() if 'C' in PARTS else []):
    if m["in_template_vocab"]: stab["members"][ph] = stab_rec(lambda l, m=m: tvec(m["template_text"], l))
for k in strat:
    t, tid = single[k]
    stab["single_tokens"][t] = {"freq_decile": int(dec[k]), **stab_rec(lambda l, t=t: tvec(t, l))}
    q = q_vec(tid); g = {}
    for l in LAYERS:
        v = Jlens.jacobians[l].float().T @ q; tw = tvec(t, l)
        g[str(l)] = {"cos_J_template": cos(v, tw), "cos_dual": cos(Sig[l] @ v, Sig[l] @ tw)}
    stab["instrument"][t] = g
if 'C' in PARTS: json.dump(stab, open(f"{RES}/covariance_stability.json", "w"))

# ------------------------------------------------------------------ plots
Ls = LAYERS
def fam_pairs_metric(Fn, cond, key):
    out = {}
    for pk in next(iter(disc[Fn][cond]["layers"].values()))["pairs"]:
        out[pk] = [disc[Fn][cond]["layers"][str(l)]["pairs"][pk].get(key, {}).get("auc", np.nan) for l in Ls]
    return out
for Fn, D in disc.items():
    conds = [c for c in D if not c.endswith("@-2")]
    if not conds: continue
    fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 4), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        for key, st in [("probe", "-"), ("template", "--"), ("J-sum", ":"), ("J-mean", "-.")]:
            for pk, v in fam_pairs_metric(Fn, cond, key).items():
                if np.all(np.isnan(v)): continue
                ax.plot(Ls, v, st, label=f"{key} {pk}")
        ax.axhline(0.5, color="k", lw=0.5); ax.set_ylim(0.3, 1.02); ax.set_title(f"{Fn} — {cond}"); ax.set_xlabel("layer"); ax.set_ylabel("pairwise AUC")
        ax.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(f"{PLT}/auc_{Fn.replace(':','_').replace('|','_')}.png", dpi=150); plt.close()
# geometry plot: within-family template cosine vs layer against controls
fig, ax = plt.subplots(figsize=(7, 4))
for Fn, pairs in geometry["families"].items():
    for pk, v in pairs.items(): ax.plot(range(64), [v[str(l)]["cos"] for l in range(64)], label=f"{Fn}: {pk}")
for name, st in [("random_same_prefix", "--"), ("random_unrelated", ":")]:
    M = np.array([[p[str(l)]["cos"] for l in LAYERS] for p in geometry["controls"][name]])
    ax.plot(LAYERS, np.median(M, 0), "k" + st, label=f"{name} (median)")
ax.set_xlabel("layer"); ax.set_ylabel("cos(t_a, t_b)"); ax.legend(fontsize=6); plt.tight_layout(); plt.savefig(f"{PLT}/template_pair_cosine.png", dpi=150); plt.close()
# stability plot
fig, ax = plt.subplots(figsize=(7, 4))
for lam in LAM:
    M = np.array([[stab["single_tokens"][t][str(l)][str(lam)]["cos_AB"] for l in LAYERS] for t in stab["single_tokens"]])
    ax.plot(LAYERS, np.median(M, 0), label=f"λ={lam} (median over 200 rows)"); ax.fill_between(LAYERS, np.percentile(M, 10, 0), np.percentile(M, 90, 0), alpha=0.15)
ax.set_xlabel("layer"); ax.set_ylabel("cos(Σ_A t, Σ_B t)"); ax.legend(); plt.tight_layout(); plt.savefig(f"{PLT}/covariance_dual_stability.png", dpi=150); plt.close()
print("analysis written; alignment", align, "DELTA", DELTA)
