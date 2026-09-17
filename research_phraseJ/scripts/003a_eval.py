"""003a (eval stage) — reliability, readout, baselines and the constituent-J ceiling for the four phrase objectives.

Inputs: outputs/003a/grads/*.pt (003a_fit), results/003a-phrase-objective/items.json, outputs/001/{acts,null_acts,sigma,unembed}.pt,
released J lens, released template stack, results/001-template-geometry/{families,contexts}.json.
Outputs: results/003a-phrase-objective/{reliability,readout,baselines,analysis}.{json,md}, plots/003a-phrase-objective/*.png
No model forwards. GPU used only for the Σ solves if available.
"""
import json, os, sys, re, itertools, collections
import numpy as np, torch
from safetensors import safe_open
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import jlens
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "003a-phrase-objective"); GR = os.path.join(ROOT, "outputs", "003a", "grads")
RES1 = os.path.join(ROOT, "results", "001-template-geometry"); O1 = os.path.join(ROOT, "outputs", "001")
PLT = os.path.join(ROOT, "plots", "003a-phrase-objective"); os.makedirs(PLT, exist_ok=True)
LENS = "/content/lenses/qwen3.6-27b"; TL = f"{LENS}/template-lens"
LAYERS = [36, 44, 52, 56, 60]; OBJ = ["logp", "lin", "logit", "odds"]; RED = ["at", "mean"]; N_FIT = int(os.environ.get("N_FIT", 20))
dev = "cuda" if torch.cuda.is_available() else "cpu"; LAM = 0.1; NBOOT = 1000; RHO_OK = 0.9
def safe(s): return re.sub(r"[^A-Za-z0-9]+", "_", s)
def cos(a, b): return float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))
def auc(p, n):
    p = np.asarray(p, float); n = np.asarray(n, float)
    return float("nan") if len(p) == 0 or len(n) == 0 else float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())

items = json.load(open(f"{RES}/items.json")); fam = json.load(open(f"{RES1}/families.json")); ctx1 = json.load(open(f"{RES1}/contexts.json"))
members = {m["phrase"]: m for F in fam["families"] for m in F["members"]}; fam_of = {m["phrase"]: F for F in fam["families"] for m in F["members"]}
A = torch.load(f"{O1}/acts.pt", weights_only=False); index = A["index"]; acts = {l: A["acts"][l].float() for l in LAYERS}
NULL = torch.load(f"{O1}/null_acts.pt", weights_only=False); null = {l: NULL[l].float() for l in LAYERS}
S = torch.load(f"{O1}/sigma.pt", weights_only=False); U = torch.load(f"{O1}/unembed.pt", weights_only=False)
W_U = U["W_U"].float(); gamma = U["final_norm_weight"]; eps = U["norm_eps"] or 1e-6
Jl = jlens.JacobianLens.load(f"{LENS}/j-lens/lens.pt")
def q_vec(t): return (1 + gamma) * W_U[t]
def rmsnorm(x): return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps)
def lens_logits(h, l, toks): return rmsnorm(Jl.transport(h, l)) @ torch.stack([q_vec(t) for t in toks]).T
def Jrow(l, t): return Jl.jacobians[l].float().T @ q_vec(t)
chol = {}
for l in LAYERS:
    mu, Sg, _ = S["pooled"][l]; tau = Sg.diagonal().mean(); chol[l] = torch.linalg.cholesky(((1 - LAM) * Sg + LAM * tau * torch.eye(Sg.shape[0])).to(dev))
def solve(l, v): return torch.cholesky_solve(v.to(dev).T.contiguous(), chol[l]).T.cpu()
def rows_for(cond, ph, pos=-1): return [i for i, r in enumerate(index) if r["cond"] == cond and r["phrase"] == ph and r["pos"] == pos]
ROUTE_ORDER = ["continent", "language", "currency", "hemisphere", "second_word_letters"]
def cue_group(i): r = index[i]; return (r["frame"] + ROUTE_ORDER.index(r["route"])) % 5
with safe_open(f"{TL}/templates+phrases_v3.safetensors", framework="pt") as f: T_all = f.get_tensor("templates")
trows = [l.rstrip("\n").split("\t", 1)[1] for l in open(f"{TL}/template_words+phrases_v3.txt")]; row_of = {t: i for i, t in enumerate(trows)}

# ---------------------------------------------------------------- load gradients, build averaged vectors
G = {}   # (name, regime) -> {"k":..,"ids":..,"kind":..,"recs":[...]}
for it in items:
    p = f"{GR}/{safe(it['name'])}__{it['regime']}.pt"
    if os.path.exists(p): G[(it["name"], it["regime"])] = torch.load(p, weights_only=False)
print(f"loaded {len(G)}/{len(items)} items")
def vec(rec, o, l, red, which, k):
    g = rec["obj"][o][l][red].float()                     # [m, d]
    return g[k:].sum(0) if which == "cond" else (g.sum(0) if which == "seq" else g[0])
def avg(recs, o, l, red, which, k, sel=None):
    rs = recs if sel is None else [recs[i] for i in sel]
    return torch.stack([vec(r, o, l, red, which, k) for r in rs]).mean(0)

# ---------------------------------------------------------------- reliability, norm, cross-regime
rel = {}
for (name, regime), d in G.items():
    recs, k = d["recs"], d["prefix_len"]; n = len(recs); key = f"{name}__{regime}"; rel[key] = {"kind": d["kind"], "n": n, "mean_logp": [float(np.mean([r["logps"][i] for r in recs])) for i in range(len(d["ids"]))], "per": {}}
    for o in OBJ:
        for red in RED:
            for l in LAYERS:
                e = {}
                for nn in (5, 10, 20):
                    if nn > n: continue
                    h1 = [i for i in range(nn) if i % 2 == 0]; h2 = [i for i in range(nn) if i % 2 == 1]
                    e[f"splithalf_cond_n{nn}"] = cos(avg(recs, o, l, red, "cond", k, h1), avg(recs, o, l, red, "cond", k, h2))
                    e[f"splithalf_seq_n{nn}"] = cos(avg(recs, o, l, red, "seq", k, h1), avg(recs, o, l, red, "seq", k, h2))
                vc = avg(recs, o, l, red, "cond", k); v1 = avg(recs, o, l, red, "first", k)
                e["norm_ratio_cond_over_g1"] = float(vc.norm() / v1.norm()); e["cos_cond_g1"] = cos(vc, v1)
                e["norm_ratio_percontext_median"] = float(np.median([float(vec(r, o, l, red, "cond", k).norm() / vec(r, o, l, red, "first", k).norm()) for r in recs]))
                rel[key]["per"][f"{o}|{red}|{l}"] = e
xreg = {}
for it in items:
    if it["regime"] != "generic_medium" or (it["name"], "natural") not in G: continue
    dn, dg = G[(it["name"], "natural")], G[(it["name"], "generic_medium")]; k = dn["prefix_len"]
    xreg[it["name"]] = {f"{o}|{red}|{l}": {"cond": cos(avg(dn["recs"], o, l, red, "cond", k), avg(dg["recs"], o, l, red, "cond", k)),
                                          "seq": cos(avg(dn["recs"], o, l, red, "seq", k), avg(dg["recs"], o, l, red, "seq", k))} for o in OBJ for red in RED for l in LAYERS}
json.dump({"reliability": rel, "cross_regime": xreg}, open(f"{RES}/reliability.json", "w"), indent=1)

# ---------------------------------------------------------------- positive control: g1^lin (source-mean) vs released J row
pc = {}
for (name, regime), d in G.items():
    if d["kind"] != "primary" or regime != "natural": continue
    pc[name] = {str(l): cos(avg(d["recs"], "lin", l, "mean", "first", d["prefix_len"]), Jrow(l, d["ids"][0])) for l in LAYERS}
print("positive control g1^lin(source-mean) vs J_rel row:", {n: {l: round(v, 3) for l, v in c.items()} for n, c in pc.items()})

# ---------------------------------------------------------------- readout on latent and held-out emission states; baselines
FAMS = {"New Zealand": "New", "South Korea": "South", "San Diego": "San", "North Carolina": "North"}
def zscore(v, l):
    s = null[l] @ v; return float(s.mean()), float(s.std() + 1e-8)
read = {}; base = {}
for ph, fname in FAMS.items():
    F = [f for f in fam["families"] if f["name"] == fname][0]; k = F["prefix_len"]
    mem = [m for m in F["members"] if (m["phrase"], "natural") in G and len(rows_for("latent", m["phrase"])) >= 10]
    if ph not in [m["phrase"] for m in mem]: continue
    sibs = [m for m in mem if m["phrase"] != ph]
    lat = {m["phrase"]: rows_for("latent", m["phrase"]) for m in mem}
    emis = {m["phrase"]: [i for j, i in enumerate(rows_for("emission_natural", m["phrase"])) if j >= len(G[(m["phrase"], "natural")]["recs"])] for m in mem}  # held-out: not used for fitting
    read[ph] = {}
    for o in OBJ:
        for red in RED:
            for l in LAYERS:
                H = acts[l]; r = {}
                v = {m["phrase"]: avg(G[(m["phrase"], "natural")]["recs"], o, l, red, "cond", k) for m in mem}
                stab = {m["phrase"]: rel[f"{m['phrase']}__natural"]["per"][f"{o}|{red}|{l}"].get(f"splithalf_cond_n{min(20, len(G[(m['phrase'],'natural')]['recs']))}", float("nan")) for m in mem}
                # pairwise AUC, own-vector direction and difference direction
                for b in sibs:
                    pa = f"{ph}|{b['phrase']}"
                    r[pa] = {"own_latent": auc(H[lat[ph]] @ v[ph], H[lat[b["phrase"]]] @ v[ph]),
                             "diff_latent": auc(H[lat[ph]] @ (v[ph] - v[b["phrase"]]), H[lat[b["phrase"]]] @ (v[ph] - v[b["phrase"]])),
                             "own_emission_heldout": auc(H[emis[ph]] @ v[ph], H[emis[b["phrase"]]] @ v[ph]) if emis[ph] and emis[b["phrase"]] else float("nan")}
                # within-family argmax accuracy with z-scored scores (chance = 1/len(mem)); unstable members' vectors still used
                zs = {m["phrase"]: zscore(v[m["phrase"]], l) for m in mem}
                Hs = torch.cat([H[lat[m["phrase"]]] for m in mem]); y = np.concatenate([[i] * len(lat[m["phrase"]]) for i, m in enumerate(mem)])
                sc = torch.stack([((Hs @ v[m["phrase"]]) - zs[m["phrase"]][0]) / zs[m["phrase"]][1] for m in mem], 1)
                r["within_family_acc_z"] = float((sc.argmax(1).numpy() == y).mean()); r["within_family_chance"] = 1 / len(mem)
                r["stable_members"] = {p: (s >= RHO_OK) for p, s in stab.items()}
                r["own_latent_mean"] = float(np.nanmean([r[f"{ph}|{b['phrase']}"]["own_latent"] for b in sibs]))
                r["own_latent_mean_headline"] = r["own_latent_mean"] if stab[ph] >= RHO_OK else 0.5   # unstable vector counts as failure
                # residual test: how much of v_cond lies in span of constituent J rows of the family
                toks = sorted({t for m in mem for t in m["ids"]}); B = torch.stack([Jrow(l, t) for t in toks]).T; Q, _ = torch.linalg.qr(B)
                r["frac_in_constituent_span"] = float((Q.T @ v[ph]).norm() ** 2 / (v[ph].norm() ** 2))
                read[ph][f"{o}|{red}|{l}"] = r
    # baselines on the same latent pairs (independent of objective)
    base[ph] = {}
    for l in LAYERS:
        H = acts[l]; b = {}
        toks_all = sorted({t for m in mem for t in m["ids"]})
        for sb in sibs:
            pa = f"{ph}|{sb['phrase']}"; La, Lb = lens_logits(H[lat[ph]], l, toks_all), lens_logits(H[lat[sb["phrase"]]], l, toks_all)
            ia, ib = [toks_all.index(t) for t in members[ph]["ids"]], [toks_all.index(t) for t in sb["ids"]]
            b[pa] = {"J_first": auc(La[:, ia[0]], Lb[:, ia[0]]) if members[ph]["ids"][0] == sb["ids"][0] else float("nan"),
                     "J_sum": auc(La[:, ia].sum(1) - La[:, ib].sum(1), Lb[:, ia].sum(1) - Lb[:, ib].sum(1)),
                     "J_mean": auc(La[:, ia].mean(1) - La[:, ib].mean(1), Lb[:, ia].mean(1) - Lb[:, ib].mean(1))}
            # constituent-J subspace ceiling and full probe, leave-one-cue-out, LDA on the pair
            ra, rb = lat[ph], lat[sb["phrase"]]; groups = sorted({cue_group(i) for i in ra + rb}); cc, fp = [], []
            for g in groups:
                tra, trb = [i for i in ra if cue_group(i) != g], [i for i in rb if cue_group(i) != g]; tea, teb = [i for i in ra if cue_group(i) == g], [i for i in rb if cue_group(i) == g]
                if min(len(tra), len(trb), len(tea), len(teb)) < 2: continue
                Xa, Xb = lens_logits(H[tra], l, toks_all), lens_logits(H[trb], l, toks_all); X = torch.cat([Xa, Xb]); mu_a, mu_b = Xa.mean(0), Xb.mean(0)
                C = torch.cov(X.T) + 1e-3 * torch.eye(X.shape[1]); w = torch.linalg.solve(C, mu_a - mu_b)
                cc.append(auc(lens_logits(H[tea], l, toks_all) @ w, lens_logits(H[teb], l, toks_all) @ w))
                d = solve(l, (H[tra].mean(0) - H[trb].mean(0))[None])[0]; fp.append(auc(H[tea] @ d, H[teb] @ d))
            b[pa]["constituent_ceiling"] = float(np.mean(cc)) if cc else float("nan"); b[pa]["full_probe"] = float(np.mean(fp)) if fp else float("nan")
            if members[ph]["in_template_vocab"] and sb["in_template_vocab"]:
                t = T_all[l, row_of[members[ph]["template_text"]]].float() - T_all[l, row_of[sb["template_text"]]].float(); b[pa]["template_diff"] = auc(H[ra] @ t, H[rb] @ t)
        base[ph][str(l)] = b
json.dump({"readout": read, "baselines": base, "positive_control": pc}, open(f"{RES}/readout.json", "w"), indent=1)

# ---------------------------------------------------------------- analysis.md
md = ["# 003a — analysis (computed numbers; interpretation in report.md)", "", f"Items loaded: {len(G)}/{len(items)}. Layers {LAYERS}. Objectives {OBJ}. Reductions {RED} (`at` = at t′, primary; `mean` = source-mean).", ""]
md += ["## Positive control: g₁^lin (source-mean) vs released J row, cosine", "", "| phrase | " + " | ".join(f"L{l}" for l in LAYERS) + " |", "|---|" + "---|" * len(LAYERS)]
for n, c in pc.items(): md.append(f"| {n} | " + " | ".join(f"{c[str(l)]:.3f}" for l in LAYERS) + " |")
md += ["", "## Reliability: split-half cosine of v_cond at n=20 (natural regime), and ‖v_cond‖/‖g₁‖ (averaged vectors), primary phrases and nulls", ""]
for red in RED:
    md += [f"### reduction = {red}", "", "| item | mean logp (tok1, tok2) | " + " | ".join(f"{o} ρ@L{l} / ratio" for o in OBJ for l in (44, 56)) + " |", "|---|---|" + "---|" * (2 * len(OBJ))]
    for key, e in rel.items():
        if not key.endswith("__natural"): continue
        cells = []
        for o in OBJ:
            for l in (44, 56):
                p = e["per"][f"{o}|{red}|{l}"]; cells.append(f"{p.get('splithalf_cond_n20', p.get('splithalf_cond_n10', float('nan'))):.2f} / {p['norm_ratio_cond_over_g1']:.3f}")
        md.append(f"| {key.replace('__natural','')} ({e['kind']}, n={e['n']}) | {e['mean_logp'][0]:+.2f}, {e['mean_logp'][-1]:+.2f} | " + " | ".join(cells) + " |")
    md.append("")
md += ["## Cross-regime cosine (natural vs generic-medium), v_cond, primary phrases", "", "| phrase | " + " | ".join(f"{o} @L{l}" for o in OBJ for l in (44, 56)) + " |", "|---|" + "---|" * (2 * len(OBJ))]
for n, x in xreg.items(): md.append(f"| {n} | " + " | ".join(f"{x[f'{o}|at|{l}']['cond']:.2f}" for o in OBJ for l in (44, 56)) + " |")
md += ["", "## Latent readout (fixed v_cond of the phrase, natural regime): mean pairwise AUC vs siblings; headline counts unstable (ρ<0.9) as 0.5", ""]
for red in RED:
    md += [f"### reduction = {red}", "", "| phrase | L | " + " | ".join(f"{o} own / headline / acc_z" for o in OBJ) + " | J_first | J_sum | J_mean | constituent ceiling | full probe |", "|---|---|" + "---|" * (len(OBJ) + 5)]
    for ph in read:
        for l in LAYERS:
            cells = [f"{read[ph][f'{o}|{red}|{l}']['own_latent_mean']:.2f} / {read[ph][f'{o}|{red}|{l}']['own_latent_mean_headline']:.2f} / {read[ph][f'{o}|{red}|{l}']['within_family_acc_z']:.2f}" for o in OBJ]
            b = base[ph][str(l)]; pairs = list(b)
            bm = lambda key: np.nanmean([b[p].get(key, np.nan) for p in pairs])
            md.append(f"| {ph} | {l} | " + " | ".join(cells) + f" | {bm('J_first'):.2f} | {bm('J_sum'):.2f} | {bm('J_mean'):.2f} | {bm('constituent_ceiling'):.2f} | {bm('full_probe'):.2f} |")
    md.append("")
md += ["## Fraction of v_cond energy inside span{constituent J rows of the family} (natural, at t′)", "", "| phrase | " + " | ".join(f"{o} @L{l}" for o in OBJ for l in (44, 56)) + " |", "|---|" + "---|" * (2 * len(OBJ))]
for ph in read: md.append(f"| {ph} | " + " | ".join(f"{read[ph][f'{o}|at|{l}']['frac_in_constituent_span']:.2f}" for o in OBJ for l in (44, 56)) + " |")
open(f"{RES}/analysis.md", "w").write("\n".join(md) + "\n")

# ---------------------------------------------------------------- plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for o in OBJ:
    ys = [np.nanmean([rel[f"{p}__natural"]["per"][f"{o}|at|{l}"].get("splithalf_cond_n20", np.nan) for p in FAMS if f"{p}__natural" in rel]) for l in LAYERS]
    axes[0].plot(LAYERS, ys, "o-", label=o)
axes[0].set_ylim(-0.1, 1.02); axes[0].set_xlabel("layer"); axes[0].set_ylabel("split-half cos of v_cond (n=20, at t′)"); axes[0].set_title("reliability, geographic phrases"); axes[0].legend()
for o in OBJ:
    ys = [np.nanmean([read[p][f"{o}|at|{l}"]["own_latent_mean"] for p in read]) for l in LAYERS]; axes[1].plot(LAYERS, ys, "o-", label=f"{o} v_cond")
axes[1].plot(LAYERS, [np.nanmean([np.nanmean([base[p][str(l)][pa]["J_sum"] for pa in base[p][str(l)]]) for p in base]) for l in LAYERS], "k--", label="J-sum")
axes[1].plot(LAYERS, [np.nanmean([np.nanmean([base[p][str(l)][pa]["constituent_ceiling"] for pa in base[p][str(l)]]) for p in base]) for l in LAYERS], "k:", label="constituent-J ceiling")
axes[1].plot(LAYERS, [np.nanmean([np.nanmean([base[p][str(l)][pa]["full_probe"] for pa in base[p][str(l)]]) for p in base]) for l in LAYERS], "k-", label="full probe")
axes[1].axhline(0.5, color="gray", lw=0.5); axes[1].set_ylim(0.3, 1.02); axes[1].set_xlabel("layer"); axes[1].set_ylabel("mean pairwise latent AUC"); axes[1].set_title("latent readout vs baselines"); axes[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{PLT}/reliability_and_readout.png", dpi=150)
print(open(f"{RES}/analysis.md").read())
