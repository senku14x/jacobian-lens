"""004 eval — matched-norm and matched-damage comparison of swap methods (design §6-§7). CPU only.
Reads results/004-causal-geometry/{items,swaps,pile_damage,first_order_control}.json; writes analysis.md and summary.json,
plots/004-causal-geometry/dose_response.png.
"""
import json, os, itertools
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "004-causal-geometry"); PLT = os.path.join(ROOT, "plots", "004-causal-geometry"); os.makedirs(PLT, exist_ok=True)
S = json.load(open(f"{RES}/swaps.json")); D = json.load(open(f"{RES}/pile_damage.json")); I = json.load(open(f"{RES}/items.json")); G = json.load(open(f"{RES}/first_order_control.json"))
RHOS = S["rhos"]; swaps = S["swaps"]; METHODS = ["PJ", "PJc", "Jc", "rand"]; FAMS = ["New", "South", "San", "North"]; NB = 2000
rng = np.random.default_rng(0)
def runs_of(rec, m, rho):
    rs = [r for r in rec["runs"] if r["rho"] == rho and (r["method"] == m or (m == "rand" and r["method"].startswith("rand")))]
    if not rs: return None
    return {k: float(np.mean([r[k] for r in rs])) for k in ("lpA", "lpB", "kl", "entropy")} | {"flip": float(np.mean([r["top1"] == tB for r in rs]))} if (tB := I["items"][rec["item"]]["tB"]) is not None else None
# per-item metrics
M = {}   # (m, rho) -> dict of arrays over items
for m in METHODS:
    for rho in RHOS:
        dB, dA, sel, flip, klv, ent, fams, pairs = [], [], [], [], [], [], [], []
        for rec in swaps:
            r = runs_of(rec, m, rho)
            if r is None: continue
            dB.append(r["lpB"] - rec["clean"]["lpB"]); dA.append(r["lpA"] - rec["clean"]["lpA"]); sel.append((r["lpB"] - rec["clean"]["lpB"]) - (r["lpA"] - rec["clean"]["lpA"]))
            flip.append(r["flip"]); klv.append(r["kl"]); ent.append(r["entropy"] - rec["clean"]["entropy"]); fams.append(rec["family"]); pairs.append((rec["A"], rec["B"]))
        M[(m, rho)] = {"dB": np.array(dB), "dA": np.array(dA), "sel": np.array(sel), "flip": np.array(flip), "kl": np.array(klv), "dent": np.array(ent), "fam": np.array(fams), "pair": pairs}
dmg = {(m, rho): {"kl": float(np.mean([d["kl"] for d in D["damage"] if d["rho"] == rho and (d["method"] == m or (m == "rand" and d["method"].startswith("rand")))])),
                  "kept": float(np.mean([d["top1_kept"] for d in D["damage"] if d["rho"] == rho and (d["method"] == m or (m == "rand" and d["method"].startswith("rand")))]))} for m in METHODS for rho in RHOS}
def boot_diff(a, b, groups=None):
    d = a - b; idx = np.arange(len(d)); bs = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(NB)]
    return {"mean": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(len(d))}
md = ["# 004 — analysis (computed; interpretation in report.md)", "", f"Items: {len(swaps)} admitted of {I['n_candidates']} candidates; by family " + str({F: int(sum(r['family'] == F for r in swaps)) for F in FAMS}) + f". First-order gate: {G['gate_pass']} (L52 slope {G['verdict']['52']['rho<=0.02']['slope']:.3f} R² {G['verdict']['52']['rho<=0.02']['r2']:.3f}; L56 slope {G['verdict']['56']['rho<=0.02']['slope']:.3f} R² {G['verdict']['56']['rho<=0.02']['r2']:.3f}).", "",
      "Doses are norm-targeted at L52 and L56: r = ρ·median‖h_L52‖ (" + ", ".join(f"ρ={rho}: {S['r_grid']['52'][i]:.1f}" for i, rho in enumerate(RHOS)) + "). Native ‖δ(α=1)‖ medians: " + ", ".join(f"{m} {np.median([r[f'native_norm_{m}'] for r in swaps]):.1f}" for m in ("PJ", "PJc", "Jc")) + ".", "",
      "## Dose response, all items (mean): Δlog P(Y_B) / Δlog P(Y_A) / selectivity / flip rate / item KL / Δentropy | pile damage KL / top-1 kept", "",
      "| method | ρ | ΔlpB | ΔlpA | selectivity | flip | KL@item | Δentropy | pile KL | pile top-1 kept |", "|---|---|---|---|---|---|---|---|---|---|"]
for m in METHODS:
    for rho in RHOS:
        x = M[(m, rho)]; d = dmg[(m, rho)]
        md.append(f"| {m} | {rho} | {x['dB'].mean():+.2f} | {x['dA'].mean():+.2f} | {x['sel'].mean():+.2f} | {x['flip'].mean():.2f} | {x['kl'].mean():.2f} | {x['dent'].mean():+.2f} | {d['kl']:.3f} | {d['kept']:.2f} |")
md += ["", "## Primary statistic: paired selectivity difference at matched norm (bootstrap 95% CI over items)", "", "| ρ | PJ − Jc | PJc − Jc | PJ − rand | Jc − rand | flip PJ / Jc / rand |", "|---|---|---|---|---|---|"]
summary = {"primary": {}}
for rho in RHOS:
    a, b, c, r = M[("PJ", rho)], M[("Jc", rho)], M[("PJc", rho)], M[("rand", rho)]
    s = {"PJ-Jc": boot_diff(a["sel"], b["sel"]), "PJc-Jc": boot_diff(c["sel"], b["sel"]), "PJ-rand": boot_diff(a["sel"], r["sel"]), "Jc-rand": boot_diff(b["sel"], r["sel"])}
    summary["primary"][str(rho)] = s
    f = lambda k: f"{s[k]['mean']:+.2f} [{s[k]['ci'][0]:+.2f}, {s[k]['ci'][1]:+.2f}]"
    md.append(f"| {rho} | {f('PJ-Jc')} | {f('PJc-Jc')} | {f('PJ-rand')} | {f('Jc-rand')} | {a['flip'].mean():.2f} / {b['flip'].mean():.2f} / {r['flip'].mean():.2f} |")
md += ["", "## Per family: selectivity (PJ / PJc / Jc / rand) and flip (PJ / Jc) at each ρ; PJ − Jc CI", "", "| family | ρ | selectivity PJ / PJc / Jc / rand | flip PJ / Jc | PJ − Jc [CI] |", "|---|---|---|---|---|"]
summary["per_family"] = {}
for F in FAMS:
    for rho in RHOS:
        a, b, c, r = M[("PJ", rho)], M[("Jc", rho)], M[("PJc", rho)], M[("rand", rho)]; msk = a["fam"] == F
        if msk.sum() == 0: continue
        bd = boot_diff(a["sel"][msk], b["sel"][msk]); summary["per_family"][f"{F}|{rho}"] = bd | {"sel": {"PJ": float(a["sel"][msk].mean()), "PJc": float(c["sel"][msk].mean()), "Jc": float(b["sel"][msk].mean()), "rand": float(r["sel"][msk].mean())}, "flip": {"PJ": float(a["flip"][msk].mean()), "Jc": float(b["flip"][msk].mean())}}
        md.append(f"| {F} (n={int(msk.sum())}) | {rho} | {a['sel'][msk].mean():+.2f} / {c['sel'][msk].mean():+.2f} / {b['sel'][msk].mean():+.2f} / {r['sel'][msk].mean():+.2f} | {a['flip'][msk].mean():.2f} / {b['flip'][msk].mean():.2f} | {bd['mean']:+.2f} [{bd['ci'][0]:+.2f}, {bd['ci'][1]:+.2f}] |")
# matched damage: interpolate selectivity vs pile KL per method onto a common grid
md += ["", "## Matched damage: mean selectivity interpolated at common pile-KL levels", ""]
grid = np.linspace(min(dmg[(m, RHOS[0])]["kl"] for m in METHODS), min(dmg[(m, RHOS[-1])]["kl"] for m in METHODS), 4)
md += ["| pile KL | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
summary["matched_damage"] = {}
for g in grid:
    row = []
    for m in METHODS:
        xs = np.array([dmg[(m, rho)]["kl"] for rho in RHOS]); ys = np.array([M[(m, rho)]["sel"].mean() for rho in RHOS]); order = np.argsort(xs); v = float(np.interp(g, xs[order], ys[order])); row.append(v); summary["matched_damage"][f"{m}|{g:.4f}"] = v
    md.append(f"| {g:.3f} | " + " | ".join(f"{v:+.2f}" for v in row) + " |")
# reversibility: for each unordered pair, compare A->B and B->A selectivity at rho=0.2
md += ["", "## Reversibility at ρ=0.2: mean selectivity by direction per unordered pair (PJ / Jc)", "", "| pair | A→B PJ / Jc | B→A PJ / Jc |", "|---|---|---|"]
a, b = M[("PJ", 0.2)], M[("Jc", 0.2)]
for p in sorted({tuple(sorted(x)) for x in a["pair"]}):
    fwd = [i for i, x in enumerate(a["pair"]) if x == p]; rev = [i for i, x in enumerate(a["pair"]) if x == (p[1], p[0])]
    if not fwd or not rev: continue
    md.append(f"| {p[0]} ↔ {p[1]} | {a['sel'][fwd].mean():+.2f} / {b['sel'][fwd].mean():+.2f} | {a['sel'][rev].mean():+.2f} / {b['sel'][rev].mean():+.2f} |")
json.dump(summary, open(f"{RES}/summary.json", "w"), indent=1)
open(f"{RES}/analysis.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for m in METHODS:
    ax[0].plot(RHOS, [M[(m, rho)]["sel"].mean() for rho in RHOS], "o-", label=m); ax[1].plot(RHOS, [M[(m, rho)]["flip"].mean() for rho in RHOS], "o-", label=m)
    ax[2].plot([dmg[(m, rho)]["kl"] for rho in RHOS], [M[(m, rho)]["sel"].mean() for rho in RHOS], "o-", label=m)
ax[0].set_xlabel("ρ (norm-targeted dose)"); ax[0].set_ylabel("selectivity Δlog P(Y_B) − Δlog P(Y_A)"); ax[1].set_xlabel("ρ"); ax[1].set_ylabel("top-1 flip rate to Y_B"); ax[2].set_xlabel("pile damage KL"); ax[2].set_ylabel("selectivity")
for x in ax: x.legend(); x.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{PLT}/dose_response.png", dpi=150)
