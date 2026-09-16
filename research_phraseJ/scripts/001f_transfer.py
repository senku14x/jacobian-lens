"""001f — Emission -> latent transfer (and reverse) on the 001 activations. No model forwards.

For each family pair (a, b) with enough emission-natural and latent contexts:
  d_emis  = (Σ_λ)^-1 (μ_a^emis - μ_b^emis)   fit on ALL emission-natural contexts, evaluated on ALL latent (fixed direction)
  d_lat   = (Σ_λ)^-1 (μ_a^lat  - μ_b^lat)    fit on ALL latent, evaluated on ALL emission-natural
  t_a - t_b (released template) on both, where both rows exist
  in-condition reference: 5-fold (emission) / leave-one-cue-out (latent) from discrimination.json
Also within-family multiclass transfer: LDA fit on emission (means + shared Σ), argmax on latent.
Writes results/001-template-geometry/transfer.json and plots/001-template-geometry/transfer_*.png
"""
import json, os, itertools
import numpy as np, torch
from safetensors import safe_open
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "001-template-geometry"); OUTD = os.path.join(ROOT, "outputs", "001"); PLT = os.path.join(ROOT, "plots", "001-template-geometry")
TL = "/content/lenses/qwen3.6-27b/template-lens"; LAM = 0.1; NBOOT = 2000
fam = json.load(open(f"{RES}/families.json")); A = torch.load(f"{OUTD}/acts.pt"); index = A["index"]; LAYERS = A["layers"]; S = torch.load(f"{OUTD}/sigma.pt")
with safe_open(f"{TL}/templates+phrases_v3.safetensors", framework="pt") as f: T = f.get_tensor("templates")
rows = [l.rstrip("\n").split("\t", 1)[1] for l in open(f"{TL}/template_words+phrases_v3.txt")]; row_of = {t: i for i, t in enumerate(rows)}
dev = "cuda"
chol = {}
for l in LAYERS:
    mu, Sg, _ = S["pooled"][l]; tau = Sg.diagonal().mean(); chol[l] = torch.linalg.cholesky(((1 - LAM) * Sg + LAM * tau * torch.eye(Sg.shape[0])).to(dev))
def solve(l, v): return torch.cholesky_solve(v.to(dev).T.contiguous(), chol[l]).T.cpu()
def rows_for(cond, ph): return [i for i, r in enumerate(index) if r["cond"] == cond and r["phrase"] == ph and r["pos"] == -1]
def auc(p, n): p = np.asarray(p); n = np.asarray(n); return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())
def auc_ci(p, n):
    rng = np.random.default_rng(0); p = np.asarray(p); n = np.asarray(n)
    v = [auc(p[rng.integers(0, len(p), len(p))], n[rng.integers(0, len(n), len(n))]) for _ in range(NBOOT)]; return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
out = {}
for F in fam["families"]:
    mem = [m for m in F["members"] if len(rows_for("emission_natural", m["phrase"])) >= 20 and len(rows_for("latent", m["phrase"])) >= 10]
    if len(mem) < 2: continue
    fo = {"members": [m["phrase"] for m in mem], "pairs": {}, "multiclass": {}}
    for a, b in itertools.combinations(mem, 2):
        pk = f"{a['phrase']}|{b['phrase']}"; fo["pairs"][pk] = {}
        ea, eb, la, lb = rows_for("emission_natural", a["phrase"]), rows_for("emission_natural", b["phrase"]), rows_for("latent", a["phrase"]), rows_for("latent", b["phrase"])
        for l in LAYERS:
            H = A["acts"][l].float(); r = {}
            d_e = solve(l, (H[ea].mean(0) - H[eb].mean(0))[None])[0]; d_l = solve(l, (H[la].mean(0) - H[lb].mean(0))[None])[0]
            r["emis_to_latent"] = {"auc": auc((H[la] @ d_e).numpy(), (H[lb] @ d_e).numpy()), "ci": auc_ci((H[la] @ d_e).numpy(), (H[lb] @ d_e).numpy())}
            r["latent_to_emis"] = {"auc": auc((H[ea] @ d_l).numpy(), (H[eb] @ d_l).numpy()), "ci": auc_ci((H[ea] @ d_l).numpy(), (H[eb] @ d_l).numpy())}
            r["cos_d_emis_d_latent"] = float(torch.nn.functional.cosine_similarity(d_e, d_l, dim=0))
            # the same in the Mahalanobis (score) metric: correlation of the two directions' scores on null-free data = d_eᵀ Σ d_l normalized
            if a["in_template_vocab"] and b["in_template_vocab"]:
                t = T[l, row_of[a["template_text"]]].float() - T[l, row_of[b["template_text"]]].float()
                r["template_on_latent"] = {"auc": auc((H[la] @ t).numpy(), (H[lb] @ t).numpy())}; r["template_on_emis"] = {"auc": auc((H[ea] @ t).numpy(), (H[eb] @ t).numpy())}
                r["cos_template_d_emis"] = float(torch.nn.functional.cosine_similarity(t, d_e, dim=0)); r["cos_template_d_latent"] = float(torch.nn.functional.cosine_similarity(t, d_l, dim=0))
            fo["pairs"][pk][str(l)] = r
    # multiclass: fit on emission, test on latent (and reverse)
    for l in LAYERS:
        H = A["acts"][l].float(); mc = {}
        for src, dst in [("emission_natural", "latent"), ("latent", "emission_natural")]:
            mus = torch.stack([H[rows_for(src, m["phrase"])].mean(0) for m in mem]); W = solve(l, mus); bias = -0.5 * (W * mus).sum(1)
            correct = total = 0
            for ci, m in enumerate(mem):
                te = rows_for(dst, m["phrase"]); pred = (H[te] @ W.T + bias).argmax(1).numpy(); correct += (pred == ci).sum(); total += len(te)
            mc[f"{src}->{dst}"] = correct / total
        mc["chance"] = 1 / len(mem); fo["multiclass"][str(l)] = mc
    out[F["name"]] = fo
json.dump(out, open(f"{RES}/transfer.json", "w"), indent=1)
# summary + plot
fig, ax = plt.subplots(figsize=(8, 4.5))
print(f"{'family':8s} | emis→latent best (L) | latent→emis best (L) | multiclass emis→latent best | template→latent")
for Fn, fo in out.items():
    e2l = [np.mean([fo["pairs"][pk][str(l)]["emis_to_latent"]["auc"] for pk in fo["pairs"]]) for l in LAYERS]
    l2e = [np.mean([fo["pairs"][pk][str(l)]["latent_to_emis"]["auc"] for pk in fo["pairs"]]) for l in LAYERS]
    mcl = [fo["multiclass"][str(l)]["emission_natural->latent"] for l in LAYERS]
    tl = [np.nanmean([fo["pairs"][pk][str(l)].get("template_on_latent", {}).get("auc", np.nan) for pk in fo["pairs"]]) for l in LAYERS]
    bi = int(np.argmax(e2l)); bj = int(np.argmax(l2e)); bk = int(np.argmax(mcl))
    tstr = f"{np.nanmax(tl):.2f}" if not np.all(np.isnan(tl)) else "-"
    print(f"{Fn:8s} | {e2l[bi]:.2f} (L{LAYERS[bi]}) | {l2e[bj]:.2f} (L{LAYERS[bj]}) | {mcl[bk]:.2f} (L{LAYERS[bk]}, chance {fo['multiclass'][str(LAYERS[0])]['chance']:.2f}) | {tstr}")
    ax.plot(LAYERS, e2l, label=f"{Fn} emission→latent"); ax.plot(LAYERS, l2e, "--", alpha=0.6, label=f"{Fn} latent→emission")
ax.axhline(0.5, color="k", lw=0.5); ax.set_ylim(0.3, 1.02); ax.set_xlabel("layer"); ax.set_ylabel("pairwise AUC (fixed direction)"); ax.legend(fontsize=6, ncol=2); plt.tight_layout(); plt.savefig(f"{PLT}/transfer_emission_latent.png", dpi=150)
print("\nlayer profile, mean over families (emis→latent / latent→emis / multiclass emis→latent):")
print("  " + "  ".join(f"L{l}:{np.mean([np.mean([fo['pairs'][pk][str(l)]['emis_to_latent']['auc'] for pk in fo['pairs']]) for fo in out.values()]):.2f}/{np.mean([np.mean([fo['pairs'][pk][str(l)]['latent_to_emis']['auc'] for pk in fo['pairs']]) for fo in out.values()]):.2f}/{np.mean([fo['multiclass'][str(l)]['emission_natural->latent'] for fo in out.values()]):.2f}" for l in LAYERS))
