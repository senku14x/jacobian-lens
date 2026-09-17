"""003a post hoc (written after reading analysis.md; labelled post hoc in the report):
(1) difference-direction latent AUC (v_a - v_b) alongside own-vector AUC, at t', L52/56/60;
(2) cross-item cosine structure of v_cond^lin (shared component across phrases, nulls, families) and the top-PC share.
Writes results/003a-phrase-objective/posthoc.md. CPU only."""
import json, numpy as np, torch, re, os, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "003a-phrase-objective"); GR = os.path.join(ROOT, "outputs", "003a", "grads")
rd = json.load(open(f"{R}/readout.json")); read = rd["readout"]; base = rd["baselines"]
out = ["# 003a — post hoc analyses (computed after the pre-registered tables were read)", "",
       "## Difference-direction latent AUC (v_a − v_b), at t′, mean over sibling pairs, vs own-vector AUC, J-sum and the full probe", "",
       "| phrase | L | " + " | ".join(f"{o} own / diff" for o in ("logp", "lin", "logit", "odds")) + " | J-sum | full probe |", "|---|---|" + "---|" * 6]
for ph in read:
    for l in (52, 56, 60):
        cells = []
        for o in ("logp", "lin", "logit", "odds"):
            r = read[ph][f"{o}|at|{l}"]; pairs = [k for k in r if "|" in k]
            cells.append(f"{np.nanmean([r[p]['own_latent'] for p in pairs]):.2f} / {np.nanmean([r[p]['diff_latent'] for p in pairs]):.2f}")
        b = base[ph][str(l)]; out.append(f"| {ph} | {l} | " + " | ".join(cells) + f" | {np.nanmean([b[p]['J_sum'] for p in b]):.2f} | {np.nanmean([b[p]['full_probe'] for p in b]):.2f} |")
def safe(s): return re.sub(r"[^A-Za-z0-9]+", "_", s)
items = json.load(open(f"{R}/items.json")); V = {}
names = [it["name"] for it in items if it["regime"] == "natural"]
for it in items:
    if it["regime"] != "natural": continue
    d = torch.load(f"{GR}/{safe(it['name'])}__natural.pt", weights_only=False); k = d["prefix_len"]
    for l in (44, 56): V[(it["name"], l)] = torch.stack([r["obj"]["lin"][l]["at"].float()[k:].sum(0) for r in d["recs"]]).mean(0)
out += ["", "## Cross-item cosine of v_cond^lin (at t′): shared component", ""]
for l in (44, 56):
    M = torch.stack([V[(n, l)] for n in names]); Mn = torch.nn.functional.normalize(M, dim=1); C = (Mn @ Mn.T).numpy()
    prim = [i for i, n in enumerate(names) if n in ("New Zealand", "South Korea", "San Diego", "North Carolina", "ice cream", "credit card")]
    nul = [i for i, n in enumerate(names) if n.startswith("null")]
    off = lambda idx: np.mean([C[i, j] for i in idx for j in idx if i < j]); cross = np.mean([C[i, j] for i in prim for j in nul])
    fam = {p: [i for i, n in enumerate(names) if n.startswith(p + " ")] for p in ("New", "South", "San", "North")}
    within = np.mean([off(v) for v in fam.values()]); between = np.mean([C[i, j] for a, b in itertools.combinations(fam, 2) for i in fam[a] for j in fam[b]])
    _, S_, _ = torch.linalg.svd(M - M.mean(0), full_matrices=False)
    out.append(f"- L{l}: primaries among themselves {off(prim):.2f}; primaries vs nulls {cross:.2f}; nulls among themselves {off(nul):.2f}; within-family siblings {within:.2f}; between families {between:.2f}; top-PC variance share across {len(names)} item vectors {(S_[0]**2/(S_**2).sum()).item():.2f}; ‖mean vector‖ / mean ‖item‖ {M.mean(0).norm().item()/M.norm(dim=1).mean().item():.2f}")
open(f"{R}/posthoc.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
