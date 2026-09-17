"""003a-controls, steps 2, 4, 5 (GPU, ~15 min).
Step 2: single-token country control in the 001 two-hop harness (4 routes x 4 frames x rotating cues), captured at all 15 layers;
        probe (leave-one-cue-out) vs released J (own-token difference score) vs logit lens; Delta_single(L) vs Delta_multi(L) from 001.
Step 4: 40 more natural contexts for New Zealand and South Korea; lin (at t') at L36-60; reliability and latent AUC vs n = 20/40/60.
Step 5: positive-control decomposition for " New" and " South": single-lag g1^lin (natural), all-target v_lin (natural), all-target v_lin (pile),
        each vs the released J row and vs each other; split-half cosines.
Outputs: results/003a-controls/{single_token_control.json, convergence.json, positive_control_decomp.json, analysis_gpu.md}; outputs/003a_controls/
"""
import json, os, sys, re, random, itertools, logging, time
import numpy as np, torch, jlens
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ekko_harness import load_model, capture
from lib.phrase_objective import PhraseJ
from datasets import load_dataset
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
for noisy in ("httpx", "huggingface_hub", "datasets", "urllib3", "filelock", "fsspec"): logging.getLogger(noisy).setLevel(logging.WARNING)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "003a-controls"); OUT = os.path.join(ROOT, "outputs", "003a_controls"); os.makedirs(RES, exist_ok=True); os.makedirs(OUT, exist_ok=True)
R3 = os.path.join(ROOT, "results", "003a-phrase-objective"); GR = os.path.join(ROOT, "outputs", "003a", "grads")
RES1 = os.path.join(ROOT, "results", "001-template-geometry"); O1 = os.path.join(ROOT, "outputs", "001")
MODEL = "/content/models/Qwen3.6-27B"; LENS = "/content/lenses/qwen3.6-27b"; SEQ = 128; SKIP = 4; TARGET = 62
ALL_L = list(range(8, 61, 4)) + [62]; L5 = [36, 44, 52, 56, 60]; LAM = 0.1
random.seed(0); torch.manual_seed(0)
model, hf, tok = load_model(MODEL)
gamma = model._final_norm.weight.detach().float(); W_U = hf.lm_head.weight.detach()
def q_of(t): return (1 + gamma) * W_U[t].float()
Jl = jlens.JacobianLens.load(f"{LENS}/j-lens/lens.pt")
def cos(a, b): return float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))
def auc(p, n):
    p = np.asarray(p, float); n = np.asarray(n, float)
    return float("nan") if len(p) == 0 or len(n) == 0 else float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())
S = torch.load(f"{O1}/sigma.pt", weights_only=False); eps = 1e-6
chol = {}
for l in ALL_L:
    mu, Sg, _ = S["pooled"][l]; tau = Sg.diagonal().mean(); chol[l] = torch.linalg.cholesky(((1 - LAM) * Sg + LAM * tau * torch.eye(Sg.shape[0])).cuda())
def solve(l, v): return torch.cholesky_solve(v.cuda().T.contiguous(), chol[l]).T.cpu()
def rmsnorm(x): return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps)
gcpu = gamma.cpu(); Wcpu = W_U.float().cpu()
def qc(t): return (1 + gcpu) * Wcpu[t]
def lens_logits(h, l, toks): return rmsnorm(Jl.transport(h, l)) @ torch.stack([qc(t) for t in toks]).T
md = ["# 003a-controls — GPU steps", ""]

# ---------------------------------------------------------------- step 2: single-token control
FACTS = {"France": {"cues": ["the Eiffel Tower", "the Louvre museum", "the city of Marseille", "the city of Bordeaux", "the city of Lyon"], "continent": "Europe", "language": "French", "currency": "euro", "hemisphere": "Northern"},
         "Japan": {"cues": ["the city of Tokyo", "the city of Kyoto", "Mount Fuji", "the city of Osaka", "the Shinkansen bullet train"], "continent": "Asia", "language": "Japanese", "currency": "yen", "hemisphere": "Northern"},
         "Brazil": {"cues": ["the city of Rio de Janeiro", "the Amazon rainforest", "the city of São Paulo", "the Copacabana beach", "the city of Brasília"], "continent": "South America", "language": "Portuguese", "currency": "real", "hemisphere": "Southern"},
         "Egypt": {"cues": ["the city of Cairo", "the pyramids of Giza", "the Nile river delta", "the city of Luxor", "the city of Alexandria"], "continent": "Africa", "language": "Arabic", "currency": "pound", "hemisphere": "Northern"},
         "Australia": {"cues": ["the Sydney Opera House", "the city of Canberra", "the Great Barrier Reef", "the city of Melbourne", "Uluru"], "continent": "Oceania", "language": "English", "currency": "dollar", "hemisphere": "Southern"},
         "Italy": {"cues": ["the Colosseum", "the city of Venice", "the city of Milan", "the city of Florence", "the Leaning Tower of Pisa"], "continent": "Europe", "language": "Italian", "currency": "euro", "hemisphere": "Northern"}}
ALIASES = {"France": ["French"], "Japan": ["Japanese"], "Brazil": ["Brazilian"], "Egypt": ["Egyptian"], "Australia": ["Australian"], "Italy": ["Italian"]}
LF = json.load(open(f"{RES1}/latent_facts.json")); ROUTES = {r: fr for r, fr in LF["routes"].items() if r != "second_word_letters"}; RORDER = list(LF["routes"])
prompts = []
for c, f in FACTS.items():
    for route, frames in ROUTES.items():
        for fi, frame in enumerate(frames):
            cue = f["cues"][(fi + RORDER.index(route)) % 5]; p = frame.format(cue=cue); low = p.lower()
            bad = [c] + ALIASES[c] + [f[route]]
            if any(re.search(r"\b" + re.escape(b.lower()) + r"\b", low) for b in bad): continue
            prompts.append({"country": c, "route": route, "frame": fi, "cue": cue, "prompt": p, "answer": f[route], "group": (fi + RORDER.index(route)) % 5})
logging.info(f"single-token prompts: {len(prompts)} ({len(prompts)/6:.0f} per country)")
acts = {l: [] for l in ALL_L}; t0 = time.time()
with torch.no_grad():
    for p in prompts:
        ids = model.encode(p["prompt"], max_length=256); _, a = capture(model, ids, max_length=256)
        for l in ALL_L: acts[l].append(a[l][0, -1].float().cpu())
acts = {l: torch.stack(v) for l, v in acts.items()}
torch.save({"prompts": prompts, "acts": acts}, f"{OUT}/single_token_acts.pt")
ctry = list(FACTS); cid = {c: tok.encode(" " + c, add_special_tokens=False)[0] for c in ctry}
rows = {c: [i for i, p in enumerate(prompts) if p["country"] == c] for c in ctry}
single = {}
for l in ALL_L:
    H = acts[l]; pr, jd, lg = [], [], []
    for a, b in itertools.combinations(ctry, 2):
        ra, rb = rows[a], rows[b]
        for g in range(5):
            tra, trb = [i for i in ra if prompts[i]["group"] != g], [i for i in rb if prompts[i]["group"] != g]; tea, teb = [i for i in ra if prompts[i]["group"] == g], [i for i in rb if prompts[i]["group"] == g]
            if min(len(tra), len(trb), len(tea), len(teb)) < 2: continue
            d = solve(l, (H[tra].mean(0) - H[trb].mean(0))[None])[0]; pr.append(auc(H[tea] @ d, H[teb] @ d))
        La, Lb = lens_logits(H[ra], l, [cid[a], cid[b]]), lens_logits(H[rb], l, [cid[a], cid[b]]); jd.append(auc(La[:, 0] - La[:, 1], Lb[:, 0] - Lb[:, 1]))
        Ga, Gb = rmsnorm(H[ra]) @ torch.stack([qc(cid[a]), qc(cid[b])]).T, rmsnorm(H[rb]) @ torch.stack([qc(cid[a]), qc(cid[b])]).T; lg.append(auc(Ga[:, 0] - Ga[:, 1], Gb[:, 0] - Gb[:, 1]))
    single[str(l)] = {"probe_cue_out": float(np.mean(pr)), "J_diff": float(np.mean(jd)), "logit_diff": float(np.mean(lg)), "delta_single": float(np.mean(pr) - np.mean(jd))}
D1 = json.load(open(f"{RES1}/discrimination.json")); multi = {}
for l in ALL_L:
    pv, jv = [], []
    for fn in ("New", "South", "North", "United", "San"):
        for pk, pr_ in D1[fn]["latent"]["layers"][str(l)]["pairs"].items(): pv.append(pr_["probe"]["loo_cue"]["auc"]); jv.append(pr_["J-sum"]["auc"])
    multi[str(l)] = {"probe_cue_out": float(np.mean(pv)), "J_sum": float(np.mean(jv)), "delta_multi": float(np.mean(pv) - np.mean(jv))}
json.dump({"single": single, "multi_from_001": multi, "n_prompts": len(prompts)}, open(f"{RES}/single_token_control.json", "w"), indent=1)
md += ["## Step 2: single-token control (6 countries, 001 harness) vs 001 multi-token families", "", "| L | single: probe | single: J-diff | single: logit-diff | Δ_single | multi: probe | multi: J-sum | Δ_multi |", "|---|---|---|---|---|---|---|---|"]
for l in ALL_L: s, m = single[str(l)], multi[str(l)]; md.append(f"| {l} | {s['probe_cue_out']:.2f} | {s['J_diff']:.2f} | {s['logit_diff']:.2f} | {s['delta_single']:+.2f} | {m['probe_cue_out']:.2f} | {m['J_sum']:.2f} | {m['delta_multi']:+.2f} |")
logging.info("step 2 done (%.0fs)", time.time() - t0)

# ---------------------------------------------------------------- step 4: convergence with more natural contexts (NZ, SK), lin at t'
pj = PhraseJ(model, L5, TARGET, SKIP)
fam = json.load(open(f"{RES1}/families.json")); members = {m["phrase"]: m for F in fam["families"] for m in F["members"]}; fam_of = {m["phrase"]: F for F in fam["families"] for m in F["members"]}
ctx1 = json.load(open(f"{RES1}/contexts.json")); nat = {}
for c in ctx1["emission_natural"]: nat.setdefault(c["phrase"], []).append(c["text"])
ds = load_dataset("NeelNanda/pile-10k", split="train")
def mine_more(phrase, skip_n, want):
    out, seen = [], 0
    for di, d in enumerate(ds):
        t = d["text"]
        for mm in re.finditer(r"(?<=\S) " + re.escape(phrase) + r"\b", t):
            left = t[:mm.start()]; ids = tok.encode(left, add_special_tokens=False)[-128:]
            if len(ids) < 32: continue
            seen += 1
            if seen <= skip_n: continue
            out.append(tok.decode(ids))
            if len(out) >= want: return out
    return out
A1 = torch.load(f"{O1}/acts.pt", weights_only=False); index = A1["index"]; lat_acts = {l: A1["acts"][l].float() for l in L5}
def rows_for(ph): return [i for i, r in enumerate(index) if r["cond"] == "latent" and r["phrase"] == ph and r["pos"] == -1]
def safe(s): return re.sub(r"[^A-Za-z0-9]+", "_", s)
conv = {}
for ph in ("New Zealand", "South Korea"):
    d0 = torch.load(f"{GR}/{safe(ph)}__natural.pt", weights_only=False); k = d0["prefix_len"]; pids = members[ph]["ids"]
    vecs = [r["obj"]["lin"] for r in d0["recs"]]                      # existing 20 (natural, first 20 of 001's 40)
    extra = mine_more(ph, 40, 40); logging.info(f"{ph}: mined {len(extra)} extra natural contexts")
    new = []
    for text in extra:
        x = model.encode(text, max_length=SEQ)
        out, lps, tp = pj.per_token_multi(x, pids, ("lin",), q_of=q_of)
        new.append({l: {"at": torch.stack([out["lin"][i][l][0] for i in range(len(pids))]).half(), "mean": torch.stack([out["lin"][i][l][1] for i in range(len(pids))]).half()} for l in L5})
    torch.save(new, f"{OUT}/{safe(ph)}_extra_lin.pt")
    allv = vecs + new; fam_ = fam_of[ph]; sibs = [m for m in fam_["members"] if m["phrase"] != ph and os.path.exists(f"{GR}/{safe(m['phrase'])}__natural.pt") and len(rows_for(m["phrase"])) >= 10]
    conv[ph] = {}
    for n in (20, 40, 60):
        if n > len(allv): continue
        conv[ph][str(n)] = {}
        for l in L5:
            vc = lambda sel: torch.stack([allv[i][l]["at"].float()[k:].sum(0) for i in sel]).mean(0)
            h1, h2 = [i for i in range(n) if i % 2 == 0], [i for i in range(n) if i % 2 == 1]; rho = cos(vc(h1), vc(h2)); v = vc(range(n))
            own, dif = [], []
            for sb in sibs:
                ds_ = torch.load(f"{GR}/{safe(sb['phrase'])}__natural.pt", weights_only=False); vb = torch.stack([r["obj"]["lin"][l]["at"].float()[k:].sum(0) for r in ds_["recs"]]).mean(0)
                ra, rb = rows_for(ph), rows_for(sb["phrase"]); H = lat_acts[l]
                own.append(auc(H[ra] @ v, H[rb] @ v)); dif.append(auc(H[ra] @ (v - vb), H[rb] @ (v - vb)))
            conv[ph][str(n)][str(l)] = {"splithalf": rho, "own_latent": float(np.mean(own)), "diff_latent": float(np.mean(dif))}
json.dump(conv, open(f"{RES}/convergence.json", "w"), indent=1)
md += ["", "## Step 4: lin (at t′) convergence with n natural contexts — split-half ρ / own AUC / diff AUC", "", "| phrase | n | " + " | ".join(f"L{l}" for l in L5) + " |", "|---|---|" + "---|" * len(L5)]
for ph in conv:
    for n in conv[ph]: md.append(f"| {ph} | {n} | " + " | ".join(f"{conv[ph][n][str(l)]['splithalf']:.2f} / {conv[ph][n][str(l)]['own_latent']:.2f} / {conv[ph][n][str(l)]['diff_latent']:.2f}" for l in L5) + " |")
logging.info("step 4 done (%.0fs)", time.time() - t0)

# ---------------------------------------------------------------- step 5: positive-control decomposition
pjA = PhraseJ(model, L5, TARGET, SKIP)
pile = []
for d in ds:
    if len(tok.encode(d["text"], add_special_tokens=False)) >= SEQ + 8: pile.append(d["text"])
    if len(pile) == 20: break
decomp = {}
for ph, first in (("New Zealand", " New"), ("South Korea", " South")):
    t = tok.encode(first, add_special_tokens=False)[0]; q = q_of(t); d0 = torch.load(f"{GR}/{safe(ph)}__natural.pt", weights_only=False)
    ctxs = nat[ph][:20]
    single_at = {l: torch.stack([r["obj"]["lin"][l]["at"].float()[0] for r in d0["recs"]]) for l in L5}          # [20, d]
    single_mean = {l: torch.stack([r["obj"]["lin"][l]["mean"].float()[0] for r in d0["recs"]]) for l in L5}
    all_nat = {l: [] for l in L5}; all_pile = {l: [] for l in L5}
    for text in ctxs:
        g = pjA.v_lin(model.encode(text, max_length=SEQ), q)
        for l in L5: all_nat[l].append(g[l])
    for text in pile:
        g = pjA.v_lin(model.encode(text, max_length=SEQ), q)
        for l in L5: all_pile[l].append(g[l])
    decomp[ph] = {}
    for l in L5:
        Jrow = Jl.jacobians[l].float().T @ qc(t); AN = torch.stack(all_nat[l]); AP = torch.stack(all_pile[l]); SA = single_at[l]; SM = single_mean[l]
        sh = lambda M: cos(M[0::2].mean(0), M[1::2].mean(0))
        decomp[ph][str(l)] = {"J_vs_single_at": cos(SA.mean(0), Jrow), "J_vs_single_mean": cos(SM.mean(0), Jrow), "J_vs_alltarget_natural": cos(AN.mean(0), Jrow), "J_vs_alltarget_pile": cos(AP.mean(0), Jrow),
                              "lag_effect_cos(single_mean, alltarget_natural)": cos(SM.mean(0), AN.mean(0)), "context_effect_cos(alltarget_natural, alltarget_pile)": cos(AN.mean(0), AP.mean(0)),
                              "splithalf_single_mean": sh(SM), "splithalf_alltarget_natural": sh(AN), "splithalf_alltarget_pile": sh(AP)}
json.dump(decomp, open(f"{RES}/positive_control_decomp.json", "w"), indent=1)
md += ["", "## Step 5: positive-control decomposition (first token; 20 natural contexts; 20 pile sequences)", "", "| phrase | L | J vs single-lag(mean) | J vs all-target natural | J vs all-target pile | lag effect | context effect | split-half single / all-nat / all-pile |", "|---|---|---|---|---|---|---|---|"]
for ph in decomp:
    for l in L5:
        s = decomp[ph][str(l)]; md.append(f"| {ph} | {l} | {s['J_vs_single_mean']:.2f} | {s['J_vs_alltarget_natural']:.2f} | {s['J_vs_alltarget_pile']:.2f} | {s['lag_effect_cos(single_mean, alltarget_natural)']:.2f} | {s['context_effect_cos(alltarget_natural, alltarget_pile)']:.2f} | {s['splithalf_single_mean']:.2f} / {s['splithalf_alltarget_natural']:.2f} / {s['splithalf_alltarget_pile']:.2f} |")
open(f"{RES}/analysis_gpu.md", "w").write("\n".join(md) + "\n"); print("\n".join(md)); logging.info("done (%.0fs)", time.time() - t0)
