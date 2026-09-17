"""004 swap harness (design §3-§6). Runs only if the first-order gate passed.

Dataset C: 001 latent two-hop prompts (New, South, North families; routes continent/language/currency/hemisphere, 4 frames)
plus a new 'state' route for the San family (the only route on which San members' answers differ; design amendment recorded
in the report). Pair (A, B) admitted per route only if answers differ; item admitted only if the clean top-1 equals the first
token of A's answer. Both directions.
Methods (one operator, design §4): PJ (v_cond lin at t', natural, n = all available), PJc (family-centred), Jc (constituent J:
prefix + two suffix rows; suffix coordinates swapped), rand (3 seeds, random unit direction), Jfirst (identity swap = clean).
Doses: norm-targeted, r = rho * median ||h_L52|| over item final tokens, rho in {0.05, 0.1, 0.2, 0.4}; applied at L52 and L56
(clamped; L56 operates on the propagated state). Native ||delta(alpha=1)|| per method recorded.
Damage: 40 pile sequences, edit at final position with each method's operator (4 representative pairs) and with random,
at each r: KL(clean || edited) and top-1 retention.
Writes results/004-causal-geometry/{items.json, swaps.json, pile_damage.json}; outputs/004/ for activations.
"""
import json, os, sys, re, random, time, itertools
import numpy as np, torch, jlens
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ekko_harness import load_model
from lib.edit_harness import make_swap_fn, make_add_fn, forward_logits_and_resid
from datasets import load_dataset
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "004-causal-geometry"); OUT = os.path.join(ROOT, "outputs", "004"); os.makedirs(RES, exist_ok=True); os.makedirs(OUT, exist_ok=True)
GR = os.path.join(ROOT, "outputs", "003a", "grads"); GRX = os.path.join(ROOT, "outputs", "003a_controls"); RES1 = os.path.join(ROOT, "results", "001-template-geometry")
MODEL = "/content/models/Qwen3.6-27B"; LENS = "/content/lenses/qwen3.6-27b"; LAYERS = [52, 56]; SEQ = 128
RHOS = [0.05, 0.1, 0.2, 0.4]; N_RAND = 3; N_PILE = 40; FAMS = ["New", "South", "San", "North"]
random.seed(0); torch.manual_seed(0); dev = "cuda"
gate = json.load(open(f"{RES}/first_order_control.json"))
assert gate["gate_pass"], "first-order gate failed; not running swaps"
model, hf, tok = load_model(MODEL)
gamma = model._final_norm.weight.detach().float().cpu(); W_U = hf.lm_head.weight.detach().float().cpu()
def q_vec(t): return (1 + gamma) * W_U[t]
Jl = jlens.JacobianLens.load(f"{LENS}/j-lens/lens.pt"); Jrow = {l: (lambda t, l=l: Jl.jacobians[l].float().T @ q_vec(t)) for l in LAYERS}
def safe(s): return re.sub(r"[^A-Za-z0-9]+", "_", s)
fam = json.load(open(f"{RES1}/families.json")); members = {m["phrase"]: m for F in fam["families"] for m in F["members"]}; fam_of = {m["phrase"]: F["name"] for F in fam["families"] for m in F["members"]}
LF = json.load(open(f"{RES1}/latent_facts.json")); FACTS, ROUTES, ALIASES = LF["facts"], {r: f for r, f in LF["routes"].items() if r != "second_word_letters"}, LF["aliases"]
ROUTES["state"] = ["Fact: The US state containing the place where {cue} is located is",
                   "The place associated with {cue} lies in the state of",
                   "Q: In which US state is the place known for {cue}? A:",
                   "Geography note: {cue} can be found in the state called"]
RORDER = list(LF["routes"]) + ["state"]

# ---------------------------------------------------------------- vectors (v_cond lin, at t', natural; all available contexts)
vec = {}
for ph in members:
    p = f"{GR}/{safe(ph)}__natural.pt"
    if not os.path.exists(p): continue
    d = torch.load(p, weights_only=False); k = d["prefix_len"]; recs = [r["obj"]["lin"] for r in d["recs"]]
    px = f"{GRX}/{safe(ph)}_extra_lin.pt"
    if os.path.exists(px): recs += torch.load(px, weights_only=False)
    vec[ph] = {"k": k, "n": len(recs), "v": {l: torch.stack([r[l]["at"].float()[k:].sum(0) for r in recs]).mean(0) for l in LAYERS}}
fam_mean = {F: {l: torch.stack([vec[p]["v"][l] for p in vec if fam_of[p] == F]).mean(0) for l in LAYERS} for F in FAMS}
print("vectors:", {p: vec[p]["n"] for p in vec}, flush=True)

# ---------------------------------------------------------------- Dataset C
def admitted_prompt(p, phrase, answer):
    low = p.lower(); bad = [phrase] + ALIASES.get(phrase, [])
    return not any(re.search(r"\b" + re.escape(b.lower()) + r"\b", low) for b in bad) and not (answer and re.search(r"\b" + re.escape(answer.lower()) + r"\b", low))
def first_tok(ans): return tok.encode(" " + ans, add_special_tokens=False)[0]
cands = []
for F in FAMS:
    mem = [p for p in vec if fam_of[p] == F]
    for A, B in itertools.permutations(mem, 2):
        for route, frames in ROUTES.items():
            aA, aB = FACTS[A].get(route), FACTS[B].get(route)
            if route == "state" and F != "San": continue
            if aA is None or aB is None or aA == aB or first_tok(aA) == first_tok(aB): continue
            for fi, frame in enumerate(frames):
                cue = FACTS[A]["cues"][(fi + RORDER.index(route)) % len(FACTS[A]["cues"])]; p = frame.format(cue=cue)
                if admitted_prompt(p, A, aA): cands.append({"family": F, "A": A, "B": B, "route": route, "frame": fi, "cue": cue, "prompt": p, "ansA": aA, "ansB": aB, "tA": first_tok(aA), "tB": first_tok(aB)})
print(f"candidate items: {len(cands)}", flush=True)
items = []; hnorm = {l: [] for l in LAYERS}
for c in cands:
    ids = model.encode(c["prompt"], max_length=256)
    logits, _ = forward_logits_and_resid(model, ids, None)
    lp = torch.log_softmax(logits[0], -1); top1 = int(logits[0].argmax())
    for l in LAYERS:
        _, r = forward_logits_and_resid(model, ids, None, resid_layer=l); hnorm[l].append(float(r[0].norm()))
    if top1 == c["tA"]:
        c = dict(c, clean_lpA=float(lp[c["tA"]]), clean_lpB=float(lp[c["tB"]]), clean_entropy=float(-(lp.exp() * lp).sum())); items.append(c)
print(f"admitted items (clean top-1 = A's answer): {len(items)} / {len(cands)}; by family: { {F: sum(i['family']==F for i in items) for F in FAMS} }", flush=True)
json.dump({"items": items, "n_candidates": len(cands), "median_hnorm": {str(l): float(np.median(hnorm[l])) for l in LAYERS}}, open(f"{RES}/items.json", "w"), indent=1)
r_grid = {str(l): [rho * float(np.median(hnorm[l])) for rho in RHOS] for l in LAYERS}

# ---------------------------------------------------------------- operators per (A,B) pair and method
def build(A, B, method, l):
    kA = vec[A]["k"]
    if method == "PJ": V = torch.stack([vec[A]["v"][l], vec[B]["v"][l]], 1); perm = [1, 0]
    elif method == "PJc": F = fam_of[A]; V = torch.stack([vec[A]["v"][l] - fam_mean[F][l], vec[B]["v"][l] - fam_mean[F][l]], 1); perm = [1, 0]
    elif method == "Jc":
        p_t = members[A]["ids"][:kA]; V = torch.stack([Jrow[l](t) for t in p_t] + [Jrow[l](members[A]["ids"][kA]), Jrow[l](members[B]["ids"][kA])], 1); m = V.shape[1]; perm = list(range(m - 2)) + [m - 1, m - 2]
    else: raise ValueError(method)
    V = V.to(dev); return V, torch.linalg.pinv(V), perm
METHODS = ["PJ", "PJc", "Jc"]
ops = {}
for c in items:
    key = (c["A"], c["B"])
    if key in ops: continue
    ops[key] = {m: {l: build(c["A"], c["B"], m, l) for l in LAYERS} for m in METHODS}

# ---------------------------------------------------------------- swaps
def metrics(logits, c):
    lp = torch.log_softmax(logits[0], -1); p = lp.exp()
    return {"lpA": float(lp[c["tA"]]), "lpB": float(lp[c["tB"]]), "top1": int(logits[0].argmax()), "entropy": float(-(p * lp).sum())}
def kl(logits_clean, logits_edit):
    lc = torch.log_softmax(logits_clean[0], -1); le = torch.log_softmax(logits_edit[0], -1); return float((lc.exp() * (lc - le)).sum())
swaps = []; t0 = time.time()
for n, c in enumerate(items):
    ids = model.encode(c["prompt"], max_length=256); clean_logits, _ = forward_logits_and_resid(model, ids, None)
    rec = {"item": n, "A": c["A"], "B": c["B"], "family": c["family"], "route": c["route"], "clean": metrics(clean_logits, c), "runs": []}
    # native ||delta(alpha=1)|| per method at L52 (from the clean residual)
    _, h52 = forward_logits_and_resid(model, ids, None, resid_layer=52)
    for m in METHODS:
        V, Vp, perm = ops[(c["A"], c["B"])][m][52]; cc = Vp @ h52[0].to(dev); rec[f"native_norm_{m}"] = float((V @ (cc[perm] - cc)).norm())
    for ri, rho in enumerate(RHOS):
        for m in METHODS:
            edits = {l: make_swap_fn(-1, *ops[(c["A"], c["B"])][m][l], target_norm=r_grid[str(l)][ri]) for l in LAYERS}
            lg, _ = forward_logits_and_resid(model, ids, edits); rec["runs"].append({"method": m, "rho": rho, **metrics(lg, c), "kl": kl(clean_logits, lg)})
        for s in range(N_RAND):
            g = torch.Generator().manual_seed(1000 * n + s); edits = {}
            for l in LAYERS:
                d = torch.randn(model.d_model, generator=g); edits[l] = make_add_fn(-1, (d / d.norm() * r_grid[str(l)][ri]))
            lg, _ = forward_logits_and_resid(model, ids, edits); rec["runs"].append({"method": f"rand{s}", "rho": rho, **metrics(lg, c), "kl": kl(clean_logits, lg)})
    swaps.append(rec)
    if (n + 1) % 20 == 0: print(f"[{n+1}/{len(items)}] swaps ({time.time()-t0:.0f}s)", flush=True); json.dump({"rhos": RHOS, "r_grid": r_grid, "swaps": swaps}, open(f"{RES}/swaps.json", "w"))
json.dump({"rhos": RHOS, "r_grid": r_grid, "swaps": swaps}, open(f"{RES}/swaps.json", "w"))
print(f"swaps done ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- pile damage
ds = load_dataset("NeelNanda/pile-10k", split="train")
natural_docs = set(json.load(open(f"{RES1}/natural_doc_ids.json"))); sigma_docs = set(json.load(open(f"{RES1}/sigma_meta.json"))["seq_ids"])
pile = []
for i in range(len(ds)):
    if i in natural_docs or i in sigma_docs: continue
    if len(tok.encode(ds[i]["text"], add_special_tokens=False)) >= SEQ + 8: pile.append(ds[i]["text"])
    if len(pile) == N_PILE: break
rep_pairs = []
for F in FAMS:
    ks = [k for k in ops if fam_of[k[0]] == F]
    if ks: rep_pairs.append(ks[0])
dmg = []
for si, text in enumerate(pile):
    ids = model.encode(text, max_length=SEQ); clean_logits, _ = forward_logits_and_resid(model, ids, None); top_c = int(clean_logits[0].argmax())
    for ri, rho in enumerate(RHOS):
        for key in rep_pairs:
            for m in METHODS:
                edits = {l: make_swap_fn(-1, *ops[key][m][l], target_norm=r_grid[str(l)][ri]) for l in LAYERS}
                lg, _ = forward_logits_and_resid(model, ids, edits); dmg.append({"seq": si, "pair": list(key), "method": m, "rho": rho, "kl": kl(clean_logits, lg), "top1_kept": int(lg[0].argmax()) == top_c})
        for s in range(N_RAND):
            g = torch.Generator().manual_seed(5000 + 10 * si + s); edits = {}
            for l in LAYERS:
                d = torch.randn(model.d_model, generator=g); edits[l] = make_add_fn(-1, d / d.norm() * r_grid[str(l)][ri])
            lg, _ = forward_logits_and_resid(model, ids, edits); dmg.append({"seq": si, "pair": None, "method": f"rand{s}", "rho": rho, "kl": kl(clean_logits, lg), "top1_kept": int(lg[0].argmax()) == top_c})
json.dump({"rhos": RHOS, "rep_pairs": [list(k) for k in rep_pairs], "damage": dmg}, open(f"{RES}/pile_damage.json", "w"))
print(f"pile damage done ({time.time()-t0:.0f}s)", flush=True)
