"""004 step 7 — first-order local control (the gate; design §6.7).

On 10 emission-natural contexts per phrase (New Zealand, South Korea, San Diego, North Carolina), the lin functional
f = q_suffix^T h_{62, t'+1} is defined under teacher forcing of the prefix token. The per-context gradient v = df/dh_{l,t'}
(003a 'at' reduction, lin, suffix token) predicts, to first order, df = v . delta for a perturbation delta of h_{l,t'}.
We perturb along d in {v_hat, 3 random unit directions} with ||delta|| = rho * ||h_{l,t'}||, rho in {0.005, 0.01, 0.02, 0.05},
rerun the forward with the edit, and regress the measured df on the predicted v . delta (all contexts, directions, rho pooled
per layer). Pass: slope within 20% of 1 and R^2 > 0.9 for rho <= 0.02 at both L52 and L56 (per design). Also reports the
averaged vector (n=20) as a secondary predictor.
Writes results/004-causal-geometry/first_order_control.json and prints PASS/FAIL (exit code 3 on FAIL).
"""
import json, os, sys, re, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ekko_harness import load_model
from lib.edit_harness import Edits, make_add_fn
from jlens.hooks import ActivationRecorder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "004-causal-geometry"); os.makedirs(RES, exist_ok=True)
GR = os.path.join(ROOT, "outputs", "003a", "grads"); RES1 = os.path.join(ROOT, "results", "001-template-geometry")
MODEL = "/content/models/Qwen3.6-27B"; LAYERS = [52, 56]; SEQ = 128; TARGET = 62
PHRASES = ["New Zealand", "South Korea", "San Diego", "North Carolina"]; RHOS = [0.005, 0.01, 0.02, 0.05]; N_CTX = 10; N_RAND = 3
random.seed(0); torch.manual_seed(0)
model, hf, tok = load_model(MODEL)
gamma = model._final_norm.weight.detach().float(); W_U = hf.lm_head.weight.detach()
def q_of(t): return (1 + gamma) * W_U[t].float()
def safe(s): return re.sub(r"[^A-Za-z0-9]+", "_", s)
fam = json.load(open(f"{RES1}/families.json")); members = {m["phrase"]: m for F in fam["families"] for m in F["members"]}
ctx1 = json.load(open(f"{RES1}/contexts.json")); nat = {}
for c in ctx1["emission_natural"]: nat.setdefault(c["phrase"], []).append(c["text"])

@torch.no_grad()
def f_value(ids_full, tprime, q, layer, delta=None):
    """f = q^T h_{62, t'+1} under teacher forcing; optional additive delta at (layer, t')."""
    with ActivationRecorder(model.layers, at=[TARGET, layer]) as rec:
        if delta is not None:
            with Edits(model.layers, {layer: make_add_fn(tprime, delta)}): model.forward(ids_full)
        else: model.forward(ids_full)
        h62 = rec.activations[TARGET][0, tprime + 1].float(); hl = rec.activations[layer][0, tprime].float()
    return float(h62 @ q.to(h62.device)), hl

out = {}; rows = {l: {"pred": [], "meas": [], "rho": [], "kind": []} for l in LAYERS}
for ph in PHRASES:
    d0 = torch.load(f"{GR}/{safe(ph)}__natural.pt", weights_only=False); pids = members[ph]["ids"]; k = d0["prefix_len"]
    q = q_of(pids[k]).cpu()                                              # suffix token functional (2-token phrases: k=1)
    avg = {l: torch.stack([r["obj"]["lin"][l]["at"].float()[k] for r in d0["recs"]]).mean(0) for l in LAYERS}
    for ci in range(min(N_CTX, len(d0["recs"]))):
        rec = d0["recs"][ci]; text = nat[ph][rec["ctx"]]
        ids = model.encode(text, max_length=SEQ); full = torch.cat([ids, torch.tensor([pids[:k]], device=ids.device)], 1)   # prefix teacher-forced
        tprime = ids.shape[1] - 1
        for l in LAYERS:
            v_local = rec["obj"]["lin"][l]["at"].float()[k]              # gradient of f wrt h_{l,t'} at this context
            f0, hl = f_value(full, tprime, q, l); hn = float(hl.norm())
            dirs = [("v_local", v_local / v_local.norm()), ("v_avg", avg[l] / avg[l].norm())] + [(f"rand{j}", torch.nn.functional.normalize(torch.randn_like(v_local), dim=0)) for j in range(N_RAND)]
            for rho in RHOS:
                for name, dhat in dirs:
                    delta = rho * hn * dhat
                    f1, _ = f_value(full, tprime, q, l, delta=delta.to(model.input_device))
                    rows[l]["pred"].append(float(v_local @ delta)); rows[l]["meas"].append(f1 - f0); rows[l]["rho"].append(rho); rows[l]["kind"].append(name)
    print(f"{ph}: done", flush=True)
verdict = {}
for l in LAYERS:
    P, M, R, K = (np.array(rows[l][k]) for k in ("pred", "meas", "rho", "kind"))
    def fit(mask):
        p, m = P[mask], M[mask]
        if len(p) < 4 or p.std() == 0: return {"slope": float("nan"), "r2": float("nan"), "n": int(mask.sum())}
        slope = float((p * m).sum() / (p * p).sum()); r2 = float(1 - ((m - slope * p) ** 2).sum() / ((m - m.mean()) ** 2).sum()); return {"slope": slope, "r2": r2, "n": int(mask.sum())}
    verdict[str(l)] = {"all": fit(np.ones(len(P), bool)), "rho<=0.02": fit(R <= 0.02), "rho<=0.02_v_local": fit((R <= 0.02) & (K == "v_local")), "rho<=0.02_random": fit((R <= 0.02) & np.char.startswith(K.astype(str), "rand")),
                       "rho<=0.02_v_avg_as_predictor_of_meas": fit((R <= 0.02) & (K == "v_avg")), "per_rho": {str(r): fit(R == r) for r in RHOS}}
    g = verdict[str(l)]["rho<=0.02"]; verdict[str(l)]["pass"] = bool(abs(g["slope"] - 1) <= 0.2 and g["r2"] > 0.9)
    print(f"L{l}: rho<=0.02 slope {g['slope']:.3f} R2 {g['r2']:.3f} (n={g['n']}) | v_local-only slope {verdict[str(l)]['rho<=0.02_v_local']['slope']:.3f} R2 {verdict[str(l)]['rho<=0.02_v_local']['r2']:.3f} | random-only slope {verdict[str(l)]['rho<=0.02_random']['slope']:.3f} R2 {verdict[str(l)]['rho<=0.02_random']['r2']:.3f} | per-rho slopes {[round(verdict[str(l)]['per_rho'][str(r)]['slope'],3) for r in RHOS]} -> {'PASS' if verdict[str(l)]['pass'] else 'FAIL'}", flush=True)
ok = all(verdict[str(l)]["pass"] for l in LAYERS)
json.dump({"verdict": verdict, "gate_pass": ok, "rows": {str(l): rows[l] for l in LAYERS}}, open(f"{RES}/first_order_control.json", "w"), indent=1)
print("GATE", "PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 3)
