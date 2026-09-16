"""001b — Residual mean/covariance per layer on pile-10k, two disjoint-sequence halves + pooled;
null activations for calibration; token frequency table.

Layers: the 13-layer grid L8..L60 step 4 plus L62 (jlens convention: output of block l).
Outputs (gitignored, outputs/001/): sigma_{half}.pt with {layer: (mu, Sigma)} fp32; null_acts.pt
with {layer: [N, d]} bf16 for 10k random positions; token_freq.json.
Documents used for emission-natural contexts (001c) are excluded here via an id list if present.
"""
import json, os, random, sys, time
import torch, transformers
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ekko_harness import load_model, capture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.path.join(ROOT, "outputs", "001"); os.makedirs(OUTD, exist_ok=True)
RES = os.path.join(ROOT, "results", "001-template-geometry")
MODEL = "/content/models/Qwen3.6-27B"
LAYERS = list(range(8, 61, 4)) + [62]
N_SEQ = int(os.environ.get("N_SEQ", 600)); SEQ_LEN = 128; SKIP = 4
N_NULL = 10000

excl = set()
p = os.path.join(RES, "natural_doc_ids.json")
if os.path.exists(p): excl = set(json.load(open(p)))

model, _hf, tok = load_model(MODEL)
ds = load_dataset("NeelNanda/pile-10k", split="train")
random.seed(0)
order = [i for i in random.sample(range(len(ds)), len(ds)) if i not in excl]

# token frequency over the first 3000 docs (for stratified row sampling in analysis)
freq = {}
for i in order[:3000]:
    for t in tok.encode(ds[i]["text"][:4000], add_special_tokens=False):
        freq[t] = freq.get(t, 0) + 1
json.dump(freq, open(os.path.join(OUTD, "token_freq.json"), "w"))

d = model.d_model
def fresh(): return {l: {"n": 0, "s": torch.zeros(d, dtype=torch.float64, device="cuda"), "ss": torch.zeros(d, d, dtype=torch.float64, device="cuda")} for l in LAYERS}
halves = {"A": fresh(), "B": fresh()}
null = {l: [] for l in LAYERS}; null_per_seq = max(1, N_NULL // N_SEQ + 1)
used = []; t0 = time.time(); n_done = 0
for i in order:
    if n_done >= N_SEQ: break
    ids = tok.encode(ds[i]["text"], add_special_tokens=False)
    if len(ids) < SEQ_LEN: continue
    start = random.randint(0, len(ids) - SEQ_LEN)
    x = torch.tensor([ids[start:start + SEQ_LEN]], device="cuda")
    _, acts = capture(model, x, max_length=SEQ_LEN)
    half = "A" if n_done % 2 == 0 else "B"
    pos = list(range(SKIP, SEQ_LEN - 1))
    npos = random.sample(pos, min(null_per_seq, len(pos)))
    for l in LAYERS:
        h = acts[l][0, pos].to(torch.float64)
        st = halves[half][l]; st["n"] += h.shape[0]; st["s"] += h.sum(0); st["ss"] += h.T @ h
        null[l].append(acts[l][0, npos].to(torch.bfloat16).cpu())
    used.append(int(i)); n_done += 1
    if n_done % 100 == 0: print(f"{n_done}/{N_SEQ} seqs, {time.time()-t0:.0f}s", flush=True)

def finalize(st):
    out = {}
    for l, v in st.items():
        n = v["n"]; mu = v["s"] / n; S = v["ss"] / n - torch.outer(mu, mu)
        out[l] = (mu.float().cpu(), S.float().cpu(), n)
    return out
res = {"A": finalize(halves["A"]), "B": finalize(halves["B"])}
pooled = {}
for l in LAYERS:
    nA, nB = res["A"][l][2], res["B"][l][2]
    sA = halves["A"][l]["s"] + halves["B"][l]["s"]; ssA = halves["A"][l]["ss"] + halves["B"][l]["ss"]; n = nA + nB
    mu = sA / n; S = ssA / n - torch.outer(mu, mu)
    pooled[l] = (mu.float().cpu(), S.float().cpu(), n)
torch.save({"layers": LAYERS, "A": res["A"], "B": res["B"], "pooled": pooled, "seq_ids": used, "seq_len": SEQ_LEN, "skip": SKIP},
           os.path.join(OUTD, "sigma.pt"))
torch.save({l: torch.cat(v)[:N_NULL] for l, v in null.items()}, os.path.join(OUTD, "null_acts.pt"))
json.dump({"n_seq": n_done, "seq_ids": used, "layers": LAYERS, "n_tokens_per_half": {h: res[h][LAYERS[0]][2] for h in "AB"}},
          open(os.path.join(RES, "sigma_meta.json"), "w"), indent=1)
print("done", n_done, "seqs; tokens per half:", {h: res[h][LAYERS[0]][2] for h in "AB"}, f"{time.time()-t0:.0f}s")
