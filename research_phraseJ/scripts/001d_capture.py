"""001d — Capture residuals at the 13-layer grid for every 001 context.

emission-*: read position = final token of the context (the position that would produce the phrase).
latent    : read positions = final prompt token (p=-1) and the token before it (p=-2).
Also records: greedy next token (latent correctness covariate) and log p(first phrase token) at the
read position (emission diagnostic). Dumps W_U and the final-norm weight for lens scoring in 001e.

Output (gitignored): outputs/001/acts.pt  {"index": [...], "acts": {layer: [N, d] bf16}, ...}
"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ekko_harness import load_model, capture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.path.join(ROOT, "outputs", "001"); RES = os.path.join(ROOT, "results", "001-template-geometry")
MODEL = "/content/models/Qwen3.6-27B"
LAYERS = list(range(8, 61, 4)) + [62]

model, hf, tok = load_model(MODEL)
ctx = json.load(open(f"{RES}/contexts.json"))
fam = json.load(open(f"{RES}/families.json"))
first_tok = {}
for F in fam["families"]:
    for m in F["members"]: first_tok[m["phrase"]] = m["ids"][0]
for P in fam["non_family_pairs"]:
    for m in P["members"]: first_tok[m["phrase"]] = m["ids"][0]

items = []
for cond in ["emission_template", "emission_natural"]:
    for c in ctx[cond]:
        items.append({"cond": cond, "phrase": c["phrase"], "text": c["text"], "positions": [-1]})
for c in ctx["latent"]:
    if c["admitted"]:
        items.append({"cond": "latent", "phrase": c["phrase"], "text": c["prompt"], "positions": [-1, -2],
                      "route": c["route"], "frame": c["frame"], "answer": c["answer"]})
print("contexts:", len(items), flush=True)

index, acts = [], {l: [] for l in LAYERS}
t0 = time.time()
with torch.no_grad():
    for n, it in enumerate(items):
        ids = model.encode(it["text"], max_length=256)
        # emission contexts: the text must end right where the phrase starts; strip a trailing space token if any
        c, a = capture(model, ids, max_length=256)
        T = ids.shape[1]
        # greedy next token and logprob of the phrase's first token at the final position
        h_final_out = a[model.n_layers - 1][0, -1]
        logits = model.unembed(h_final_out[None])[0].float()
        lp = torch.log_softmax(logits, -1)
        ft = first_tok.get(it["phrase"])
        greedy = int(logits.argmax())
        for p in it["positions"]:
            rec = dict(it); rec.pop("text"); rec["pos"] = p; rec["seq_len"] = T
            rec["greedy_next"] = tok.decode([greedy]); rec["greedy_id"] = greedy
            rec["logp_first_token"] = float(lp[ft]) if ft is not None and p == -1 else None
            rec["last_token"] = tok.decode([int(ids[0, p])])
            index.append(rec)
            for l in LAYERS: acts[l].append(a[l][0, p].to(torch.bfloat16).cpu())
        if (n + 1) % 200 == 0: print(f"{n+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)

out = {"layers": LAYERS, "index": index, "acts": {l: torch.stack(v) for l, v in acts.items()}}
torch.save(out, f"{OUTD}/acts.pt")
# unembedding + final norm for lens scoring on CPU
W_U = hf.lm_head.weight.detach().to(torch.bfloat16).cpu()
norm_w = model._final_norm.weight.detach().float().cpu() if hasattr(model, "_final_norm") else None
torch.save({"W_U": W_U, "final_norm_weight": norm_w, "norm_eps": getattr(getattr(model, "_final_norm", None), "eps", None)}, f"{OUTD}/unembed.pt")
json.dump({"n_items": len(items), "n_rows": len(index), "layers": LAYERS, "seconds": time.time() - t0}, open(f"{RES}/capture_meta.json", "w"), indent=1)
print("done", len(index), "rows in", f"{time.time()-t0:.0f}s")
