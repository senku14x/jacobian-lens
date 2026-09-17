"""000 — Environment verification after a runtime rebuild (no science numbers).

Checks, in order: released J/R/template artifacts load with the expected shapes and provenance;
the model loads in bf16; the CLAUDE.md smoke test reproduces (released J on the web-spinner prompt:
`spider` rank 4 @L40, rank 8 @L46); pile-10k is cached; a scalar backward through jlens' recorder
runs (the phrase-J code path) and reports peak GPU memory.
"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jlens
from lib.ekko_harness import load_model, capture
from lib.phrase_objective import PhraseJ

MODEL = "/content/models/Qwen3.6-27B"; LENS = "/content/lenses/qwen3.6-27b"
t0 = time.time()

# --- artifacts
for kind in ("j", "r"):
    ck = torch.load(f"{LENS}/{kind}-lens/lens.pt", map_location="cpu", weights_only=True)
    J = ck["J"]; ls = sorted(J); tgt = ck.get("provenance", {}).get("target_layer")
    print(f"{kind}-lens: layers {ls[0]}..{ls[-1]} ({len(ls)}), d={ck['d_model']}, n_prompts={ck['n_prompts']}, "
          f"provenance target_layer={tgt} skip_first={ck.get('provenance', {}).get('skip_first')}, "
          f"||J[target]-I||_F={float((J[max(ls)].float() - torch.eye(ck['d_model'])).norm()):.3f}, dtype={J[ls[0]].dtype}")
from safetensors import safe_open
with safe_open(f"{LENS}/template-lens/templates+phrases_v3.safetensors", framework="pt") as f:
    keys = list(f.keys()); T = f.get_slice("templates"); print("template stack keys", keys, "shape", T.get_shape(), "metadata", f.metadata())
rows = open(f"{LENS}/template-lens/template_words+phrases_v3.txt").read().splitlines(); print("template rows:", len(rows))
pj_dir = f"{LENS}/template-lens/passages"; print("passage files:", sorted(os.listdir(pj_dir)))

# --- dataset
from datasets import load_dataset
ds = load_dataset("NeelNanda/pile-10k", split="train"); print("pile-10k:", len(ds), "rows", f"({time.time()-t0:.0f}s)")

# --- model + smoke test
model, hf, tok = load_model(MODEL)
print(f"model loaded: {model} ({time.time()-t0:.0f}s), GPU alloc {torch.cuda.memory_allocated()/1e9:.1f} GB")
Jl = jlens.JacobianLens.load(f"{LENS}/j-lens/lens.pt")
prompt = "Fact: The number of legs on the animal that spins webs is"
lens_logits, model_logits, ids = Jl.apply(model, prompt, layers=[40, 46, 55], positions=[-1])
spider = tok.encode(" spider", add_special_tokens=False)[0]
for l in (40, 46, 55):
    lg = lens_logits[l][0]; rank = int((lg > lg[spider]).sum()); top = [tok.decode([int(t)]) for t in lg.topk(6).indices]
    print(f"  L{l}: spider rank {rank}  top6 {top}")
print("  model top-3:", [tok.decode([int(t)]) for t in model_logits[0].topk(3).indices])

# --- phrase-J code path smoke (one scalar backward, released convention)
LAYERS = list(range(8, 61, 4)); pj = PhraseJ(model, LAYERS, 62, 4)
gamma = model._final_norm.weight.detach().float().cpu(); W_U = hf.lm_head.weight.detach().float().cpu()
x = model.encode(ds[0]["text"], max_length=128)
torch.cuda.reset_peak_memory_stats(); t1 = time.time()
g = pj.v_lin(x, (1 + gamma) * W_U[spider])
print(f"v_lin scalar backward: {time.time()-t1:.1f}s, peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB, "
      f"cos vs released J row @L40 = {float(torch.nn.functional.cosine_similarity(g[40], Jl.jacobians[40].float().T @ ((1+gamma)*W_U[spider]), dim=0)):.4f} (single prompt vs 25-prompt mean; not a gate)")
print(f"done ({time.time()-t0:.0f}s)")
