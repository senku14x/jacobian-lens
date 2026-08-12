"""D2: fit J_own -- the convention-matched baseline the whole study has lacked.

Every J and R in artifacts 007 and 008 is the RELEASED lens. So every claim of
the form "operator X beats/loses to the J-lens" bundles the thing being tested
with corpus mismatch, convention mismatch, and estimation noise. The fix has been
listed as outstanding three times, deferred each time on cost, and the cost was
never measured. D1 measured it: 186 s/prompt at the released convention, so
n=25 is ~1.3 h on this box. It was never expensive.

Fitted at EXACTLY the released convention so the comparison is controlled:
  corpus      NeelNanda/pile-10k
  n_prompts   25
  max_seq_len 128
  skip_first  4          (released; NOT jlens' default of 16)
  target      62         (= n_layers - 2, read from the released lens)
  dim_batch   8          (32 OOMs at 128 tokens on an 80 GiB H100)

Source layers default to the three bank layers, because that is where every
effect measurement in this project lives and the backward cost is independent of
how many source layers are recorded.

What this unlocks, in order of importance:
  1. J_loc(x) vs J_own      -- the ACTUAL cost of context averaging, isolated
  2. J_own   vs released J  -- how much of the 2-4x gap was corpus/convention
  3. a matched noise floor: two halves of the corpus give two J_own estimates,
     and their disagreement bounds how large a "real" operator difference must
     be before it means anything. Nothing in this project has had that floor.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ekko import harness as H  # noqa: E402
import jlens  # noqa: E402

MODEL = os.environ.get("EKKO_MODEL", "Qwen/Qwen3.6-27B")
LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
OUT = os.environ.get("EKKO_OUT", "research/outputs/D2")
LAYERS = [int(x) for x in os.environ.get("EKKO_LAYERS", "31").split(",")]
N_PROMPTS = int(os.environ.get("EKKO_N_PROMPTS", "25"))
DIM_BATCH = int(os.environ.get("EKKO_DIMB", "8"))
MAX_SEQ = 128
SKIP_FIRST = 4
SPLIT_HALVES = os.environ.get("EKKO_HALVES", "1") == "1"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    jl = H.load_released_lens(LENS_DIR, "j")
    target = jl.target_layer
    prov = dict(jl.provenance)
    model, hf, tok = H.load_model(MODEL)
    print(f"{model!r} target={target} released_provenance={prov}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    prompts, it = [], iter(ds)
    while len(prompts) < N_PROMPTS:
        r = next(it)
        if len(r["text"]) > 2000:
            prompts.append(r["text"][:4000])
    print(f"corpus: {len(prompts)} pile-10k documents", flush=True)

    # Fit the two disjoint halves and MERGE them for the full lens. jlens.merge
    # is an n_prompts-weighted mean of disjoint-subset fits, so this costs n=25
    # prompts total -- the same as one full fit -- and yields the twin-fit noise
    # floor for free. No comparison in this project has had that floor.
    # (3 source layers at seq=128 OOMs on an 80 GiB H100: the 1-layer fit already
    # peaks at 72.7 GiB. Layers are fitted one at a time.)
    runs = {"halfA": prompts[0::2], "halfB": prompts[1::2]}

    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git,
           "model": MODEL, "target_layer": target, "layers": LAYERS,
           "n_prompts": N_PROMPTS, "dim_batch": DIM_BATCH, "max_seq_len": MAX_SEQ,
           "skip_first": SKIP_FIRST, "released_provenance": prov, "fits": {}}

    fitted, lens_objs = {}, {}
    for name, pr in runs.items():
        t1 = time.time()
        torch.cuda.empty_cache()
        lens = jlens.fit(model, pr, source_layers=LAYERS, target_layer=target,
                         dim_batch=DIM_BATCH, max_seq_len=MAX_SEQ,
                         skip_first=SKIP_FIRST,
                         checkpoint_path=f"{OUT}/ckpt_{name}.pt", checkpoint_every=5)
        dt = time.time() - t1
        n_used = int(getattr(lens, "n_prompts", len(pr)))
        fitted[name] = {l: lens.jacobians[l].detach().cpu().float() for l in LAYERS}
        rep["fits"][name] = {"n_prompts_requested": len(pr), "n_prompts_used": n_used,
                             "seconds": dt, "sec_per_prompt": dt / max(n_used, 1)}
        print(f"  [{name}] fitted on {n_used}/{len(pr)} prompts in {dt/60:.1f} min "
              f"({dt/max(n_used,1):.0f} s/prompt)  ({time.time()-t0:.0f}s)", flush=True)
        if n_used != len(pr):
            print(f"    WARNING: {len(pr)-n_used} prompts silently dropped "
                  f"(too short for skip_first={SKIP_FIRST})", flush=True)
        torch.save(fitted[name], f"{OUT}/{LENS_DIR}_{name}.pt")
        lens_objs[name] = lens
        torch.cuda.empty_cache()

    merged = jlens.JacobianLens.merge([lens_objs["halfA"], lens_objs["halfB"]])
    fitted["all"] = {l: merged.jacobians[l].detach().cpu().float() for l in LAYERS}
    rep["fits"]["all"] = {"n_prompts_used": int(getattr(merged, "n_prompts", -1)),
                          "note": "n_prompts-weighted merge of halfA and halfB"}
    torch.save(fitted["all"], f"{OUT}/{LENS_DIR}_all.pt")
    print(f"  [all] merged -> n_prompts={getattr(merged,'n_prompts','?')}",
          flush=True)
    for k in list(lens_objs):
        del lens_objs[k]
    del merged
    torch.cuda.empty_cache()

    # ---- how far is J_own from the released J, and what is the noise floor? --
    Jrel = {l: jl.jacobians[l].float() for l in LAYERS}
    rl = H.load_released_lens(LENS_DIR, "r")
    Rrel = {l: rl.jacobians[l].float() for l in LAYERS}

    def rel_fro(a, b):
        return float((a - b).norm() / b.norm())

    def cos_flat(a, b):
        return float(torch.nn.functional.cosine_similarity(
            a.flatten()[None], b.flatten()[None]).item())

    cmp = {}
    for l in LAYERS:
        e = {"J_own_vs_released_J": {
                "rel_fro": rel_fro(fitted["all"][l], Jrel[l]),
                "cos": cos_flat(fitted["all"][l], Jrel[l])},
             "released_R_vs_released_J": {
                "rel_fro": rel_fro(Rrel[l], Jrel[l]),
                "cos": cos_flat(Rrel[l], Jrel[l])},
             "diag_mean_J_own": float(torch.diagonal(fitted["all"][l]).mean()),
             "diag_mean_J_rel": float(torch.diagonal(Jrel[l]).mean())}
        if SPLIT_HALVES:
            e["twin_fit_noise_floor"] = {
                "rel_fro": rel_fro(fitted["halfA"][l], fitted["halfB"][l]),
                "cos": cos_flat(fitted["halfA"][l], fitted["halfB"][l])}
        cmp[str(l)] = e
        print(f"\n  L{l}:  ||J_own - J_rel||/||J_rel|| = "
              f"{e['J_own_vs_released_J']['rel_fro']:.4f} "
              f"(cos {e['J_own_vs_released_J']['cos']:.4f})", flush=True)
        print(f"        ||R_rel - J_rel||/||J_rel||   = "
              f"{e['released_R_vs_released_J']['rel_fro']:.4f} "
              f"(cos {e['released_R_vs_released_J']['cos']:.4f})", flush=True)
        if SPLIT_HALVES:
            print(f"        twin-fit floor ||A-B||/||B||  = "
                  f"{e['twin_fit_noise_floor']['rel_fro']:.4f} "
                  f"(cos {e['twin_fit_noise_floor']['cos']:.4f})   <-- noise floor",
                  flush=True)
    rep["comparison"] = cmp

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
