"""D1 (blocking): measure what fitting a lens actually costs on this box.

The whole study compared operators fitted by me against the RELEASED J and R.
That comparison is uncontrolled -- the released lenses were fitted on a corpus
and convention I did not reproduce -- and the fix (J_own) was deferred three
times citing cost WITHOUT EVER MEASURING THE COST. The plan document says to
time a few prompts and multiply. This does that.

Measures, on the released convention (skip_first=4, pile-10k, max_seq_len=128):

  * seconds per prompt at the 3 bank layers, and at all 63 layers
  * peak GPU memory, and the largest dim_batch that fits
  * the extrapolated wall clock for n=25 prompts (the released n) in both cases

The cost model from jlens/fitting.py is one forward plus ceil(d_model/dim_batch)
backward passes per prompt, INDEPENDENT of how many source layers are recorded --
so fitting 3 layers should cost nearly the same as fitting all 63, and only
memory should differ. That prediction is checked rather than assumed, because if
it holds then J_own for the 3 bank layers is the same price as a full-depth lens
and there was never a cost argument for skipping it.

Also verifies autograd works at all on this GatedDeltaNet hybrid, which the
whole project routed around: every measurement so far was deliberately
autograd-free because torch.func/double-backward were untested here. A plain
VJP is a weaker requirement, but it has never been run on this model.
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
OUT = os.environ.get("EKKO_OUT", "research/outputs/D1")
N_TIME = int(os.environ.get("EKKO_N_TIME", "2"))
DIM_BATCHES = [int(x) for x in os.environ.get("EKKO_DIMB", "8,32,128").split(",")]
BANK_LAYERS = [16, 31, 46]
SKIP_FIRST = 4
MAX_SEQ = 128
N_RELEASED = 25


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    jl = H.load_released_lens(LENS_DIR, "j")
    target = jl.target_layer
    model, hf, tok = H.load_model(MODEL)
    print(f"{model!r} target={target}  ({time.time()-t0:.0f}s)", flush=True)

    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    prompts, it = [], iter(ds)
    while len(prompts) < N_TIME:
        r = next(it)
        if len(r["text"]) > 2000:
            prompts.append(r["text"][:4000])

    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git,
           "model": MODEL, "target_layer": target, "d_model": model.d_model,
           "n_timed": N_TIME, "skip_first": SKIP_FIRST, "max_seq_len": MAX_SEQ,
           "runs": []}

    # Every config OOMed at max_seq_len=128 without checkpointing: 55 GiB of
    # weights plus the backward graph for 64 blocks exceeds 80 GiB. Gradient
    # checkpointing trades recompute for activation memory and is the standard
    # fix; shorter sequences are the fallback. Both are swept here so the answer
    # is "it costs X" or "it does not fit", not "it is expensive".
    CONFIGS = [(BANK_LAYERS, "bank3", 128, False), (BANK_LAYERS, "bank3", 128, True),
               (BANK_LAYERS, "bank3", 64, True), (BANK_LAYERS, "bank3", 32, True),
               (list(range(target)), "alldepth", 128, True)]
    for src, tag, seqlen, ckpt in CONFIGS:
        if ckpt:
            try:
                hf.gradient_checkpointing_enable()
                hf.config.use_cache = False
            except Exception as e:
                print(f"  gradient checkpointing unavailable: {e}", flush=True)
                continue
        else:
            try:
                hf.gradient_checkpointing_disable()
            except Exception:
                pass
        for db in DIM_BATCHES:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            try:
                lens = jlens.fit(model, prompts, source_layers=src,
                                 target_layer=target, dim_batch=db,
                                 max_seq_len=seqlen, skip_first=SKIP_FIRST,
                                 checkpoint_path=None, checkpoint_every=None)
                dt = (time.time() - t1) / len(prompts)
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
                row = {"tag": tag, "n_source_layers": len(src), "dim_batch": db,
                       "max_seq_len": seqlen, "grad_ckpt": ckpt,
                       "sec_per_prompt": dt, "peak_gib": peak,
                       "n_prompts_fitted": int(getattr(lens, "n_prompts", -1)),
                       "est_hours_n25": dt * N_RELEASED / 3600, "ok": True}
                del lens
            except Exception as e:  # OOM or autograd failure on the hybrid
                row = {"tag": tag, "n_source_layers": len(src), "dim_batch": db,
                       "max_seq_len": seqlen, "grad_ckpt": ckpt,
                       "ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
            rep["runs"].append(row)
            if row["ok"]:
                print(f"  [{tag:<9s}] L={len(src):<3d} db={db:<4d} seq={seqlen:<4d} "
                      f"ckpt={int(ckpt)} {row['sec_per_prompt']:7.1f} s/prompt  "
                      f"peak={row['peak_gib']:5.1f} GiB "
                      f"-> n=25 in {row['est_hours_n25']:.2f} h  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"  [{tag:<9s}] L={len(src):<3d} db={db:<4d} seq={seqlen:<4d} "
                      f"ckpt={int(ckpt)} FAILED {row['error'][:70]}", flush=True)
            torch.cuda.empty_cache()

    ok = [r for r in rep["runs"] if r["ok"]]
    if ok:
        best = min(ok, key=lambda r: r["sec_per_prompt"])
        rep["best"] = best
        b3 = [r for r in ok if r["tag"] == "bank3"]
        ad = [r for r in ok if r["tag"] == "alldepth"]
        if b3 and ad:
            rep["layers_are_free"] = {
                "bank3_best_s": min(r["sec_per_prompt"] for r in b3),
                "alldepth_best_s": min(r["sec_per_prompt"] for r in ad),
                "ratio": min(r["sec_per_prompt"] for r in ad)
                         / min(r["sec_per_prompt"] for r in b3)}
        print(f"\nbest: {best['tag']} dim_batch={best['dim_batch']} "
              f"{best['sec_per_prompt']:.1f} s/prompt -> "
              f"J_own at n=25 costs {best['est_hours_n25']:.2f} h", flush=True)
        if "layers_are_free" in rep:
            print(f"all-depth / 3-layer time ratio = "
                  f"{rep['layers_are_free']['ratio']:.2f} "
                  f"(cost model predicts ~1.0)", flush=True)

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
