"""A4 / M2: direction identification by ablation. Span-immune causal gate.

Whose intermediate direction, when projected out of the residual stream, costs
the model the most probability mass on the correct answer? This never requires
an operator to predict the effect of arbitrary held-out deltas, so it is immune
to the identifiability confound that makes M1 transfer hard to read.

Scoring is judge-free: Delta log-prob of the correct answer token at the final
prompt position, clean minus ablated. Larger = the ablated direction mattered
more. (The R-lens post used 8 samples + a GPT-5.4-nano autorater; we trade that
for determinism and no external dependency.)

Band: T_sec exists only at the bank layers, so every operator -- including J and
R, which have all 63 -- is ablated at the SAME layers. Comparing a 3-layer
ablation against a 32-layer one would confound operator quality with band width.

Controls: matched-norm random directions (N_RANDOM seeds), and a general-damage
check -- the fraction of positions on held-out pile text whose top-1 token is
unchanged under the same ablation. An operator that wins by breaking the model
is not identifying a direction.
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
OPS = os.environ.get("EKKO_OPS", "research/outputs/A1_A6")
OUT = os.environ.get("EKKO_OUT", "research/outputs/A4_m2")
BANDS = json.loads(os.environ.get("EKKO_BANDS", '{"L16":[16],"L16_31_46":[16,31,46]}'))
N_ITEMS = int(os.environ.get("EKKO_N_ITEMS", "40"))
N_RANDOM = int(os.environ.get("EKKO_N_RANDOM", "3"))
N_BOOT = 2000
SEED = 0
DEV = "cuda:0"


class Ablate:
    def __init__(self, model, dirs):
        self.model, self.dirs, self.h = model, dirs, []

    def _mk(self, v):
        def fn(mod, args, out):
            t = out if torch.is_tensor(out) else out[0]
            vv = v.to(t.dtype)
            new = t - (t @ vv)[..., None] * vv
            return new if torch.is_tensor(out) else (new, *out[1:])
        return fn

    def __enter__(self):
        for l, v in self.dirs.items():
            self.h.append(self.model.layers[l].register_forward_hook(self._mk(v)))
        return self

    def __exit__(self, *e):
        for x in self.h:
            x.remove()
        self.h = []


@torch.no_grad()
def final_logits(model, ids):
    with jlens.ActivationRecorder(model.layers, at=[model.n_layers - 1]) as rec:
        model.forward(ids)
        h = rec.activations[model.n_layers - 1]
    return model.unembed(h[0, -1]).float()


@torch.no_grad()
def all_logits(model, ids):
    with jlens.ActivationRecorder(model.layers, at=[model.n_layers - 1]) as rec:
        model.forward(ids)
        h = rec.activations[model.n_layers - 1]
    return model.unembed(h[0]).float()


def boot_mean(vals, seed=21):
    v = torch.tensor(vals)
    g = torch.Generator().manual_seed(seed)
    d = torch.stack([v[torch.randint(len(v), (len(v),), generator=g)].mean()
                     for _ in range(N_BOOT)])
    return v.mean().item(), torch.quantile(d, .025).item(), torch.quantile(d, .975).item(), d


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    torch.manual_seed(SEED)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    model, hf, tok = H.load_model(MODEL)
    jl = H.load_released_lens(LENS_DIR, "j")
    rl = H.load_released_lens(LENS_DIR, "r")
    gamma = model._final_norm.weight.detach().float()
    W_U = model._lm_head.weight.detach()
    print(f"{model!r}  ({time.time()-t0:.0f}s)", flush=True)

    items = json.load(open("data/experiments/probe-swap.json"))["items"]
    g = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(items), generator=g).tolist()
    work = []
    for i in order:
        if len(work) >= N_ITEMS:
            break
        it = items[i]
        wid = H.single_token_id(tok, it["intermediate"])
        aid = H.single_token_id(tok, it["answer"])
        if wid is None or aid is None:
            continue
        ids = model.encode(it["prompt"], max_length=256)
        lg = final_logits(model, ids)
        lp = torch.log_softmax(lg, -1)[aid].item()
        if int(lg.argmax()) != aid:
            continue
        work.append({"name": it.get("name", str(i)), "ids": ids, "wid": wid,
                     "aid": aid, "clean_lp": lp})
    print(f"usable items: {len(work)}  (single-token + correct greedy clean)", flush=True)

    # general-damage probe text
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    probe = []
    for r in ds:
        if len(r["text"]) > 1500:
            probe.append(model.encode(r["text"], max_length=128))
        if len(probe) == 4:
            break
    clean_top1 = [all_logits(model, p).argmax(-1) for p in probe]

    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git,
           "n_items": len(work), "scoring": "delta log-prob of correct answer token",
           "bands": BANDS, "results": {}}

    for band_name, band in BANDS.items():
        ops = {"logit": {l: None for l in band},
               "J": {l: jl.jacobians[l] for l in band},
               "R": {l: rl.jacobians[l] for l in band}}
        for nm in ("T_sec^D4+D1", "T_sec^D4", "T_RC"):
            d = {}
            for l in band:
                p = f"{OPS}/{LENS_DIR}_L{l}_operators.pt"
                if os.path.exists(p):
                    t = torch.load(p, map_location="cpu", weights_only=True)
                    if nm in t:
                        d[l] = t[nm]
            if len(d) == len(band):
                ops[nm] = d

        def vec(T, wid):
            u = gamma * W_U[wid].float()
            v = u if T is None else T.to(DEV, torch.float32).T @ u
            return v / v.norm().clamp_min(1e-12)

        band_res = {}
        for nm, mats in ops.items():
            drops = []
            for w in work:
                dirs = {l: vec(mats[l], w["wid"]) for l in band}
                with Ablate(model, dirs):
                    lg = final_logits(model, w["ids"])
                drops.append(w["clean_lp"] - torch.log_softmax(lg, -1)[w["aid"]].item())
            m, lo, hi, dr = boot_mean(drops)
            # general damage under the same ablation, using a fixed token's direction
            dirs = {l: vec(mats[l], work[0]["wid"]) for l in band}
            with Ablate(model, dirs):
                keep = [float((all_logits(model, p).argmax(-1) == c).float().mean())
                        for p, c in zip(probe, clean_top1)]
            band_res[nm] = {"mean_drop": m, "ci": [lo, hi],
                            "top1_kept_on_pile": sum(keep) / len(keep), "_draws": dr}
            print(f"  [{band_name}] {nm:<12s} Δlogp={m:+.4f} CI[{lo:+.3f},{hi:+.3f}] "
                  f"top1_kept={band_res[nm]['top1_kept_on_pile']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        rnd_all = []
        for s in range(N_RANDOM):
            gg = torch.Generator(device=DEV).manual_seed(100 + s)
            drops = []
            for w in work:
                dirs = {}
                for l in band:
                    v = torch.randn(model.d_model, device=DEV, generator=gg)
                    dirs[l] = v / v.norm()
                with Ablate(model, dirs):
                    lg = final_logits(model, w["ids"])
                drops.append(w["clean_lp"] - torch.log_softmax(lg, -1)[w["aid"]].item())
            rnd_all.append(drops)
        flat = [sum(x) / len(x) for x in zip(*rnd_all)]
        m, lo, hi, dr = boot_mean(flat)
        band_res["random"] = {"mean_drop": m, "ci": [lo, hi], "n_seeds": N_RANDOM,
                              "top1_kept_on_pile": None, "_draws": dr}
        print(f"  [{band_name}] {'random':<12s} Δlogp={m:+.4f} CI[{lo:+.3f},{hi:+.3f}]",
              flush=True)

        paired = {}
        for a in band_res:
            for b in ("random", "R", "J"):
                if a == b or b not in band_res:
                    continue
                dl = band_res[a]["_draws"] - band_res[b]["_draws"]
                lo2, hi2 = torch.quantile(dl, .025).item(), torch.quantile(dl, .975).item()
                paired[f"{a}-{b}"] = {
                    "delta": band_res[a]["mean_drop"] - band_res[b]["mean_drop"],
                    "lo": lo2, "hi": hi2, "ci_clear": bool(lo2 > 0 or hi2 < 0)}
        for k, v in paired.items():
            if v["ci_clear"]:
                print(f"    {k:<22s} {v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] *",
                      flush=True)
        rep["results"][band_name] = {
            "operators": {k: {kk: vv for kk, vv in v.items() if kk != "_draws"}
                          for k, v in band_res.items()}, "paired": paired}

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
