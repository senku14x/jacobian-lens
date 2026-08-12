"""M2: direction identification by ablation — the PRIMARY gate signal.

Whose intermediate direction, when removed from the residual stream, costs the
model the most accuracy? This is the R-lens post's own causal metric, and it is
the gate signal that does NOT require an operator to predict the effect of
arbitrary held-out deltas — so it is immune to the span/identifiability
confound that makes M1 transfer hard to read (see 004's span diagnostic).

Protocol (adapted from the R-lens post):
  - two-hop items from data/experiments/probe-swap.json; `intermediate` is the
    bridge entity, `answer` the correct continuation.
  - for operator T and layer l, the intermediate's direction is the norm-folded
    probe vector  v = T_l^T (gamma * u_w)  (doc Part 2 §7, same convention for
    every operator).
  - ablate: h <- h - <v_hat, h> v_hat, applied at every prompt position at each
    layer in the band.
  - score: greedy next-token == `answer`. Objective, no autorater. (The post
    used 8 samples + a GPT-5.4-nano autorater; we trade that for determinism.)
  - control: a matched-norm random direction, redrawn per (item, layer).

BAND. T_sec/T_RC exist only at the three bank layers, so the band is those same
three layers for EVERY operator including J and R. Comparing a 3-layer ablation
against a 32-layer one would confound operator quality with band width.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ekko import harness as H  # noqa: E402

MODEL = os.environ.get("EKKO_MODEL", "Qwen/Qwen3.6-27B")
LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
OPS_DIR = os.environ.get("EKKO_OPS", "research/outputs/004_secant")
BANK = os.environ.get("EKKO_BANK", "research/outputs/003_bank")
OUT = os.environ.get("EKKO_OUT", "research/outputs/005_m2")
N_ITEMS = int(os.environ.get("EKKO_N_ITEMS", "40"))
N_RANDOM = int(os.environ.get("EKKO_N_RANDOM", "3"))
SEED = 0


class Ablator:
    """Project a per-layer direction out of the residual stream."""

    def __init__(self, model, dirs: dict[int, torch.Tensor]):
        self.model, self.dirs, self.handles = model, dirs, []

    def _hook(self, v):
        def fn(module, args, output):
            t = output if torch.is_tensor(output) else output[0]
            vv = v.to(t.dtype)
            proj = (t @ vv)[..., None] * vv
            new = t - proj
            return new if torch.is_tensor(output) else (new, *output[1:])
        return fn

    def __enter__(self):
        for l, v in self.dirs.items():
            self.handles.append(self.model.layers[l].register_forward_hook(self._hook(v)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def greedy_next(model, ids):
    with jl_recorder(model) as rec:
        model.forward(ids)
        h = rec.activations[model.n_layers - 1]
    return int(model.unembed(h[0, -1]).argmax())


def jl_recorder(model):
    import jlens
    return jlens.ActivationRecorder(model.layers, at=[model.n_layers - 1])


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    torch.manual_seed(SEED)
    model, hf, tok = H.load_model(MODEL)
    jl = H.load_released_lens(LENS_DIR, "j")
    rl = H.load_released_lens(LENS_DIR, "r")
    man = json.load(open(f"{BANK}/{LENS_DIR}_manifest.json"))
    LAYERS = man["layers"]
    dev = model.input_device
    gamma = model._final_norm.weight.detach().float()
    W_U = model._lm_head.weight.detach()

    ops = {"I": {l: None for l in LAYERS},
           "J": {l: jl.jacobians[l] for l in LAYERS},
           "R": {l: rl.jacobians[l] for l in LAYERS}}
    for name in ("T_sec", "T_RC", "T_smooth"):
        d = {}
        for l in LAYERS:
            p = f"{OPS_DIR}/{LENS_DIR}_L{l}_operators.pt"
            if os.path.exists(p):
                d[l] = torch.load(p, map_location="cpu", weights_only=True)[name]
        if len(d) == len(LAYERS):
            ops[name] = d
    print(f"operators: {list(ops)}  band={LAYERS}  ({time.time()-t0:.0f}s)", flush=True)

    items = json.load(open("data/experiments/probe-swap.json"))["items"]
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(items), generator=g)[:N_ITEMS].tolist()

    # keep only items the model gets right clean, and whose intermediate is single-token
    work = []
    for i in idx:
        it = items[i]
        wid = H.single_token_id(tok, it["intermediate"])
        aid = H.single_token_id(tok, it["answer"])
        if wid is None or aid is None:
            continue
        ids = model.encode(it["prompt"], max_length=256)
        if greedy_next(model, ids) == aid:
            work.append({"name": it.get("name", str(i)), "ids": ids, "wid": wid, "aid": aid})
    print(f"usable items: {len(work)}/{N_ITEMS} (single-token + correct clean)", flush=True)
    if not work:
        raise SystemExit("no usable items")

    def probe_vec(T, wid):
        """Norm-folded probe direction v = T^T (gamma * u_w), unit-normalised."""
        u = gamma * W_U[wid].float()
        v = u if T is None else T.to(dev, torch.float32).T @ u
        return v / v.norm().clamp_min(1e-12)

    results = {}
    for name, mats in ops.items():
        n_ok = 0
        for w in work:
            dirs = {l: probe_vec(mats[l], w["wid"]) for l in LAYERS}
            with Ablator(model, dirs):
                n_ok += int(greedy_next(model, w["ids"]) == w["aid"])
        acc = n_ok / len(work)
        results[name] = {"acc_after_ablation": acc, "n_correct": n_ok}
        print(f"  {name:<10s} acc after ablating its direction = {acc:.3f} "
              f"({n_ok}/{len(work)})   ({time.time()-t0:.0f}s)", flush=True)

    # matched-norm random control: same geometry, no operator information
    rnd = []
    for r in range(N_RANDOM):
        n_ok = 0
        for w in work:
            dirs = {}
            for l in LAYERS:
                v = torch.randn(model.d_model, device=dev)
                dirs[l] = v / v.norm()
            with Ablator(model, dirs):
                n_ok += int(greedy_next(model, w["ids"]) == w["aid"])
        rnd.append(n_ok / len(work))
    results["random_control"] = {"acc_after_ablation": sum(rnd) / len(rnd),
                                 "per_seed": rnd}
    print(f"  {'random':<10s} acc after ablating a matched-norm random dir = "
          f"{sum(rnd)/len(rnd):.3f}  {rnd}", flush=True)

    base = 1.0  # every kept item is correct clean by construction
    print(f"\n  clean accuracy on kept items = {base:.3f}", flush=True)
    print(f"\n  {'operator':<12s}{'acc':>8s}{'drop':>8s}{'drop vs random':>16s}", flush=True)
    ctrl = results["random_control"]["acc_after_ablation"]
    for name in list(ops) + ["random_control"]:
        a = results[name]["acc_after_ablation"]
        print(f"  {name:<12s}{a:>8.3f}{base-a:>8.3f}{(ctrl-a):>16.3f}", flush=True)

    report = {"model": MODEL, "lens_dir": LENS_DIR, "band": LAYERS,
              "n_items": len(work), "clean_acc": base, "results": results,
              "protocol": "greedy next-token == answer; ablate v=T^T(gamma*u_w) "
                          "at every position at each band layer"}
    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
