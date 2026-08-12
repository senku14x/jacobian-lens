"""D3: test modified backward rules at the level of ONE block, before any lens fit.

Rationale (the review's own first experiment): propagating a rule through the
whole upper stack and fitting a lens is expensive, and pointless if the rule is
already worse locally. So measure, for a single decoder block:

    true    Delta y = block(x + delta) - block(x)          (finite differences)
    pred    J_rule(x) . delta                              (autograd JVP under the rule)

and compare cosine, relative error and magnitude ratio.

DESIGN POINT THAT DECIDES HOW TO READ THIS. The exact Jacobian is by definition
the best linear predictor of the true effect as eps -> 0, so every LRP-style rule
-- which deliberately deviates from the true gradient -- MUST lose to plain
autograd at small eps. A ranking at eps -> 0 is close to tautological. These
rules can only win at FINITE eps, where the truth is a secant and a modified
backward can implicitly resemble a path average. The eps sweep and the crossover
point are therefore the result, not any single number.

Blocks are chosen one of each type, since Qwen3.6-27B is a hybrid:
  full_attention   16 of 64 layers  -- q/k norm, output gate, softmax rules
  linear_attention 48 of 64 layers  -- GatedDeltaNet gate/state rules

The JVP is taken with forward-mode AD if it works on this architecture and
double-backward otherwise; both are untested here, which is why the rest of the
project was built autograd-free. Whichever path is used is recorded.
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
from ekko import rules as RU  # noqa: E402

MODEL = os.environ.get("EKKO_MODEL", "Qwen/Qwen3.6-27B")
LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
BANK = os.environ.get("EKKO_BANK", "research/outputs/B1_bank")
OUT = os.environ.get("EKKO_OUT", "research/outputs/D3")
N_SITES = int(os.environ.get("EKKO_SITES", "24"))
EPS_MULT = [float(x) for x in os.environ.get("EKKO_EPS", "0.001,0.01,0.05,0.2,1.0").split(",")]
DEV = "cuda:0"


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def block_type(model, l):
    blk = model.layers[l]
    if getattr(blk, "linear_attn", None) is not None:
        return "linear_attention"
    return "full_attention"


def jvp_forward_mode(fn, x, v):
    from torch.func import jvp
    return jvp(fn, (x,), (v,))[1]


def jvp_double_backward(fn, x, v):
    xx = x.detach().clone().requires_grad_(True)
    y = fn(xx)
    u = torch.zeros_like(y, requires_grad=True)
    (g,) = torch.autograd.grad(y, xx, grad_outputs=u, create_graph=True)
    (out,) = torch.autograd.grad(g, u, grad_outputs=v, retain_graph=False)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    model, hf, tok = H.load_model(MODEL)
    # SDPA/flash kernels have no forward-mode (or double-backward) derivative:
    # every full-attention rule failed with
    #   "derivative for aten::_scaled_dot_product_flash_attention_backward is
    #    not implemented".
    # Eager attention also routes through nn.functional.softmax, which is what
    # the value-only and tempered-softmax rules patch, so this is required for
    # those rules to apply at all rather than merely to avoid the crash.
    try:
        hf.config._attn_implementation = "eager"
        hf.set_attn_implementation("eager")
    except Exception as e:
        print(f"  could not force eager attention: {e}", flush=True)
    types_by_layer = {l: block_type(model, l) for l in range(model.n_layers)}
    n_full = sum(1 for v in types_by_layer.values() if v == "full_attention")
    print(f"{model!r}  full_attention={n_full}/{model.n_layers}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    # one representative block of each type, near the middle of the stack
    picks = {}
    for want in ("full_attention", "linear_attention"):
        cands = [l for l, t in types_by_layer.items() if t == want and 8 <= l <= 50]
        if cands:
            picks[want] = min(cands, key=lambda l: abs(l - 31))
    print(f"blocks under test: {picks}", flush=True)

    pack = torch.load(f"{BANK}/{LENS_DIR}_L31.pt", weights_only=False)
    meta, base_ids = pack["meta"], pack["base_ids"]
    med = pack["median_h_norm"]
    # BOTH delta families, as specified: native counterfactual deltas (D4) and
    # small isotropic deltas (D1). The first run used D4 only.
    g = torch.Generator().manual_seed(5)
    fam_sites = {}
    for fam, cond in (("D4_native", lambda m: m["family"] == "D4" and m["eps"] < 0),
                      ("D1_isotropic", lambda m: m["family"] == "D1"
                       and abs(m["eps"] - 0.05) < 1e-12)):
        cand = [i for i, m in enumerate(meta) if cond(m)]
        if not cand:
            continue
        o = torch.randperm(len(cand), generator=g)[:N_SITES].tolist()
        fam_sites[fam] = [cand[i] for i in o]
    print("delta families: " + ", ".join(f"{k}={len(v)}" for k, v in fam_sites.items()),
          flush=True)

    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git,
           "blocks": picks, "n_sites": {k: len(v) for k, v in fam_sites.items()}, "eps_mult": EPS_MULT,
           "layer_types": {"full_attention": n_full,
                           "linear_attention": model.n_layers - n_full},
           "results": {}}

    jvp_mode = None
    for btype, layer in picks.items():
      for famname, sites in fam_sites.items():
        blk = model.layers[layer]
        print(f"\n{'='*74}\n{btype}  block {layer}  deltas={famname}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        acc = {}
        for si, idx in enumerate(sites):
            m = meta[idx]
            ids = torch.tensor([base_ids[m["base"]]], device=model.input_device)
            ctx, acts = H.capture(model, ids)
            # input to this block = output of the previous block
            x = (acts[layer - 1] if layer > 0 else acts[0]).detach()
            kw = ctx.expand(1)
            dirv = pack["dir"][idx].to(DEV, torch.float32)
            pos = m["pos"]

            def f(z):
                out = blk(z, **kw)
                return out if torch.is_tensor(out) else out[0]

            # ---- JVP under each rule (choose the working autograd path once)
            preds = {}
            for rule in RU.RULES:
                v = torch.zeros_like(x)
                v[0, pos] = dirv.to(x.dtype)
                try:
                    with RU.apply_rule(blk, rule):
                        if jvp_mode in (None, "forward"):
                            try:
                                out = jvp_forward_mode(f, x, v)
                                jvp_mode = "forward"
                            except Exception:
                                out = jvp_double_backward(f, x, v)
                                jvp_mode = "double_backward"
                        else:
                            out = jvp_double_backward(f, x, v)
                    preds[rule] = out[0, pos:].float().sum(0)
                except Exception as e:
                    if si == 0:
                        print(f"    rule {rule} unavailable: "
                              f"{type(e).__name__}: {str(e)[:90]}", flush=True)
                    preds[rule] = None

            # ---- true finite effect at each eps
            with torch.no_grad():
                y0 = f(x)
                for e in EPS_MULT:
                    xx = x.clone()
                    xx[0, pos] += (dirv * (e * med)).to(x.dtype)
                    dy = (f(xx) - y0)[0, pos:].float().sum(0)
                    for rule, p in preds.items():
                        if p is None:
                            continue
                        pe = p * (e * med)
                        k = (rule, f"{e:g}")
                        a = acc.setdefault(k, {"cos": [], "rel": [], "mag": []})
                        a["cos"].append(float(cos(pe[None], dy[None])))
                        a["rel"].append(float((pe - dy).norm() / dy.norm().clamp_min(1e-9)))
                        a["mag"].append(float(pe.norm() / dy.norm().clamp_min(1e-9)))
            if (si + 1) % 8 == 0:
                print(f"    {si+1}/{len(sites)} sites  ({time.time()-t0:.0f}s)",
                      flush=True)

        res = {}
        for (rule, e), a in acc.items():
            res.setdefault(e, {})[rule] = {
                "cos": sum(a["cos"]) / len(a["cos"]),
                "rel_err": sum(a["rel"]) / len(a["rel"]),
                "mag_ratio": sum(a["mag"]) / len(a["mag"]), "n": len(a["cos"])}
        rep["results"][f"{btype}|{famname}"] = {"layer": layer, "family": famname,
                                                "by_eps": res}
        # all three metrics reported, not cosine alone
        for metric, fmt in (("cos", "{:>11.4f}"), ("rel_err", "{:>11.3f}"),
                            ("mag_ratio", "{:>11.3f}")):
            print(f"\n  [{metric}] {'rule':<26s}"
                  + "".join(f"{('e=' + f'{x:g}'):>11s}" for x in EPS_MULT))
            for rule in RU.RULES:
                if not any(rule in res.get(f"{e:g}", {}) for e in EPS_MULT):
                    continue
                row = "".join(
                    fmt.format(res.get(f"{e:g}", {}).get(rule, {}).get(metric, float("nan")))
                    for e in EPS_MULT)
                print(f"           {rule:<26s}{row}", flush=True)

    rep["jvp_mode"] = jvp_mode
    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\njvp path used: {jvp_mode}\nwrote {OUT}/{LENS_DIR}.json "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
