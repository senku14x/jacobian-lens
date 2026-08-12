"""C1: reference-anchored J-lens -- a coordinate-origin fix to the READOUT.

The released lens is lens(h) = softmax(W_U norm(J h)), confirmed against the
paper: the operator is applied to the ABSOLUTE activation, with no reference
subtraction and no intercept anywhere. But a Jacobian transports DIFFERENCES.
Taylor expansion licenses

    F(h) ~= F(r) + J (h - r)

for a reference activation r, and only the r = 0, F(0) = 0 special case reduces
to F(h) ~= J h. So the released readout implicitly assumes the upper stack is
linear THROUGH THE ORIGIN, which it is not.

This changes nothing about causal transport: effects are differences, so
F(h+delta) - F(h) = J delta whatever the anchor is. Every effect-prediction
number in this project is therefore untouched. It is purely a readout question
and is evaluated only on readout metrics.

Variants (mu_h, mu_F estimated on pile-10k at the released lens's own
convention -- skip_first=4, 128 tokens):

    logit           unembed(h)                    logit-lens baseline
    J               unembed(J h)                  released, the thing to beat
    J_centred       unembed(J (h - mu_h))
    J_anchored      unembed(mu_F + J (h - mu_h))  the Taylor-licensed form
    J_anchored_pos  same, with position-conditioned mu_h(pos)
    INTERCEPT       unembed(mu_F)                 <-- the control that decides it

The intercept control is not optional. The tuned lens's known pathology is that
an affine bias improves output metrics while ignoring the inspected activation,
and the workspace paper reports exactly that in early tuned-lens layers ("skips
ahead to the output rather than surface those intermediates"). If unembed(mu_F)
alone -- a CONSTANT, identical for every prompt -- scores non-trivially on
pass@k, then any gain from anchoring is that pathology and not a better lens.

Metric is the frozen pass@k from data/evaluations/README.md: mean over items of
the fraction of intermediates whose min-over-layers rank <= k. Evaluated ONLY on
the permanent 10-items-per-category calibration slice; the frozen sets are not
touched.
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

MODEL = os.environ.get("EKKO_MODEL", "Qwen/Qwen3.6-27B")
LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
OUT = os.environ.get("EKKO_OUT", "research/outputs/C1")
N_CALIB = int(os.environ.get("EKKO_N_CALIB", "10"))
N_MU = int(os.environ.get("EKKO_N_MU", "25"))       # pile prompts for mu, released n
MU_LEN = int(os.environ.get("EKKO_MU_LEN", "128"))
SKIP_FIRST = 4                                       # released convention
SEED = 0
SETS = ["multihop", "multilingual", "order-ops", "poetry", "typo", "association"]
DEV = "cuda:0"


def readout_position(tok, prompt: str, slug: str) -> int:
    ids = tok.encode(prompt, add_special_tokens=False)
    if slug != "poetry":
        return len(ids) - 1
    nl = [i for i, t in enumerate(ids) if "\n" in tok.decode([t])]
    return nl[-1] if nl else len(ids) - 1


def target_ids(tok, intermediate) -> list[int]:
    forms = intermediate if isinstance(intermediate, list) else [intermediate]
    out = []
    for f in forms:
        if isinstance(f, str):
            t = H.single_token_id(tok, f)
            if t is not None:
                out.append(t)
    return out


@torch.no_grad()
def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    jl = H.load_released_lens(LENS_DIR, "j")
    rl = H.load_released_lens(LENS_DIR, "r")
    tgt = jl.target_layer
    model, hf, tok = H.load_model(MODEL)
    layers = sorted(jl.jacobians)
    print(f"{model!r} target={tgt} layers={len(layers)}  ({time.time()-t0:.0f}s)",
          flush=True)

    # ---------- mu_h per layer and mu_F, on the released fitting corpus ----
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    texts, it = [], iter(ds)
    while len(texts) < N_MU:
        r = next(it)
        if len(r["text"]) > 1500:
            texts.append(r["text"])
    sum_h = torch.zeros(len(layers), model.d_model, dtype=torch.float64, device=DEV)
    sum_F = torch.zeros(model.d_model, dtype=torch.float64, device=DEV)
    # position-conditioned: bucket by absolute position, capped
    POSB = 16
    sum_hp = torch.zeros(POSB, len(layers), model.d_model, dtype=torch.float64, device=DEV)
    cnt_p = torch.zeros(POSB, dtype=torch.float64, device=DEV)
    n_tok = 0
    for i, txt in enumerate(texts):
        ids = model.encode(txt, max_length=MU_LEN)
        _, acts = H.capture(model, ids, max_length=MU_LEN)
        T = acts[layers[0]].shape[1]
        keep = slice(SKIP_FIRST, T - 1)
        nk = max(0, (T - 1) - SKIP_FIRST)
        if nk <= 0:
            continue
        pos_idx = torch.arange(SKIP_FIRST, T - 1, device=DEV)
        buck = (pos_idx * POSB // max(T - 1, 1)).clamp(max=POSB - 1)
        for li, l in enumerate(layers):
            hh = acts[l][0, keep].double()
            sum_h[li] += hh.sum(0)
            sum_hp[:, li].index_add_(0, buck, hh)
        sum_F += acts[tgt][0, keep].double().sum(0)
        cnt_p.index_add_(0, buck, torch.ones_like(buck, dtype=torch.float64))
        n_tok += nk
        if (i + 1) % 5 == 0:
            print(f"  mu: {i+1}/{N_MU} prompts, {n_tok} positions "
                  f"({time.time()-t0:.0f}s)", flush=True)
    mu_h = (sum_h / n_tok).float()                       # [L, d]
    mu_F = (sum_F / n_tok).float()                       # [d]
    mu_hp = (sum_hp / cnt_p.clamp_min(1)[:, None, None]).float()   # [POSB, L, d]
    print(f"  mu over {n_tok} positions; ||mu_h[L31]||={mu_h[layers.index(31)].norm():.3f} "
          f"||mu_F||={mu_F.norm():.3f}", flush=True)

    # ---------- calibration slice ----------------------------------------
    # Reproduce 002_calib_passk.py's draw EXACTLY -- one generator seeded once and
    # shared across sets, so this is the same permanent calibration slice and not a
    # fresh sample from the frozen evaluation sets.
    work = []
    g = torch.Generator().manual_seed(SEED)
    for slug in SETS:
        items = json.load(open(f"data/evaluations/lens-eval-{slug}.json"))["items"]
        perm = torch.randperm(len(items), generator=g)[:N_CALIB].tolist()
        for i in perm:
            it_ = items[i]
            groups = [target_ids(tok, x) for x in it_["intermediates"]]
            groups = [gp for gp in groups if gp]
            if not groups:
                continue
            work.append((slug, it_["prompt"],
                         readout_position(tok, it_["prompt"], slug), groups))
    print(f"calibration slice: {len(work)} items, "
          f"{sum(len(w[3]) for w in work)} intermediates", flush=True)

    resid = torch.zeros(len(work), len(layers), model.d_model, dtype=torch.float32)
    posb = torch.zeros(len(work), dtype=torch.long)
    for n, (slug, pr, pos, _) in enumerate(work):
        _, acts = H.capture(model, pr, max_length=512)
        T = acts[layers[0]].shape[1]
        posb[n] = min(POSB - 1, pos * POSB // max(T, 1))
        for li, l in enumerate(layers):
            resid[n, li] = acts[l][0, pos].float().cpu()
        if n % 20 == 0:
            print(f"  captured {n}/{len(work)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------- rank every variant ---------------------------------------
    maxg = max(len(w[3]) for w in work)
    BETAS = [0.1, 0.25, 0.5, 1.0]
    variants = (["logit", "logit_centred", "J", "J_centred"]
                + [f"J_anchored_b{b:g}" for b in BETAS]
                + ["J_anchored_normmatched", "J_anchored_pos", "R", "R_anchored",
                   "INTERCEPT"])
    ranks = {v: torch.full((len(work), len(layers), maxg), 10 ** 9, dtype=torch.long)
             for v in variants}

    for li, l in enumerate(layers):
        h = resid[:, li].to(DEV)
        m = mu_h[li].to(DEV)
        mp = mu_hp[posb.to(DEV), li]
        Jm = jl.jacobians[l].to(DEV, torch.float32)
        Rm = rl.jacobians[l].to(DEV, torch.float32)
        preds = {
            "logit": h,
            "logit_centred": h - m,
            "J": h @ Jm.T,
            "J_centred": (h - m) @ Jm.T,
            "J_anchored_pos": mu_F + (h - mp) @ Jm.T,
            "R": h @ Rm.T,
            "R_anchored": mu_F + (h - m) @ Rm.T,
            "INTERCEPT": mu_F.expand(h.shape[0], -1),
        }
        for b in BETAS:
            preds[f"J_anchored_b{b:g}"] = b * mu_F + (h - m) @ Jm.T
        # norm-matched anchor: rescale mu_F so it does not simply swamp the
        # h-dependent term, which is what beta=1 does (||mu_F||=170 vs ||J(h-mu)||)
        zc = (h - m) @ Jm.T
        sc = zc.norm(dim=-1, keepdim=True) / mu_F.norm().clamp_min(1e-6)
        preds["J_anchored_normmatched"] = sc * mu_F + zc
        for v, z in preds.items():
            lg = model.unembed(z).float()
            for n, w in enumerate(work):
                for gi, tids in enumerate(w[3]):
                    t = torch.tensor(tids, device=DEV)
                    best = lg[n, t].max()
                    ranks[v][n, li, gi] = int((lg[n] > best).sum())
            del lg
        del Jm, Rm
        if li % 16 == 0:
            print(f"  ranked layer {l}/{tgt}  ({time.time()-t0:.0f}s)", flush=True)

    def passk(Rk, k, lo, hi):
        vals = []
        for n, w in enumerate(work):
            mm = Rk[n, lo:hi, : len(w[3])].min(0).values
            vals.append((mm < k).float().mean().item())
        return sum(vals) / len(vals)

    half = len(layers) // 2
    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git,
           "n_items": len(work), "n_mu_positions": n_tok, "target_layer": tgt,
           "results": {}}
    print(f"\n{'variant':<16s} {'pass@1':>8s} {'pass@10':>8s} {'pass@10 1st half':>18s} "
          f"{'mean first layer top-10':>24s}")
    for v in variants:
        Rk = ranks[v]
        first = []
        for n, w in enumerate(work):
            for gi in range(len(w[3])):
                hit = (Rk[n, :, gi] < 10).nonzero()
                if len(hit):
                    first.append(int(layers[int(hit[0])]))
        rep["results"][v] = {
            "pass@1": passk(Rk, 1, 0, len(layers)),
            "pass@10": passk(Rk, 10, 0, len(layers)),
            "pass@10_first_half": passk(Rk, 10, 0, half),
            "mean_first_layer_top10": (sum(first) / len(first)) if first else None,
            "frac_ever_top10": len(first) / max(sum(len(w[3]) for w in work), 1)}
        r_ = rep["results"][v]
        fl = r_["mean_first_layer_top10"]
        print(f"{v:<16s} {r_['pass@1']:>8.4f} {r_['pass@10']:>8.4f} "
              f"{r_['pass@10_first_half']:>18.4f} "
              f"{(f'{fl:.1f}' if fl else 'n/a'):>24s}")

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
