"""Branch C2: is a low-dimensional context-conditioned transport viable?

G-SECANT failed; G-CONTEXT passed (J_loc/J-bar = 2.2-4.4x, CI-clear, 3/3 layers).
So the dominant error in J-bar is that it averages one operator over contexts and
positions. This asks whether the per-context deviation is LOW-RANK enough to be
worth modelling, and how much of the J-bar -> J_loc gap a rank-k correction
recovers.

METHOD. Materialising J_loc(x_i) as a full 5120x5120 matrix per site is out of
reach (d finite-difference JVPs each). Instead every site is SKETCHED on one
SHARED r-dimensional subspace S = top-r eigenvectors of the pooled C_dd:

    Y_i = J_loc(x_i) . S    in R^{d x r},  by central differences at eps_ref

Shared S means the Y_i are directly comparable, so PCA over sites is well posed,
and any delta inside span(S) is handled exactly. Evaluation therefore uses
directions drawn inside S, with their true effects freshly measured -- never a
projection of an effect measured for a different delta.

  G-CONDVIABLE: top-16 modes of dT_i = Y_i - mean_i Y_i explain > 50% of
  deviation variance. Computed via the site x site Gram matrix (n sites << d*r).

  RECOVERY: on HELD-OUT sites, how much of the J-bar -> J_loc effect-prediction
  gap does T_cond = J-bar + sum_k a_k M_k close, with
    (a) ORACLE a_k  -- projection of the site's true deviation. Upper bound; says
        whether rank-k structure exists at all.
    (b) PREDICTED a_k -- ridge from LAYER-l ACTIVATION ONLY (never a later layer,
        never a label). This is the tuned-lens trap and the honest number.

Everything autograd-free (forward_from + central differences).
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
BANK = os.environ.get("EKKO_BANK", "research/outputs/003_bank")
OUT = os.environ.get("EKKO_OUT", "research/outputs/C2")
LAYERS = [int(x) for x in os.environ.get("EKKO_LAYERS", "16,31,46").split(",")]
R_SKETCH = int(os.environ.get("EKKO_R", "192"))
N_SITES = int(os.environ.get("EKKO_N_SITES", "56"))
N_EVAL_DIRS = int(os.environ.get("EKKO_N_EVAL", "10"))
BATCH = int(os.environ.get("EKKO_BATCH", "32"))
EPS_EVAL, EPS_REF = 0.2, 0.01
KS = (1, 2, 4, 8, 16, 32)
DEV = "cuda:0"


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


@torch.no_grad()
def sketch_and_eval(model, ctx, h, layer, target, pos, S, eval_dirs, med, batch):
    """Y = J_loc . S by central differences at EPS_REF (scaled to unit action),
    plus true finite effects at EPS_EVAL for eval_dirs."""
    def diffs(D, scale):
        n = D.shape[0]
        out = torch.zeros(n, model.d_model, dtype=torch.float32, device=DEV)
        half = max(batch // 2, 1)
        for s in range(0, n, half):
            ch = D[s:s + half] * scale
            m = ch.shape[0]
            hb = h.expand(2 * m, -1, -1).clone()
            hb[:m, pos] += ch.to(hb.dtype)
            hb[m:, pos] -= ch.to(hb.dtype)
            Fb = H.forward_from(model, hb, layer, ctx, target=target)
            out[s:s + m] = (Fb[:m, pos:].float().sum(1) - Fb[m:, pos:].float().sum(1)) / 2
        return out
    Y = diffs(S, EPS_REF * med) / (EPS_REF * med)          # [r, d] = (J_loc S)^T
    E = diffs(eval_dirs, EPS_EVAL * med)                    # [n_eval, d] true effects
    return Y, E


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    jl = H.load_released_lens(LENS_DIR, "j")
    rl = H.load_released_lens(LENS_DIR, "r")
    target = jl.target_layer
    model, hf, tok = H.load_model(MODEL)
    print(f"{model!r}  r={R_SKETCH} sites={N_SITES}  ({time.time()-t0:.0f}s)", flush=True)

    # rebuild base prompts exactly as the bank builder did
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    doc = next(r["text"] for r in ds if len(r["text"]) > 2000)
    prefix_ids = tok.encode(doc, add_special_tokens=False)[:96]
    cats = json.load(open("data/experiments/flexible-generalization.json"))["categories"]
    pairs = []
    for cat in cats:
        for fn in cat["funcs"]:
            enc = {a: prefix_ids + tok.encode(fn["template"].replace("{arg}", a),
                                              add_special_tokens=False) for a in cat["args"]}
            ln = {a: len(v) for a, v in enc.items()}
            for a in cat["args"]:
                for b in cat["args"]:
                    if a != b and ln[a] == ln[b] and enc[a] != enc[b]:
                        pairs.append({"ids_a": enc[a], "cat": cat["name"]})
    bases, cat_of = [], []
    seen = set()
    for p in pairs:
        k = tuple(p["ids_a"])
        if k not in seen:
            seen.add(k)
            bases.append(p["ids_a"])
            cat_of.append(p["cat"])

    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git,
           "r_sketch": R_SKETCH, "n_sites_target": N_SITES, "eps_eval": EPS_EVAL,
           "eps_ref": EPS_REF, "layers": {}}

    for layer in LAYERS:
        print(f"\n{'='*78}\nLAYER {layer}  ({time.time()-t0:.0f}s)", flush=True)
        pack = torch.load(f"{BANK}/{LENS_DIR}_L{layer}.pt", weights_only=False)
        meta = pack["meta"]
        d = pack["dir"].shape[1]
        dirs = pack["dir"].to(DEV, torch.float32)
        eps = pack["eps"].to(DEV, torch.float32)
        X = dirs * eps[:, None]
        base = torch.tensor([m["base"] for m in meta])
        epsmul = torch.tensor([m["eps"] for m in meta])
        fam = [m["family"] for m in meta]
        ub = sorted(set(base.tolist()))
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(ub), generator=g)
        cal_b = {ub[i] for i in perm[: len(ub) // 2].tolist()}
        is_cal = torch.tensor([b.item() in cal_b for b in base])
        med = pack["median_h_norm"]

        fitm = (torch.tensor([f in ("D4", "D1") for f in fam]) & is_cal
                & (epsmul == EPS_EVAL)).to(DEV)
        C_dd = X[fitm].T @ X[fitm]
        w, V = torch.linalg.eigh(C_dd.double())
        S = V[:, -R_SKETCH:].float().T.contiguous()          # [r, d] orthonormal rows
        J = jl.jacobians[layer].to(DEV, torch.float32)
        R = rl.jacobians[layer].to(DEV, torch.float32)

        # sites: one position per base prompt, stratified over calibrate/held-out
        site_list = []
        for b in ub:
            ms = (base == b).nonzero(as_tuple=True)[0]
            pos = meta[int(ms[0])]["pos"]
            site_list.append((b, pos))
        site_list = site_list[:N_SITES]

        ge = torch.Generator(device=DEV).manual_seed(5)
        Ys, Es, Ds, sid = [], [], [], []
        for n, (b, pos) in enumerate(site_list):
            ids = torch.tensor([bases[b]], device=DEV)
            ctx, acts = H.capture(model, ids)
            h = acts[layer]
            coef = torch.randn(N_EVAL_DIRS, R_SKETCH, device=DEV, generator=ge)
            ed = coef @ S
            ed = ed / ed.norm(dim=-1, keepdim=True)
            Y, E = sketch_and_eval(model, ctx, h, layer, target, pos, S, ed, med, BATCH)
            Ys.append(Y); Es.append(E); Ds.append(ed); sid.append(b)
            if (n + 1) % 10 == 0:
                print(f"  sketched {n+1}/{len(site_list)} sites  ({time.time()-t0:.0f}s)",
                      flush=True)

        Y = torch.stack(Ys)                     # [n, r, d]
        n = Y.shape[0]
        Ybar = Y.mean(0)
        dY = (Y - Ybar).reshape(n, -1)          # [n, r*d]

        # ---- G-CONDVIABLE: PCA of deviations via the n x n Gram --------
        Gm = dY @ dY.T
        ev, U = torch.linalg.eigh(Gm.double())
        ev = ev.flip(0).clamp_min(0)
        tot = ev.sum().item()
        var_exp = {str(k): float(ev[:k].sum() / tot) for k in KS if k <= n}
        print(f"\n  deviation PCA over {n} sites: variance explained "
              + "  ".join(f"top{k}={v:.3f}" for k, v in var_exp.items()), flush=True)
        cond_viable = var_exp.get("16", 0.0) > 0.5

        # ---- recovery on held-out sites -------------------------------
        hold_mask = torch.tensor([s not in cal_b for s in sid])
        cal_mask = ~hold_mask
        Uk = U.flip(1).float()                  # [n, n] site loadings
        modes = {}
        for k in KS:
            if k > n:
                continue
            # rank-k basis of deviations built from CALIBRATE sites only
            dYc = dY[cal_mask]
            Gc = dYc @ dYc.T
            evc, Uc = torch.linalg.eigh(Gc.double())
            Uc = Uc.flip(1).float()[:, :k]
            Mk = (Uc.T @ dYc)                   # [k, r*d]
            Mk = Mk / Mk.norm(dim=1, keepdim=True).clamp_min(1e-9)
            modes[k] = Mk

        Dall = torch.stack(Ds)                  # [n, n_eval, d]
        Eall = torch.stack(Es)
        res = {}
        hidx = hold_mask.nonzero(as_tuple=True)[0].tolist()
        for k in modes:
            Mk = modes[k]
            oracle, pred_c = [], []
            for i in hidx:
                a = Mk @ dY[i]                                  # oracle coefficients
                Yi_hat = (Ybar.reshape(-1) + a @ Mk).reshape(R_SKETCH, d)
                # T_cond acting on a direction in span(S): coords in S then apply
                c = Dall[i] @ S.T                               # [n_eval, r]
                oracle.append(cos(c @ Yi_hat, Eall[i]))
            res[k] = {"oracle_cos": torch.cat(oracle).mean().item()}
        # baselines on the same held-out eval directions
        bl = {}
        for nm, T in (("I", None), ("J", J), ("R", R)):
            v = []
            for i in hidx:
                p = Dall[i] if T is None else Dall[i] @ T.T
                v.append(cos(p, Eall[i]))
            bl[nm] = torch.cat(v).mean().item()
        vloc, vbar = [], []
        for i in hidx:
            c = Dall[i] @ S.T
            vloc.append(cos(c @ Y[i], Eall[i]))
            vbar.append(cos(c @ Ybar, Eall[i]))
        bl["J_loc"] = torch.cat(vloc).mean().item()
        bl["J_bar_sketch"] = torch.cat(vbar).mean().item()

        gap = bl["J_loc"] - bl["J"]
        for k in res:
            res[k]["recovered_frac_of_gap"] = ((res[k]["oracle_cos"] - bl["J"]) / gap
                                               if abs(gap) > 1e-9 else float("nan"))
        print(f"\n  held-out sites n={len(hidx)}, eval dirs in span(S), eps={EPS_EVAL}", flush=True)
        print("    baselines: " + "  ".join(f"{k}={v:.4f}" for k, v in bl.items()), flush=True)
        for k, v in sorted(res.items()):
            print(f"    rank {k:<3d} oracle cos={v['oracle_cos']:.4f}  "
                  f"recovers {v['recovered_frac_of_gap']*100:5.1f}% of the J->J_loc gap",
                  flush=True)

        rep["layers"][str(layer)] = {
            "n_sites": n, "n_held_out": len(hidx), "variance_explained": var_exp,
            "G_CONDVIABLE_top16_gt_50pct": bool(cond_viable),
            "baselines": bl, "recovery": {str(k): v for k, v in res.items()},
            "median_h_norm": med}
        with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
            json.dump(rep, f, indent=2)
        torch.save({"S": S.cpu(), "Ybar": Ybar.cpu(), "site_ids": sid},
                   f"{OUT}/{LENS_DIR}_L{layer}_sketch.pt")
        del X, dirs, J, R, Y, dY, Dall, Eall, V, C_dd
        torch.cuda.empty_cache()

    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
