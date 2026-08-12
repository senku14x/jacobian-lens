"""C3: spectrally shrunk J-lens and mode-wise J/R fusion.

Diagnosis this follows from: T_sec failed with d^2 = 26.2M free parameters
estimated from ~1700 held-out deltas, and collapsed to ~0 under a
template-disjoint split. The operators here keep the SAME model class (a single
fixed d x d matrix, complete outside any calibration span) but cut the free
parameters to d or fewer by reweighting an existing basis instead of learning a
new operator.

Two families, both anchored on the released J:

  spectral shrinkage      J - cI = U diag(s) V^T,   T = cI + U diag(a_i s_i) V^T
  mode-wise J/R fusion    R - J  = U diag(s) V^T,   T = J  + U diag(a_i s_i) V^T

The residual identity motivates c = 1 as the prior an unreliable mode should
collapse toward; c is also fitted as a control.

The fit is closed form and decoupled. With T = A + sum_i a_i s_i u_i v_i^T,

    T delta = A delta + sum_i a_i f_i,      f_i = s_i (v_i . delta) u_i

and because U is ORTHONORMAL, the normal-equation matrix

    G_ij = sum_n <f_i^n, f_j^n> = s_i s_j sum_n (v_i.d_n)(v_j.d_n) <u_i,u_j>

is DIAGONAL. So each mode has a scalar ridge solution a_i = b_i / (G_ii + lam),
and clipping to [0,1] afterwards is the exact solution of the box-constrained
problem rather than an approximation. No d x d solve, no conditioning problem.

Everything is evaluated under the leak-repaired splits from B7 -- base,
unordered-pair, template, category -- because the base-prompt split is what made
T_sec look good. An operator that only wins on the base split has learned
templates. Reported on both cosine and relative residual error, since J's
relerr ~ 1.0 (no better than predicting zero) is invisible to cosine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
BANK = os.environ.get("EKKO_BANK", "research/outputs/B1_bank")
OUT = os.environ.get("EKKO_OUT", "research/outputs/C3")
LAYERS = [int(x) for x in os.environ.get("EKKO_LAYERS", "16,31,46").split(",")]
RANKS = [int(x) for x in os.environ.get("EKKO_RANKS", "64,512,5120").split(",")]
LAMS = [float(x) for x in os.environ.get("EKKO_LAMS", "0.0,0.01,0.1").split(",")]
TARGET = os.environ.get("EKKO_TARGET", "d_one_sum")     # native one-sided primary
NATIVE = os.environ.get("EKKO_NATIVE", "1") == "1"
N_BOOT = 2000
DEV = "cuda:0"


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def relerr(p, y):
    return (p - y).norm(dim=-1) / y.norm(dim=-1).clamp_min(1e-12)


def boot(vals, groups, seed=29):
    g = torch.Generator().manual_seed(seed)
    us = sorted(set(groups.tolist()))
    idx = {u: (groups == u).nonzero(as_tuple=True)[0].to(vals.device) for u in us}
    d = []
    for _ in range(N_BOOT):
        pick = torch.randint(len(us), (len(us),), generator=g).tolist()
        d.append(vals[torch.cat([idx[us[i]] for i in pick])].mean().item())
    t = torch.tensor(d)
    return vals.mean().item(), torch.quantile(t, .025).item(), torch.quantile(t, .975).item()


def fit_modes(U, S, Vt, A, Xf, Yf, lam_rel, rank):
    """Closed-form per-mode coefficients; see module docstring for why diagonal.

    Returns a_i in [0,1] for the top `rank` modes (zero beyond).
    """
    r = min(rank, S.shape[0])
    coef = Xf @ Vt[:r].T                       # [n, r]  (v_i . delta)
    resid = Yf - Xf @ A.T                      # [n, d]  what A leaves unexplained
    Gii = (S[:r] ** 2) * (coef ** 2).sum(0)                       # [r]
    bi = S[:r] * (coef * (resid @ U[:, :r])).sum(0)               # [r]
    lam = lam_rel * Gii.mean().clamp_min(1e-12)
    a = bi / (Gii + lam)
    return a.clamp(0.0, 1.0), a


def build(A, U, S, Vt, a, rank):
    r = a.shape[0]
    return A + (U[:, :r] * (a * S[:r])) @ Vt[:r]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    from ekko import harness as H
    from B7_splits_targets_anchors import build_base_meta
    jl = H.load_released_lens(LENS_DIR, "j")
    rl = H.load_released_lens(LENS_DIR, "r")
    info, _, _ = build_base_meta()
    print(f"metadata for {len(info)} bases ({time.time()-t0:.0f}s)", flush=True)

    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git,
           "target": TARGET, "native": NATIVE, "ranks": RANKS, "lams": LAMS,
           "layers": {}}

    for layer in LAYERS:
        print(f"\n{'='*78}\nLAYER {layer}  ({time.time()-t0:.0f}s)", flush=True)
        pack = torch.load(f"{BANK}/{LENS_DIR}_L{layer}.pt", weights_only=False)
        meta = pack["meta"]
        d = pack["dir"].shape[1]
        X = (pack["dir"].to(DEV, torch.float32)
             * pack["eps"].to(DEV, torch.float32)[:, None])
        Y = pack[TARGET].to(DEV, torch.float32)
        base = torch.tensor([m["base"] for m in meta])
        epsmul = torch.tensor([m["eps"] for m in meta])
        isD4 = torch.tensor([m["family"] == "D4" for m in meta])
        selm = isD4 & ((epsmul < 0) if NATIVE else (epsmul == 0.2))

        J = jl.jacobians[layer].to(DEV, torch.float32)
        R = rl.jacobians[layer].to(DEV, torch.float32)
        I = torch.eye(d, device=DEV)

        # bases for the two families
        fam_defs = {}
        Uj, Sj, Vtj = torch.linalg.svd((J - I).double(), full_matrices=False)
        fam_defs["shrinkJ"] = (I, Uj.float(), Sj.float(), Vtj.float())
        Ur, Sr, Vtr = torch.linalg.svd((R - J).double(), full_matrices=False)
        fam_defs["fuseJR"] = (J, Ur.float(), Sr.float(), Vtr.float())
        print(f"  spectra: (J-I) s1={Sj[0]:.3f} s_med={Sj[len(Sj)//2]:.4f}  "
              f"(R-J) s1={Sr[0]:.3f} s_med={Sr[len(Sr)//2]:.4f}", flush=True)

        Lrep = {"n_total": int(selm.sum()), "svd": {
            "J_minus_I_top": [float(x) for x in Sj[:5]],
            "R_minus_J_top": [float(x) for x in Sr[:5]],
            "J_minus_I_energy_top64": float((Sj[:64] ** 2).sum() / (Sj ** 2).sum()),
            "R_minus_J_energy_top64": float((Sr[:64] ** 2).sum() / (Sr ** 2).sum())}}

        for level in ("base", "template", "category"):
            def key(i):
                ii = info[int(base[i])]
                if level == "base":
                    return str(int(base[i]))
                if level == "category":
                    return ii["cat"]
                return f"{ii['cat']}/{ii['func']}"
            kk = [key(i) for i in range(len(meta))]
            uniq = sorted(set(kk))
            g = torch.Generator().manual_seed(0)
            perm = torch.randperm(len(uniq), generator=g)
            cal = {uniq[i] for i in perm[: max(1, len(uniq) // 2)].tolist()}
            is_cal = torch.tensor([k in cal for k in kk])
            gid = torch.tensor([uniq.index(k) for k in kk])

            fm = (selm & is_cal).to(DEV)
            hm = (selm & ~is_cal).to(DEV)
            Xf, Yf, Xh, Yh = X[fm], Y[fm], X[hm], Y[hm]
            gh = gid[hm.cpu()]

            ops = {"J": J, "R": R, "I": I}
            # 3-scalar affine blend, the baseline no fitted operator may lose to
            F_ = torch.stack([Xf @ J.T, Xf @ R.T, Xf], -1)
            G3 = torch.einsum("ndi,ndj->ij", F_, F_).double()
            b3 = torch.einsum("ndi,nd->i", F_, Yf).double()
            w = torch.linalg.solve(G3 + 1e-6 * torch.eye(3, device=DEV,
                                                         dtype=torch.float64), b3)
            ops["aJ+bR+cI"] = w[0].item() * J + w[1].item() * R + w[2].item() * I

            coefs = {}
            for famname, (A, Um, Sm, Vtm) in fam_defs.items():
                for rank in RANKS:
                    for lam in LAMS:
                        a, araw = fit_modes(Um, Sm, Vtm, A, Xf, Yf, lam, rank)
                        nm = f"{famname}_r{rank}_l{lam:g}"
                        ops[nm] = build(A, Um, Sm, Vtm, a, rank)
                        coefs[nm] = {"n_free": int(a.shape[0]),
                                     "mean_alpha": float(a.mean()),
                                     "frac_alpha_gt_half": float((a > .5).float().mean()),
                                     "frac_clipped_hi": float((araw >= 1).float().mean()),
                                     "frac_clipped_lo": float((araw <= 0).float().mean())}
                # single global alpha control: same basis, ONE parameter
                a1, _ = fit_modes(Um, Sm, Vtm, A, Xf, Yf, 0.0, Sm.shape[0])
                g1 = (a1 * Sm).sum() / Sm.sum().clamp_min(1e-9)
                ops[f"{famname}_globalalpha"] = build(
                    A, Um, Sm, Vtm, torch.full_like(Sm, float(g1)), Sm.shape[0])
                # random-basis control: same #coefficients, meaningless directions
                gg = torch.Generator(device=DEV).manual_seed(41)
                Qu, _ = torch.linalg.qr(torch.randn(d, 512, device=DEV, generator=gg))
                Qv, _ = torch.linalg.qr(torch.randn(d, 512, device=DEV, generator=gg))
                ar, _ = fit_modes(Qu, Sm[:512], Qv.T, A, Xf, Yf, 0.0, 512)
                ops[f"{famname}_randbasis512"] = build(A, Qu, Sm[:512], Qv.T, ar, 512)

            row = {"n_fit": int(fm.sum()), "n_hold": int(hm.sum()),
                   "n_groups": len(set(gh.tolist())), "blend_w": [float(x) for x in w],
                   "coefs": coefs, "ops": {}}
            for nm, T in ops.items():
                p = Xh @ T.T
                c, lo, hi = boot(cos(p, Yh), gh)
                re_, _, _ = boot(relerr(p, Yh), gh)
                row["ops"][nm] = {"cos": c, "ci": [lo, hi], "rel_err": re_}
            Lrep[level] = row

            base_c = row["ops"]["J"]["cos"]
            best = sorted((v["cos"], k) for k, v in row["ops"].items())[-4:][::-1]
            print(f"  [{level:<9s}] groups={row['n_groups']:<4d} n={row['n_hold']:<5d} "
                  f"J={base_c:.3f}(re {row['ops']['J']['rel_err']:.2f}) "
                  f"R={row['ops']['R']['cos']:.3f} "
                  f"blend={row['ops']['aJ+bR+cI']['cos']:.3f}", flush=True)
            for c_, k_ in best:
                print(f"      {k_:<26s} cos={c_:.4f} "
                      f"CI[{row['ops'][k_]['ci'][0]:.3f},{row['ops'][k_]['ci'][1]:.3f}] "
                      f"relerr={row['ops'][k_]['rel_err']:.3f}", flush=True)

        # save the best operators for readout evaluation in C1
        rep["layers"][str(layer)] = Lrep
        torch.save({k: v.cpu() for k, v in ops.items()
                    if k in ("aJ+bR+cI",) or "shrinkJ" in k or "fuseJR" in k},
                   f"{OUT}/{LENS_DIR}_L{layer}_ops.pt")
        del X, Y, ops
        torch.cuda.empty_cache()

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
