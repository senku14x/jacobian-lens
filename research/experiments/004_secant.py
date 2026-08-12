"""Stage 1 baselines + Stage 2 secant POC, with the span diagnostic.

Operators scored on held-out sites (split by base prompt, so no site leakage):

  I          identity / logit-lens transport
  J          released J-lens
  R          released R-lens
  J+lam*I    the 1-parameter shrinkage competitor (lambda chosen on calibrate)
  J_loc(x)   LOCAL LINEARISATION at this site, estimated from the eps=0.01
             antithetic measurement and extrapolated linearly to the test eps.
             Separates context-averaging error (J_loc >> J) from finite-response
             error (T_sec >> J_loc).
  T_sec      C_Ddelta (C_dd + eta I)^-1, fit on calibrate sites
  T_smooth   T_sec fit on ISOTROPIC D1 ONLY. By Stein this is the SmoothGrad-J
             estimator, so "T_sec beats T_smooth on D4" is exactly the
             circularity control: does finite transport carry content beyond
             smoothing the Jacobian?
  T_RC(lam)  (C_Ddelta + lam R)(C_dd + lam I)^-1, ridge toward R

THE SPAN DIAGNOSTIC. T_sec is only identified on span(C_dd); ridge attenuates
it elsewhere. A held-out direction with most of its energy outside that span is
mispredicted BY CONSTRUCTION, which is indistinguishable from "finite transport
does not generalise". So before any transfer number is interpreted we report,
per held-out delta, the captured fraction ||P_r delta|| / ||delta|| against the
top-r eigenvectors of C_dd, and we report every metric BINNED by that fraction.
A deficit that disappears in the high-capture bin is non-identifiability, not
non-generalisation.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ekko import harness as H  # noqa: E402

LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
BANK = os.environ.get("EKKO_BANK", "research/outputs/003_bank")
OUT = os.environ.get("EKKO_OUT", "research/outputs/004_secant")
TARGET_KEY = os.environ.get("EKKO_TARGET", "d_odd_sum")
DEV = "cuda:0"
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
SHRINK_GRID = (0.0, 0.1, 0.3, 1.0, 3.0)
EPS_EVAL = 0.2          # the "intervention-realistic" scale (doc §4.1)
EPS_REF = 0.01          # the local-linearisation reference scale
CAPTURE_RANK = 512      # r for the span projector


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def fit_T(C_Dd, C_dd, eta, anchor=None, lam=0.0):
    """Ridge / anchored-ridge solve, fp64 for the inverse."""
    d = C_dd.shape[0]
    A = C_dd.double() + (eta * torch.diag(C_dd).mean().double() + lam) * torch.eye(d, device=C_dd.device, dtype=torch.float64)
    B = C_Dd.double() + (lam * anchor.double() if anchor is not None else 0.0)
    return torch.linalg.solve(A.T, B.T).T.float()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    man = json.load(open(f"{BANK}/{LENS_DIR}_manifest.json"))
    jl = H.load_released_lens(LENS_DIR, "j")
    rl = H.load_released_lens(LENS_DIR, "r")
    report = {"lens_dir": LENS_DIR, "target_key": TARGET_KEY, "layers": {},
              "eps_eval": EPS_EVAL, "capture_rank": CAPTURE_RANK}

    for layer in man["layers"]:
        print(f"\n{'='*70}\nLAYER {layer}  ({time.time()-t0:.0f}s)", flush=True)
        pack = torch.load(f"{BANK}/{LENS_DIR}_L{layer}.pt", weights_only=False)
        meta = pack["meta"]
        d = pack["dir"].shape[1]
        dirs = pack["dir"].to(DEV, torch.float32)
        eps = pack["eps"].to(DEV, torch.float32)
        Y = pack[TARGET_KEY].to(DEV, torch.float32)
        X = dirs * eps[:, None]                      # the actual perturbation
        fam = [m["family"] for m in meta]
        base = torch.tensor([m["base"] for m in meta])
        epsmul = torch.tensor([m["eps"] for m in meta])

        # --- split by base prompt (no site leakage) ---------------------
        ub = sorted(set(base.tolist()))
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(ub), generator=g)
        cal_b = {ub[i] for i in perm[: len(ub) // 2].tolist()}
        is_cal = torch.tensor([b.item() in cal_b for b in base])
        print(f"  {len(meta)} deltas | calibrate sites {len(cal_b)}/{len(ub)} "
              f"| cal deltas {int(is_cal.sum())} hold {int((~is_cal).sum())}", flush=True)

        # --- operators --------------------------------------------------
        J = jl.jacobians[layer].to(DEV, torch.float32)
        R = rl.jacobians[layer].to(DEV, torch.float32)
        I = torch.eye(d, device=DEV)

        def accum(mask):
            Xm, Ym = X[mask], Y[mask]
            return Ym.T @ Xm, Xm.T @ Xm

        m_all_cal = is_cal.to(DEV)
        m_d1_cal = torch.tensor([f == "D1" for f in fam], device=DEV) & m_all_cal
        m_d4_cal = torch.tensor([f == "D4" for f in fam], device=DEV) & m_all_cal
        # fit sets: D4+D1 (the doc's fitting families); D1 only (= SmoothGrad-J)
        C_Dd, C_dd = accum(m_d4_cal | m_d1_cal)
        C_Dd1, C_dd1 = accum(m_d1_cal)

        evals = torch.linalg.eigvalsh(C_dd.double())
        cond = (evals[-1] / evals.clamp_min(1e-30)[0]).item()
        eff_rank = (evals.sum() ** 2 / (evals ** 2).sum()).item()
        print(f"  C_dd: cond={cond:.3e}  effective_rank={eff_rank:.1f} / {d}", flush=True)

        # ridge chosen on calibrate D4 at the evaluation scale
        sel = (epsmul == EPS_EVAL).to(DEV)
        m_sel_cal = m_d4_cal & sel
        best_eta, best = RIDGE_GRID[0], -2.0
        for eta in RIDGE_GRID:
            T = fit_T(C_Dd, C_dd, eta)
            c = cos(X[m_sel_cal] @ T.T, Y[m_sel_cal]).mean().item()
            if c > best:
                best_eta, best = eta, c
        T_sec = fit_T(C_Dd, C_dd, best_eta)
        T_smooth = fit_T(C_Dd1, C_dd1, best_eta)
        print(f"  ridge eta={best_eta:g} (cal D4 cos={best:.4f})", flush=True)

        best_lam, bl = LAMBDA_GRID[0], -2.0
        for lam in LAMBDA_GRID:
            T = fit_T(C_Dd, C_dd, best_eta, anchor=R, lam=lam * torch.diag(C_dd).mean().item())
            c = cos(X[m_sel_cal] @ T.T, Y[m_sel_cal]).mean().item()
            if c > bl:
                best_lam, bl = lam, c
        T_RC = fit_T(C_Dd, C_dd, best_eta, anchor=R,
                     lam=best_lam * torch.diag(C_dd).mean().item())
        best_sh, bs = 0.0, -2.0
        for s in SHRINK_GRID:
            c = cos(X[m_sel_cal] @ (J + s * I).T, Y[m_sel_cal]).mean().item()
            if c > bs:
                best_sh, bs = s, c
        print(f"  T_RC lambda={best_lam:g}  |  J+lam*I lam={best_sh:g}", flush=True)

        ops = {"I": I, "J": J, "R": R, f"J+{best_sh:g}I": J + best_sh * I,
               "T_smooth": T_smooth, "T_sec": T_sec, "T_RC": T_RC}

        # --- local linearisation J_loc(x) -------------------------------
        # key a delta by (base, pos, family, tag); the eps=EPS_REF row of the
        # same key is the local JVP, extrapolated linearly to the test scale.
        ref = {}
        for i, m in enumerate(meta):
            if abs(m["eps"] - EPS_REF) < 1e-12:
                ref[(m["base"], m["pos"], m["family"], m["tag"])] = i
        jloc_src = torch.full((len(meta),), -1, dtype=torch.long)
        for i, m in enumerate(meta):
            k = (m["base"], m["pos"], m["family"], m["tag"])
            if k in ref:
                jloc_src[i] = ref[k]
        has_loc = jloc_src >= 0

        # --- span diagnostic --------------------------------------------
        w, V = torch.linalg.eigh(C_dd.double())
        Vr = V[:, -CAPTURE_RANK:].float()                 # top-r eigenvectors
        Xn = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        captured = (Xn @ Vr).norm(dim=-1)                 # ||P_r xhat|| in [0,1]

        # --- evaluate ----------------------------------------------------
        hold = (~is_cal).to(DEV)
        layer_rep = {"cond": cond, "effective_rank": eff_rank, "eta": best_eta,
                     "lambda_RC": best_lam, "shrink": best_sh,
                     "n_cal": int(is_cal.sum()), "n_hold": int((~is_cal).sum()),
                     "families": {}, "capture": {}, "capture_bins": {}}

        for family in ("D4", "D1", "D2"):
            fm = torch.tensor([f == family for f in fam], device=DEV)
            layer_rep["capture"][family] = {
                "mean": captured[fm & hold].mean().item(),
                "median": captured[fm & hold].median().item()}
            for e in sorted({float(x) for x in epsmul.tolist()}):
                m = fm & hold & (epsmul == e).to(DEV)
                if m.sum() < 10:
                    continue
                row = {"n": int(m.sum())}
                for name, T in ops.items():
                    row[name] = cos(X[m] @ T.T, Y[m]).mean().item()
                if family != "D2":
                    ml = m & has_loc.to(DEV)
                    if ml.sum() >= 10:
                        scale = (eps[ml] / eps[jloc_src[ml.cpu()].to(DEV)])
                        pred = Y[jloc_src[ml.cpu()].to(DEV)] * scale[:, None]
                        row["J_loc"] = cos(pred, Y[ml]).mean().item()
                lbl = "native" if e < 0 else f"{e:g}"
                layer_rep["families"].setdefault(family, {})[lbl] = row

        # --- paired bootstrap over held-out SITES on the primary metric ----
        # (D4, eps=EPS_EVAL, cos vs the summed antithetic target). Resampling
        # sites rather than deltas: deltas within a site are not independent.
        prim = torch.tensor([f == "D4" for f in fam], device=DEV) & hold & (epsmul == EPS_EVAL).to(DEV)
        if prim.sum() >= 20:
            pi = prim.nonzero(as_tuple=True)[0]
            sites = base[pi.cpu()]
            usites = sorted(set(sites.tolist()))
            per_op = {k: cos(X[pi] @ T.T, Y[pi]) for k, T in ops.items()}
            gb = torch.Generator().manual_seed(1)
            boots = {k: [] for k in ops}
            for _ in range(2000):
                pick = [usites[i] for i in torch.randint(len(usites), (len(usites),), generator=gb).tolist()]
                sel_idx = torch.cat([(sites == s).nonzero(as_tuple=True)[0] for s in pick])
                for k in ops:
                    boots[k].append(per_op[k][sel_idx.to(DEV)].mean().item())
            bt = {k: torch.tensor(v) for k, v in boots.items()}
            ci = {k: {"mean": per_op[k].mean().item(),
                      "lo": torch.quantile(bt[k], 0.025).item(),
                      "hi": torch.quantile(bt[k], 0.975).item()} for k in ops}
            paired = {}
            for a, b in (("T_sec", "J"), ("T_sec", "R"), ("T_sec", "T_smooth"),
                         ("T_RC", "R"), ("T_RC", "T_sec"), ("R", "J")):
                dlt = bt[a] - bt[b]
                paired[f"{a}-{b}"] = {"delta": per_op[a].mean().item() - per_op[b].mean().item(),
                                      "lo": torch.quantile(dlt, 0.025).item(),
                                      "hi": torch.quantile(dlt, 0.975).item(),
                                      "p_gt0": (dlt > 0).float().mean().item()}
            layer_rep["bootstrap"] = {"n_sites": len(usites), "n_deltas": int(prim.sum()),
                                      "ci": ci, "paired": paired}
            print(f"\n  --- paired bootstrap, D4 eps={EPS_EVAL}, {len(usites)} held-out sites ---",
                  flush=True)
            for k, v in paired.items():
                star = "*" if (v["lo"] > 0 or v["hi"] < 0) else " "
                print(f"   {k:<18s} Δcos={v['delta']:+.4f}  95% CI [{v['lo']:+.4f}, {v['hi']:+.4f}] "
                      f"P(>0)={v['p_gt0']:.3f} {star}", flush=True)

        # metrics binned by captured fraction, at the evaluation scale
        for family in ("D4", "D1", "D2"):
            fm = torch.tensor([f == family for f in fam], device=DEV) & hold
            fm = fm & ((epsmul == EPS_EVAL).to(DEV) if family != "D2" else torch.ones_like(fm))
            if fm.sum() < 20:
                continue
            bins = {}
            for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 1.01)):
                m = fm & (captured >= lo) & (captured < hi)
                if m.sum() < 10:
                    continue
                bins[f"{lo}-{hi}"] = {"n": int(m.sum()),
                                      **{k: cos(X[m] @ T.T, Y[m]).mean().item()
                                         for k, T in ops.items()}}
            layer_rep["capture_bins"][family] = bins

        report["layers"][str(layer)] = layer_rep

        # --- print -------------------------------------------------------
        names = list(ops) + ["J_loc"]
        for family, per_eps in layer_rep["families"].items():
            print(f"\n  --- {family} (held-out) | mean cos(T*delta, Delta) ---", flush=True)
            print("   eps      n   " + "".join(f"{n:>10s}" for n in names), flush=True)
            for e, row in per_eps.items():
                print(f"   {e:<7s}{row['n']:>5d}   " +
                      "".join(f"{row.get(n, float('nan')):>10.4f}" for n in names), flush=True)
            print(f"   captured fraction (top-{CAPTURE_RANK}): "
                  f"mean={layer_rep['capture'][family]['mean']:.3f}", flush=True)
        for family, bins in layer_rep["capture_bins"].items():
            if not bins:
                continue
            print(f"\n  --- {family} binned by captured fraction (eps={EPS_EVAL}) ---", flush=True)
            print("   bin        n   " + "".join(f"{n:>10s}" for n in list(ops)), flush=True)
            for b, row in bins.items():
                print(f"   {b:<9s}{row['n']:>5d}   " +
                      "".join(f"{row[n]:>10.4f}" for n in ops), flush=True)

        torch.save({"T_sec": T_sec.cpu(), "T_RC": T_RC.cpu(), "T_smooth": T_smooth.cpu()},
                   f"{OUT}/{LENS_DIR}_L{layer}_operators.pt")
        del J, R, I, T_sec, T_RC, T_smooth, X, Y, dirs, C_Dd, C_dd, C_Dd1, C_dd1, V, Vr
        torch.cuda.empty_cache()

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
