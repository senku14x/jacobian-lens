"""A1 (per-eps secant fits) + A6 (J_loc vs J-bar across layers) + T_RC.

No model load -- pure linear algebra over the verified bank.

A1 fixes the eps-weighting bug: a pooled accumulator over all scales weights
each sample by ||delta||^2, so eps=1.0 samples carry 10^4x the weight of
eps=0.01 ones and the "pooled" operator is really T_sec(eps~1). Every operator
here is fit and evaluated at a single scale. Two fit families are kept --
T_sec^D4 (natural directions only) and T_sec^D4+D1 (span-filled) -- because
their difference is itself a finding about whether the secant needs isotropic
coverage to be identified.

A6 computes J_loc (the exact local Jacobian action at that site, from the
eps=0.01 antithetic measurement extrapolated linearly) against the released
J-bar, with a paired bootstrap over HELD-OUT BASE PROMPTS on the ratio.

All CIs: 2000 resamples of held-out base prompts (not deltas -- deltas within a
prompt are not independent).
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

LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
BANK = os.environ.get("EKKO_BANK", "research/outputs/003_bank")
OUT = os.environ.get("EKKO_OUT", "research/outputs/A1_A6")
TARGET_KEY = os.environ.get("EKKO_TARGET", "d_odd_sum")
DEV = "cuda:0"
RIDGE_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
SHRINK_GRID = (0.0, 0.1, 0.3, 1.0, 3.0)
EPS_EVAL, EPS_REF = 0.2, 0.01
N_BOOT = 2000


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def solve(C_Dd, C_dd, eta_abs, anchor=None, lam=0.0):
    d = C_dd.shape[0]
    A = C_dd.double() + (eta_abs + lam) * torch.eye(d, device=C_dd.device, dtype=torch.float64)
    B = C_Dd.double() + (lam * anchor.double() if anchor is not None else 0.0)
    return torch.linalg.solve(A.T, B.T).T.float()


def boot_ci(per_delta: dict, sites: torch.Tensor, seed=1):
    """Paired bootstrap over held-out base prompts. per_delta: name -> [n] cos."""
    us = sorted(set(sites.tolist()))
    idx = [(sites == s).nonzero(as_tuple=True)[0] for s in us]
    g = torch.Generator().manual_seed(seed)
    draws = {k: [] for k in per_delta}
    for _ in range(N_BOOT):
        pick = torch.randint(len(us), (len(us),), generator=g).tolist()
        sel = torch.cat([idx[i] for i in pick]).to(next(iter(per_delta.values())).device)
        for k, v in per_delta.items():
            draws[k].append(v[sel].mean().item())
    return {k: torch.tensor(v) for k, v in draws.items()}, len(us)


def summarize(draws, per_delta, pairs):
    out = {"n_prompts": None, "point": {}, "ci": {}, "paired": {}}
    for k, v in per_delta.items():
        out["point"][k] = v.mean().item()
        out["ci"][k] = [torch.quantile(draws[k], 0.025).item(),
                        torch.quantile(draws[k], 0.975).item()]
    for a, b in pairs:
        if a not in draws or b not in draws:
            continue
        dl = draws[a] - draws[b]
        lo, hi = torch.quantile(dl, 0.025).item(), torch.quantile(dl, 0.975).item()
        out["paired"][f"{a}-{b}"] = {
            "delta": per_delta[a].mean().item() - per_delta[b].mean().item(),
            "lo": lo, "hi": hi, "p_gt0": (dl > 0).float().mean().item(),
            "ci_clear": bool(lo > 0 or hi < 0),
            "half_width": (hi - lo) / 2}
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    man = json.load(open(f"{BANK}/{LENS_DIR}_manifest.json"))
    jl = H.load_released_lens(LENS_DIR, "j")
    rl = H.load_released_lens(LENS_DIR, "r")
    githash = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bank": BANK,
           "target_key": TARGET_KEY, "git": githash, "n_boot": N_BOOT, "layers": {}}

    for layer in man["layers"]:
        print(f"\n{'='*80}\nLAYER {layer}   ({time.time()-t0:.0f}s)", flush=True)
        pack = torch.load(f"{BANK}/{LENS_DIR}_L{layer}.pt", weights_only=False)
        meta = pack["meta"]
        d = pack["dir"].shape[1]
        dirs = pack["dir"].to(DEV, torch.float32)
        eps = pack["eps"].to(DEV, torch.float32)
        Y = pack[TARGET_KEY].to(DEV, torch.float32)
        X = dirs * eps[:, None]
        fam = [m["family"] for m in meta]
        base = torch.tensor([m["base"] for m in meta])
        epsmul = torch.tensor([m["eps"] for m in meta])
        F = {f: torch.tensor([x == f for x in fam]) for f in ("D1", "D2", "D4")}

        ub = sorted(set(base.tolist()))
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(ub), generator=g)
        cal_b = {ub[i] for i in perm[: len(ub) // 2].tolist()}
        is_cal = torch.tensor([b.item() in cal_b for b in base])
        hold = ~is_cal

        J = jl.jacobians[layer].to(DEV, torch.float32)
        R = rl.jacobians[layer].to(DEV, torch.float32)
        I = torch.eye(d, device=DEV)

        # local-linearisation source index
        ref = {(m["base"], m["pos"], m["family"], m["tag"]): i
               for i, m in enumerate(meta) if abs(m["eps"] - EPS_REF) < 1e-12}
        src = torch.tensor([ref.get((m["base"], m["pos"], m["family"], m["tag"]), -1)
                            for m in meta])
        has_loc = src >= 0

        def jloc_pred(mask):
            si = src[mask].to(DEV)
            md = mask.to(DEV)
            return Y[si] * (eps[md] / eps[si])[:, None]

        eps_labels = [f"{e:g}" for e in sorted({float(x) for x in epsmul.tolist() if x > 0})]
        eps_labels.append("native")
        L = {"d_model": d, "n_cal_prompts": len(cal_b), "n_hold_prompts": len(ub) - len(cal_b),
             "fits": {}, "cross_eps": {}, "A6": {}}
        fitted: dict[str, dict] = {}

        # ---------------- A1: fit per scale ---------------------------
        for lab in eps_labels:
            at = (epsmul < 0) if lab == "native" else (epsmul == float(lab))
            cal = (at & is_cal)
            sets = {"D4": F["D4"] & cal, "D4+D1": (F["D4"] | F["D1"]) & cal}
            entry = {}
            for tag, m in sets.items():
                if m.sum() < 40:
                    entry[tag] = {"skipped": f"only {int(m.sum())} samples"}
                    continue
                md = m.to(DEV)
                C_Dd, C_dd = Y[md].T @ X[md], X[md].T @ X[md]
                sc = torch.diag(C_dd).mean().item()
                selc = (F["D4"] & cal).to(DEV)
                best, bv = RIDGE_GRID[0], -2.0
                for v in RIDGE_GRID:
                    T = solve(C_Dd, C_dd, v * sc)
                    c = cos(X[selc] @ T.T, Y[selc]).mean().item()
                    if c > bv:
                        best, bv = v, c
                T = solve(C_Dd, C_dd, best * sc)
                fitted[f"T_sec^{tag}@{lab}"] = {"T": T, "eta": best, "n_fit": int(m.sum())}
                entry[tag] = {"eta": best, "n_fit": int(m.sum()), "cal_D4_cos": bv}
            # T_RC anchored on released R, fit on D4+D1 at this scale
            m = ((F["D4"] | F["D1"]) & cal)
            if m.sum() >= 40:
                md = m.to(DEV)
                C_Dd, C_dd = Y[md].T @ X[md], X[md].T @ X[md]
                sc = torch.diag(C_dd).mean().item()
                eta = fitted[f"T_sec^D4+D1@{lab}"]["eta"]
                selc = (F["D4"] & cal).to(DEV)
                bl, bv = LAMBDA_GRID[0], -2.0
                for v in LAMBDA_GRID:
                    T = solve(C_Dd, C_dd, eta * sc, R, v * sc)
                    c = cos(X[selc] @ T.T, Y[selc]).mean().item()
                    if c > bv:
                        bl, bv = v, c
                fitted[f"T_RC@{lab}"] = {"T": solve(C_Dd, C_dd, eta * sc, R, bl * sc),
                                         "lambda": bl}
                entry["T_RC_lambda"] = bl
            L["fits"][lab] = entry

        # shrinkage lambda on calibrate D4 @ EPS_EVAL
        selc = (F["D4"] & is_cal & (epsmul == EPS_EVAL)).to(DEV)
        sh, bv = 0.0, -2.0
        for v in SHRINK_GRID:
            c = cos(X[selc] @ (J + v * I).T, Y[selc]).mean().item()
            if c > bv:
                sh, bv = v, c
        L["shrink_lambda"] = sh

        # ---------------- evaluate at each scale ----------------------
        for lab in eps_labels:
            at = (epsmul < 0) if lab == "native" else (epsmul == float(lab))
            m = F["D4"] & hold & at
            if m.sum() < 10:
                continue
            md = m.to(DEV)
            per = {"I": cos(X[md], Y[md]), "J": cos(X[md] @ J.T, Y[md]),
                   "R": cos(X[md] @ R.T, Y[md]),
                   f"J+{sh:g}I": cos(X[md] @ (J + sh * I).T, Y[md])}
            for tag in ("D4", "D4+D1"):
                k = f"T_sec^{tag}@{lab}"
                if k in fitted:
                    per[f"T_sec^{tag}"] = cos(X[md] @ fitted[k]["T"].T, Y[md])
            if f"T_RC@{lab}" in fitted:
                per["T_RC"] = cos(X[md] @ fitted[f"T_RC@{lab}"]["T"].T, Y[md])
            ml = m & has_loc
            if ml.sum() >= 10 and lab != f"{EPS_REF:g}":
                # restrict every operator to the same rows for a paired comparison
                keep = ml[m]
                per = {k: v[keep.to(DEV)] for k, v in per.items()}
                per["J_loc"] = cos(jloc_pred(ml), Y[ml.to(DEV)])
                sites = base[ml]
            else:
                sites = base[m]
            draws, npr = boot_ci(per, sites)
            s = summarize(draws, per, [("T_sec^D4", "J"), ("T_sec^D4+D1", "J"),
                                       ("T_sec^D4", "J_loc"), ("J_loc", "J"),
                                       ("T_RC", "R"), ("R", "J")])
            s["n_prompts"], s["n_deltas"] = npr, int(next(iter(per.values())).shape[0])
            L.setdefault("eval", {})[lab] = s
            names = ["I", "J", "R", f"J+{sh:g}I", "T_sec^D4", "T_sec^D4+D1", "T_RC", "J_loc"]
            print(f"\n  eps={lab:<7s} n={s['n_deltas']:<5d} prompts={npr}", flush=True)
            print("    " + "".join(f"{n:>13s}" for n in names), flush=True)
            print("    " + "".join(
                f"{s['point'].get(n, float('nan')):>13.4f}" for n in names), flush=True)
            print("    " + "".join(
                (f"[{s['ci'][n][0]:.2f},{s['ci'][n][1]:.2f}]".rjust(13) if n in s["ci"] else
                 "".rjust(13)) for n in names), flush=True)

        # ---------------- cross-eps generalisation --------------------
        cx = {}
        for fl in eps_labels:
            k = f"T_sec^D4+D1@{fl}"
            if k not in fitted:
                continue
            row = {}
            for el in eps_labels:
                at = (epsmul < 0) if el == "native" else (epsmul == float(el))
                m = (F["D4"] & hold & at).to(DEV)
                if m.sum() < 10:
                    continue
                row[el] = cos(X[m] @ fitted[k]["T"].T, Y[m]).mean().item()
            cx[fl] = row
        L["cross_eps"] = cx
        print(f"\n  cross-eps  (fit \\ eval): " + "".join(f"{e:>10s}" for e in eps_labels), flush=True)
        for fl, row in cx.items():
            print(f"    fit@{fl:<8s}" + "".join(
                f"{row.get(e, float('nan')):>10.4f}" for e in eps_labels), flush=True)

        # ---------------- A6: J_loc / J-bar ---------------------------
        m = F["D4"] & hold & (epsmul == EPS_EVAL) & has_loc
        if m.sum() >= 10:
            md = m.to(DEV)
            per = {"J": cos(X[md] @ J.T, Y[md]), "R": cos(X[md] @ R.T, Y[md]),
                   "J_loc": cos(jloc_pred(m), Y[md])}
            draws, npr = boot_ci(per, base[m], seed=7)
            ratio = draws["J_loc"] / draws["J"].clamp_min(1e-9)
            L["A6"] = {"n_prompts": npr, "n_deltas": int(m.sum()),
                       "J": per["J"].mean().item(), "R": per["R"].mean().item(),
                       "J_loc": per["J_loc"].mean().item(),
                       "ratio_point": per["J_loc"].mean().item() / per["J"].mean().item(),
                       "ratio_ci": [torch.quantile(ratio, 0.025).item(),
                                    torch.quantile(ratio, 0.975).item()]}
            print(f"\n  A6 J_loc/J-bar @eps={EPS_EVAL}: J={L['A6']['J']:.4f} "
                  f"J_loc={L['A6']['J_loc']:.4f} ratio={L['A6']['ratio_point']:.2f} "
                  f"95% CI [{L['A6']['ratio_ci'][0]:.2f}, {L['A6']['ratio_ci'][1]:.2f}]",
                  flush=True)

        # save operators at EPS_EVAL for downstream model-side tasks
        save, cfg = {}, {"layer": layer, "target_row": 62, "skip_first": 4,
                         "eps": EPS_EVAL, "target_key": TARGET_KEY, "git": githash,
                         "utc": rep["utc"], "bank": BANK,
                         "n_cal_prompts": len(cal_b), "shrink_lambda": sh}
        for tag in ("T_sec^D4", "T_sec^D4+D1", "T_RC"):
            k = f"{tag}@{EPS_EVAL:g}"
            if k in fitted:
                save[tag] = fitted[k]["T"].cpu()
                cfg[tag] = {kk: vv for kk, vv in fitted[k].items() if kk != "T"}
        torch.save(save, f"{OUT}/{LENS_DIR}_L{layer}_operators.pt")
        with open(f"{OUT}/{LENS_DIR}_L{layer}_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        rep["layers"][str(layer)] = L
        del X, Y, dirs, J, R, I, fitted
        torch.cuda.empty_cache()

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json   ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
