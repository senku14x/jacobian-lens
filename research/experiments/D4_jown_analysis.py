"""D4: redo the study's headline comparisons with J_own, and get a noise floor.

Three things this settles that nothing before it could.

1. THE NOISE FLOOR, in the units the study actually reports. halfA and halfB are
   two fits of the SAME estimator on disjoint halves of the same corpus, so any
   difference between them is pure estimation noise. Their paired difference in
   cos(T.delta, true) is the threshold a "real" operator difference must clear.
   Several margins in artifacts 007 and 008 were ~0.005 and were reported as
   CI-clear; a CI can be tight and still sit under the noise floor of the
   estimator itself, because bootstrapping over prompts does not resample the
   FIT.

2. CONTEXT AVERAGING, isolated at last. Every "J_loc / J_bar = 2-4x" number in
   007 compared an input-specific estimator against the RELEASED J, bundling
   context averaging with corpus mismatch, convention mismatch and estimation
   noise. J_own is fitted at the released convention on our own corpus, so
   J_loc / J_own is the context-averaging cost alone, and J_own / J_released is
   everything else.

3. WHETHER ANY 007/008 CONCLUSION MOVES when the baseline is controlled. The
   secant, the blend and the spectral operators are all re-anchored on J_own and
   re-evaluated under the leak-repaired template-disjoint split.

No model load: everything reads the stored bank and the fitted matrices.
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
JOWN = os.environ.get("EKKO_JOWN", "research/outputs/D2")
OUT = os.environ.get("EKKO_OUT", "research/outputs/D4")
LAYER = int(os.environ.get("EKKO_LAYER", "31"))
N_BOOT = 2000
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def relerr(p, y):
    return (p - y).norm(dim=-1) / y.norm(dim=-1).clamp_min(1e-12)


def boot(per, groups, pairs, seed=37):
    g = torch.Generator().manual_seed(seed)
    us = sorted(set(groups.tolist()))
    idx = {u: (groups == u).nonzero(as_tuple=True)[0].to(DEV) for u in us}
    draws = {k: [] for k in per}
    for _ in range(N_BOOT):
        pick = torch.randint(len(us), (len(us),), generator=g).tolist()
        sel = torch.cat([idx[us[i]] for i in pick])
        for k, v in per.items():
            draws[k].append(v[sel].mean().item())
    D = {k: torch.tensor(v) for k, v in draws.items()}
    out = {}
    for k, v in per.items():
        out[k] = {"cos": v.mean().item(),
                  "ci": [torch.quantile(D[k], .025).item(),
                         torch.quantile(D[k], .975).item()]}
    for a, b in pairs:
        if a in D and b in D:
            d = D[a] - D[b]
            lo, hi = torch.quantile(d, .025).item(), torch.quantile(d, .975).item()
            out[f"{a}-{b}"] = {"delta": per[a].mean().item() - per[b].mean().item(),
                               "lo": lo, "hi": hi,
                               "ci_clear": bool(lo > 0 or hi < 0)}
    return out


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

    own = {}
    for nm in ("all", "halfA", "halfB"):
        p = f"{JOWN}/{LENS_DIR}_{nm}.pt"
        if os.path.exists(p):
            d = torch.load(p, map_location="cpu", weights_only=True)
            if LAYER in d:
                own[nm] = d[LAYER].to(DEV, torch.float32)
    if "all" not in own:
        print(f"J_own not found in {JOWN}; run D2_fit_jown.py first")
        return
    print(f"loaded J_own variants: {list(own)}  ({time.time()-t0:.0f}s)", flush=True)

    pack = torch.load(f"{BANK}/{LENS_DIR}_L{LAYER}.pt", weights_only=False)
    meta = pack["meta"]
    d = pack["dir"].shape[1]
    X = pack["dir"].to(DEV, torch.float32) * pack["eps"].to(DEV, torch.float32)[:, None]
    base = torch.tensor([m["base"] for m in meta])
    epsmul = torch.tensor([m["eps"] for m in meta])
    isD4 = torch.tensor([m["family"] == "D4" for m in meta])
    J = jl.jacobians[LAYER].to(DEV, torch.float32)
    R = rl.jacobians[LAYER].to(DEV, torch.float32)
    I = torch.eye(d, device=DEV)

    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git,
           "layer": LAYER, "targets": {}}

    # matrix-level distances
    def rf(a, b):
        return float((a - b).norm() / b.norm())
    rep["matrix"] = {
        "J_own_vs_J_rel": rf(own["all"], J),
        "R_rel_vs_J_rel": rf(R, J),
        "J_own_vs_R_rel": rf(own["all"], R),
        "diag_mean": {"J_own": float(torch.diagonal(own["all"]).mean()),
                      "J_rel": float(torch.diagonal(J).mean()),
                      "R_rel": float(torch.diagonal(R).mean())}}
    if "halfA" in own and "halfB" in own:
        rep["matrix"]["twin_fit_floor"] = rf(own["halfA"], own["halfB"])
    print(f"\n  ||J_own - J_rel||/||J_rel||     = {rep['matrix']['J_own_vs_J_rel']:.4f}")
    print(f"  ||R_rel - J_rel||/||J_rel||     = {rep['matrix']['R_rel_vs_J_rel']:.4f}")
    if "twin_fit_floor" in rep["matrix"]:
        print(f"  ||J_A - J_B||/||J_B||  (floor)  = "
              f"{rep['matrix']['twin_fit_floor']:.4f}   <-- estimation noise",
              flush=True)

    for tname, ykey, selm in (("native_one_sided", "d_one_sum", isD4 & (epsmul < 0)),
                              ("norm0.2_antithetic", "d_odd_sum", isD4 & (epsmul == 0.2))):
        Y = pack[ykey].to(DEV, torch.float32)
        # template-disjoint split (the leak-repaired one from 007)
        kk = [f"{info[int(b)]['cat']}/{info[int(b)]['func']}" for b in base]
        uniq = sorted(set(kk))
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(uniq), generator=g)
        cal = {uniq[i] for i in perm[: max(1, len(uniq) // 2)].tolist()}
        is_cal = torch.tensor([k in cal for k in kk])
        hm = (selm & ~is_cal).to(DEV)
        gid = torch.tensor([uniq.index(k) for k in kk])[hm.cpu()]
        Xh, Yh = X[hm], Y[hm]

        # J_loc from the stored eps=0.01 antithetic measurement, same rows
        ref = {(m["base"], m["pos"], m["tag"]): i for i, m in enumerate(meta)
               if m["family"] == "D4" and abs(m["eps"] - 0.01) < 1e-12}
        rows = hm.nonzero(as_tuple=True)[0].tolist()
        src = torch.tensor([ref.get((meta[i]["base"], meta[i]["pos"], meta[i]["tag"]), -1)
                            for i in rows])
        per = {"I": cos(Xh, Yh), "J_rel": cos(Xh @ J.T, Yh), "R_rel": cos(Xh @ R.T, Yh)}
        for nm, M in own.items():
            per[f"J_own_{nm}" if nm != "all" else "J_own"] = cos(Xh @ M.T, Yh)
        if (src >= 0).all():
            Yref = pack["d_odd_sum"].to(DEV, torch.float32)[src.to(DEV)]
            eref = pack["eps"].to(DEV, torch.float32)[src.to(DEV)]
            eh = pack["eps"].to(DEV, torch.float32)[hm]
            per["J_loc"] = cos(Yref * (eh / eref)[:, None], Yh)

        pairs = [("J_own", "J_rel"), ("J_own", "R_rel"), ("J_loc", "J_own"),
                 ("J_loc", "J_rel"), ("R_rel", "J_rel")]
        if "J_own_halfA" in per and "J_own_halfB" in per:
            pairs.append(("J_own_halfA", "J_own_halfB"))
        st = boot(per, gid, pairs)
        rep["targets"][tname] = {"n": int(hm.sum()), "n_groups": len(set(gid.tolist())),
                                 "stats": st,
                                 "rel_err": {k: float(relerr(Xh @ (own["all"] if k == "J_own"
                                                                   else (J if k == "J_rel" else R)).T,
                                                             Yh).mean())
                                             for k in ("J_rel", "R_rel", "J_own")}}
        print(f"\n  === {tname}  (n={int(hm.sum())} dirs / "
              f"{len(set(gid.tolist()))} templates, template-disjoint) ===", flush=True)
        for k in ("I", "J_rel", "R_rel", "J_own", "J_own_halfA", "J_own_halfB", "J_loc"):
            if k in st:
                print(f"     {k:<14s} {st[k]['cos']:.4f}  "
                      f"CI[{st[k]['ci'][0]:.3f},{st[k]['ci'][1]:.3f}]", flush=True)
        for k in list(st):
            if isinstance(st[k], dict) and "ci_clear" in st[k]:
                flag = "*" if st[k]["ci_clear"] else " "
                print(f"       {k:<26s} {st[k]['delta']:+.4f} "
                      f"[{st[k]['lo']:+.4f},{st[k]['hi']:+.4f}] {flag}", flush=True)
        if "J_own_halfA-J_own_halfB" in st:
            fl = abs(st["J_own_halfA-J_own_halfB"]["delta"])
            hw = (st["J_own_halfA-J_own_halfB"]["hi"]
                  - st["J_own_halfA-J_own_halfB"]["lo"]) / 2
            rep["targets"][tname]["noise_floor_cos"] = {"abs_delta": fl, "ci_halfwidth": hw}
            print(f"\n     NOISE FLOOR: two fits of the SAME estimator differ by "
                  f"{fl:.4f} in cos (CI halfwidth {hw:.4f}).", flush=True)
            print(f"     Any operator margin below ~{max(fl, hw):.4f} is inside "
                  f"estimation noise, however tight its bootstrap CI.", flush=True)

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
