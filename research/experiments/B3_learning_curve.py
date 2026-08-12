"""B3: does T_sec's advantage converge, and does it transfer?

Broadening the bank 4.2x roughly doubled T_sec's held-out in-family accuracy but
left both span-immune checks (Stein, in-span) unchanged. Two readings survive:

  (a) the fit is still rank-starved -- more data keeps helping on every axis, we
      simply have not reached the plateau; or
  (b) the fit converges on the family it was trained on and nowhere else, i.e.
      T_sec is a distribution-matching object rather than a transport operator.

They differ in the SHAPE of the curves, not in any single number. Fit T_sec on
1/8, 1/4, 1/2, 1 of the calibrate base prompts and track, per fraction:

  * held-out D4 cos          -- in-family, the axis that improved
  * held-out D2 cos          -- cross-family, unprojected
  * captured fraction of held-out D2 energy in span(C_dd) -- identifiability
  * effective rank of C_dd

(b) predicts D4 rising toward a plateau while D2 stays flat EVEN AS captured
fraction rises -- the fit becomes better identified on directions it still
cannot transport to. (a) predicts D2 tracking captured fraction upward.

Subsampling is over BASE PROMPTS, not deltas: deltas within a prompt are not
independent, and subsampling deltas would inflate the effective sample size.
No model load -- reads the stored bank only.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
BANK = os.environ.get("EKKO_BANK", "research/outputs/B1_bank")
OUT = os.environ.get("EKKO_OUT", "research/outputs/B3")
LAYERS = [int(x) for x in os.environ.get("EKKO_LAYERS", "16,31,46").split(",")]
FRACS = [float(x) for x in os.environ.get("EKKO_FRACS", "0.125,0.25,0.5,1.0").split(",")]
EPS_EVAL = 0.2
ETA_REL = float(os.environ.get("EKKO_ETA_REL", "1e-3"))
N_BOOT = 2000
DEV = "cuda:0"


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def boot_ci(vals, groups, seed=3):
    """Paired bootstrap over base prompts (groups), not over deltas."""
    g = torch.Generator().manual_seed(seed)
    us = sorted(set(groups.tolist()))
    idx = {u: (groups == u).nonzero(as_tuple=True)[0] for u in us}
    draws = []
    for _ in range(N_BOOT):
        pick = torch.randint(len(us), (len(us),), generator=g).tolist()
        sel = torch.cat([idx[us[i]] for i in pick])
        draws.append(vals[sel.to(vals.device)].mean().item())
    d = torch.tensor(draws)
    return (vals.mean().item(), torch.quantile(d, .025).item(),
            torch.quantile(d, .975).item())


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bank": BANK,
           "fracs": FRACS, "eps_eval": EPS_EVAL, "eta_rel": ETA_REL, "layers": {}}

    for layer in LAYERS:
        print(f"\n{'='*74}\nLAYER {layer}   ({time.time()-t0:.0f}s)", flush=True)
        pack = torch.load(f"{BANK}/{LENS_DIR}_L{layer}.pt", weights_only=False)
        meta = pack["meta"]
        d = pack["dir"].shape[1]
        dirs = pack["dir"].to(DEV, torch.float32)
        eps = pack["eps"].to(DEV, torch.float32)
        Y = pack["d_odd_sum"].to(DEV, torch.float32)
        X = dirs * eps[:, None]
        base = torch.tensor([m["base"] for m in meta])
        epsmul = torch.tensor([m["eps"] for m in meta])
        fam = [m["family"] for m in meta]

        # same calibrate/hold split as every other script (seed 0 over bases)
        ub = sorted(set(base.tolist()))
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(ub), generator=g)
        cal_bases = [ub[i] for i in perm[: len(ub) // 2].tolist()]
        is_cal = torch.tensor([b.item() in set(cal_bases) for b in base])
        hold = ~is_cal

        at_eps = epsmul == EPS_EVAL
        isD4 = torch.tensor([f == "D4" for f in fam])
        isD1 = torch.tensor([f == "D1" for f in fam])
        isD2 = torch.tensor([f == "D2" for f in fam])
        ev = {"D4": (isD4 & hold & at_eps).to(DEV),
              "D2": (isD2 & hold & at_eps).to(DEV)}

        rows = []
        for frac in FRACS:
            k = max(1, int(round(frac * len(cal_bases))))
            sub = set(cal_bases[:k])           # nested subsets: 1/8 subset of 1/4 ...
            m_cal = torch.tensor([b.item() in sub for b in base])
            fitm = ((isD4 | isD1) & m_cal & at_eps).to(DEV)
            Xf, Yf = X[fitm], Y[fitm]
            C_dd = Xf.T @ Xf
            C_Dd = Yf.T @ Xf
            tr = torch.diagonal(C_dd).sum().item() / d
            A = C_dd.double() + ETA_REL * tr * torch.eye(d, device=DEV, dtype=torch.float64)
            T = torch.linalg.solve(A.T, C_Dd.double().T).T.float()

            evals, V = torch.linalg.eigh(C_dd.double())
            evals = evals.flip(0).clamp_min(0)
            eff = int((evals > 1e-6 * evals[0]).sum())
            Vtop = V.flip(1)[:, :eff].float()

            row = {"frac": frac, "n_bases": k, "n_fit_deltas": int(fitm.sum()),
                   "eff_rank": eff}
            for nm, m in ev.items():
                pred = X[m] @ T.T
                c = cos(pred, Y[m])
                mu, lo, hi = boot_ci(c, base[m.cpu()])
                row[f"cos_{nm}"] = mu
                row[f"ci_{nm}"] = [lo, hi]
                # identifiability: fraction of each held-out direction's energy
                # inside the identified subspace
                D = dirs[m]
                cap = ((D @ Vtop).norm(dim=1) / D.norm(dim=1)).mean().item()
                row[f"captured_{nm}"] = cap
            rows.append(row)
            print(f"  frac={frac:<6g} bases={k:<4d} deltas={row['n_fit_deltas']:<6d} "
                  f"eff_rank={eff:<5d} | D4 cos={row['cos_D4']:.4f} "
                  f"cap={row['captured_D4']:.3f} | D2 cos={row['cos_D2']:.4f} "
                  f"cap={row['captured_D2']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

        # shape summary: last-half slope on each axis, normalised by its own level
        def slope(key):
            a, b = rows[len(rows) // 2][key], rows[-1][key]
            return (b - a) / max(abs(a), 1e-9)
        rep["layers"][str(layer)] = {
            "rows": rows,
            "rel_gain_second_half": {k: slope(k) for k in
                                     ("cos_D4", "cos_D2", "captured_D2", "eff_rank")}}
        s = rep["layers"][str(layer)]["rel_gain_second_half"]
        print(f"  relative gain over the last doubling: D4 {s['cos_D4']:+.1%}  "
              f"D2 {s['cos_D2']:+.1%}  captured_D2 {s['captured_D2']:+.1%}  "
              f"eff_rank {s['eff_rank']:+.1%}", flush=True)
        del dirs, eps, Y, X
        torch.cuda.empty_cache()

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
