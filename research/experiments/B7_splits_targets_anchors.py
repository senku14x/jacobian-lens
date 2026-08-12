"""B7: offline re-analysis addressing three review findings at once.

All of this runs on the stored bank -- no model, no fitting loop.

(1) SPLIT LEAKAGE. The bank splits calibrate/held-out by base prompt. A base is
    keyed by its own token ids, so the pair (a->b) and the pair (b->a) live on
    DIFFERENT bases and can land on opposite sides of the split -- and their
    deltas are exactly negatives of each other. Direction-level leakage inflates
    every in-family number. Same for two arguments filled into one template.
    Four nested split levels are evaluated here:
        base            (what the study used)
        unordered_pair  group {a,b} within a template
        template        no template crosses the split
        category        no category crosses the split
    Bootstrap groups are raised to match the split level, since the split level
    is the independent unit.

(2) TARGET ECOLOGY. The study's primary target was the ANTITHETIC effect at a
    normalised scale, Delta_odd at eps = 0.2*median||h||. For a natural
    activation delta, h+delta is a state the model actually reaches; h-delta is
    an extrapolation, and 0.2*median||h|| is not the natural magnitude. The
    ecological target is the one-sided effect at NATIVE magnitude,
    Delta_one = F(h + (h(x')-h(x))) - F(h). Both are reported; native one-sided
    is now primary and normalised antithetic is a diagnostic.

(3) ANCHORING. The study fitted an R-anchored operator but never a J-anchored
    one, despite J being the stronger baseline on the behavioural gate. Note
        A + (C_Dd - A C_dd)(C_dd + lam I)^-1 == (C_Dd + lam A)(C_dd + lam I)^-1
    so the J-anchored operator is the existing anchored solve with anchor=J.
    Also fits the affine blend a*J + b*R + c*I by least squares, which is the
    cheapest baseline that no fitted operator should lose to.

Metrics: cosine (comparable with earlier work) AND relative residual error
||pred - true|| / ||true||, which cosine hides -- an operator can point the right
way with badly wrong magnitude.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
BANK = os.environ.get("EKKO_BANK", "research/outputs/B1_bank")
OUT = os.environ.get("EKKO_OUT", "research/outputs/B7")
LAYERS = [int(x) for x in os.environ.get("EKKO_LAYERS", "16,31,46").split(",")]
LAMBDAS = [float(x) for x in os.environ.get("EKKO_LAMS", "0.03,0.1,0.3,1.0").split(",")]
ETA_REL = 1e-3
N_BOOT = 2000
DEV = "cuda:0"


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def relerr(pred, true):
    return (pred - true).norm(dim=-1) / true.norm(dim=-1).clamp_min(1e-12)


def solve(C_Dd, C_dd, eta_abs, anchor=None, lam=0.0):
    d = C_dd.shape[0]
    A = C_dd.double() + (eta_abs + lam) * torch.eye(d, device=C_dd.device,
                                                    dtype=torch.float64)
    B = C_Dd.double() + (lam * anchor.double() if anchor is not None else 0.0)
    return torch.linalg.solve(A.T, B.T).T.float()


def boot_groups(vals, groups, seed=17):
    g = torch.Generator().manual_seed(seed)
    us = sorted(set(groups.tolist()))
    idx = {u: (groups == u).nonzero(as_tuple=True)[0].to(vals.device) for u in us}
    d = []
    for _ in range(N_BOOT):
        pick = torch.randint(len(us), (len(us),), generator=g).tolist()
        sel = torch.cat([idx[us[i]] for i in pick])
        d.append(vals[sel].mean().item())
    t = torch.tensor(d)
    return vals.mean().item(), torch.quantile(t, .025).item(), torch.quantile(t, .975).item()


def build_base_meta():
    """base index -> (category, template_idx, arg, unordered_pair_key).

    Rebuilt from the deterministic builder, then keyed by the SAME base_ids the
    bank stores, so an index mismatch is impossible rather than merely unlikely.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transformers import AutoTokenizer
    from B1_bank_broad import build_pairs, MAX_BASES, MODEL
    tok = AutoTokenizer.from_pretrained(MODEL)
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    doc = next(r["text"] for r in ds if len(r["text"]) > 2000)
    prefix = tok.encode(doc, add_special_tokens=False)[:96]
    pairs, _ = build_pairs(tok, prefix)
    by: dict[tuple, list] = {}
    for p in pairs:
        by.setdefault(tuple(p["ids_a"]), []).append(p)
    keys = list(by)[:MAX_BASES]
    info = {}
    for i, k in enumerate(keys):
        p0 = by[k][0]
        info[i] = {"cat": p0["cat"], "func": p0["func"], "arg": p0["arg"],
                   "ids": list(k)}
    return info, by, keys


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ekko import harness as H
    jl = H.load_released_lens(LENS_DIR, "j")
    rl = H.load_released_lens(LENS_DIR, "r")
    info, by, keys = build_base_meta()
    print(f"rebuilt metadata for {len(info)} bases  ({time.time()-t0:.0f}s)", flush=True)

    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git,
           "bank": BANK, "lambdas": LAMBDAS, "layers": {}}

    for layer in LAYERS:
        print(f"\n{'='*78}\nLAYER {layer}  ({time.time()-t0:.0f}s)", flush=True)
        pack = torch.load(f"{BANK}/{LENS_DIR}_L{layer}.pt", weights_only=False)
        meta = pack["meta"]
        d = pack["dir"].shape[1]
        dirs = pack["dir"].to(DEV, torch.float32)
        epsv = pack["eps"].to(DEV, torch.float32)
        base = torch.tensor([m["base"] for m in meta])
        epsmul = torch.tensor([m["eps"] for m in meta])
        fam = [m["family"] for m in meta]
        isD4 = torch.tensor([f == "D4" for f in fam])
        J = jl.jacobians[layer].to(DEV, torch.float32)
        R = rl.jacobians[layer].to(DEV, torch.float32)
        X = dirs * epsv[:, None]

        # ---- split keys at four levels -------------------------------------
        def key_for(i, level):
            b = int(base[i])
            ii = info[b]
            if level == "base":
                return b
            if level == "category":
                return ii["cat"]
            if level == "template":
                return (ii["cat"], ii["func"])
            # unordered_pair: {arg, alt} within a template
            return (ii["cat"], ii["func"],
                    tuple(sorted((ii["arg"], str(meta[i]["tag"])))))

        Ltab = {}
        for level in ("base", "unordered_pair", "template", "category"):
            keys_all = [key_for(i, level) for i in range(len(meta))]
            uniq = sorted({k for k in keys_all}, key=str)
            g = torch.Generator().manual_seed(0)
            perm = torch.randperm(len(uniq), generator=g)
            cal_keys = {uniq[i] for i in perm[: max(1, len(uniq) // 2)].tolist()}
            is_cal = torch.tensor([k in cal_keys for k in keys_all])
            gid = torch.tensor([uniq.index(k) for k in keys_all])

            # ---- two targets ------------------------------------------------
            for tname, ykey, selmask in (
                    ("native_one_sided", "d_one_sum", isD4 & (epsmul < 0)),
                    ("norm0.2_antithetic", "d_odd_sum", isD4 & (epsmul == 0.2))):
                Y = pack[ykey].to(DEV, torch.float32)
                fitm = (selmask & is_cal).to(DEV)
                holdm = (selmask & ~is_cal).to(DEV)
                if int(fitm.sum()) < 50 or int(holdm.sum()) < 50:
                    continue
                Xf, Yf = X[fitm], Y[fitm]
                C_dd = Xf.T @ Xf
                C_Dd = Yf.T @ Xf
                eta = ETA_REL * torch.diagonal(C_dd).sum().item() / d
                tr = torch.diagonal(C_dd).sum().item() / d

                ops = {"I": torch.eye(d, device=DEV), "J": J, "R": R,
                       "T_sec": solve(C_Dd, C_dd, eta)}
                for lam in LAMBDAS:
                    ops[f"T_J@{lam:g}"] = solve(C_Dd, C_dd, eta, anchor=J, lam=lam * tr)
                    ops[f"T_R@{lam:g}"] = solve(C_Dd, C_dd, eta, anchor=R, lam=lam * tr)

                # affine blend aJ + bR + cI, least squares on the calibrate set
                F_ = torch.stack([Xf @ J.T, Xf @ R.T, Xf], -1)          # [n,d,3]
                G = torch.einsum("ndi,ndj->ij", F_, F_).double()
                b_ = torch.einsum("ndi,nd->i", F_, Yf).double()
                w = torch.linalg.solve(G + 1e-6 * torch.eye(3, device=DEV,
                                                            dtype=torch.float64), b_)
                ops["aJ+bR+cI"] = (w[0].item() * J + w[1].item() * R
                                   + w[2].item() * torch.eye(d, device=DEV))

                Xh, Yh = X[holdm], Y[holdm]
                gh = gid[holdm.cpu()]
                row = {"n_fit": int(fitm.sum()), "n_hold": int(holdm.sum()),
                       "n_groups_hold": len(set(gh.tolist())),
                       "blend_w": [float(x) for x in w], "ops": {}}
                for nm, T in ops.items():
                    p = Xh @ T.T
                    c, lo, hi = boot_groups(cos(p, Yh), gh)
                    re_, _, _ = boot_groups(relerr(p, Yh), gh)
                    row["ops"][nm] = {"cos": c, "ci": [lo, hi], "rel_err": re_}
                Ltab[(level, tname)] = row

                best_J = max((k for k in ops if k.startswith("T_J@")),
                             key=lambda k: row["ops"][k]["cos"])
                print(f"  [{level:<15s}|{tname:<19s}] groups_hold="
                      f"{row['n_groups_hold']:<4d} n={row['n_hold']:<5d}  "
                      f"J={row['ops']['J']['cos']:.3f} R={row['ops']['R']['cos']:.3f} "
                      f"T_sec={row['ops']['T_sec']['cos']:.3f} "
                      f"blend={row['ops']['aJ+bR+cI']['cos']:.3f} "
                      f"{best_J}={row['ops'][best_J]['cos']:.3f}  "
                      f"(relerr J={row['ops']['J']['rel_err']:.2f} "
                      f"{best_J}={row['ops'][best_J]['rel_err']:.2f})", flush=True)

        rep["layers"][str(layer)] = {f"{a}|{b}": v for (a, b), v in Ltab.items()}
        del dirs, X, epsv
        torch.cuda.empty_cache()

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
