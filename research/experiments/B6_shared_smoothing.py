"""B6: does the smoothing correction survive averaging into a fixed matrix?

B4 established that SmoothGrad-J beats the exact local Jacobian by +0.17..+0.27,
and that the gain is DENOISING (orthogonal-only smoothing recovers 99-109% of it)
rather than path-averaging -- so it is not structurally barred from a fixed
operator. That does not mean a fixed operator gets it. The obvious next move is
to refit the released estimator at smoothed operating points, which is hours of
backward passes. This script is the cheap precursor that decides whether that run
is worth starting.

The skeptical case for expecting nothing: J_bar already averages over 25 prompts
x many positions, and hidden states of different prompts differ by O(1)*||h||,
which DWARFS a smoothing ball of radius sigma ~ 0.2*||h||. J_bar may already be
smoothed far past anything sigma adds. G-CONDVIABLE showed one input-conditional
correction evaporating under exactly this kind of averaging.

Test, with no fitting at all. Sketch every operator on S = top-r eigenvectors of
the pooled C_dd, so a d x d operator becomes a d x r matrix we can actually hold:

    Y_J(x)  = J_loc(x) . S            r JVPs per site
    Y_SG(x) = E_u[J(h_x+u)] . S       K*r JVPs per site
    C(x)    = Y_SG(x) - Y_J(x)        the smoothing correction at x

Then two questions:

  Q1 (is it shared?)  ||mean_x C(x)||_F^2 / mean_x ||C(x)||_F^2. Near 1 means the
     correction is essentially the same operator at every input and survives
     averaging; near 0 means it is input-specific and cancels.

  Q2 (does it help the FIXED lens?)  Estimate C_bar on calibrate sites only, then
     on HELD-OUT sites score  (J_bar + C_bar).delta  against the true effect and
     compare with J_bar.delta. This is exactly what a smoothed-operating-point
     lens would buy, obtained without fitting one. Also scored against the
     sketched J_bar, so the comparison is not confounded by the sketch itself.

Controls: a random-correction control with C_bar's Frobenius norm, and the
same-site (oracle) correction as an upper bound.
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
BANK = os.environ.get("EKKO_BANK", "research/outputs/B1_bank")
OUT = os.environ.get("EKKO_OUT", "research/outputs/B6")
LAYERS = [int(x) for x in os.environ.get("EKKO_LAYERS", "16,31").split(",")]
R = int(os.environ.get("EKKO_R", "160"))
K = int(os.environ.get("EKKO_K", "4"))
SIGMA = float(os.environ.get("EKKO_SIGMA", "0.2"))
N_SITES = int(os.environ.get("EKKO_SITES", "48"))
EPS_EVAL = 0.2
EPS_REF = 0.01
BATCH = int(os.environ.get("EKKO_BATCH", "48"))
N_BOOT = 2000
DEV = "cuda:0"


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def boot(per, groups, pairs, seed=13):
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
    out = {"n_sites_boot": len(us)}
    for k, v in per.items():
        out[k] = {"mean": v.mean().item(),
                  "ci": [torch.quantile(D[k], .025).item(),
                         torch.quantile(D[k], .975).item()]}
    for a, b in pairs:
        d = D[a] - D[b]
        lo, hi = torch.quantile(d, .025).item(), torch.quantile(d, .975).item()
        out[f"{a}-{b}"] = {"delta": per[a].mean().item() - per[b].mean().item(),
                           "lo": lo, "hi": hi, "ci_clear": bool(lo > 0 or hi < 0)}
    return out


@torch.no_grad()
def jvp_block(model, ctx, h, layer, target, pos, U, D_, eps_probe):
    """J(h+U_i) . D_i for matched rows, central difference. Returns [n,d]."""
    n = U.shape[0]
    out = torch.zeros(n, model.d_model, dtype=torch.float32, device=DEV)
    half = max(BATCH // 2, 1)
    for s in range(0, n, half):
        u, dd = U[s:s + half], D_[s:s + half]
        m = u.shape[0]
        hb = h.expand(2 * m, -1, -1).clone()
        hb[:m, pos] += (u + eps_probe * dd).to(hb.dtype)
        hb[m:, pos] += (u - eps_probe * dd).to(hb.dtype)
        Fb = H.forward_from(model, hb, layer, ctx, target=target)
        out[s:s + m] = (Fb[:m, pos:].float().sum(1) - Fb[m:, pos:].float().sum(1)) / 2
    return out / eps_probe


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    jl = H.load_released_lens(LENS_DIR, "j")
    target = jl.target_layer
    model, hf, tok = H.load_model(MODEL)
    print(f"{model!r}  r={R} K={K} sigma={SIGMA} sites={N_SITES}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git,
           "r": R, "K": K, "sigma": SIGMA, "eps_eval": EPS_EVAL, "layers": {}}

    for layer in LAYERS:
        print(f"\n{'='*78}\nLAYER {layer}   ({time.time()-t0:.0f}s)", flush=True)
        pack = torch.load(f"{BANK}/{LENS_DIR}_L{layer}.pt", weights_only=False)
        meta, base_ids = pack["meta"], pack["base_ids"]
        d = pack["dir"].shape[1]
        dirs = pack["dir"].to(DEV, torch.float32)
        Y = pack["d_odd_sum"].to(DEV, torch.float32)
        med = pack["median_h_norm"]
        base = torch.tensor([m["base"] for m in meta])
        epsmul = torch.tensor([m["eps"] for m in meta])
        fam = [m["family"] for m in meta]
        J = jl.jacobians[layer].to(DEV, torch.float32)

        # sketch basis from the pooled fit set (same convention as C2/A5)
        fitm = (torch.tensor([f in ("D4", "D1") for f in fam])
                & (epsmul == EPS_EVAL)).to(DEV)
        Xf = dirs[fitm] * pack["eps"].to(DEV, torch.float32)[fitm][:, None]
        w, V = torch.linalg.eigh((Xf.T @ Xf).double())
        S = V.flip(1)[:, :R].float().contiguous()          # [d, r]

        # sites: held-out D4 rows at the eval scale
        isD4 = torch.tensor([f == "D4" for f in fam])
        sel = (isD4 & (epsmul == EPS_EVAL)).nonzero(as_tuple=True)[0]
        groups: dict[tuple, list] = {}
        for i in sel.tolist():
            groups.setdefault((meta[i]["base"], meta[i]["pos"]), []).append(i)
        gk = sorted(groups)
        gg = torch.Generator().manual_seed(3)
        gk = [gk[i] for i in torch.randperm(len(gk), generator=gg)[:N_SITES].tolist()]
        n_cal = len(gk) // 2
        cal_sites, hold_sites = gk[:n_cal], gk[n_cal:]
        print(f"  sites: {len(cal_sites)} calibrate / {len(hold_sites)} held out, "
              f"r={R}", flush=True)

        gsm = torch.Generator(device=DEV).manual_seed(23)
        YJ, YSG, order = {}, {}, cal_sites + hold_sites
        for si, (bi, pos) in enumerate(order):
            ctx, acts = H.capture(
                model, torch.tensor([base_ids[bi]], device=model.input_device))
            h = acts[layer]
            Z = torch.zeros(R, d, device=DEV)
            YJ[(bi, pos)] = jvp_block(model, ctx, h, layer, target, pos,
                                      Z, S.T, EPS_REF * med)
            acc = torch.zeros(R, d, device=DEV)
            for _ in range(K):
                u = torch.randn(d, device=DEV, generator=gsm)
                U = (u / u.norm()).expand(R, d) * (SIGMA * med)
                acc += jvp_block(model, ctx, h, layer, target, pos,
                                 U, S.T, EPS_REF * med)
            YSG[(bi, pos)] = acc / K
            if (si + 1) % 8 == 0:
                print(f"    sketched {si+1}/{len(order)} sites "
                      f"({time.time()-t0:.0f}s)", flush=True)

        # ---- Q1: is the correction shared across inputs? -------------------
        Cs = torch.stack([YSG[k] - YJ[k] for k in order])          # [n, r, d]
        Cbar_all = Cs.mean(0)
        shared = (Cbar_all.pow(2).sum() / Cs.pow(2).sum(dim=(1, 2)).mean()).item()
        # per-site cosine against the mean correction, a distributional view
        flat = Cs.reshape(len(order), -1)
        cos_to_mean = cos(flat, Cbar_all.reshape(1, -1).expand_as(flat))
        print(f"\n  Q1 shared-energy ||mean C||²/mean||C||² = {shared:.4f}", flush=True)
        print(f"     per-site cos to mean correction: median {cos_to_mean.median():.3f} "
              f"[{cos_to_mean.quantile(.1):.3f}, {cos_to_mean.quantile(.9):.3f}]",
              flush=True)

        # ---- Q2: does the MEAN correction help the FIXED lens? -------------
        Cbar = torch.stack([YSG[k] - YJ[k] for k in cal_sites]).mean(0)   # [r, d]
        gr = torch.Generator(device=DEV).manual_seed(31)
        Crnd = torch.randn(R, d, device=DEV, generator=gr)
        Crnd *= Cbar.norm() / Crnd.norm()

        rows = [i for (bi, pos) in hold_sites for i in groups[(bi, pos)]]
        ridx = torch.tensor(rows)
        Dh = dirs[ridx.to(DEV)]
        Xd = Dh * pack["eps"].to(DEV, torch.float32)[ridx.to(DEV)][:, None]
        Ytrue = Y[ridx.to(DEV)]
        site_of = torch.tensor([hold_sites.index((meta[i]["base"], meta[i]["pos"]))
                                for i in rows])
        coef = Xd @ S                                          # [n, r] sketch coords

        def apply_corr(Cm):
            return coef @ Cm                                   # [n, d]

        per = {
            "J_bar": cos(Xd @ J.T, Ytrue),
            "J_bar_sketch": cos((coef @ (S.T @ J.T)), Ytrue),
            "J_bar+Cbar": cos((coef @ (S.T @ J.T)) + apply_corr(Cbar), Ytrue),
            "J_bar+Crandom": cos((coef @ (S.T @ J.T)) + apply_corr(Crnd), Ytrue),
            "J_loc": cos(torch.stack([coef[j] @ YJ[hold_sites[site_of[j]]]
                                      for j in range(len(rows))]), Ytrue),
            "SG_J_oracle": cos(torch.stack([coef[j] @ YSG[hold_sites[site_of[j]]]
                                            for j in range(len(rows))]), Ytrue),
        }
        per["J_loc+Cbar"] = cos(
            torch.stack([coef[j] @ (YJ[hold_sites[site_of[j]]] + Cbar)
                         for j in range(len(rows))]), Ytrue)

        st = boot(per, site_of, [("J_bar+Cbar", "J_bar_sketch"),
                                 ("J_bar+Crandom", "J_bar_sketch"),
                                 ("J_loc+Cbar", "J_loc"),
                                 ("SG_J_oracle", "J_loc")])
        st["shared_energy"] = shared
        st["cos_to_mean_median"] = cos_to_mean.median().item()
        st["n_dirs"] = len(rows)
        rep["layers"][str(layer)] = st

        print(f"\n  Q2 held-out sites, n={len(rows)} dirs / {len(hold_sites)} sites",
              flush=True)
        for k in ("J_bar", "J_bar_sketch", "J_bar+Crandom", "J_bar+Cbar",
                  "J_loc", "J_loc+Cbar", "SG_J_oracle"):
            v = st[k]
            print(f"     {k:<16s} {v['mean']:.4f}  CI[{v['ci'][0]:.3f},{v['ci'][1]:.3f}]",
                  flush=True)
        for k, v in st.items():
            if isinstance(v, dict) and v.get("ci_clear"):
                print(f"       {k:<26s} {v['delta']:+.4f} "
                      f"[{v['lo']:+.4f},{v['hi']:+.4f}] *", flush=True)

        del dirs, Y, Cs, YJ, YSG
        torch.cuda.empty_cache()

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
