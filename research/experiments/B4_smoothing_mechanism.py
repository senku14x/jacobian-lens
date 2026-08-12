"""B4: WHY does smoothing the local Jacobian beat the exact local Jacobian?

SmoothGrad-J is the largest single effect in this study -- at L31 it scores .862
against J_loc .590 and J_bar .188, and it beats the EXACT local Jacobian by
+0.272 CI-clear. An operator that beats the exact derivative at predicting a
finite effect is not a curiosity; it is the finding with the most headroom. This
script asks which of two mechanisms produces it, because they have opposite
consequences for whether a *lens* can capture it.

  H_path   -- at eps=0.2 the true effect is the SECANT from h to h+delta, not
              the tangent at h. Isotropic smoothing partially averages J along
              the path, so it approximates the mean-value operator. This
              advantage is delta-dependent and NO fixed matrix can capture it.

  H_denoise-- J(h) at the exact operating point is atypically ill-conditioned
              (saturated SiLU gates, RMSNorm geometry), and averaging over ANY
              small ball regularises it. This advantage is largely
              delta-independent and a fitted operator COULD capture it.

Three discriminating measurements, all sharing one machinery:

  A. sigma x eps grid. SG-J's prediction is computed by a linear probe at
     EPS_REF and is therefore linear in the eval scale; cosine is scale
     invariant. So ONE SG-J(sigma) computation serves every eval eps, and the
     grid costs 3 sigmas, not 9 cells. H_path predicts the optimal sigma tracks
     eps (average over the path you actually traverse). H_denoise predicts a
     constant sigma* set by an intrinsic noise scale.

  B. parallel vs orthogonal smoothing at fixed sigma. Draw the base
     perturbation u either along +-delta_hat (maximal path averaging) or inside
     the orthogonal complement of delta (no path averaging at all, pure
     conditioning change). H_path predicts orthogonal smoothing loses the gain;
     H_denoise predicts it keeps most of it.

  C. T_IG positive control. The path-integrated operator
     (1/M) sum_m J(h + t_m delta) . delta with t_m = (m+.5)/M is, by the
     fundamental theorem of calculus, EXACTLY the finite effect up to
     discretisation. It should score ~1.0. If it does not, the measurement
     framework is wrong and every other number here is suspect. It also fixes
     the ceiling that any operator is competing against.

Autograd-free throughout: every Jacobian-vector product is a central difference
at EPS_REF, which is the JVP up to O(eps^2).
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
OPS = os.environ.get("EKKO_OPS", "research/outputs/B2_A1_A6")
OUT = os.environ.get("EKKO_OUT", "research/outputs/B4")
LAYERS = [int(x) for x in os.environ.get("EKKO_LAYERS", "16,31,46").split(",")]
SIGMAS = [float(x) for x in os.environ.get("EKKO_SIGMAS", "0.05,0.2,0.5").split(",")]
EPS_EVAL = [float(x) for x in os.environ.get("EKKO_EPS_EVAL", "0.05,0.2,1.0").split(",")]
SIGMA_DECOMP = float(os.environ.get("EKKO_SIGMA_DECOMP", "0.2"))
EPS_REF = 0.01
K = int(os.environ.get("EKKO_K", "8"))
M_IG = int(os.environ.get("EKKO_M_IG", "8"))
N_SITES = int(os.environ.get("EKKO_SITES", "60"))
BATCH = int(os.environ.get("EKKO_BATCH", "48"))
N_BOOT = 2000
DEV = "cuda:0"


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def boot(per, groups, pairs, seed=5):
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
    out = {"n_prompts": len(us)}
    for k, v in per.items():
        out[k] = {"mean": v.mean().item(),
                  "ci": [torch.quantile(D[k], .025).item(),
                         torch.quantile(D[k], .975).item()]}
    for a, b in pairs:
        if a in D and b in D:
            d = D[a] - D[b]
            lo, hi = torch.quantile(d, .025).item(), torch.quantile(d, .975).item()
            out[f"{a}-{b}"] = {"delta": per[a].mean().item() - per[b].mean().item(),
                               "lo": lo, "hi": hi, "ci_clear": bool(lo > 0 or hi < 0)}
    return out


@torch.no_grad()
def jvp_at(model, ctx, h, layer, target, pos, U, D, eps_probe):
    """Central-difference J(h+u_i) . d_i for matched rows of U and D.

    U [n,d] base-point offsets, D [n,d] unit probe directions. Builds a batch
    whose i-th element sits at h+U_i and is probed with +-eps_probe*D_i, so each
    row can have its OWN base point -- which the parallel/orthogonal decomposition
    requires and the shared-u version of SmoothGrad-J cannot express.
    """
    n = U.shape[0]
    out = torch.zeros(n, model.d_model, dtype=torch.float32, device=DEV)
    half = max(BATCH // 2, 1)
    for s in range(0, n, half):
        u, dd = U[s:s + half], D[s:s + half]
        m = u.shape[0]
        hb = h.expand(2 * m, -1, -1).clone()
        hb[:m, pos] += (u + eps_probe * dd).to(hb.dtype)
        hb[m:, pos] += (u - eps_probe * dd).to(hb.dtype)
        Fb = H.forward_from(model, hb, layer, ctx, target=target)
        out[s:s + m] = (Fb[:m, pos:].float().sum(1) - Fb[m:, pos:].float().sum(1)) / 2
    return out / eps_probe          # -> J . d_hat  (per unit direction)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    jl = H.load_released_lens(LENS_DIR, "j")
    rl = H.load_released_lens(LENS_DIR, "r")
    target = jl.target_layer
    model, hf, tok = H.load_model(MODEL)
    print(f"{model!r}  sigmas={SIGMAS} eps={EPS_EVAL} K={K} sites={N_SITES}"
          f"  ({time.time()-t0:.0f}s)", flush=True)

    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git,
           "sigmas": SIGMAS, "eps_eval": EPS_EVAL, "eps_ref": EPS_REF, "K": K,
           "M_IG": M_IG, "sigma_decomp": SIGMA_DECOMP, "layers": {}}

    for layer in LAYERS:
        print(f"\n{'='*78}\nLAYER {layer}   ({time.time()-t0:.0f}s)", flush=True)
        pack = torch.load(f"{BANK}/{LENS_DIR}_L{layer}.pt", weights_only=False)
        assert "base_ids" in pack, "bank lacks base_ids -- run B2_patch_bank_bases.py"
        meta, base_ids = pack["meta"], pack["base_ids"]
        d = pack["dir"].shape[1]
        dirs = pack["dir"].to(DEV, torch.float32)
        med = pack["median_h_norm"]
        base = torch.tensor([m["base"] for m in meta])
        epsmul = torch.tensor([m["eps"] for m in meta])
        fam = [m["family"] for m in meta]

        ub = sorted(set(base.tolist()))
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(ub), generator=g)
        cal = {ub[i] for i in perm[: len(ub) // 2].tolist()}
        hold = torch.tensor([b.item() not in cal for b in base])

        J = jl.jacobians[layer].to(DEV, torch.float32)
        R = rl.jacobians[layer].to(DEV, torch.float32)
        ops_pt = torch.load(f"{OPS}/{LENS_DIR}_L{layer}_operators.pt", weights_only=True)
        T_sec = ops_pt["T_sec^D4+D1"].to(DEV)

        # ---- pick held-out D4 sites; one row-set shared by every condition ----
        isD4 = torch.tensor([f == "D4" for f in fam])
        anchor = (isD4 & hold & (epsmul == EPS_EVAL[0])).nonzero(as_tuple=True)[0]
        groups: dict[tuple, list] = {}
        for i in anchor.tolist():
            groups.setdefault((meta[i]["base"], meta[i]["pos"]), []).append(i)
        gk = sorted(groups)
        gg = torch.Generator().manual_seed(7)
        gk = [gk[i] for i in torch.randperm(len(gk), generator=gg)[:N_SITES].tolist()]
        # index every (base,pos,tag) at every eval eps, so all conditions and all
        # eps score EXACTLY the same underlying directions
        bykey = {(m["base"], m["pos"], m["tag"], round(m["eps"], 6)): i
                 for i, m in enumerate(meta) if m["family"] == "D4"}
        rows, sites = [], []
        for (bi, pos) in gk:
            tags = [meta[i]["tag"] for i in groups[(bi, pos)]]
            keep = [t for t in tags
                    if all((bi, pos, t, round(e, 6)) in bykey for e in EPS_EVAL)]
            if not keep:
                continue
            sites.append((bi, pos, keep))
            rows.extend((bi, pos, t) for t in keep)
        n = len(rows)
        print(f"  {len(sites)} sites, {n} directions, all present at every eps",
              flush=True)

        ridx = {e: torch.tensor([bykey[(b_, p_, t_, round(e, 6))] for b_, p_, t_ in rows])
                for e in EPS_EVAL}
        Dhat = dirs[ridx[EPS_EVAL[0]].to(DEV)]           # unit dirs, eps-independent
        gsrc = torch.tensor([b_ for b_, _, _ in rows])   # bootstrap over base prompts

        # ---- run every condition site by site (one capture per site) ---------
        conds = ([(f"SG_iso@{s:g}", s, "iso") for s in SIGMAS]
                 + [(f"SG_par@{SIGMA_DECOMP:g}", SIGMA_DECOMP, "par"),
                    (f"SG_orth@{SIGMA_DECOMP:g}", SIGMA_DECOMP, "orth"),
                    ("J_loc", 0.0, "point")])
        REF_ISO = f"SG_iso@{SIGMA_DECOMP:g}"
        PAR, ORTH = f"SG_par@{SIGMA_DECOMP:g}", f"SG_orth@{SIGMA_DECOMP:g}"
        pred = {c[0]: torch.zeros(n, d, device=DEV) for c in conds}
        pred_ig = {e: torch.zeros(n, d, device=DEV) for e in EPS_EVAL}
        gsm = torch.Generator(device=DEV).manual_seed(11)
        off, done = 0, 0
        for (bi, pos, keep) in sites:
            m = len(keep)
            sl = slice(off, off + m)
            Dh = Dhat[sl]
            ctx, acts = H.capture(
                model, torch.tensor([base_ids[bi]], device=model.input_device))
            h = acts[layer]

            for name, sig, mode in conds:
                if mode == "point":
                    pred[name][sl] = jvp_at(model, ctx, h, layer, target, pos,
                                            torch.zeros(m, d, device=DEV), Dh, EPS_REF * med)
                    continue
                # build all K draws as one [m*K, d] block so the forward pass is
                # batched over draws as well as directions
                if mode == "iso":
                    u = torch.randn(K, d, device=DEV, generator=gsm)
                    u = u / u.norm(dim=-1, keepdim=True)
                    U = u[:, None, :].expand(K, m, d).reshape(K * m, d) * (sig * med)
                elif mode == "par":
                    s_ = torch.randint(2, (K * m, 1), device=DEV,
                                       generator=gsm) * 2.0 - 1.0
                    U = Dh.repeat(K, 1) * s_ * (sig * med)
                else:  # orth: isotropic, then projected off each row's own direction
                    Dr = Dh.repeat(K, 1)
                    u = torch.randn(K * m, d, device=DEV, generator=gsm)
                    u = u - (u * Dr).sum(-1, keepdim=True) * Dr
                    U = u / u.norm(dim=-1, keepdim=True) * (sig * med)
                e_ = jvp_at(model, ctx, h, layer, target, pos,
                            U, Dh.repeat(K, 1), EPS_REF * med)
                pred[name][sl] = e_.view(K, m, d).mean(0)

            # path-integrated control, per eval eps (its path length depends on eps)
            for e in EPS_EVAL:
                mag = (pack["eps"][ridx[e]][sl.start:sl.stop]).to(DEV, torch.float32)
                # The target is the ANTITHETIC effect Delta_odd = [F(h+d)-F(h-d)]/2,
                # and [F(h+d)-F(h-d)]/2 = d . mean_{t in [-1,1]} J(h+t d). The one-
                # sided integral over t in [0,1] equals Delta_one instead and is the
                # wrong identity for this target -- it scored .84 where the control
                # must read ~1.0. Sample t symmetrically on [-1,1].
                t = -1.0 + (torch.arange(M_IG, device=DEV) + 0.5) * 2.0 / M_IG
                U = (Dh[None] * (t[:, None] * mag[None])[..., None]).reshape(M_IG * m, d)
                e_ = jvp_at(model, ctx, h, layer, target, pos,
                            U, Dh.repeat(M_IG, 1), EPS_REF * med)
                pred_ig[e][sl] = e_.view(M_IG, m, d).mean(0) * mag[:, None]

            off += m
            done += 1
            if done % 10 == 0:
                print(f"    {done}/{len(sites)} sites  ({time.time()-t0:.0f}s)", flush=True)

        # ---- score every condition at every eval eps -------------------------
        Lrep = {"n_sites": len(sites), "n_dirs": n, "grid": {}}
        for e in EPS_EVAL:
            ii = ridx[e].to(DEV)
            Ytrue = pack["d_odd_sum"][ridx[e]].to(DEV, torch.float32)
            X = dirs[ii] * pack["eps"][ridx[e]].to(DEV, torch.float32)[:, None]
            per = {"J": cos(X @ J.T, Ytrue), "R": cos(X @ R.T, Ytrue),
                   "T_sec": cos(X @ T_sec.T, Ytrue), "T_IG": cos(pred_ig[e], Ytrue)}
            for name, _, _ in conds:
                per[name] = cos(pred[name], Ytrue)   # scale-invariant, one fit per sigma
            pairs = ([(f"SG_iso@{s:g}", "J_loc") for s in SIGMAS]
                     + [(ORTH, "J_loc"), (PAR, "J_loc"), (ORTH, REF_ISO), (PAR, REF_ISO),
                        ("T_IG", REF_ISO), ("T_IG", "J_loc"), ("J_loc", "J")])
            st = boot(per, gsrc, pairs)
            # which smoothing radius wins at this eval scale -- the sigma*(eps)
            # trajectory is what separates H_path from H_denoise
            st["sigma_star"] = max(SIGMAS, key=lambda s: st[f"SG_iso@{s:g}"]["mean"])
            Lrep["grid"][f"{e:g}"] = st
            print(f"\n  --- eval eps={e:g}  (n={n} dirs / {st['n_prompts']} prompts) "
                  f"| sigma*={st['sigma_star']:g} ---", flush=True)
            for k in (["J", "R", "T_sec", "J_loc"]
                      + [f"SG_iso@{s:g}" for s in SIGMAS] + [PAR, ORTH, "T_IG"]):
                v = st[k]
                print(f"     {k:<14s} {v['mean']:.4f}  CI[{v['ci'][0]:.3f},{v['ci'][1]:.3f}]",
                      flush=True)
            for k, v in st.items():
                if isinstance(v, dict) and "ci_clear" in v and v["ci_clear"]:
                    print(f"       {k:<20s} {v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] *",
                          flush=True)
        rep["layers"][str(layer)] = Lrep
        del dirs, pred, pred_ig
        torch.cuda.empty_cache()

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
