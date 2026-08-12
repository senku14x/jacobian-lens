"""A2 (SmoothGrad-J / Stein) + A5 (in-span transfer) + Stage-0 re-verification.

One 27B load serves both. Autograd-free throughout: every Jacobian action is a
central finite difference through forward_from.

A2 -- SmoothGrad-J. The Stein control. SG-J . d  =  mean over K isotropic base
perturbations u_k of [J_loc at (h + sigma u_k)] . d, each term a central
difference at eps_ref extrapolated to the test scale. If T_sec ~= SG-J then the
secant is "just a smoothed Jacobian" and is not a new object. Reported with
cos(T_sec . d, SG-J . d) directly, not only via their separate scores.

A5 -- in-span transfer. T_sec is identified only on span(C_dd); a held-out D1/D2
direction lying mostly outside it is mispredicted BY CONSTRUCTION. So we project
held-out directions onto the top-r eigenvectors of C_dd, renormalise to the same
magnitude, MEASURE THE TRUE EFFECT OF THE PROJECTED DIRECTION with a fresh
forward_from pass, and score operators there. Only this number is admissible as
transfer evidence.
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
OPS = os.environ.get("EKKO_OPS", "research/outputs/A1_A6")
OUT = os.environ.get("EKKO_OUT", "research/outputs/A2_A5")
LAYERS = [int(x) for x in os.environ.get("EKKO_LAYERS", "16,31,46").split(",")]
DEV = "cuda:0"
EPS_EVAL, EPS_REF = 0.2, 0.01
K_SMOOTH = int(os.environ.get("EKKO_K", "8"))
BATCH = int(os.environ.get("EKKO_BATCH", "32"))
N_INSPAN = int(os.environ.get("EKKO_N_INSPAN", "160"))
RANKS = [int(x) for x in os.environ.get("EKKO_RANKS", "117,313,615").split(",")]
N_SG_SITES = int(os.environ.get("EKKO_SG_SITES", "0"))  # 0 = all held-out sites
N_BOOT = 2000


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def boot(per: dict, sites, pairs, seed=3):
    us = sorted(set(sites.tolist()))
    idx = [(sites == s).nonzero(as_tuple=True)[0] for s in us]
    g = torch.Generator().manual_seed(seed)
    dr = {k: [] for k in per}
    dev = next(iter(per.values())).device
    for _ in range(N_BOOT):
        pick = torch.randint(len(us), (len(us),), generator=g).tolist()
        sel = torch.cat([idx[i] for i in pick]).to(dev)
        for k, v in per.items():
            dr[k].append(v[sel].mean().item())
    dr = {k: torch.tensor(v) for k, v in dr.items()}
    out = {"n_prompts": len(us), "point": {k: v.mean().item() for k, v in per.items()},
           "ci": {k: [torch.quantile(dr[k], .025).item(), torch.quantile(dr[k], .975).item()]
                  for k in per}, "paired": {}}
    for a, b in pairs:
        if a in dr and b in dr:
            dl = dr[a] - dr[b]
            lo, hi = torch.quantile(dl, .025).item(), torch.quantile(dl, .975).item()
            out["paired"][f"{a}-{b}"] = {
                "delta": per[a].mean().item() - per[b].mean().item(), "lo": lo, "hi": hi,
                "p_gt0": (dl > 0).float().mean().item(), "ci_clear": bool(lo > 0 or hi < 0),
                "half_width": (hi - lo) / 2}
    return out


@torch.no_grad()
def effects(model, ctx, h, layer, target, pos, deltas, batch):
    """Central-difference effects (summed over current-and-future) for [n,d] deltas."""
    n = deltas.shape[0]
    out = torch.zeros(n, model.d_model, dtype=torch.float32, device=DEV)
    half = max(batch // 2, 1)
    for s in range(0, n, half):
        ch = deltas[s:s + half]
        m = ch.shape[0]
        hb = h.expand(2 * m, -1, -1).clone()
        hb[:m, pos] += ch.to(hb.dtype)
        hb[m:, pos] -= ch.to(hb.dtype)
        Fb = H.forward_from(model, hb, layer, ctx, target=target)
        out[s:s + m] = ((Fb[:m, pos:].float().sum(1) - Fb[m:, pos:].float().sum(1)) / 2)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    man = json.load(open(f"{BANK}/{LENS_DIR}_manifest.json"))
    jl = H.load_released_lens(LENS_DIR, "j")
    rl = H.load_released_lens(LENS_DIR, "r")
    target = jl.target_layer
    model, hf, tok = H.load_model(MODEL)
    print(f"{model!r}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- Stage-0 invariants (HARD-STOP if regressed) -------------------
    ids = model.encode("The capital of France is the city of", max_length=128)
    ctx0, acts0 = H.capture(model, ids)
    with torch.no_grad():
        a = model.unembed(acts0[model.n_layers - 1][0]).float()
        b = hf(input_ids=ids, use_cache=False).logits[0].float()
    id_err = (a - b).abs().max().item()
    rt_err = max((H.forward_from(model, acts0[l], l, ctx0, target=target).float()
                  - acts0[target].float()).abs().max().item() for l in LAYERS)
    print(f"STAGE-0 recheck: identity={id_err:.3e}  round-trip={rt_err:.3e}", flush=True)
    if id_err != 0.0 or rt_err != 0.0:
        raise SystemExit("HARD-STOP: Stage-0 invariant regressed")

    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git,
           "stage0": {"identity_max_err": id_err, "roundtrip_max_err": rt_err},
           "K_smooth": K_SMOOTH, "eps_eval": EPS_EVAL, "eps_ref": EPS_REF, "layers": {}}

    for layer in LAYERS:
        print(f"\n{'='*78}\nLAYER {layer}   ({time.time()-t0:.0f}s)", flush=True)
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
        ub = sorted(set(base.tolist()))
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(ub), generator=g)
        cal_b = {ub[i] for i in perm[: len(ub) // 2].tolist()}
        is_cal = torch.tensor([b_.item() in cal_b for b_ in base])
        hold = ~is_cal
        med = pack["median_h_norm"]

        ops_pt = torch.load(f"{OPS}/{LENS_DIR}_L{layer}_operators.pt", weights_only=True)
        T_sec = ops_pt["T_sec^D4+D1"].to(DEV)
        T_secD4 = ops_pt["T_sec^D4"].to(DEV)
        T_RC = ops_pt["T_RC"].to(DEV)
        J = jl.jacobians[layer].to(DEV, torch.float32)
        R = rl.jacobians[layer].to(DEV, torch.float32)

        # C_dd eigenbasis from the calibrate fit set at EPS_EVAL
        fitm = ((torch.tensor([f in ("D4", "D1") for f in fam])) & is_cal
                & (epsmul == EPS_EVAL)).to(DEV)
        C_dd = X[fitm].T @ X[fitm]
        w, V = torch.linalg.eigh(C_dd.double())
        V = V.float()
        L = {"median_h_norm": med, "A2": {}, "A5": {}}

        # ---- rebuild prompts / contexts for held-out sites -------------
        # base index -> token ids: read from the pack itself. Banks are made
        # self-describing by B2_patch_bank_bases.py (verified by re-measuring
        # stored deltas); re-deriving prompts from builder code here is exactly
        # the bug that crashed the first broadened-bank run.
        assert "base_ids" in pack, (
            f"{BANK}: bank lacks base_ids -- run B2_patch_bank_bases.py first")
        base_ids = pack["base_ids"]

        # ================= A2 : SmoothGrad-J =========================
        sel = (torch.tensor([f == "D4" for f in fam]) & hold & (epsmul == EPS_EVAL))
        sel_idx = sel.nonzero(as_tuple=True)[0]
        groups: dict[tuple, list] = {}
        for i in sel_idx.tolist():
            groups.setdefault((meta[i]["base"], meta[i]["pos"]), []).append(i)
        if N_SG_SITES and len(groups) > N_SG_SITES:
            # deterministic site subsample; restrict every A2 row to survivors
            gk = sorted(groups)
            gg = torch.Generator().manual_seed(7)
            keep_k = {gk[i] for i in torch.randperm(len(gk), generator=gg)[:N_SG_SITES].tolist()}
            groups = {k: v for k, v in groups.items() if k in keep_k}
            sel_idx = torch.tensor(sorted(i for v in groups.values() for i in v))
            sel = torch.zeros_like(sel)
            sel[sel_idx] = True  # keep boot grouping (base[sel]) aligned
        sg_pred = torch.zeros(len(sel_idx), d, device=DEV)
        pos_in_sel = {int(v): k for k, v in enumerate(sel_idx.tolist())}
        gsm = torch.Generator(device=DEV).manual_seed(11)
        done = 0
        for (bi, pos), idxs in groups.items():
            ids_t = torch.tensor([base_ids[bi]], device=DEV)
            ctx, acts = H.capture(model, ids_t)
            h = acts[layer]
            D = dirs[torch.tensor(idxs, device=DEV)]           # unit dirs [m,d]
            acc = torch.zeros(len(idxs), d, device=DEV)
            for k in range(K_SMOOTH):
                u = torch.randn(d, device=DEV, generator=gsm)
                u = u / u.norm() * (EPS_EVAL * med)
                hk = h.clone()
                hk[0, pos] += u.to(hk.dtype)
                e = effects(model, ctx, hk, layer, target, pos, D * (EPS_REF * med), BATCH)
                acc += e * (EPS_EVAL / EPS_REF)
            acc /= K_SMOOTH
            for j, i in enumerate(idxs):
                sg_pred[pos_in_sel[i]] = acc[j]
            done += 1
            if done % 20 == 0:
                print(f"  A2 {done}/{len(groups)} sites  ({time.time()-t0:.0f}s)", flush=True)

        si = sel_idx.to(DEV)
        Ytrue = Y[si]
        Xs = X[si]
        per = {"J": cos(Xs @ J.T, Ytrue), "R": cos(Xs @ R.T, Ytrue),
               "T_sec": cos(Xs @ T_sec.T, Ytrue), "T_sec^D4": cos(Xs @ T_secD4.T, Ytrue),
               "T_RC": cos(Xs @ T_RC.T, Ytrue), "SmoothGrad_J": cos(sg_pred, Ytrue)}
        # J_loc at the clean base, same rows
        ref = {(m["base"], m["pos"], m["family"], m["tag"]): i
               for i, m in enumerate(meta) if abs(m["eps"] - EPS_REF) < 1e-12}
        src = torch.tensor([ref.get((meta[i]["base"], meta[i]["pos"], meta[i]["family"],
                                     meta[i]["tag"]), -1) for i in sel_idx.tolist()])
        if (src >= 0).all():
            sd = src.to(DEV)
            per["J_loc"] = cos(Y[sd] * (eps[si] / eps[sd])[:, None], Ytrue)
        st = boot(per, base[sel], [("T_sec", "SmoothGrad_J"), ("T_sec", "J"),
                                   ("J_loc", "SmoothGrad_J"), ("J_loc", "T_sec")], seed=5)
        st["cos_Tsec_vs_SGJ"] = cos(Xs @ T_sec.T, sg_pred).mean().item()
        st["n_deltas"] = int(sel.sum())
        L["A2"] = st
        print(f"\n  --- A2 SmoothGrad-J (K={K_SMOOTH}), D4 held-out @ eps={EPS_EVAL}, "
              f"n={st['n_deltas']} deltas / {st['n_prompts']} prompts ---", flush=True)
        for k, v in st["point"].items():
            print(f"    {k:<14s} {v:.4f}  CI [{st['ci'][k][0]:.3f},{st['ci'][k][1]:.3f}]", flush=True)
        print(f"    cos(T_sec.d, SG-J.d) = {st['cos_Tsec_vs_SGJ']:.4f}", flush=True)
        for k, v in st["paired"].items():
            print(f"    {k:<22s} {v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] "
                  f"clear={v['ci_clear']}", flush=True)

        # ================= A5 : in-span transfer =====================
        L["A5"] = {}
        for r in RANKS:
            Vr = V[:, -r:]
            res_r = {}
            for family in ("D1", "D2"):
                fm = (torch.tensor([f == family for f in fam]) & hold
                      & (epsmul == EPS_EVAL)).nonzero(as_tuple=True)[0]
                if len(fm) == 0:
                    continue
                gsel = torch.Generator().manual_seed(13)
                take = fm[torch.randperm(len(fm), generator=gsel)[:N_INSPAN]]
                gr: dict[tuple, list] = {}
                for i in take.tolist():
                    gr.setdefault((meta[i]["base"], meta[i]["pos"]), []).append(i)
                dl_in, Y_in, keep = [], [], []
                for (bi, pos), idxs in gr.items():
                    ids_t = torch.tensor([base_ids[bi]], device=DEV)
                    ctx, acts = H.capture(model, ids_t)
                    h = acts[layer]
                    D = dirs[torch.tensor(idxs, device=DEV)]
                    Din = (D @ Vr) @ Vr.T
                    nn = Din.norm(dim=-1, keepdim=True)
                    ok = (nn[:, 0] > 1e-4)
                    if ok.sum() == 0:
                        continue
                    Din = Din[ok] / nn[ok] * (EPS_EVAL * med)
                    e = effects(model, ctx, h, layer, target, pos, Din, BATCH)
                    dl_in.append(Din)
                    Y_in.append(e)
                    keep += [idxs[j] for j in ok.nonzero(as_tuple=True)[0].tolist()]
                if not dl_in:
                    continue
                Din = torch.cat(dl_in)
                Yin = torch.cat(Y_in)
                capt = ((dirs[torch.tensor(keep, device=DEV)] @ Vr).norm(dim=-1))
                perr = {"I": cos(Din, Yin), "J": cos(Din @ J.T, Yin), "R": cos(Din @ R.T, Yin),
                        "T_sec": cos(Din @ T_sec.T, Yin), "T_sec^D4": cos(Din @ T_secD4.T, Yin),
                        "T_RC": cos(Din @ T_RC.T, Yin)}
                sb = boot(perr, base[torch.tensor(keep)],
                          [("T_sec", "J"), ("T_sec", "R"), ("T_RC", "R")], seed=9)
                sb["n_deltas"] = len(keep)
                sb["mean_captured_before_projection"] = capt.mean().item()
                res_r[family] = sb
                print(f"\n  --- A5 in-span r={r} {family}: n={len(keep)} "
                      f"(captured before proj {capt.mean():.3f}) ---", flush=True)
                print("     " + "  ".join(f"{k}={v:.4f}" for k, v in sb["point"].items()),
                      flush=True)
                for k, v in sb["paired"].items():
                    print(f"     {k:<12s} {v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] "
                          f"clear={v['ci_clear']}", flush=True)
            L["A5"][str(r)] = res_r

        rep["layers"][str(layer)] = L
        with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
            json.dump(rep, f, indent=2)
        del X, Y, dirs, J, R, T_sec, T_secD4, T_RC, V, C_dd
        torch.cuda.empty_cache()

    print(f"\nwrote {OUT}/{LENS_DIR}.json   ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
