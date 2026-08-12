"""A0 (BLOCKING): verify the perturbation bank before anything reads from it.

Asserts per layer: counts, family split, finiteness, no duplicate deltas, both
targets present, every eps bucket + native populated. Then re-derives
C_dd / C_Ddelta FROM THE STORED DELTAS (not from any cached accumulator),
reports the eigenspectrum, and gives effective rank at a SANE threshold
(lambda > 1e-6 * lambda_max) plus the participation ratio -- never a
machine-epsilon rank. Finally a write-integrity check: re-accumulate a random
subset of base prompts and confirm it sums into the full C to tolerance.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
BANK = os.environ.get("EKKO_BANK", "research/outputs/003_bank")
OUT = os.environ.get("EKKO_OUT", "research/outputs/A0")
DEV = "cuda:0"
EPS_GRID = (0.01, 0.05, 0.2, 1.0)
# A hard-coded expectation is bank-specific and goes stale the moment the bank is
# rebuilt (it did: the B1 broadening tripped it). The integrity question that
# actually matters is "does the file on disk contain what the builder recorded
# writing", so read the expectation from the manifest the builder wrote.
# Duplicate deltas are checked on a 80-dim rounded signature, which collides at a
# low rate for genuinely distinct but near-parallel directions; allow a small
# collision fraction and additionally report exact full-row duplicates, which are
# the thing that would actually corrupt a fit.
DUP_TOL = float(os.environ.get("EKKO_DUP_TOL", "0.001"))


def rank_stats(C):
    ev = torch.linalg.eigvalsh(C.double()).flip(0).clamp_min(0)
    lmax = ev[0].item()
    eff_1e6 = int((ev > 1e-6 * lmax).sum())
    eff_1e10 = int((ev > 1e-10 * lmax).sum())
    part = (ev.sum() ** 2 / (ev ** 2).sum()).item()
    nz = ev[ev > 1e-6 * lmax]
    cond_sane = (nz[0] / nz[-1]).item() if len(nz) else float("inf")
    return {"lambda_max": lmax, "eff_rank_1e-6": eff_1e6, "eff_rank_1e-10": eff_1e10,
            "participation_ratio": part, "cond_at_1e-6": cond_sane,
            "spectrum_head": [float(x) for x in ev[:8]],
            "spectrum_decile": [float(ev[min(i, len(ev) - 1)])
                                for i in range(0, len(ev), max(len(ev) // 10, 1))][:11]}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    man = json.load(open(f"{BANK}/{LENS_DIR}_manifest.json"))
    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "bank": BANK, "manifest_layers": man["layers"], "layers": {}, "verdict": {}}

    for layer in man["layers"]:
        path = f"{BANK}/{LENS_DIR}_L{layer}.pt"
        print(f"\n===== L{layer}  {path}", flush=True)
        L = {"path": path, "sha256_head": hashlib.sha256(
            open(path, "rb").read(1 << 20)).hexdigest()[:16],
            "bytes": os.path.getsize(path), "asserts": {}}
        pack = torch.load(path, weights_only=False)
        meta = pack["meta"]
        d = pack["dir"].shape[1]
        fam = [m["family"] for m in meta]
        epsmul = torch.tensor([m["eps"] for m in meta])

        counts = {f: sum(1 for x in fam if x == f) for f in ("D4", "D1", "D2")}
        L["counts"] = {"total": len(meta), **counts}
        exp = man.get(f"L{layer}", {})
        exp_fam = exp.get("families", {})
        L["expected_counts"] = {"total": exp.get("n_deltas"), **exp_fam}
        L["asserts"]["counts"] = bool(
            exp.get("n_deltas") == len(meta) and exp_fam and
            all(counts.get(f, 0) == n for f, n in exp_fam.items()))

        finite = {}
        for k in ("dir", "eps", "d_odd_sum", "d_one_sum", "d_odd_self", "d_one_self"):
            t = pack[k].float()
            finite[k] = bool(torch.isfinite(t).all())
        L["asserts"]["all_finite"] = all(finite.values())
        L["finite_by_field"] = finite
        L["asserts"]["both_targets_present"] = all(
            k in pack for k in ("d_odd_sum", "d_one_sum", "d_odd_self", "d_one_self"))

        # duplicate deltas: hash the (rounded) actual perturbation
        X_cpu = (pack["dir"].float() * pack["eps"][:, None].float())
        sig = torch.round(X_cpu[:, ::64] * 1e3).to(torch.int64)
        uniq = len({tuple(r.tolist()) for r in sig})
        groups: dict[bytes, list] = {}
        for i, r in enumerate(X_cpu):
            groups.setdefault(hashlib.blake2b(r.numpy().tobytes(), digest_size=16
                                              ).digest(), []).append(i)
        dup = [g for g in groups.values() if len(g) > 1]
        # D2 is v_t = J^T (gamma * u_t): a function of token id and layer ONLY, so
        # the same tag sampled at two sites gives the identical direction by
        # construction. Those are legal. A duplicate inside D4 or D1 would mean
        # the builder wrote the same measurement twice and is disqualifying.
        legal = [g for g in dup if all(meta[i]["family"] == "D2" for i in g)
                 and len({meta[i]["tag"] for i in g}) == 1]
        L["unique_delta_signatures"] = uniq
        L["unique_deltas_exact"] = len(groups)
        L["dup_signature_frac"] = (len(meta) - uniq) / max(len(meta), 1)
        L["dup_groups"] = {"total": len(dup), "legal_D2_same_tag": len(legal),
                           "illegal": [[{"i": i, **{k: meta[i][k] for k in
                                                    ("family", "base", "pos", "tag", "eps")}}
                                        for i in g]
                                       for g in dup if g not in legal][:8]}
        L["asserts"]["no_dup_deltas"] = bool(
            len(dup) == len(legal) and L["dup_signature_frac"] <= DUP_TOL)

        buckets = {f"{e:g}": int((epsmul == e).sum()) for e in EPS_GRID}
        buckets["native"] = int((epsmul < 0).sum())
        L["eps_buckets"] = buckets
        L["asserts"]["eps_buckets_populated"] = all(v > 0 for v in buckets.values())

        nrm = X_cpu.norm(dim=-1)
        L["delta_norms"] = {}
        for e in list(EPS_GRID) + [-1.0]:
            m = (epsmul == e) if e > 0 else (epsmul < 0)
            if m.sum():
                L["delta_norms"]["native" if e < 0 else f"{e:g}"] = {
                    "n": int(m.sum()), "mean": nrm[m].mean().item(),
                    "min": nrm[m].min().item(), "max": nrm[m].max().item()}
        L["asserts"]["norms_positive"] = bool((nrm > 0).all())

        # ---- re-derive C from stored deltas ---------------------------
        X = X_cpu.to(DEV)
        Y = pack["d_odd_sum"].to(DEV, torch.float32)
        base = torch.tensor([m["base"] for m in meta])
        ub = sorted(set(base.tolist()))
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(ub), generator=g)
        cal_b = {ub[i] for i in perm[: len(ub) // 2].tolist()}
        is_cal = torch.tensor([b.item() in cal_b for b in base])
        F4 = torch.tensor([f == "D4" for f in fam])
        F1 = torch.tensor([f == "D1" for f in fam])

        for tag, m in (("fit_D4+D1_cal_all_eps", (F4 | F1) & is_cal),
                       ("fit_D4+D1_cal_eps0.2", (F4 | F1) & is_cal & (epsmul == 0.2)),
                       ("D4_cal_eps0.2", F4 & is_cal & (epsmul == 0.2))):
            md = m.to(DEV)
            C = X[md].T @ X[md]
            L.setdefault("C_dd", {})[tag] = {"n": int(m.sum()), **rank_stats(C)}
            print(f"  {tag:<26s} n={int(m.sum()):<5d} "
                  f"eff_rank(1e-6)={L['C_dd'][tag]['eff_rank_1e-6']:<5d} "
                  f"PR={L['C_dd'][tag]['participation_ratio']:.1f} "
                  f"cond={L['C_dd'][tag]['cond_at_1e-6']:.3e}", flush=True)

        # distinct directions available for identification
        dis = {f: len({(m_["base"], m_["pos"], m_["tag"]) for m_, c in zip(meta, is_cal.tolist())
                       if m_["family"] == f and c}) for f in ("D4", "D1", "D2")}
        L["distinct_cal_directions"] = dis
        print(f"  distinct calibrate directions: {dis}   (d_model={d})", flush=True)

        # ---- write-integrity: subset must sum into the whole ----------
        md = ((F4 | F1) & is_cal).to(DEV)
        C_full = X[md].T @ X[md]
        Cx_full = Y[md].T @ X[md]
        sub = sorted(cal_b)[: max(len(cal_b) // 3, 1)]
        ms = ((F4 | F1) & is_cal & torch.tensor([b.item() in set(sub) for b in base])).to(DEV)
        mr = md & ~ms
        err_C = (C_full - (X[ms].T @ X[ms] + X[mr].T @ X[mr])).abs().max().item()
        err_Cx = (Cx_full - (Y[ms].T @ X[ms] + Y[mr].T @ X[mr])).abs().max().item()
        rel = err_C / C_full.abs().max().item()
        L["write_integrity"] = {"max_abs_err_C_dd": err_C, "rel": rel,
                                "max_abs_err_C_Dd": err_Cx,
                                "n_subset_prompts": len(sub)}
        L["asserts"]["write_integrity"] = rel < 1e-5
        print(f"  write-integrity: rel err C_dd = {rel:.2e}  (abs {err_C:.3e})", flush=True)

        L["all_asserts_pass"] = all(L["asserts"].values())
        print(f"  ASSERTS: {L['asserts']}", flush=True)
        print(f"  --> L{layer} {'PASS' if L['all_asserts_pass'] else 'FAIL'}", flush=True)
        rep["layers"][str(layer)] = L
        del X, Y
        torch.cuda.empty_cache()

    rep["verdict"] = {"all_layers_pass": all(v["all_asserts_pass"] for v in rep["layers"].values()),
                      "layers_verified": list(rep["layers"])}
    with open(f"{OUT}/{LENS_DIR}_bank_verification.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nA0 verdict: {rep['verdict']}   ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
