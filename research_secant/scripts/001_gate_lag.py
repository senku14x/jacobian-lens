"""001 gate G_lag (design 001, Amendment 2 §5). Runs at L48 before the main v2 run.

A  implementation/leakage: spaced-source estimator vs exact single-source perturbations on the SAME samples.
   Pass: cos >= 0.95 for B1-B4 and FULL32.
B  documentation of the v1 defect: B3 split-half cos, old all-positions sign estimator (K=32, R=2) vs spaced (K=32, R=4).
C  reliability: spaced B3 split-half cos at the planned budget; if < 0.7, R is doubled for the main run.
Writes results/001-lag-bucket-poetry/gate_lag.json.
"""
import json, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
from engine import Engine
from mfl import (lag_profile_spaced, lag_profile_exact, buckets_from_profile, lag_buckets, valid_positions, cos)

ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "results", "001-lag-bucket-poetry"); os.makedirs(OUT, exist_ok=True)
POETRY = "/workspace/workspace-bench/evals/poetry/items.json"
BUCKETS = {"B0": (0, 0), "B1": (1, 3), "B2": (4, 8), "B3": (9, 16), "B4": (17, 32)}
L = 48
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def hosts(tok, n):
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    out, i = [], 100
    while len(out) < n:
        x = tok(ds[i]["text"], truncation=True, max_length=128, add_special_tokens=False)["input_ids"]
        if len(x) == 128:
            out.append(x)
        i += 1
    return out


def main():
    items = json.load(open(POETRY))["items"][:2]
    eng = Engine(l0=L); tok = eng.tok
    eng.to_gpu_bf16()
    hs = []
    for it in items:
        ids = tok(it["prompt"], add_special_tokens=False)["input_ids"]
        nl = [i for i, t in enumerate(ids) if "\n" in tok.decode([t])][-1]
        X, _ = eng.capture(torch.tensor([ids]), [L]); hs.append(X[L][0, nl].clone())
    HX, hkw = eng.capture(torch.tensor(hosts(tok, 32)), [L])
    eng.to_cpu_bf16(); eng.build_fp32_upper()
    U = eng.U
    Np = len(valid_positions(128))
    res = {"A": [], "B": [], "C": []}
    for ii, h in enumerate(hs):
        # ---- A: 8 hosts, 2 draws, spaced vs exact on identical samples
        S, C, samples, _ = lag_profile_spaced(eng, HX[L][:8], hkw, L, h, R=2, seed=11 + ii)
        m_sp = S.sum(0) / C.sum(0).clamp_min(1)[:, None].to(S.device)
        Se, Ce = lag_profile_exact(eng, HX[L][:8], hkw, L, h, samples)
        m_ex = Se / Ce.clamp_min(1)[:, None].to(Se.device)
        b_sp, b_ex = buckets_from_profile(m_sp, BUCKETS, Np), buckets_from_profile(m_ex, BUCKETS, Np)
        a = {n: {"cos_resid": float(cos(b_sp[n], b_ex[n])), "cos_vocab": float(cos(U @ b_sp[n], U @ b_ex[n]))}
             for n in b_sp}
        res["A"].append(a); log(f"A item{ii}", json.dumps({n: round(v['cos_resid'], 4) for n, v in a.items()}))
        # ---- B: old all-positions sign estimator, split halves (K=32, R=2)
        est, hv = lag_buckets(eng, HX[L], hkw, L, h, BUCKETS, R=2, seed=21 + ii)
        old = {n: float(cos(U @ hv[n][0].to(U.device), U @ hv[n][1].to(U.device))) for n in ("B1", "B2", "B3", "B4")}
        # ---- spaced at the planned budget (K=32, R=4): split halves
        S, C, _, _ = lag_profile_spaced(eng, HX[L], hkw, L, h, R=4, seed=31 + ii)
        mh = [S[k] / C[k].clamp_min(1)[:, None].to(S.device) for k in (0, 1)]
        bh = [buckets_from_profile(m, BUCKETS, Np) for m in mh]
        new = {n: float(cos(U @ bh[0][n], U @ bh[1][n])) for n in ("B1", "B2", "B3", "B4", "FULL32")}
        res["B"].append({"old_split_half_vocab_cos": old, "spaced_split_half_vocab_cos": new})
        log(f"B item{ii} old", {k: round(v, 3) for k, v in old.items()}, "spaced", {k: round(v, 3) for k, v in new.items()})
        res["C"].append(new["B3"])
    passA = all(a[n]["cos_resid"] >= 0.95 for a in res["A"] for n in ("B1", "B2", "B3", "B4", "FULL32"))
    passC = min(res["C"]) >= 0.7
    res["verdict"] = {"A_pass": passA, "C_pass": passC, "R_main": 4 if passC else 8}
    json.dump(res, open(os.path.join(OUT, "gate_lag.json"), "w"), indent=1)
    log("verdict", res["verdict"])


if __name__ == "__main__":
    main()
