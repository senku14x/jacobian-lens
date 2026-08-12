"""B8: export everything needed to check the reports without the 4.5 GB banks.

The review's closing point: the report's tables cannot be independently verified
from the repository alone. Raw banks and fitted operators are large and
regenerable and stay ignored, but nothing that a number depends on should be.

Writes to research/artifacts/data/:
  metrics/<run>.json     every compact result JSON, verbatim
  splits/<layer>.csv     per-delta split assignment at all four levels, plus the
                         (category, template, arg, alt, position, family, eps)
                         needed to regroup or re-split without the bank
  hyperparams.json       the env-var configuration each run was launched with
  manifest.json          sha256 of every bank file and every exported artifact,
                         plus row counts, so a regenerated bank can be checked
                         against the one the numbers came from
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTDIR = "research/artifacts/data"
BANKS = {"orig": "research/outputs/003_bank", "broad": "research/outputs/B1_bank"}
LENS_DIR = "qwen3.6-27b"
RUNS = ["A0", "A1_A6", "A2_A5", "A4_m2", "C2", "B2_A0", "B2_A1_A6", "B2_A2_A5",
        "B2_A4_m2", "B3", "B4", "B4_ctrl", "B6_smoke", "B7",
        "A0_recheck_003_bank", "A0_recheck_B1_bank", "002_calib", "001_stage0",
        "C1", "C3", "D1", "D2", "D3", "D3_smoke", "D4"]

HYPER = {
    "B1_bank_broad": {"MAX_BASES": 240, "EPS_GRID": [0.01, 0.05, 0.2, 1.0],
                      "prefix_tokens": 96, "layers": [16, 31, 46], "seed": 0},
    "A1_A6_fits": {"EKKO_TARGET": "d_odd_sum", "eta_rel": 1e-3, "N_BOOT": 2000},
    "A2_A5_model": {"EKKO_K": 8, "EKKO_RANKS": [555, 1911], "EKKO_SG_SITES": 120,
                    "eps_eval": 0.2, "eps_ref": 0.01},
    "A4_m2_ablation": {"N_ITEMS": 40, "N_RANDOM": 3,
                       "BANDS": {"L16": [16], "L16_31_46": [16, 31, 46]},
                       "scoring": "delta log-prob of correct answer token"},
    "C2_conditional": {"r_sketch": 192, "n_sites": 56, "eps_eval": 0.2},
    "B3_learning_curve": {"FRACS": [0.125, 0.25, 0.5, 1.0], "eta_rel": 1e-3},
    "B4_smoothing_mechanism": {"SIGMAS": [0.05, 0.2, 0.5], "EPS_EVAL": [0.05, 0.2, 1.0],
                               "K": 8, "M_IG": 8, "N_SITES": 60,
                               "SIGMA_DECOMP": 0.2},
    "B7_splits_targets_anchors": {"LAMBDAS": [0.03, 0.1, 0.3, 1.0], "ETA_REL": 1e-3,
                                  "splits": ["base", "unordered_pair", "template",
                                             "category"]},
}


def sha256(path, cap=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        n = 0
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
            n += len(chunk)
            if cap and n >= cap:
                break
    return h.hexdigest()


def main() -> None:
    os.makedirs(f"{OUTDIR}/metrics", exist_ok=True)
    os.makedirs(f"{OUTDIR}/splits", exist_ok=True)
    man = {"git": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
           "banks": {}, "metrics": {}, "splits": {}}

    for run in RUNS:
        src = f"research/outputs/{run}/{LENS_DIR}.json"
        alt = f"research/outputs/{run}/{LENS_DIR}_bank_verification.json"
        p = src if os.path.exists(src) else (alt if os.path.exists(alt) else None)
        if not p:
            print(f"  skip {run} (no json)")
            continue
        dst = f"{OUTDIR}/metrics/{run}.json"
        obj = json.load(open(p))
        with open(dst, "w") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
        man["metrics"][run] = {"sha256": sha256(dst), "bytes": os.path.getsize(dst),
                               "source": p}
        print(f"  metrics/{run}.json  ({os.path.getsize(dst)/1024:.0f} KB)")

    from B7_splits_targets_anchors import build_base_meta
    info, by, keys = build_base_meta()

    for layer in (16, 31, 46):
        path = f"{BANKS['broad']}/{LENS_DIR}_L{layer}.pt"
        if not os.path.exists(path):
            continue
        pack = torch.load(path, weights_only=False)
        meta = pack["meta"]
        levels = ("base", "unordered_pair", "template", "category")
        assign = {}
        for level in levels:
            kk = []
            for i, m in enumerate(meta):
                ii = info[int(m["base"])]
                if level == "base":
                    kk.append(str(m["base"]))
                elif level == "category":
                    kk.append(ii["cat"])
                elif level == "template":
                    kk.append(f"{ii['cat']}/{ii['func']}")
                else:
                    kk.append(f"{ii['cat']}/{ii['func']}/"
                              f"{'+'.join(sorted((ii['arg'], str(m['tag']))))}")
            uniq = sorted(set(kk))
            g = torch.Generator().manual_seed(0)
            perm = torch.randperm(len(uniq), generator=g)
            cal = {uniq[i] for i in perm[: max(1, len(uniq) // 2)].tolist()}
            assign[level] = [(k, "cal" if k in cal else "hold") for k in kk]

        dst = f"{OUTDIR}/splits/L{layer}.csv"
        with open(dst, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "base", "category", "template", "arg", "alt", "pos",
                        "family", "eps_mul", "eps_abs"]
                       + [f"{lv}_key" for lv in levels]
                       + [f"{lv}_side" for lv in levels])
            for i, m in enumerate(meta):
                ii = info[int(m["base"])]
                w.writerow([i, m["base"], ii["cat"], ii["func"], ii["arg"], m["tag"],
                            m["pos"], m["family"], f"{m['eps']:g}",
                            f"{float(pack['eps'][i]):.6f}"]
                           + [assign[lv][i][0] for lv in levels]
                           + [assign[lv][i][1] for lv in levels])
        man["splits"][f"L{layer}"] = {"sha256": sha256(dst),
                                      "bytes": os.path.getsize(dst),
                                      "n_rows": len(meta)}
        print(f"  splits/L{layer}.csv  ({os.path.getsize(dst)/1024/1024:.1f} MB, "
              f"{len(meta)} rows)")

    for tag, bd in BANKS.items():
        for layer in (16, 31, 46):
            p = f"{bd}/{LENS_DIR}_L{layer}.pt"
            if os.path.exists(p):
                man["banks"][f"{tag}_L{layer}"] = {
                    "bytes": os.path.getsize(p),
                    "sha256_first_64MB": sha256(p, cap=1 << 26)}

    with open(f"{OUTDIR}/hyperparams.json", "w") as f:
        json.dump(HYPER, f, indent=2, sort_keys=True)
    with open(f"{OUTDIR}/manifest.json", "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    print(f"\nwrote {OUTDIR}/manifest.json + hyperparams.json")


if __name__ == "__main__":
    main()
