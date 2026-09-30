"""001 analysis — implements the pre-registered analysis of designs/001-lag-bucket-poetry.md (Amendments 1-3).
Committed before any aggregate of v2 was viewed.

Primary (per layer, single-token rhymes, paired on items): B2 and B3 vs FULL_U — log-rank Wilcoxon + top-10 McNemar,
Holm over 6 tests each; success needs both Holm p < .05, hit gain >= .10, median-rank ratio >= 2, decoy AUC_own gain
(Wilcoxon p < .05) at the same layer, and the same direction in both source halves. Kill: no bucket B0-B4 at any
layer improves the median rank over FULL_U by >= 1.5x. Secondary: cross-validated best layer, WITEM, FULL32, positive
control (line-two tokens by lag), host split-half reliability, generic-token guardrail, qualitative samples.

Usage: python 001_analysis.py [--dry]   (--dry: run everything, print only structure; used to test the code)
"""
import argparse, json, math, os, random, collections
import numpy as np
from scipy.stats import wilcoxon, binomtest, spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
IN = os.path.join(ROOT, "results", "001-lag-bucket-poetry", "v2", "cells.jsonl")
OUT = os.path.join(ROOT, "results", "001-lag-bucket-poetry", "v2")
BUCKETS = ["B0", "B1", "B2", "B3", "B4"]
READOUTS = BUCKETS + ["WITEM", "FULL32", "FULL_U", "JCB", "JNP", "LOGIT"]
PRIMARY = ["B2", "B3"]
LAG_CLASSES = {"lag0": (0, 0), "lag1-3": (1, 3), "lag4-8": (4, 8), "lag9-16": (9, 16)}


def lr(r):
    return math.log10(r + 1)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, m, running = [0.0] * len(ps), len(ps), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * ps[i]))
        adj[i] = running
    return adj


def auc_own(row, n):
    """Fraction of the OTHER items' distinct rhyme tokens ranked below this item's own rhyme token (ties 0.5).
    The decoy set contains the own token once; it is removed by its rank (distinct tokens have distinct ranks)."""
    own = row["rank"][n]
    others = list(row["decoy_ranks"][n])
    if own in others:
        others.remove(own)
    if not others:
        return float("nan")
    return float(np.mean([1.0 if r > own else (0.5 if r == own else 0.0) for r in others]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--items_file", default=None, help="restrict to the Amendment-4 subset")
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(IN)]
    rows = [r for r in rows if r["single"]]
    if a.items_file:
        keep = set(json.load(open(a.items_file))["names"])
        rows = [r for r in rows if r["name"] in keep]
    layers = sorted({r["layer"] for r in rows})
    by = collections.defaultdict(dict)                       # by[layer][name] = row
    for r in rows:
        by[r["layer"]][r["name"]] = r
    names_all = sorted(set.intersection(*[set(by[l]) for l in layers]))   # items complete at every layer
    S = {"n_items": len(names_all), "layers": layers, "per_layer": {}, "primary": [], "kill": {}, "pc": {},
         "reliability": {}, "guardrail": {}, "cv_best_layer": {}}

    # ---------------- descriptive per layer
    for l in layers:
        R = {}
        for n in READOUTS:
            rk = np.array([by[l][x]["rank"][n] for x in names_all])
            au = np.array([auc_own(by[l][x], n) for x in names_all])
            R[n] = {"median_rank": float(np.median(rk)), "mean_log10_rank": float(np.mean([lr(v) for v in rk])),
                    "top10": float(np.mean(rk < 10)), "top100": float(np.mean(rk < 100)),
                    "mean_auc_own": float(np.nanmean(au))}
        S["per_layer"][l] = R

    # ---------------- primary tests
    tests = []
    for l in layers:
        for b in PRIMARY:
            xb = np.array([lr(by[l][x]["rank"][b]) for x in names_all])
            xf = np.array([lr(by[l][x]["rank"]["FULL_U"]) for x in names_all])
            pw = wilcoxon(xb, xf).pvalue if np.any(xb != xf) else 1.0
            hb = np.array([by[l][x]["rank"][b] < 10 for x in names_all])
            hf = np.array([by[l][x]["rank"]["FULL_U"] < 10 for x in names_all])
            bc, cb = int(np.sum(hb & ~hf)), int(np.sum(~hb & hf))
            pm = binomtest(bc, bc + cb, 0.5).pvalue if bc + cb > 0 else 1.0
            ab = np.array([auc_own(by[l][x], b) for x in names_all]); af = np.array([auc_own(by[l][x], "FULL_U") for x in names_all])
            ok = ~np.isnan(ab) & ~np.isnan(af)
            pa = wilcoxon(ab[ok], af[ok]).pvalue if np.any(ab[ok] != af[ok]) else 1.0
            med_ratio = (np.median([by[l][x]["rank"]["FULL_U"] for x in names_all]) + 1) / \
                        (np.median([by[l][x]["rank"][b] for x in names_all]) + 1)
            halves = {}
            for src_name, pred in (("jlens_source", lambda s: s == "jlens_source"), ("staging", lambda s: s != "jlens_source")):
                sub = [x for x in names_all if pred(by[l][x]["source"])]
                if sub:
                    halves[src_name] = {
                        "n": len(sub),
                        "hit_gain": float(np.mean([by[l][x]["rank"][b] < 10 for x in sub]) - np.mean([by[l][x]["rank"]["FULL_U"] < 10 for x in sub])),
                        "median_ratio": float((np.median([by[l][x]["rank"]["FULL_U"] for x in sub]) + 1) / (np.median([by[l][x]["rank"][b] for x in sub]) + 1))}
            tests.append({"layer": l, "bucket": b, "p_wilcoxon": float(pw), "p_mcnemar": float(pm), "p_auc": float(pa),
                          "hit_gain": float(hb.mean() - hf.mean()), "discordant_b_c": [bc, cb],
                          "median_rank_ratio": float(med_ratio),
                          "auc_gain": float(np.nanmean(ab - af)), "source_halves": halves,
                          "bucket_better_logrank": bool(np.mean(xb) < np.mean(xf))})
    pw_adj = holm([t["p_wilcoxon"] for t in tests]); pm_adj = holm([t["p_mcnemar"] for t in tests])
    for t, a1, a2 in zip(tests, pw_adj, pm_adj):
        t["p_wilcoxon_holm"], t["p_mcnemar_holm"] = a1, a2
        t["success"] = bool(a1 < .05 and a2 < .05 and t["hit_gain"] >= .10 and t["median_rank_ratio"] >= 2
                            and t["bucket_better_logrank"] and t["p_auc"] < .05 and t["auc_gain"] > 0
                            and all(h["hit_gain"] > 0 and h["median_ratio"] > 1 for h in t["source_halves"].values()))
    S["primary"] = tests
    S["success_any"] = any(t["success"] for t in tests)
    kill_ratios = {f"{b}@L{l}": (S["per_layer"][l]["FULL_U"]["median_rank"] + 1) / (S["per_layer"][l][b]["median_rank"] + 1)
                   for l in layers for b in BUCKETS}
    S["kill"] = {"median_ratio_by_bucket_layer": kill_ratios, "killed": all(v < 1.5 for v in kill_ratios.values())}

    # ---------------- positive control: line-two tokens by lag class, per readout (mean log-rank)
    for l in layers:
        M = {}
        for n in BUCKETS + ["FULL_U"]:
            M[n] = {}
            for cls, (lo, hi) in LAG_CLASSES.items():
                v = [lr(rk) for x in names_all for lag, rk in by[l][x]["pc"][n] if lo <= lag <= hi]
                M[n][cls] = float(np.mean(v)) if v else None
        # paired per item: B0 vs B3 on lag-0 tokens; B1 vs B3 on lag 1-3 tokens
        def paired(bk, cls):
            lo, hi = LAG_CLASSES[cls]
            xa, xb = [], []
            for x in names_all:
                pa = [lr(rk) for lag, rk in by[l][x]["pc"][bk] if lo <= lag <= hi]
                pb = [lr(rk) for lag, rk in by[l][x]["pc"]["B3"] if lo <= lag <= hi]
                if pa and pb:
                    xa.append(np.mean(pa)); xb.append(np.mean(pb))
            xa, xb = np.array(xa), np.array(xb)
            return {"n": len(xa), "mean_diff": float(np.mean(xa - xb)) if len(xa) else None,
                    "p": float(wilcoxon(xa, xb).pvalue) if len(xa) > 5 and np.any(xa != xb) else None}
        pcB0, pcB1 = paired("B0", "lag0"), paired("B1", "lag1-3")
        S["pc"][l] = {"matrix_mean_log10_rank": M, "B0_vs_B3_on_lag0": pcB0, "B1_vs_B3_on_lag1-3": pcB1,
                      "pass": bool(pcB0["mean_diff"] is not None and pcB0["mean_diff"] < 0 and pcB0["p"] is not None and pcB0["p"] < .05
                                   and pcB1["mean_diff"] is not None and pcB1["mean_diff"] < 0 and pcB1["p"] is not None and pcB1["p"] < .05)}

    # ---------------- reliability: host split-half Spearman of own-rank over items
    for l in layers:
        S["reliability"][l] = {}
        for n in BUCKETS + ["WITEM", "FULL32", "FULL_U"]:
            h = [by[l][x]["half_rank"].get(n) for x in names_all]
            h = [v for v in h if v]
            if len(h) > 5:
                S["reliability"][l][n] = float(spearmanr([v[0] for v in h], [v[1] for v in h]).correlation)

    # ---------------- guardrail: share of top-10 slots held by tokens present in >= 50% of items' top-10
    for l in layers:
        S["guardrail"][l] = {}
        for n in READOUTS:
            cnt = collections.Counter(t for x in names_all for t in set(by[l][x]["top10"][n]))
            generic = {t for t, c in cnt.items() if c >= 0.5 * len(names_all)}
            slots = [t for x in names_all for t in by[l][x]["top10"][n]]
            S["guardrail"][l][n] = {"generic_tokens": sorted(generic)[:20], "generic_slot_share": float(np.mean([t in generic for t in slots]))}

    # ---------------- secondary: cross-validated best layer (choose on half A, score on half B, and vice versa)
    for n in BUCKETS + ["WITEM", "FULL32", "FULL_U"]:
        vals = []
        for x in names_all:
            hr = {l: by[l][x]["half_rank"].get(n) for l in layers}
            if any(v is None for v in hr.values()):
                continue
            la = min(layers, key=lambda l: hr[l][0]); lb = min(layers, key=lambda l: hr[l][1])
            vals.append(0.5 * (lr(hr[la][1]) + lr(hr[lb][0])))
        S["cv_best_layer"][n] = {"mean_log10_rank": float(np.mean(vals)) if vals else None, "n": len(vals)}

    if a.dry:
        print("dry run ok:", {"n_items": S["n_items"], "layers": layers, "n_tests": len(tests)}); return
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

    # ---------------- qualitative samples (L48 if present): 10 random + 3 best + 3 worst (B3 vs FULL_U)
    lq = 48 if 48 in layers else layers[len(layers) // 2]
    rnd = random.Random(0).sample(names_all, min(10, len(names_all)))
    diff = sorted(names_all, key=lambda x: lr(by[lq][x]["rank"]["B3"]) - lr(by[lq][x]["rank"]["FULL_U"]))
    pick = [("random", x) for x in rnd] + [("B3 best vs FULL_U", x) for x in diff[:3]] + [("B3 worst vs FULL_U", x) for x in diff[-3:]]
    lines = [f"# 001 v2 qualitative samples (L{lq}; 10 random seed 0 + 3 best + 3 worst B3-vs-FULL_U)\n"]
    for tag, x in pick:
        r = by[lq][x]
        lines.append(f"## {x} — rhyme `{r['word']}` (δ_item {r['delta_item']}) — {tag}\n")
        for n in ["B0", "B1", "B2", "B3", "WITEM", "FULL_U", "JCB"]:
            lines.append(f"- **{n}** (rank {r['rank'][n]}): " + ", ".join(repr(t) for t in r["top10"][n]))
        lines.append("")
    open(os.path.join(OUT, "qualitative.md"), "w").write("\n".join(lines))
    print(json.dumps({"success_any": S["success_any"], "killed": S["kill"]["killed"], "n_items": S["n_items"]}))


if __name__ == "__main__":
    main()
