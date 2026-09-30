"""002 analysis — the pre-registered analysis of designs/002-future-words-own-text.md. Committed before any real
002 data existed (only a 1-item smoke file, never analysed).

Primary: per layer, matching window (B2 for lag class L2 = 4-8, B3 for L3 = 9-16) vs FULL_U on per-item mean
log10(rank+1) of the item's own eligible future words in that class; paired Wilcoxon, Holm over 6 tests. Success:
Holm p < .05, mean diff <= -0.30 (>= 2x geometric-mean rank), and specificity (own-vs-decoy AUC gain, Wilcoxon p < .05,
mean > 0). Kill: no (class in L2/L3/L4, layer) with matching-window mean diff <= -0.176 (1.5x). Positive control:
B0 beats B3 on L0 targets (p < .05). Secondary: all-token version, reliability, per-lag curve, qualitative samples.

Usage: python 002_analysis.py [--cells FILE] [--dry]
"""
import argparse, json, math, os, random, collections
import numpy as np
from scipy.stats import wilcoxon, spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DIR = os.path.join(ROOT, "results", "002-future-words-own-text")
CLASSES = {"L0": (0, 0), "L1": (1, 3), "L2": (4, 8), "L3": (9, 16), "L4": (17, 32)}
MATCH = {"L0": "B0", "L1": "B1", "L2": "B2", "L3": "B3", "L4": "B4"}
READOUTS = ["B0", "B1", "B2", "B3", "B4", "FULL32", "FULL_U", "JCB", "JNP", "LOGIT"]


def lr(r):
    return math.log10(r + 1)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def class_mean(row, readout, cls, key="ranks", tkey="targets"):
    lo, hi = CLASSES[cls]
    v = [lr(r) for (tid, lag), r in zip(row[tkey], row[key][readout]) if lo <= lag <= hi]
    return float(np.mean(v)) if v else None


def auc_own_decoy(row, readout, cls):
    lo, hi = CLASSES[cls]
    own = [r for (t, lag), r in zip(row["targets"], row["ranks"][readout]) if lo <= lag <= hi]
    dec = [r for (t, lag), r in zip(row["decoy_targets"], row["decoy_ranks"][readout]) if lo <= lag <= hi]
    if not own or not dec:
        return None
    return float(np.mean([1.0 if o < d else 0.5 if o == d else 0.0 for o in own for d in dec]))


def paired(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 6 or not np.any(xs != ys):
        return {"n": int(len(xs)), "mean_diff": float(np.mean(xs - ys)) if len(xs) else None, "p": None}
    return {"n": int(len(xs)), "mean_diff": float(np.mean(xs - ys)), "p": float(wilcoxon(xs, ys).pvalue)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default=os.path.join(DIR, "cells.jsonl"))
    ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    rows = [json.loads(x) for x in open(a.cells)]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["layer"]][r["id"]] = r
    layers = sorted(by)
    ids = sorted(set.intersection(*[set(by[l]) for l in layers]))
    S = {"n_items": len(ids), "layers": layers, "primary": [], "kill": {}, "pc": {}, "secondary": {}, "descriptive": {}}

    # descriptive: per layer x class x readout, median over items of class-mean log-rank (content targets)
    for l in layers:
        S["descriptive"][l] = {c: {n: (lambda v: float(np.median(v)) if v else None)(
            [m for m in (class_mean(by[l][i], n, c) for i in ids) if m is not None]) for n in READOUTS} for c in CLASSES}

    # primary
    tests = []
    for l in layers:
        for c in ("L2", "L3"):
            w = MATCH[c]
            pairs = [(class_mean(by[l][i], w, c), class_mean(by[l][i], "FULL_U", c)) for i in ids]
            pairs = [(x, y) for x, y in pairs if x is not None]
            res = paired([x for x, _ in pairs], [y for _, y in pairs])
            au = [(auc_own_decoy(by[l][i], w, c), auc_own_decoy(by[l][i], "FULL_U", c)) for i in ids]
            au = [(x, y) for x, y in au if x is not None and y is not None]
            spec = paired([x for x, _ in au], [y for _, y in au])
            tests.append({"layer": l, "class": c, "window": w, **{f"rank_{k}": v for k, v in res.items()},
                          **{f"spec_{k}": v for k, v in spec.items()}})
    adj = holm([t["rank_p"] if t["rank_p"] is not None else 1.0 for t in tests])
    for t, p in zip(tests, adj):
        t["rank_p_holm"] = p
        t["success"] = bool(p < .05 and t["rank_mean_diff"] is not None and t["rank_mean_diff"] <= -0.30
                            and t["spec_p"] is not None and t["spec_p"] < .05 and t["spec_mean_diff"] > 0)
    S["primary"] = tests; S["success_any"] = any(t["success"] for t in tests)

    # kill (effect-size based)
    kd = {}
    for l in layers:
        for c in ("L2", "L3", "L4"):
            pairs = [(class_mean(by[l][i], MATCH[c], c), class_mean(by[l][i], "FULL_U", c)) for i in ids]
            pairs = [(x, y) for x, y in pairs if x is not None]
            kd[f"{c}@L{l}"] = float(np.mean([x - y for x, y in pairs])) if pairs else None
    S["kill"] = {"mean_log10_diff_matching_minus_FULL_U": kd,
                 "killed": all(v is None or v > -0.176 for v in kd.values())}

    # positive control: B0 vs B3 on L0 targets (content; and all-token secondary)
    for l in layers:
        S["pc"][l] = {}
        for key, tkey, lab in (("ranks", "targets", "content"), ("all_ranks", "all_targets", "all_tokens")):
            pairs = [(class_mean(by[l][i], "B0", "L0", key, tkey), class_mean(by[l][i], "B3", "L0", key, tkey)) for i in ids]
            pairs = [(x, y) for x, y in pairs if x is not None]
            res = paired([x for x, _ in pairs], [y for _, y in pairs])
            res["pass"] = bool(res["p"] is not None and res["p"] < .05 and res["mean_diff"] < 0)
            S["pc"][l][lab] = res

    # secondary: all-token primary analogue; reliability; per-lag curve
    sec = {"all_tokens": [], "reliability": {}, "per_lag_median_rank": {}}
    for l in layers:
        for c in ("L2", "L3"):
            w = MATCH[c]
            pairs = [(class_mean(by[l][i], w, c, "all_ranks", "all_targets"), class_mean(by[l][i], "FULL_U", c, "all_ranks", "all_targets")) for i in ids]
            pairs = [(x, y) for x, y in pairs if x is not None]
            sec["all_tokens"].append({"layer": l, "class": c, **paired([x for x, _ in pairs], [y for _, y in pairs])})
        sec["reliability"][l] = {}
        for n in ["B2", "B3", "FULL_U"]:
            h0, h1 = [], []
            for i in ids:
                r = by[l][i]
                if r["targets"]:
                    h0.append(np.mean([lr(x) for x in r["half_ranks"][n][0]])); h1.append(np.mean([lr(x) for x in r["half_ranks"][n][1]]))
            sec["reliability"][l][n] = float(spearmanr(h0, h1).correlation) if len(h0) > 5 else None
        curve = {}
        for n in ["B0", "B1", "B2", "B3", "B4", "FULL_U", "JCB"]:
            per = collections.defaultdict(list)
            for i in ids:
                for (t, lag), r in zip(by[l][i]["all_targets"], by[l][i]["all_ranks"][n]):
                    per[lag].append(r)
            curve[n] = {lag: float(np.median(v)) for lag, v in sorted(per.items())}
        sec["per_lag_median_rank"][l] = curve
    S["secondary"] = sec

    if a.dry:
        print("dry run ok:", {"n_items": S["n_items"], "layers": layers, "n_tests": len(tests)}); return
    json.dump(S, open(os.path.join(DIR, "summary.json"), "w"), indent=1)

    lq = 48 if 48 in layers else layers[len(layers) // 2]
    have = [i for i in ids if class_mean(by[lq][i], "B3", "L3") is not None]
    rnd = random.Random(0).sample(have, min(10, len(have)))
    diff = sorted(have, key=lambda i: class_mean(by[lq][i], "B3", "L3") - class_mean(by[lq][i], "FULL_U", "L3"))
    pick = [("random", i) for i in rnd] + [("B3 best vs FULL_U", i) for i in diff[:3]] + [("B3 worst vs FULL_U", i) for i in diff[-3:]]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    lines = [f"# 002 qualitative samples (L{lq}; 10 random seed 0 + 3 best + 3 worst on L3 = 9-16 ahead)\n"]
    for tag, i in pick:
        r = by[lq][i]
        l3 = [(t, lag, rk) for (t, lag), rk in zip(r["targets"], r["ranks"]["B3"]) if 9 <= lag <= 16]
        lines.append(f"## {i} — read at t={r['t']} — {tag}")
        lines.append(f"future content words at lags 9–16 (word, lag, rank under B3 / FULL_U): "
                     + ", ".join(f"({tok.decode([t]).strip()!r}, lag {lag}, {rk} / {r['ranks']['FULL_U'][r['targets'].index([t, lag])]})" for t, lag, rk in l3))
        for n in ["B0", "B1", "B2", "B3", "FULL_U", "JCB"]:
            lines.append(f"- **{n}**: " + ", ".join(repr(x) for x in r["top10"][n]))
        lines.append("")
    open(os.path.join(DIR, "qualitative.md"), "w").write("\n".join(lines))
    print(json.dumps({"success_any": S["success_any"], "killed": S["kill"]["killed"], "n_items": S["n_items"]}))


if __name__ == "__main__":
    main()
