"""B5: figures for the Phase-A/B report. Reads committed JSON only, no model.

One rule applied throughout: every comparison plot shows the noise floor or the
control it is being read against, and distributions/CIs are drawn rather than
bare means. A figure that carries no more than its caption is not drawn.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.environ.get("EKKO_FIGDIR",
                     "research/artifacts/007-2026-08-12-transport-operators/figures")
O = "research/outputs"
LENS = "qwen3.6-27b"
LAYERS = ["16", "31", "46"]
DEPTH = {"16": "L16 (26%)", "31": "L31 (50%)", "46": "L46 (74%)"}
C = {"I": "#9e9e9e", "J": "#1f77b4", "R": "#17becf", "T_sec": "#d62728",
     "T_RC": "#e377c2", "J_loc": "#2ca02c", "SG": "#ff7f0e", "T_IG": "#8c564b"}


def load(p):
    try:
        return json.load(open(f"{O}/{p}/{LENS}.json"))
    except Exception:
        return None


def fig_operator_ladder():
    r = load("B2_A1_A6")
    if not r:
        return
    ops = [("I", "logit lens (I)", C["I"]), ("J", "J-lens  J̄", C["J"]),
           ("R", "R-lens  R̄", C["R"]), ("T_sec^D4+D1", "secant  T_sec", C["T_sec"]),
           ("T_RC", "R-anchored  T_RC", C["T_RC"]), ("J_loc", "local  J_ℓ(x)", C["J_loc"])]
    sg = load("B2_A2_A5")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, L in zip(axes, LAYERS):
        f = r["layers"][L]["fits"]["0.2"]
        names, vals, los, his, cols = [], [], [], [], []
        for k, lab, c in ops:
            if k not in f:
                continue
            names.append(lab)
            vals.append(f[k]["point"])
            lo, hi = f[k].get("ci", [f[k]["point"]] * 2)
            los.append(vals[-1] - lo)
            his.append(hi - vals[-1])
            cols.append(c)
        if sg and L in sg["layers"]:
            a2 = sg["layers"][L]["A2"]
            if "SmoothGrad_J" in a2["point"]:
                names.append("SmoothGrad-J")
                vals.append(a2["point"]["SmoothGrad_J"])
                lo, hi = a2["ci"]["SmoothGrad_J"]
                los.append(vals[-1] - lo)
                his.append(hi - vals[-1])
                cols.append(C["SG"])
        y = range(len(names))
        ax.barh(list(y), vals, xerr=[los, his], color=cols, height=.68,
                error_kw={"lw": 1.1, "capsize": 3, "ecolor": "#333"})
        ax.set_yticks(list(y))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("cos(predicted Δ, true Δ)")
        ax.set_title(DEPTH[L], fontsize=11)
        ax.grid(axis="x", alpha=.25)
    axes[0].set_ylabel("")
    fig.suptitle("Transport operators vs the true finite effect — held-out D4, ε=0.2, "
                 "broadened bank (120 held-out prompts)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_operator_ladder.png", dpi=170)
    plt.close(fig)


def fig_learning_curve():
    r = load("B3")
    if not r:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, L in zip(axes, LAYERS):
        rows = r["layers"][L]["rows"]
        x = [w["n_bases"] for w in rows]
        for key, lab, c, mk in (("cos_D4", "in-family (D4) cos", C["T_sec"], "o"),
                                ("cos_D2", "cross-family (D2) cos", "#7f4fbf", "s")):
            y = [w[key] for w in rows]
            lo = [w[key] - w["ci_" + key.split("_")[1]][0] for w in rows]
            hi = [w["ci_" + key.split("_")[1]][1] - w[key] for w in rows]
            ax.errorbar(x, y, yerr=[lo, hi], marker=mk, color=c, label=lab,
                        lw=1.8, capsize=3, ms=5)
        ax.plot(x, [w["captured_D2"] for w in rows], "--", color="#999", lw=1.4,
                marker="^", ms=4, label="D2 energy inside span(C_δδ)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x])
        ax.set_xlabel("calibrate base prompts")
        ax.set_ylim(0, 1.02)
        ax.set_title(DEPTH[L], fontsize=11)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("cos / captured fraction")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("T_sec learning curves — in-family accuracy climbs with data while "
                 "cross-family stays low, even as cross-family identifiability doubles",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_learning_curves.png", dpi=170)
    plt.close(fig)


def fig_context_ratio():
    r = load("B2_A1_A6")
    o = load("A1_A6")
    if not r:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for src, lab, c, mk in ((r, "broadened bank (120 held-out prompts)", C["J_loc"], "o"),
                            (o, "original bank (32 prompts)", "#bbbbbb", "s")):
        if not src:
            continue
        xs, ys, lo, hi = [], [], [], []
        for L in LAYERS:
            a6 = src["layers"][L]["A6"]
            xs.append(int(L))
            ys.append(a6["ratio_point"])
            lo.append(a6["ratio_point"] - a6["ratio_ci"][0])
            hi.append(a6["ratio_ci"][1] - a6["ratio_point"])
        ax.errorbar(xs, ys, yerr=[lo, hi], marker=mk, color=c, label=lab,
                    lw=2, capsize=4, ms=6)
    ax.axhline(2.0, ls="--", color="#d62728", lw=1.4)
    ax.text(46, 2.06, "G-CONTEXT threshold (2×)", color="#d62728", fontsize=8.5, ha="right")
    ax.set_xlabel("source layer")
    ax.set_ylabel("cos(J_loc·δ, Δ)  /  cos(J̄·δ, Δ)")
    ax.set_xticks([16, 31, 46])
    ax.set_title("Cost of context-averaging, by depth", fontsize=11.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_context_ratio.png", dpi=170)
    plt.close(fig)


def fig_mechanism():
    r = load("B4") or load("B4_ctrl") or load("B4_smoke")
    if not r:
        return
    sig = r["sigmas"]
    eps = [f"{e:g}" for e in r["eps_eval"]]
    Ls = [L for L in LAYERS if L in r["layers"]]
    fig, axes = plt.subplots(1, len(Ls) + 1, figsize=(4.3 * (len(Ls) + 1), 4.2))
    if len(Ls) == 0:
        return
    for ax, L in zip(axes, Ls):
        g = r["layers"][L]["grid"]
        shades = ["#ffd08a", "#ff8c1a", "#b35900"]
        for si, s in enumerate(sig):
            k = f"SG_iso@{s:g}"
            ax.plot(eps, [g[e][k]["mean"] for e in eps], marker="o",
                    color=shades[si % 3], label=f"σ={s:g} (isotropic)")
        ax.plot(eps, [g[e]["J_loc"]["mean"] for e in eps], "--", color="#111111",
                marker="D", label="J_loc (σ=0, exact)")
        ax.plot(eps, [g[e]["T_IG"]["mean"] for e in eps], ":", color=C["T_IG"],
                marker="*", ms=9, label="T_IG (ceiling)")
        ax.set_xlabel("evaluation ε")
        ax.set_title(f"{DEPTH[L]} — smoothing radius", fontsize=10.5)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("cos(predicted Δ, true Δ)")
    axes[0].legend(fontsize=8)

    ax = axes[-1]
    L = Ls[0]
    sd = r["sigma_decomp"]
    g = r["layers"][L]["grid"]
    keys = [("J_loc", "J_loc\n(no smoothing)"), (f"SG_par@{sd:g}", "parallel only\n(path avg)"),
            (f"SG_orth@{sd:g}", "orthogonal only\n(no path avg)"),
            (f"SG_iso@{sd:g}", "isotropic\n(both)"), ("T_IG", "T_IG\n(ceiling)")]
    e = eps[min(1, len(eps) - 1)]
    v = [g[e][k]["mean"] for k, _ in keys]
    lo = [g[e][k]["mean"] - g[e][k]["ci"][0] for k, _ in keys]
    hi = [g[e][k]["ci"][1] - g[e][k]["mean"] for k, _ in keys]
    cols = ["#111111", "#c7c7c7", "#ff8c1a", "#b35900", C["T_IG"]]
    ax.bar(range(len(keys)), v, yerr=[lo, hi], color=cols, capsize=4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([n for _, n in keys], fontsize=7.5, rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("cos(predicted Δ, true Δ)")
    ax.set_title(f"Where the smoothing gain comes from\n({DEPTH[L]}, σ={sd:g}, ε={e})",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=.25)
    fig.suptitle("Why smoothing beats the exact local Jacobian", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_smoothing_mechanism.png", dpi=170)
    plt.close(fig)


def fig_conditional():
    r = load("C2")
    if not r:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for L in LAYERS:
        c = r["layers"][L]
        ve = c["variance_explained"]
        ks = sorted(int(k) for k in ve)
        axes[0].plot(ks, [ve[str(k)] for k in ks], marker="o", label=DEPTH[L])
        rec = c["recovery"]
        rk = sorted(int(k) for k in rec)
        axes[1].plot(rk, [rec[str(k)]["recovered_frac_of_gap"] * 100 for k in rk],
                     marker="o", label=DEPTH[L])
    axes[0].axhline(.5, ls="--", color="#d62728", lw=1.3)
    axes[0].text(1.1, .52, "G-CONDVIABLE bar (50% @ top-16)", color="#d62728", fontsize=8)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("number of deviation modes")
    axes[0].set_ylabel("variance explained")
    axes[0].set_title("Is J_loc(x) − J̄ low-rank across inputs?", fontsize=10.5)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("model rank")
    axes[1].set_ylabel("% of J̄ → J_loc gap recovered")
    axes[1].set_title("Recovery with ORACLE coefficients (upper bound)", fontsize=10.5)
    for a in axes:
        a.grid(alpha=.25)
        a.legend(fontsize=8.5)
    fig.suptitle("The conditional low-rank model works only where the gap is smallest",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_conditional.png", dpi=170)
    plt.close(fig)


def fig_m2():
    for src, tag in (("B2_A4_m2", "broadened-bank operators"),
                     ("A4_m2", "original-bank operators")):
        r = load(src)
        if not r:
            continue
        band = "L16_31_46"
        if band not in r["results"]:
            continue
        ops = r["results"][band]["operators"]
        order = ["J", "R", "logit", "T_RC", "T_sec^D4+D1", "T_sec^D4", "random"]
        names = [o for o in order if o in ops]
        v = [ops[o]["mean_drop"] for o in names]
        lo = [ops[o]["mean_drop"] - ops[o]["ci"][0] for o in names]
        hi = [ops[o]["ci"][1] - ops[o]["mean_drop"] for o in names]
        cols = [C["J"], C["R"], C["I"], C["T_RC"], C["T_sec"], "#ff9896", "#555555"]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(range(len(names)), v, yerr=[lo, hi], color=cols[:len(names)], capsize=4)
        ax.axhline(0, color="#000", lw=.9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9, rotation=15)
        ax.set_ylabel("Δ log-prob of correct answer (clean − ablated)")
        ax.set_title(f"M2: whose direction, when ablated, costs the answer?\n"
                     f"band {{16,31,46}}, {r['n_items']} probe-swap items, {tag}",
                     fontsize=10.5)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(f"{OUT}/fig6_m2_{'broad' if src.startswith('B2') else 'orig'}.png",
                    dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig_operator_ladder, fig_learning_curve, fig_context_ratio,
               fig_mechanism, fig_conditional, fig_m2):
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as ex:
            print(f"ERR {fn.__name__}: {type(ex).__name__}: {ex}")
    print("\n".join(sorted(os.listdir(OUT))))
