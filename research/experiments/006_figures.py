"""Figures for the Stage-1/2 artifact.

F1  epsilon dose-response: cos(T*delta, Delta) vs scale, per family, per layer.
    The scientific core — where each operator stops predicting the true effect.
F2  paired bootstrap over held-out sites: Delta-cos with 95% CIs.
F3  span diagnostic: captured-fraction distribution, and cos binned by it.
    Reads as "is a transfer deficit identifiability or generalisation?"
F4  C_dd spectrum: eigenvalue decay, i.e. how rank-starved the fit is.
F5  M2 ablation accuracy (if 005 has run).

Palette: the validated categorical order, assigned by operator identity and held
fixed across every figure, so a colour always means the same operator.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
SEC = os.environ.get("EKKO_SEC", "research/outputs/004_secant")
M2 = os.environ.get("EKKO_M2", "research/outputs/005_m2")
BANK = os.environ.get("EKKO_BANK", "research/outputs/003_bank")
FIG = os.environ.get("EKKO_FIG", "research/artifacts/001-2026-08-11-secant-poc/figures")

# colour follows the entity, never its rank
C = {"I": "#8a8a85", "J": "#2a78d6", "R": "#eb6834", "J_loc": "#4a3aa7",
     "T_smooth": "#e87ba4", "T_sec": "#e34948", "T_RC": "#1baf7a", "shrink": "#008300"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
ORDER = ["I", "J", "R", "shrink", "J_loc", "T_smooth", "T_sec", "T_RC"]


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def opkey(row, name):
    if name != "shrink":
        return row.get(name)
    for k in row:
        if k.startswith("J+"):
            return row[k]
    return None


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    rep = json.load(open(f"{SEC}/{LENS_DIR}.json"))
    layers = sorted(rep["layers"], key=int)
    fams = ["D4", "D1", "D2"]

    # ---- F1 dose-response ---------------------------------------------
    rows = [f for f in fams if any(f in rep["layers"][l]["families"] for l in layers)]
    fig, axes = plt.subplots(len(rows), len(layers), figsize=(4.2 * len(layers), 3.3 * len(rows)),
                             squeeze=False, facecolor=SURF)
    for r, family in enumerate(rows):
        for c, l in enumerate(layers):
            ax = axes[r][c]
            style(ax)
            per = rep["layers"][l]["families"].get(family, {})
            xs_all = sorted([e for e in per if e != "native"], key=float)
            for name in ORDER:
                ys = [opkey(per[e], name) for e in xs_all]
                if all(y is None for y in ys):
                    continue
                xs = [float(e) for e, y in zip(xs_all, ys) if y is not None]
                yy = [y for y in ys if y is not None]
                lbl = name if name != "shrink" else "J+λI"
                ax.plot(xs, yy, "-o", color=C[name], lw=2, ms=5, label=lbl, zorder=3)
            ax.set_xscale("log")
            ax.set_ylim(-0.05, 1.02)
            if r == 0:
                ax.set_title(f"layer {l}", color=INK, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{family}\ncos(Tδ, Δ)", color=INK, fontsize=10)
            if r == len(rows) - 1:
                ax.set_xlabel("ε  (× median ‖h‖)", color=INK2, fontsize=9)
    axes[0][-1].legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=2, loc="lower left")
    fig.suptitle(f"Finite-effect prediction vs perturbation scale — {LENS_DIR}, held-out sites",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/f1_dose_response.png", dpi=170, facecolor=SURF)
    plt.close(fig)

    # ---- F2 bootstrap forest -------------------------------------------
    have = [l for l in layers if "bootstrap" in rep["layers"][l]]
    if have:
        keys = list(rep["layers"][have[0]]["bootstrap"]["paired"])
        fig, axes = plt.subplots(1, len(have), figsize=(4.6 * len(have), 3.4),
                                 squeeze=False, facecolor=SURF)
        for c, l in enumerate(have):
            ax = axes[0][c]
            style(ax)
            pr = rep["layers"][l]["bootstrap"]["paired"]
            ys = range(len(keys))
            for y, k in zip(ys, keys):
                v = pr[k]
                sig = (v["lo"] > 0) or (v["hi"] < 0)
                col = C["T_sec"] if sig else INK2
                ax.plot([v["lo"], v["hi"]], [y, y], color=col, lw=2.5, zorder=3)
                ax.plot([v["delta"]], [y], "o", color=col, ms=7, zorder=4,
                        markeredgecolor=SURF, markeredgewidth=1.5)
            ax.axvline(0, color=INK2, lw=1, ls="--", zorder=2)
            ax.set_yticks(list(ys))
            ax.set_yticklabels(keys if c == 0 else [""] * len(keys), fontsize=9, color=INK)
            ax.set_title(f"layer {l}  (n={rep['layers'][l]['bootstrap']['n_sites']} sites)",
                         color=INK, fontsize=11)
            ax.set_xlabel("Δ mean cos  (95% CI)", color=INK2, fontsize=9)
        fig.suptitle("Paired bootstrap over held-out sites — D4, ε=0.2", color=INK, fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{FIG}/f2_bootstrap.png", dpi=170, facecolor=SURF)
        plt.close(fig)

    # ---- F3 span diagnostic --------------------------------------------
    fig, axes = plt.subplots(1, len(layers), figsize=(4.4 * len(layers), 3.4),
                             squeeze=False, facecolor=SURF)
    for c, l in enumerate(layers):
        ax = axes[0][c]
        style(ax)
        bins = rep["layers"][l].get("capture_bins", {})
        labels, xs = [], []
        for family in fams:
            for b in bins.get(family, {}):
                labels.append(f"{family}\n{b}")
                xs.append((family, b))
        if not xs:
            continue
        w = 0.8 / max(len(ORDER), 1)
        for i, name in enumerate(ORDER):
            vals = [opkey(bins[f][b], name) for f, b in xs]
            if all(v is None for v in vals):
                continue
            pos = [j + i * w - 0.4 for j in range(len(xs))]
            ax.bar(pos, [v if v is not None else 0 for v in vals], width=w * 0.9,
                   color=C[name], zorder=3, label=name if c == 0 else None)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(labels, fontsize=7, color=INK2)
        ax.set_title(f"layer {l}", color=INK, fontsize=11)
        if c == 0:
            ax.set_ylabel("cos(Tδ, Δ)", color=INK, fontsize=10)
    axes[0][0].legend(frameon=False, fontsize=7, labelcolor=INK2, ncol=2)
    fig.suptitle("Span diagnostic: prediction binned by captured fraction ‖P_r δ̂‖ "
                 "— a deficit that vanishes at high capture is non-identifiability, "
                 "not non-generalisation", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/f3_span.png", dpi=170, facecolor=SURF)
    plt.close(fig)

    # ---- F4 spectrum ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor=SURF)
    style(ax)
    for i, l in enumerate(layers):
        p = f"{BANK}/{LENS_DIR}_L{l}.pt"
        if not os.path.exists(p):
            continue
        pack = torch.load(p, weights_only=False)
        X = (pack["dir"].float() * pack["eps"][:, None].float())
        ev = torch.linalg.eigvalsh(X.T @ X).flip(0).clamp_min(1e-20)
        ax.plot((ev / ev[0]).numpy(), color=list(C.values())[i], lw=2,
                label=f"layer {l}", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue index", color=INK2, fontsize=9)
    ax.set_ylabel("λ / λ₁", color=INK, fontsize=10)
    ax.set_title("C_δδ spectrum — how rank-starved the secant fit is", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(f"{FIG}/f4_spectrum.png", dpi=170, facecolor=SURF)
    plt.close(fig)

    # ---- F5 M2 ----------------------------------------------------------
    mp = f"{M2}/{LENS_DIR}.json"
    if os.path.exists(mp):
        m = json.load(open(mp))
        res = m["results"]
        names = [k for k in ORDER if k in res] + ["random_control"]
        names = [n for n in names if n in res]
        fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURF)
        style(ax)
        drops = [m["clean_acc"] - res[n]["acc_after_ablation"] for n in names]
        cols = [C.get(n, INK2) for n in names]
        ax.bar(range(len(names)), drops, color=cols, zorder=3, width=0.68)
        ctrl = m["clean_acc"] - res["random_control"]["acc_after_ablation"]
        ax.axhline(ctrl, color=INK2, lw=1.4, ls="--", zorder=4)
        ax.text(len(names) - 0.4, ctrl, " matched-norm random", color=INK2,
                fontsize=8, va="bottom", ha="right")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("random_control", "random") for n in names],
                           fontsize=9, color=INK)
        ax.set_ylabel("accuracy drop from ablating\nthe operator's direction", color=INK, fontsize=10)
        ax.set_title(f"M2 direction identification — band {m['band']}, n={m['n_items']} items",
                     color=INK, fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{FIG}/f5_m2_ablation.png", dpi=170, facecolor=SURF)
        plt.close(fig)

    print("figures ->", FIG)
    for f in sorted(os.listdir(FIG)):
        print("  ", f)


if __name__ == "__main__":
    main()
