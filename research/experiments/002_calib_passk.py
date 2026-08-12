"""§4.4 harness validation: does the published R>J early-layer gap replicate?

Runs pass@k for the released J and R lenses on a PERMANENT CALIBRATION SLICE —
10 items per category, fixed seed. These items are burned: they are never
reported as a headline result. The remaining ~85% of each eval set stays frozen
for the final claim.

Metric, per data/evaluations/README.md: pass@k = mean over items of the fraction
of `intermediates` whose min-over-layers lens rank <= k. Reported over all
layers and over the first half (the R-lens post's two headline aggregations).

Readout position: the final prompt token for every set except poetry, which
uses the last newline token (end of line 1 of the couplet).
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ekko import harness as H  # noqa: E402

MODEL = os.environ.get("EKKO_MODEL", "Qwen/Qwen3.6-27B")
LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
OUT = os.environ.get("EKKO_OUT", "research/outputs/002_calib")
N_CALIB = int(os.environ.get("EKKO_N_CALIB", "10"))
SEED = 0
SETS = ["multihop", "multilingual", "order-ops", "poetry", "typo", "association"]


def readout_position(tok, prompt: str, slug: str) -> int:
    """Index of the scored token. Poetry: last newline. Otherwise: final token."""
    ids = tok.encode(prompt, add_special_tokens=False)
    if slug != "poetry":
        return len(ids) - 1
    nl = [i for i, t in enumerate(ids) if "\n" in tok.decode([t])]
    return nl[-1] if nl else len(ids) - 1


def target_ids(tok, intermediate) -> list[int]:
    """Single-token ids for an intermediate (a str, or a synonym list)."""
    forms = intermediate if isinstance(intermediate, list) else [intermediate]
    out = []
    for f in forms:
        if not isinstance(f, str):
            continue
        tid = H.single_token_id(tok, f)
        if tid is not None:
            out.append(tid)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    model, hf, tok = H.load_model(MODEL)
    jl = H.load_released_lens(LENS_DIR, "j")
    rl = H.load_released_lens(LENS_DIR, "r")
    tgt = jl.target_layer
    layers = list(range(tgt + 1))
    half = tgt // 2
    print(f"{model!r} target={tgt}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- collect the calibration slice ---------------------------------
    g = torch.Generator().manual_seed(SEED)
    work = []  # (slug, item_idx, prompt, pos, [ [tid,...] per intermediate ])
    for slug in SETS:
        items = json.load(open(f"data/evaluations/lens-eval-{slug}.json"))["items"]
        perm = torch.randperm(len(items), generator=g)[:N_CALIB].tolist()
        for i in perm:
            it = items[i]
            groups = [target_ids(tok, x) for x in it["intermediates"]]
            groups = [g_ for g_ in groups if g_]
            if not groups:
                continue
            work.append((slug, it["name"], it["prompt"],
                         readout_position(tok, it["prompt"], slug), groups))
    print(f"calibration slice: {len(work)} items over {len(SETS)} sets", flush=True)

    # ---- one clean forward per item, keep the residual at every layer ---
    resid = torch.zeros(len(work), len(layers), model.d_model, dtype=torch.float32)
    for n, (slug, name, prompt, pos, _) in enumerate(work):
        _, acts = H.capture(model, prompt, max_length=512)
        for li, l in enumerate(layers):
            resid[n, li] = acts[l][0, pos].float().cpu()
        if n % 20 == 0:
            print(f"  captured {n}/{len(work)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- ranks: outer loop over layers so each Jacobian moves once ------
    dev = model.input_device
    ranks = {}
    for tag, lens in (("J", jl), ("R", rl)):
        R = torch.full((len(work), len(layers), max(len(w[4]) for w in work)), 10**9,
                       dtype=torch.long)
        for li, l in enumerate(layers):
            Jm = lens.jacobians[l].to(dev, torch.float32)
            with torch.no_grad():
                logits = model.unembed(resid[:, li].to(dev) @ Jm.T).float()
                for n, w in enumerate(work):
                    for gi, tids in enumerate(w[4]):
                        t = torch.tensor(tids, device=dev)
                        best = logits[n, t].max()
                        R[n, li, gi] = int((logits[n] > best).sum())
            del Jm
            if li % 16 == 0:
                print(f"  {tag} layer {l}/{tgt}  ({time.time()-t0:.0f}s)", flush=True)
        ranks[tag] = R

    # ---- pass@k ---------------------------------------------------------
    def passk(R, k, lo, hi):
        """mean over items of the fraction of intermediates with min-rank <= k."""
        vals = []
        for n, w in enumerate(work):
            m = R[n, lo:hi, : len(w[4])].min(0).values
            vals.append((m < k).float().mean().item())
        return vals

    report = {"model": MODEL, "lens_dir": LENS_DIR, "n_items": len(work),
              "n_calib_per_set": N_CALIB, "seed": SEED, "results": {}}
    print(f"\n{'set':<14s} {'n':>3s}  {'J all':>7s} {'R all':>7s} {'Δ':>7s}   "
          f"{'J 1st½':>7s} {'R 1st½':>7s} {'Δ':>7s}", flush=True)
    for slug in SETS + ["ALL"]:
        idx = [n for n, w in enumerate(work) if slug == "ALL" or w[0] == slug]
        if not idx:
            continue
        row = {}
        for span, lo, hi in (("all", 0, len(layers)), ("first_half", 0, half)):
            for tag in ("J", "R"):
                v = passk(ranks[tag], 10, lo, hi)
                row[f"{tag}_{span}"] = sum(v[i] for i in idx) / len(idx)
        report["results"][slug] = {**row, "n": len(idx)}
        print(f"{slug:<14s} {len(idx):>3d}  {row['J_all']:>7.3f} {row['R_all']:>7.3f} "
              f"{row['R_all']-row['J_all']:>+7.3f}   {row['J_first_half']:>7.3f} "
              f"{row['R_first_half']:>7.3f} {row['R_first_half']-row['J_first_half']:>+7.3f}",
              flush=True)

    # earliest layer reaching top-10, averaged over intermediates that ever do
    for tag in ("J", "R"):
        R = ranks[tag]
        firsts = []
        for n, w in enumerate(work):
            for gi in range(len(w[4])):
                hit = (R[n, :, gi] < 10).nonzero()
                if len(hit):
                    firsts.append(int(hit[0]))
        report[f"{tag}_mean_first_top10_layer"] = sum(firsts) / max(len(firsts), 1)
        report[f"{tag}_n_ever_top10"] = len(firsts)
    print(f"\nmean first layer reaching top-10:  "
          f"J={report['J_mean_first_top10_layer']:.1f} (n={report['J_n_ever_top10']})  "
          f"R={report['R_mean_first_top10_layer']:.1f} (n={report['R_n_ever_top10']})", flush=True)

    torch.save({k: v for k, v in ranks.items()}, f"{OUT}/{LENS_DIR}_ranks.pt")
    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
