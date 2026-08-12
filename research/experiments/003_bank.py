"""Stage 1: the causal ruler — the perturbation bank.

For each site (prompt, layer, position) and each perturbation d, measure the
TRUE finite effect of the intervention by running the upper stack on h+d:

    D_one = F(h+d) - F(h)          one-sided, matches an actual activation patch
    D_odd = [F(h+d) - F(h-d)]/2    antithetic, cancels even Taylor terms

each reduced two ways: SUMMED over current-and-future positions (this is the
reduction J-bar / R-bar are estimators of, so it is the primary target), and
SELF (position p only).

Families:
  D4  natural activation deltas  d = h_l(x')[p] - h_l(x)[p] for token-aligned
      minimal pairs (same template, different argument). PRIMARY.
  D1  isotropic random directions. Identification, scale sweep, span-filling.
  D2  lens-token directions v_t = J_l^T (gamma * u_t). HELD OUT for transfer.

Minimal pairs are embedded in a pile-10k prefix so the perturbed position sits
deep in a ~128-token context, matching the distribution J/R were fitted on
(skip_first=4, max_seq_len=128). Perturbing a 10-token prompt would put every
site off-distribution for the released operators and confound the comparison.
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
OUT = os.environ.get("EKKO_OUT", "research/outputs/003_bank")
MAX_LEN = 128
BATCH = int(os.environ.get("EKKO_BATCH", "32"))
N_D1_PER_SITE = int(os.environ.get("EKKO_N_D1", "6"))
N_D2_PER_SITE = int(os.environ.get("EKKO_N_D2", "4"))
EPS_GRID = (0.01, 0.05, 0.2, 1.0)
SEED = 0


# --------------------------------------------------------------------------
def build_minimal_pairs(tok, prefix_ids: list[int]):
    """Token-aligned minimal pairs from flexible-generalization.

    Returns list of dicts: prefix + template filled with arg vs arg', kept only
    when both tokenize to the same length and differ in one contiguous block.
    """
    cats = json.load(open("data/experiments/flexible-generalization.json"))["categories"]
    pairs = []
    for cat in cats:
        for fn in cat["funcs"]:
            enc = {}
            for a in cat["args"]:
                ids = prefix_ids + tok.encode(fn["template"].replace("{arg}", a),
                                              add_special_tokens=False)
                enc[a] = ids
            lens_ = {a: len(v) for a, v in enc.items()}
            for a in cat["args"]:
                for b in cat["args"]:
                    if a == b or lens_[a] != lens_[b]:
                        continue
                    ia, ib = enc[a], enc[b]
                    diff = [i for i in range(len(ia)) if ia[i] != ib[i]]
                    if not diff:
                        continue
                    pairs.append({"cat": cat["name"], "func": fn["name"], "arg": a, "alt": b,
                                  "ids_a": ia, "ids_b": ib,
                                  "diff_lo": diff[0], "diff_hi": diff[-1], "T": len(ia)})
    return pairs


def site_positions(pair, n: int = 4) -> list[int]:
    """Positions to perturb: the changed token, and points after it."""
    lo, hi, T = pair["diff_lo"], pair["diff_hi"], pair["T"]
    cands = sorted({lo, hi, (hi + T - 1) // 2, T - 1})
    return [p for p in cands if 0 <= p < T][:n]


# --------------------------------------------------------------------------
@torch.no_grad()
def measure(model, ctx, h_clean, layer, target, pos, dirs, eps_vals, batch):
    """True finite effects for a stack of unit directions at one site.

    Args:
        h_clean: ``[1, T, d]`` clean residual at ``layer``.
        dirs: ``[n, d]`` unit directions (fp32, on device).
        eps_vals: ``[n]`` scale for each direction.
    Returns dict of ``[n, d]`` fp16 CPU tensors.
    """
    n = dirs.shape[0]
    F0 = H.forward_from(model, h_clean, layer, ctx, target=target)  # [1,T,d]
    f0_sum = F0[0, pos:].float().sum(0)
    f0_self = F0[0, pos].float()

    out = {k: torch.zeros(n, model.d_model, dtype=torch.float16)
           for k in ("d_odd_sum", "d_one_sum", "d_odd_self", "d_one_self")}
    half = max(batch // 2, 1)
    for s in range(0, n, half):
        chunk = dirs[s:s + half] * eps_vals[s:s + half, None]
        m = chunk.shape[0]
        hb = h_clean.expand(2 * m, -1, -1).clone()
        hb[:m, pos] += chunk.to(hb.dtype)
        hb[m:, pos] -= chunk.to(hb.dtype)
        Fb = H.forward_from(model, hb, layer, ctx, target=target)
        fp_sum, fm_sum = Fb[:m, pos:].float().sum(1), Fb[m:, pos:].float().sum(1)
        fp_self, fm_self = Fb[:m, pos].float(), Fb[m:, pos].float()
        out["d_odd_sum"][s:s + m] = ((fp_sum - fm_sum) / 2).half().cpu()
        out["d_one_sum"][s:s + m] = (fp_sum - f0_sum).half().cpu()
        out["d_odd_self"][s:s + m] = ((fp_self - fm_self) / 2).half().cpu()
        out["d_one_self"][s:s + m] = (fp_self - f0_self).half().cpu()
    return out


# --------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    torch.manual_seed(SEED)
    model, hf, tok = H.load_model(MODEL)
    jl = H.load_released_lens(LENS_DIR, "j")
    target = jl.target_layer
    LAYERS = [int(round(f * target)) for f in (0.25, 0.50, 0.75)]
    dev = model.input_device
    print(f"{model!r} target={target} layers={LAYERS}  ({time.time()-t0:.0f}s)", flush=True)

    # pile-10k prefix, so perturbed positions sit deep in a ~128-token context
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    doc = next(r["text"] for r in ds if len(r["text"]) > 2000)
    prefix_ids = tok.encode(doc, add_special_tokens=False)[:96]
    print(f"prefix: {len(prefix_ids)} tokens", flush=True)

    pairs = build_minimal_pairs(tok, prefix_ids)
    print(f"minimal pairs: {len(pairs)} (aligned, from flexible-generalization)", flush=True)
    if not pairs:
        raise SystemExit("no token-aligned minimal pairs; cannot build D4")

    # gamma-folded unembedding rows for D2
    gamma = model._final_norm.weight.detach().float()
    W_U = model._lm_head.weight.detach()  # [V, d]

    # base prompts = unique ids_a, so D1/D2 share operating points with D4
    bases: dict[tuple, dict] = {}
    for p in pairs:
        bases.setdefault(tuple(p["ids_a"]), {"ids": p["ids_a"], "pairs": []})["pairs"].append(p)
    print(f"base prompts: {len(bases)}", flush=True)

    manifest = {"model": MODEL, "lens_dir": LENS_DIR, "layers": LAYERS, "target": target,
                "eps_grid": EPS_GRID, "seed": SEED, "max_len": MAX_LEN,
                "prefix_tokens": len(prefix_ids), "n_base_prompts": len(bases)}

    for layer in LAYERS:
        rows, store = [], {k: [] for k in
                           ("dir", "eps", "d_odd_sum", "d_one_sum", "d_odd_self", "d_one_self")}
        norms = []
        Jm = jl.jacobians[layer].to(dev, torch.float32)
        n_done = 0
        for key, base in bases.items():
            ids = torch.tensor([base["ids"]], device=dev)
            ctx, acts = H.capture(model, ids)
            h = acts[layer]
            norms.append(h[0].float().norm(dim=-1).median().item())
            med = norms[-1]

            # alternate-prompt residuals for this base, for D4
            alt_h = {}
            for p in base["pairs"]:
                k = tuple(p["ids_b"])
                if k not in alt_h:
                    _, aacts = H.capture(model, torch.tensor([p["ids_b"]], device=dev))
                    alt_h[k] = aacts[layer]

            for pos in site_positions(base["pairs"][0]):
                dirs, epss, meta = [], [], []
                # --- D4 natural ---
                for p in base["pairs"]:
                    d = (alt_h[tuple(p["ids_b"])][0, pos] - h[0, pos]).float()
                    nn = d.norm().item()
                    if nn < 1e-6:
                        continue
                    u = d / nn
                    for eps in EPS_GRID:
                        dirs.append(u); epss.append(eps * med)
                        meta.append(("D4", p["alt"], eps, nn))
                    dirs.append(u); epss.append(nn)          # native magnitude
                    meta.append(("D4", p["alt"], -1.0, nn))
                # --- D1 isotropic: every direction at every scale, so the
                #     local-linearisation baseline and the eps->0 check are
                #     computable per direction rather than across directions ---
                for j in range(N_D1_PER_SITE):
                    u = torch.randn(model.d_model, device=dev)
                    u = u / u.norm()
                    for eps in EPS_GRID:
                        dirs.append(u); epss.append(eps * med)
                        meta.append(("D1", str(j), eps, float("nan")))
                # --- D2 lens-token (transfer only) ---
                for _ in range(N_D2_PER_SITE):
                    t = torch.randint(W_U.shape[0], (1,)).item()
                    v = Jm.T @ (gamma * W_U[t].float())
                    if v.norm() < 1e-6:
                        continue
                    dirs.append(v / v.norm()); epss.append(0.2 * med)
                    meta.append(("D2", str(t), 0.2, float("nan")))

                D = torch.stack(dirs)
                E = torch.tensor(epss, device=dev, dtype=torch.float32)
                res = measure(model, ctx, h, layer, target, pos, D, E, BATCH)
                store["dir"].append(D.half().cpu())
                store["eps"].append(E.cpu())
                for k in ("d_odd_sum", "d_one_sum", "d_odd_self", "d_one_self"):
                    store[k].append(res[k])
                for fam, tagv, eps, nat in meta:
                    rows.append({"base": n_done, "pos": pos, "family": fam,
                                 "tag": tagv, "eps": eps, "native_norm": nat,
                                 "median_h_norm": med})
            n_done += 1
            if n_done % 8 == 0:
                print(f"  L{layer}: {n_done}/{len(bases)} bases, {len(rows)} deltas "
                      f"({time.time()-t0:.0f}s)", flush=True)
        del Jm

        pack = {k: torch.cat(v) for k, v in store.items()}
        pack["meta"] = rows
        pack["median_h_norm"] = float(torch.tensor(norms).median())
        torch.save(pack, f"{OUT}/{LENS_DIR}_L{layer}.pt")
        fams = {}
        for r in rows:
            fams[r["family"]] = fams.get(r["family"], 0) + 1
        print(f"L{layer}: {len(rows)} deltas {fams} median||h||={pack['median_h_norm']:.2f} "
              f"-> {OUT}/{LENS_DIR}_L{layer}.pt  ({time.time()-t0:.0f}s)", flush=True)
        manifest[f"L{layer}"] = {"n_deltas": len(rows), "families": fams,
                                 "median_h_norm": pack["median_h_norm"]}

    with open(f"{OUT}/{LENS_DIR}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nbank done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
