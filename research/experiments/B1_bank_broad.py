"""B1: broadened perturbation bank.

Same construction as 003_bank.py -- token-aligned minimal pairs on a 96-token
pile-10k prefix, eps grid + native, antithetic, summed and self targets -- but
over 10 categories x 4 templates x 6 arguments instead of 4 x 4 x 4.

Motivation (A0): the original D4 family spans only ~117 dims of 5120, so every
off-family transfer number is identifiability-limited. See expanded_templates.md.

Arguments are filtered at build time to those that keep the pair token-aligned;
drops are counted and reported, not silently absorbed.
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
OUT = os.environ.get("EKKO_OUT", "research/outputs/B1_bank")
LAYERS = [int(x) for x in os.environ.get("EKKO_LAYERS", "16,31,46").split(",")]
BATCH = int(os.environ.get("EKKO_BATCH", "32"))
N_D1 = int(os.environ.get("EKKO_N_D1", "4"))
N_D2 = int(os.environ.get("EKKO_N_D2", "3"))
MAX_BASES = int(os.environ.get("EKKO_MAX_BASES", "240"))
EPS_GRID = (0.01, 0.05, 0.2, 1.0)
SEED = 0

CATEGORIES = {
    "countries": (["France", "Canada", "China", "Egypt", "Japan", "Brazil"], [
        "The capital of {arg} is the city of",
        "Most people in {arg} speak the language called",
        "{arg} is a country on the continent of",
        "The currency used every day in {arg} is the"]),
    "months": (["January", "February", "March", "April", "June", "July"], [
        "The month that comes right after {arg} is",
        "In the northern hemisphere {arg} falls in the season of",
        "The month {arg} has a total number of days equal to",
        "A holiday that many people celebrate during {arg} is"]),
    "animals": (["dog", "cat", "horse", "sheep", "mouse", "bear"], [
        "The young of a {arg} is normally called a",
        "A {arg} is best described as an animal of the class",
        "The sound most often made by a {arg} is written as",
        "In the wild a {arg} mainly feeds on"]),
    "numbers": (["three", "four", "five", "six", "seven", "eight"], [
        "The number that comes immediately after {arg} is",
        "A polygon with {arg} sides is normally called a",
        "Doubling the number {arg} gives you the number",
        "Written as a digit, the word {arg} becomes"]),
    "colors": (["red", "blue", "green", "yellow", "black", "purple"], [
        "The colour {arg} is most often associated with",
        "Mixing the colour {arg} with plain white paint produces",
        "In French the colour {arg} is written as",
        "A traffic signal showing {arg} instructs a driver to"]),
    "elements": (["iron", "gold", "silver", "copper", "carbon", "oxygen"], [
        "The chemical symbol for {arg} is written as",
        "At room temperature the element {arg} exists as a",
        "One common industrial use of {arg} is in making",
        "The element {arg} was historically extracted from"]),
    "professions": (["doctor", "teacher", "lawyer", "farmer", "pilot", "nurse"], [
        "The place where a {arg} usually works is called a",
        "To qualify as a {arg} a person must first study",
        "A tool used every day by a {arg} is the",
        "The main responsibility of a {arg} is to"]),
    "sports": (["tennis", "soccer", "golf", "boxing", "chess", "hockey"], [
        "The playing area used for {arg} is normally called a",
        "A single game of {arg} is won by the player who",
        "The equipment most associated with {arg} is the",
        "Professional {arg} is governed internationally by the"]),
    "body": (["hand", "foot", "heart", "brain", "liver", "lung"], [
        "The main function of the human {arg} is to",
        "A doctor who specialises in the {arg} is called a",
        "The {arg} is located in the part of the body known as the",
        "Damage to the {arg} most commonly results in"]),
    "instruments": (["piano", "guitar", "violin", "drums", "flute", "trumpet"], [
        "A person who plays the {arg} is called a",
        "The {arg} belongs to the family of instruments known as",
        "Sound is produced on a {arg} by",
        "A famous piece of music written for the {arg} is"]),
}


def build_pairs(tok, prefix_ids):
    pairs, dropped = [], {"unaligned": 0, "identical": 0}
    for cat, (args, templates) in CATEGORIES.items():
        for ti, tpl in enumerate(templates):
            enc = {a: prefix_ids + tok.encode(tpl.replace("{arg}", a), add_special_tokens=False)
                   for a in args}
            ln = {a: len(v) for a, v in enc.items()}
            for a in args:
                for b in args:
                    if a == b:
                        continue
                    if ln[a] != ln[b]:
                        dropped["unaligned"] += 1
                        continue
                    ia, ib = enc[a], enc[b]
                    df = [i for i in range(len(ia)) if ia[i] != ib[i]]
                    if not df:
                        dropped["identical"] += 1
                        continue
                    pairs.append({"cat": cat, "func": f"t{ti}", "arg": a, "alt": b,
                                  "ids_a": ia, "ids_b": ib,
                                  "diff_lo": df[0], "diff_hi": df[-1], "T": len(ia)})
    return pairs, dropped


def site_positions(pair, n=3):
    lo, hi, T = pair["diff_lo"], pair["diff_hi"], pair["T"]
    return [p for p in sorted({lo, hi, (hi + T - 1) // 2, T - 1}) if 0 <= p < T][:n]


@torch.no_grad()
def measure(model, ctx, h, layer, target, pos, dirs, epsv, batch):
    n = dirs.shape[0]
    F0 = H.forward_from(model, h, layer, ctx, target=target)
    f0s, f0e = F0[0, pos:].float().sum(0), F0[0, pos].float()
    out = {k: torch.zeros(n, model.d_model, dtype=torch.float16)
           for k in ("d_odd_sum", "d_one_sum", "d_odd_self", "d_one_self")}
    half = max(batch // 2, 1)
    for s in range(0, n, half):
        ch = dirs[s:s + half] * epsv[s:s + half, None]
        m = ch.shape[0]
        hb = h.expand(2 * m, -1, -1).clone()
        hb[:m, pos] += ch.to(hb.dtype)
        hb[m:, pos] -= ch.to(hb.dtype)
        Fb = H.forward_from(model, hb, layer, ctx, target=target)
        ps, ms_ = Fb[:m, pos:].float().sum(1), Fb[m:, pos:].float().sum(1)
        pe, me = Fb[:m, pos].float(), Fb[m:, pos].float()
        out["d_odd_sum"][s:s + m] = ((ps - ms_) / 2).half().cpu()
        out["d_one_sum"][s:s + m] = (ps - f0s).half().cpu()
        out["d_odd_self"][s:s + m] = ((pe - me) / 2).half().cpu()
        out["d_one_self"][s:s + m] = (pe - f0e).half().cpu()
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    torch.manual_seed(SEED)
    model, hf, tok = H.load_model(MODEL)
    jl = H.load_released_lens(LENS_DIR, "j")
    target = jl.target_layer
    dev = model.input_device
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    doc = next(r["text"] for r in ds if len(r["text"]) > 2000)
    prefix_ids = tok.encode(doc, add_special_tokens=False)[:96]

    pairs, dropped = build_pairs(tok, prefix_ids)
    bases: dict[tuple, dict] = {}
    for p in pairs:
        bases.setdefault(tuple(p["ids_a"]), {"ids": p["ids_a"], "pairs": []})["pairs"].append(p)
    keys = list(bases)[:MAX_BASES]
    bases = {k: bases[k] for k in keys}
    print(f"pairs={len(pairs)} dropped={dropped} bases={len(bases)} "
          f"(cap {MAX_BASES})  ({time.time()-t0:.0f}s)", flush=True)

    gamma = model._final_norm.weight.detach().float()
    W_U = model._lm_head.weight.detach()
    man = {"model": MODEL, "lens_dir": LENS_DIR, "layers": LAYERS, "target": target,
           "eps_grid": EPS_GRID, "seed": SEED, "prefix_tokens": len(prefix_ids),
           "n_base_prompts": len(bases), "n_pairs": len(pairs), "dropped": dropped,
           "categories": {c: len(v[0]) for c, v in CATEGORIES.items()},
           "n_templates": sum(len(v[1]) for v in CATEGORIES.values())}

    for layer in LAYERS:
        rows, store = [], {k: [] for k in
                           ("dir", "eps", "d_odd_sum", "d_one_sum", "d_odd_self", "d_one_self")}
        norms = []
        Jm = jl.jacobians[layer].to(dev, torch.float32)
        for n_done, (key, bp) in enumerate(bases.items()):
            ids = torch.tensor([bp["ids"]], device=dev)
            ctx, acts = H.capture(model, ids)
            h = acts[layer]
            med = h[0].float().norm(dim=-1).median().item()
            norms.append(med)
            alt = {}
            for p in bp["pairs"]:
                k = tuple(p["ids_b"])
                if k not in alt:
                    _, aa = H.capture(model, torch.tensor([p["ids_b"]], device=dev))
                    alt[k] = aa[layer]
            for pos in site_positions(bp["pairs"][0]):
                dirs, epss, meta = [], [], []
                for p in bp["pairs"]:
                    dvec = (alt[tuple(p["ids_b"])][0, pos] - h[0, pos]).float()
                    nn = dvec.norm().item()
                    if nn < 1e-6:
                        continue
                    u = dvec / nn
                    for e in EPS_GRID:
                        dirs.append(u); epss.append(e * med); meta.append(("D4", p["alt"], e, nn))
                    dirs.append(u); epss.append(nn); meta.append(("D4", p["alt"], -1.0, nn))
                for j in range(N_D1):
                    u = torch.randn(model.d_model, device=dev)
                    u = u / u.norm()
                    for e in EPS_GRID:
                        dirs.append(u); epss.append(e * med)
                        meta.append(("D1", f"{n_done}_{j}", e, float("nan")))
                for j in range(N_D2):
                    t = torch.randint(W_U.shape[0], (1,)).item()
                    v = Jm.T @ (gamma * W_U[t].float())
                    if v.norm() < 1e-6:
                        continue
                    dirs.append(v / v.norm()); epss.append(0.2 * med)
                    meta.append(("D2", str(t), 0.2, float("nan")))
                D = torch.stack(dirs)
                E = torch.tensor(epss, device=dev, dtype=torch.float32)
                res = measure(model, ctx, h, layer, target, pos, D, E, BATCH)
                store["dir"].append(D.half().cpu()); store["eps"].append(E.cpu())
                for k in ("d_odd_sum", "d_one_sum", "d_odd_self", "d_one_self"):
                    store[k].append(res[k])
                for f_, tg, e, nat in meta:
                    rows.append({"base": n_done, "pos": pos, "family": f_, "tag": tg,
                                 "eps": e, "native_norm": nat, "median_h_norm": med})
            if (n_done + 1) % 20 == 0:
                print(f"  L{layer}: {n_done+1}/{len(bases)} bases, {len(rows)} deltas "
                      f"({time.time()-t0:.0f}s)", flush=True)
        del Jm
        pack = {k: torch.cat(v) for k, v in store.items()}
        pack["meta"] = rows
        pack["median_h_norm"] = float(torch.tensor(norms).median())
        torch.save(pack, f"{OUT}/{LENS_DIR}_L{layer}.pt")
        fams = {}
        for r in rows:
            fams[r["family"]] = fams.get(r["family"], 0) + 1
        print(f"L{layer}: {len(rows)} deltas {fams} -> {OUT}/{LENS_DIR}_L{layer}.pt "
              f"({time.time()-t0:.0f}s)", flush=True)
        man[f"L{layer}"] = {"n_deltas": len(rows), "families": fams,
                            "median_h_norm": pack["median_h_norm"]}
        with open(f"{OUT}/{LENS_DIR}_manifest.json", "w") as f:
            json.dump(man, f, indent=2)
    print(f"\nB1 bank done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
