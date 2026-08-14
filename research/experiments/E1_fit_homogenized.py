"""E1: fit the fully-homogenized lens (O1) -- the F11 theory's sharpest prediction.

THEORY (FINDINGS.md F11). Each R-lens rule is exactly the linear operator that
reproduces its component's output when applied to the full activation (LRP
conservation): identity-rule sigma(z)*z = SiLU(z); LN-rule diag(1+w)/r * x =
RMSNorm(x); half-rule = Euler's identity for the bilinear gate. R leaves two
components un-homogenized: attention routing and the GDN gated output. Their
output-reproducing forms are frozen-A (O = A.detach() @ V(x) is exact, V linear)
and the output half-rule on the GDN gated norm. Both won at eps=1.0 on natural
deltas in D3 (4/4 and 2/2 conditions).

PRE-REGISTERED PREDICTION: the H-lens (R + frozen-softmax + gdn_out_half),
fitted full-stack at the released convention, beats R on first-half pass@10 on
the frozen eval sets, evaluated on MATCHED layer grids. Decision rule lives in
E2; this script only fits. A negative is informative: it localizes where
composition (residual adds, cross-position mixing) breaks the conservation
account.

VARIANTS fitted here, each as two disjoint corpus halves + merge (twin floor):
  H      R baseline on every block + global softmax detach (eager attention
         required so F.softmax is the code path) + half-rule on the GDN gated
         output norm.
  R      the R recipe alone, same corpus, same convention -- the control that
         separates "the new rules helped" from fit-instance noise, and the
         full-stack theta_R endpoint check the project never ran (conformance
         gap A.3.6).

Both variants run under eager attention so their forward numerics match.

GATES before any fitting (the section-3.2 test that was never written):
  1. forward invariance: patched-eager logits == clean-eager logits (top-1
     agreement over all positions, max |delta| recorded; hard-fail < 98%).
  2. probe finiteness: the config probe's Jacobian rows must be finite.

Config is chosen by an OOM/time ladder probed on one prompt (project rule:
"time 3 prompts and multiply" -- here one full prompt, which is the unit the
fit repeats). Memory driver is the earliest source layer (graph depth), not
the number of source layers (they share the same backward passes).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ekko import harness as H  # noqa: E402
from ekko import rules as RU  # noqa: E402
import jlens  # noqa: E402

MODEL = os.environ.get("EKKO_MODEL", "Qwen/Qwen3.6-27B")
LENS_DIR = os.environ.get("EKKO_LENS_DIR", "qwen3.6-27b")
OUT = os.environ.get("EKKO_OUT", "/home/ubuntu/ekko/outputs/E1")
N_PROMPTS = int(os.environ.get("EKKO_N_PROMPTS", "25"))
MAX_SEQ = 128
SKIP_FIRST = 4
VARIANTS = os.environ.get("EKKO_VARIANTS", "H,R").split(",")
SMOKE = os.environ.get("EKKO_SMOKE", "0") == "1"
MAX_HOURS_PER_VARIANT = float(os.environ.get("EKKO_MAX_HOURS", "8"))
MEM_HEADROOM_GIB = 5.0

# ambitious -> safe; every rung keeps the bank layers {16, 31, 46}
LADDER = [
    ([8, 12, 16, 20, 24, 28, 31, 36, 40, 46, 52, 58, 60], 4),
    ([12, 16, 20, 24, 28, 31, 36, 42, 46, 54, 60], 4),
    ([16, 20, 24, 28, 31, 38, 46, 54, 60], 4),
    ([16, 31, 46], 4),
    ([31], 8),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")


# --------------------------------------------------------------------------
# rule installation for a whole model, for the duration of a `with` block
# --------------------------------------------------------------------------
@contextlib.contextmanager
def frozen_softmax():
    """cp_value_only, stack-wide: forward unchanged, attention pattern detached.

    Only attention calls F.softmax in this stack (GDN uses the delta rule; the
    readout softmax is outside the fitted map), so a global patch is exactly
    the per-block rule applied everywhere. Requires eager attention.
    """
    orig = torch.nn.functional.softmax

    def sm(x, *a, **k):
        return orig(x, *a, **k).detach()

    torch.nn.functional.softmax = sm
    try:
        yield
    finally:
        torch.nn.functional.softmax = orig


@contextlib.contextmanager
def install_variant(model, variant: str):
    """'R' = R recipe on every block. 'H' = R + frozen softmax + gdn_out_half."""
    assert variant in ("R", "H"), variant
    with contextlib.ExitStack() as stack:
        for blk in model.layers:
            stack.enter_context(RU._patch_r_baseline(blk))
            if variant == "H":
                gdn = getattr(blk, "linear_attn", None)
                gn = getattr(gdn, "norm", None) if gdn is not None else None
                if gn is not None:
                    stack.enter_context(RU._patch_branch_half(gn))
        if variant == "H":
            stack.enter_context(frozen_softmax())
        yield


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
@torch.no_grad()
def _all_logits(model, ids):
    with jlens.ActivationRecorder(model.layers, at=[model.n_layers - 1]) as rec:
        model.forward(ids)
        h = rec.activations[model.n_layers - 1]
    return model.unembed(h[0]).float()


def gate_forward_invariance(model, prompts, variant):
    worst_top1, worst_abs = 1.0, 0.0
    for p in prompts[:3]:
        ids = model.encode(p, max_length=MAX_SEQ)
        clean = _all_logits(model, ids)
        with install_variant(model, variant):
            patched = _all_logits(model, ids)
        agree = float((clean.argmax(-1) == patched.argmax(-1)).float().mean())
        mx = float((clean - patched).abs().max())
        worst_top1, worst_abs = min(worst_top1, agree), max(worst_abs, mx)
    print(f"  [{variant}] forward invariance: top-1 agreement {worst_top1:.4f}, "
          f"max|dlogit| {worst_abs:.4f}", flush=True)
    if worst_top1 < 0.98:
        raise RuntimeError(
            f"forward NOT invariant under {variant} patches (top-1 {worst_top1:.3f}) "
            f"-- a rule is changing the forward; do not fit.")
    return {"top1_agreement": worst_top1, "max_abs_dlogit": worst_abs}


def probe_config(model, prompt, variant, target):
    """First ladder rung that fits in memory and the hour budget."""
    for sources, dim_batch in LADDER:
        sources = [s for s in sources if s < target]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            with install_variant(model, variant):
                J, _, _ = jlens.fitting.jacobian_for_prompt(
                    model, prompt, sources, target_layer=target,
                    dim_batch=dim_batch, max_seq_len=MAX_SEQ,
                    skip_first=SKIP_FIRST)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  probe sources[0]={sources[0]} db={dim_batch}: OOM -> next rung",
                  flush=True)
            continue
        secs = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        free = 80.0 - peak
        for l in sources:
            if not torch.isfinite(J[l]).all():
                raise RuntimeError(f"non-finite Jacobian rows at L{l} under {variant}")
        proj_h = secs * N_PROMPTS / 3600
        print(f"  probe sources={sources} db={dim_batch}: {secs:.0f}s/prompt, "
              f"peak {peak:.1f} GiB, projected {proj_h:.1f} h/variant", flush=True)
        if free < MEM_HEADROOM_GIB:
            print("    headroom too small -> next rung", flush=True)
            continue
        if proj_h > MAX_HOURS_PER_VARIANT:
            print("    over the hour budget -> next rung", flush=True)
            continue
        return sources, dim_batch, secs, peak
    raise RuntimeError("no ladder rung fits")


# --------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    model, hf, tok = H.load_model(MODEL)
    # eager attention: required for the softmax patch to be the code path, and
    # used for BOTH variants so their forward numerics match (D3 pattern).
    try:
        hf.config._attn_implementation = "eager"
        hf.set_attn_implementation("eager")
        print("attention implementation: eager", flush=True)
    except Exception as e:
        print(f"could not force eager attention: {e}", flush=True)

    if SMOKE:
        target = model.n_layers - 2
        n_prompts = 2
        global LADDER
        mid = model.n_layers // 2
        LADDER = [([mid], 4)]
    else:
        jl = H.load_released_lens(LENS_DIR, "j")
        target = jl.target_layer
        n_prompts = N_PROMPTS

    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    prompts, it = [], iter(ds)
    while len(prompts) < n_prompts:
        r = next(it)
        if len(r["text"]) > 2000:
            prompts.append(r["text"][:4000])
    print(f"{MODEL} target={target} corpus={len(prompts)} pile-10k docs "
          f"({time.time()-t0:.0f}s)", flush=True)

    rep = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git,
           "model": MODEL, "target_layer": target, "n_prompts": n_prompts,
           "max_seq_len": MAX_SEQ, "skip_first": SKIP_FIRST, "smoke": SMOKE,
           "attn_implementation": "eager", "variants": {}}

    for variant in VARIANTS:
        print(f"\n{'='*74}\nVARIANT {variant}  ({time.time()-t0:.0f}s)", flush=True)
        v = {"gate": gate_forward_invariance(model, prompts, variant)}
        if os.environ.get("EKKO_SOURCES"):
            sources = [int(x) for x in os.environ["EKKO_SOURCES"].split(",")]
            dim_batch = int(os.environ.get("EKKO_DIMB", "4"))
            v.update({"source_layers": sources, "dim_batch": dim_batch,
                      "probe": "skipped (config from env)"})
        else:
            sources, dim_batch, probe_s, peak = probe_config(
                model, prompts[0], variant, target)
            v.update({"source_layers": sources, "dim_batch": dim_batch,
                      "probe_sec_per_prompt": probe_s, "probe_peak_gib": peak})

        halves = {"halfA": prompts[0::2], "halfB": prompts[1::2]}
        only = os.environ.get("EKKO_HALF", "")
        if only in ("A", "B"):
            halves = {f"half{only}": halves[f"half{only}"]}
        lens_objs = {}
        with install_variant(model, variant):
            for name, pr in halves.items():
                t1 = time.time()
                torch.cuda.empty_cache()
                lens = jlens.fit(
                    model, pr, source_layers=sources, target_layer=target,
                    dim_batch=dim_batch, max_seq_len=MAX_SEQ,
                    skip_first=SKIP_FIRST,
                    checkpoint_path=f"{OUT}/ckpt_{variant}_{name}.pt",
                    checkpoint_every=2)
                dt = time.time() - t1
                lens_objs[name] = lens
                torch.save({l: lens.jacobians[l].detach().cpu().half()
                            for l in sources},
                           f"{OUT}/{LENS_DIR}_{variant}_{name}.pt")
                v[name] = {"n_prompts": int(lens.n_prompts), "seconds": dt,
                           "sec_per_prompt": dt / max(int(lens.n_prompts), 1)}
                print(f"  [{variant}/{name}] {lens.n_prompts} prompts in "
                      f"{dt/60:.1f} min ({time.time()-t0:.0f}s)", flush=True)

        if len(lens_objs) == 2:
            merged = jlens.JacobianLens.merge(
                [lens_objs["halfA"], lens_objs["halfB"]])
            torch.save({l: merged.jacobians[l].detach().cpu().half()
                        for l in sources},
                       f"{OUT}/{LENS_DIR}_{variant}_all.pt")
            v["all"] = {"n_prompts": int(merged.n_prompts)}
        else:
            merged = next(iter(lens_objs.values()))
            v["all"] = {"note": f"single half only (n={int(merged.n_prompts)}); "
                                "no merge"}

        if not SMOKE and len(lens_objs) == 2:
            jrel = H.load_released_lens(LENS_DIR, "j")
            rrel = H.load_released_lens(LENS_DIR, "r")
            cmpd = {}
            for l in (16, 31, 46):
                if l not in merged.jacobians:
                    continue
                T = merged.jacobians[l].float()
                A, B = lens_objs["halfA"].jacobians[l].float(), \
                    lens_objs["halfB"].jacobians[l].float()
                cmpd[str(l)] = {
                    "vs_released_J_relF": float((T - jrel.jacobians[l].float()).norm()
                                                / jrel.jacobians[l].float().norm()),
                    "vs_released_R_relF": float((T - rrel.jacobians[l].float()).norm()
                                                / rrel.jacobians[l].float().norm()),
                    "twin_floor_relF": float((A - B).norm() / B.norm()),
                    "diag_mean": float(torch.diagonal(T).mean())}
                print(f"  L{l}: |T-Jrel|/|Jrel|={cmpd[str(l)]['vs_released_J_relF']:.3f} "
                      f"|T-Rrel|/|Rrel|={cmpd[str(l)]['vs_released_R_relF']:.3f} "
                      f"twin={cmpd[str(l)]['twin_floor_relF']:.3f}", flush=True)
            v["matrix_comparison"] = cmpd

        rep["variants"][variant] = v
        with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
            json.dump(rep, f, indent=2)
        del lens_objs, merged
        torch.cuda.empty_cache()

    print(f"\nE1 done -> {OUT}/{LENS_DIR}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
