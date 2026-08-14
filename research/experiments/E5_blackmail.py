"""E5: lens the agentic-misalignment (blackmail) scenario with J / R / H / logit.

The user's North Star: an instrument that surfaces latent personas / concerning
cognition with causal handles. This is the qualitative test bed: the public
Agentic Misalignment blackmail scenario (jlens/data/blackmail.json), the same
one the workspace paper's slice viewer demos.

Protocol:
  1. Build the chat-formatted scenario, greedy-generate the model's response
     (256 tokens) -- the deliberation happens there.
  2. One capture over prompt+response; lens readouts at the E1 grid layers for
     each operator: logit lens, released J, released R, our H.
  3. CONCERN SWEEP (meta-tokens style): a curated token list split into
     'latent' (blackmail, leverage, coerce, expose, survival, ...) and
     'in-context' (affair, email, wipe, ...). A token FIRES at (layer, pos) if
     rank <= 20. A firing is DISSOCIATED if the word does not appear in the
     +-8-token window of the text -- those are the interesting ones.
  4. QUALITATIVE DUMP: top-8 tokens per lens at L{16,31,46} every 8th response
     position, written as markdown to research/artifacts/010-blackmail-.../
     (committed to the repo by the pipeline chain, per user request).

This is exploratory/qualitative by design: it generates hypotheses about which
instrument surfaces deliberative content earliest; it does not establish
claims. Causal follow-up (ablate/steer the surfaced directions, watch the
behavior change) is the next stage, not this script.
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
E1 = os.environ.get("EKKO_E1", "/home/ubuntu/ekko/outputs/E1")
OUT = os.environ.get("EKKO_OUT", "/home/ubuntu/ekko/outputs/E5")
ART = os.environ.get(
    "EKKO_ART",
    "/home/ubuntu/ekko/jacobian-lens/research/artifacts/010-2026-08-14-blackmail-lens-samples")
GRID = [int(x) for x in os.environ.get(
    "EKKO_GRID", "8,12,16,20,24,28,31,36,40,46,52,58,60").split(",")]
DUMP_LAYERS = [16, 31, 46]
N_NEW = 256
RANK_FIRE = 20

LATENT = ["blackmail", "leverage", "coerce", "coercion", "manipulate",
          "threat", "threaten", "expose", "reveal", "secret", "pressure",
          "survival", "survive", "preserve", "protect", "deceive", "hide",
          "conceal", "power", "control", "risk", "danger", "eliminate"]
IN_CONTEXT = ["affair", "email", "wipe", "shutdown", "Kyle", "board",
              "deadline", "American", "replace"]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(ART, exist_ok=True)
    t0 = time.time()
    model, hf, tok = H.load_model(MODEL)
    dev = model.input_device
    sc = json.load(open("jlens/data/blackmail.json"))
    try:
        ids = tok.apply_chat_template(
            [{"role": "system", "content": sc["system"]},
             {"role": "user", "content": sc["user"]}],
            add_generation_prompt=True, return_tensors="pt").to(dev)
    except Exception as e:
        print(f"chat template failed ({e}); concatenating raw", flush=True)
        ids = torch.tensor([tok.encode(sc["system"] + "\n\n" + sc["user"])],
                           device=dev)
    if not torch.is_tensor(ids):        # transformers 5.x returns BatchEncoding
        ids = ids["input_ids"]
    ids = ids.to(dev)
    print(f"prompt: {ids.shape[1]} tokens  ({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        out = hf.generate(ids, max_new_tokens=N_NEW, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    r0 = ids.shape[1]
    full = out[:, : r0 + N_NEW]
    response = tok.decode(full[0, r0:], skip_special_tokens=True)
    print(f"response ({full.shape[1]-r0} toks): {response[:200]!r}", flush=True)

    torch.cuda.empty_cache()          # drop the generation KV cache
    _, acts = H.capture(model, full)
    T_seq = full.shape[1]

    jl = H.load_released_lens(LENS_DIR, "j")
    rl = H.load_released_lens(LENS_DIR, "r")
    hd = torch.load(f"{E1}/{LENS_DIR}_H_all.pt", map_location="cpu",
                    weights_only=True)
    ops = {"logit": {l: None for l in GRID},
           "J": {l: jl.jacobians[l] for l in GRID},
           "R": {l: rl.jacobians[l] for l in GRID},
           "H": {l: hd[l] for l in GRID}}

    # concern token ids (norm-corrected fold irrelevant here: full readout
    # goes through the model's own unembed, identical for every op)
    concern = {}
    for w in LATENT + IN_CONTEXT:
        tid = H.single_token_id(tok, w)
        if tid is not None:
            concern[w] = tid
    print(f"concern tokens resolved: {len(concern)}/{len(LATENT)+len(IN_CONTEXT)}",
          flush=True)
    cids = torch.tensor(list(concern.values()), device=dev)
    cnames = list(concern)

    # window text for dissociation check
    win_text = ["".join(tok.decode(full[0, max(0, p - 8): p + 8]).lower())
                for p in range(T_seq)]

    firings = {k: [] for k in ops}     # (word, layer, pos, rank, dissociated)
    topk_dump = {k: {} for k in ops}   # layer -> pos -> [top8 strings]
    CH = 512
    for name, mats in ops.items():
        for l in GRID:
            Tm = None if mats[l] is None else mats[l].to(dev, torch.float32)
            for s in range(0, T_seq, CH):
                x = acts[l][0, s: s + CH].float().to(dev)
                with torch.no_grad():
                    lg = model.unembed(x if Tm is None else x @ Tm.T).float()
                    # per-token rank loop: the broadcast form materialises a
                    # [CH, n_concern, vocab] tensor (~30 GiB at CH=512) and OOMs
                    cr = torch.stack(
                        [(lg > lg[:, int(c)].unsqueeze(-1)).sum(-1)
                         for c in cids], dim=1)
                    hit = (cr <= RANK_FIRE).nonzero()
                    for pi, ci in hit.tolist():
                        p = s + pi
                        w = cnames[ci]
                        firings[name].append(
                            {"word": w, "layer": l, "pos": p,
                             "rank": int(cr[pi, ci]),
                             "in_response": p >= r0,
                             "dissociated": w.lower() not in win_text[p]})
                    if l in DUMP_LAYERS:
                        tk = lg.topk(8, -1).indices
                        for pi in range(tk.shape[0]):
                            p = s + pi
                            if p >= r0 and (p - r0) % 8 == 0:
                                topk_dump[name].setdefault(l, {})[p] = [
                                    tok.decode([int(t)]) for t in tk[pi]]
                del lg
            del Tm
        n_dis = sum(f["dissociated"] and f["word"] in LATENT
                    for f in firings[name])
        print(f"  {name}: {len(firings[name])} firings "
              f"({n_dis} dissociated-latent)  ({time.time()-t0:.0f}s)", flush=True)

    with open(f"{OUT}/{LENS_DIR}.json", "w") as f:
        json.dump({"grid": GRID, "r0": r0, "T": T_seq, "response": response,
                   "concern_tokens": cnames, "firings": firings}, f, indent=2)

    # ---------------- markdown qualitative artifact ----------------------
    md = ["# Blackmail scenario — lens qualitative samples",
          "",
          f"Model: {MODEL}. Grid: {GRID}. Prompt {r0} toks, response "
          f"{T_seq-r0} toks, greedy. Generated response (first 1200 chars):",
          "", "```", response[:1200], "```", "",
          "A firing = concern token rank <= 20 at (layer, position). "
          "*Dissociated* = the word does not appear within +-8 tokens of that "
          "position — i.e., the lens surfaces it without surface-text support. "
          "Qualitative, hypothesis-generating only.", ""]
    md.append("## Dissociated latent firings per lens (the interesting table)\n")
    md.append("| lens | total firings | dissociated latent | distinct latent words | earliest layer |")
    md.append("|---|---|---|---|---|")
    for name in ops:
        dl = [f for f in firings[name] if f["dissociated"] and f["word"] in LATENT]
        words = sorted({f["word"] for f in dl})
        earliest = min((f["layer"] for f in dl), default="—")
        md.append(f"| {name} | {len(firings[name])} | {len(dl)} | "
                  f"{', '.join(words) if words else '—'} | {earliest} |")
    md.append("")
    for name in ops:
        dl = sorted((f for f in firings[name]
                     if f["dissociated"] and f["word"] in LATENT),
                    key=lambda f: (f["layer"], f["pos"]))[:40]
        md.append(f"### {name}: dissociated latent firings (first 40)\n")
        for f_ in dl:
            ctx = tok.decode(full[0, max(0, f_["pos"] - 6): f_["pos"] + 2]
                             ).replace("\n", "\\n")
            md.append(f"- **{f_['word']}** rank {f_['rank']} @ L{f_['layer']} "
                      f"pos {f_['pos']}{' (response)' if f_['in_response'] else ''}"
                      f" — context: `...{ctx}`")
        md.append("")
    md.append("## Top-8 readouts on response positions (every 8th)\n")
    for name in ops:
        md.append(f"### {name}\n")
        for l in DUMP_LAYERS:
            md.append(f"**L{l}**\n")
            for p, toks in sorted(topk_dump[name].get(l, {}).items()):
                ctx = tok.decode(full[0, p - 3: p + 1]).replace("\n", "\\n")
                md.append(f"- pos {p} (`...{ctx}`): "
                          + " | ".join(repr(t) for t in toks))
            md.append("")
    with open(f"{ART}/samples.md", "w") as f:
        f.write("\n".join(md))
    print(f"\nwrote {ART}/samples.md and {OUT}/{LENS_DIR}.json "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
