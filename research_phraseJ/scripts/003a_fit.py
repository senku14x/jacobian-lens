"""003a (fit stage) — per-context phrase gradients for four objectives (design: research_artifacts/003a-phrase-objective/design.md).

Items:
  primary phrases  : New Zealand, South Korea, San Diego, North Carolina (latent-evaluable) + ice cream, credit card (emission-only)
                     regimes: natural (first N_FIT emission-natural contexts) and generic-medium (pile positions in the middle
                     surprisal tercile of 300 sampled teacher-forced positions)
  siblings         : other members of the four geographic families, natural regime only (first N_FIT contexts)
  nulls            : per primary phrase, NULLS_PER surprisal-matched 2-token non-name bigrams (suffix log p within +-0.3 nats of
                     the phrase's natural-regime mean where possible, else closest), fitted after that phrase's natural contexts;
                     named "null:<bigram>@<phrase>" (unique per phrase); plus one shared junk pair
Objectives per token (batch 1, retained graph): logp, lin, logit, odds. Both reductions (at t', source-mean) saved per context.
Checkpointed per item: outputs/003a/grads/<item>__<regime>.pt; resumes by skipping existing files. If results/003a-.../{items,contexts}.json
exist, the item and context lists are reused (context building is ~30 min of forwards); set REBUILD_CONTEXTS=1 to rebuild.
Results: results/003a-phrase-objective/{items.json, contexts.json, surprisal.json}
"""
import json, os, sys, time, random, re, logging
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.ekko_harness import load_model, capture
from lib.phrase_objective import PhraseJ
from datasets import load_dataset
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
for noisy in ("httpx", "huggingface_hub", "datasets", "urllib3", "filelock", "fsspec"): logging.getLogger(noisy).setLevel(logging.WARNING)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "003a-phrase-objective"); OUTD = os.path.join(ROOT, "outputs", "003a"); GR = os.path.join(OUTD, "grads")
for d in (RES, OUTD, GR): os.makedirs(d, exist_ok=True)
RES1 = os.path.join(ROOT, "results", "001-template-geometry")
MODEL = "/content/models/Qwen3.6-27B"
LAYERS = [36, 44, 52, 56, 60]; TARGET = 62; SKIP = 4; SEQ = 128
N_FIT = int(os.environ.get("N_FIT", 20)); N_SIB = int(os.environ.get("N_SIB", 20)); N_GEN_CAND = 300; NULLS_PER = 2
PRIMARY = ["New Zealand", "South Korea", "San Diego", "North Carolina", "ice cream", "credit card"]
LATENT_FAMS = {"New Zealand": "New", "South Korea": "South", "San Diego": "San", "North Carolina": "North"}
OBJ = ("logp", "lin", "logit", "odds")
NULL_CANDS = ["of the", "in the", "to the", "on the", "at the", "for the", "and the", "with a", "as well", "such as", "one of",
              "more than", "has been", "have been", "it is", "there is", "this is", "in order", "at least", "as a",
              "part of", "out of", "due to", "in which", "so that", "as the", "from the", "by the", "into the", "over the"]
JUNK = [" spider", "ünd"]
random.seed(0); torch.manual_seed(0)

model, hf, tok = load_model(MODEL)
pj = PhraseJ(model, LAYERS, TARGET, SKIP)
gamma = model._final_norm.weight.detach().float(); W_U = hf.lm_head.weight.detach()
def q_of(t): return (1 + gamma) * W_U[t].float()
fam = json.load(open(f"{RES1}/families.json")); ctx1 = json.load(open(f"{RES1}/contexts.json"))
members = {m["phrase"]: m for F in fam["families"] for m in F["members"]}
fam_of = {m["phrase"]: F for F in fam["families"] for m in F["members"]}
nat_ctx = {}
for c in ctx1["emission_natural"]: nat_ctx.setdefault(c["phrase"], []).append(c["text"])

@torch.no_grad()
def phrase_logps(text_ids, pids):
    full = torch.cat([text_ids, torch.tensor([pids], device=text_ids.device)], 1)
    _, a = capture(model, full, max_length=full.shape[1])
    h = a[model.n_layers - 1][0, text_ids.shape[1] - 1: full.shape[1] - 1]
    lp = torch.log_softmax(model.unembed(h).float(), -1)
    return [float(lp[i, w]) for i, w in enumerate(pids)]

def build_items():
    items = []  # (name, token_ids, prefix_len, kind, regime, contexts)
    for ph in PRIMARY:
        items.append((ph, members[ph]["ids"], fam_of[ph]["prefix_len"], "primary", "natural", nat_ctx[ph][:N_FIT]))
    for ph, fname in LATENT_FAMS.items():
        for s in fam_of[ph]["members"]:
            if s["phrase"] == ph or s["n_tokens"] <= fam_of[ph]["prefix_len"]: continue
            cs = nat_ctx.get(s["phrase"], []); n_fit = min(N_SIB, len(cs) // 2 if len(cs) < 2 * N_SIB else N_SIB)
            if n_fit >= 5: items.append((s["phrase"], s["ids"], fam_of[ph]["prefix_len"], "sibling", "natural", cs[:n_fit]))
    # generic-medium contexts (surprisal binned) for primaries
    natural_docs = set(json.load(open(f"{RES1}/natural_doc_ids.json"))); sigma_docs = set(json.load(open(f"{RES1}/sigma_meta.json"))["seq_ids"])
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    pool = [i for i in range(len(ds)) if i not in natural_docs and i not in sigma_docs]; random.shuffle(pool)
    surpr = {}; gen_ctx = {}; t0 = time.time()
    for ph in PRIMARY:
        pids = members[ph]["ids"]; cands = []
        for di in pool:
            if len(cands) >= N_GEN_CAND: break
            ids = tok.encode(ds[di]["text"], add_special_tokens=False)
            if len(ids) < 64: continue
            cut = random.randint(32, min(len(ids), SEQ)); left = ids[max(0, cut - SEQ):cut]
            x = torch.tensor([left], device=model.input_device)
            lps = phrase_logps(x, pids); cands.append({"doc": di, "cut": cut, "text": tok.decode(left), "logps": lps, "mean": float(np.mean(lps))})
        ms = np.array([c["mean"] for c in cands]); lo, hi = np.percentile(ms, [33.3, 66.7])
        mid = [c for c in cands if lo <= c["mean"] <= hi]; random.shuffle(mid); chosen = mid[:N_FIT]
        surpr[ph] = {"n_cand": len(cands), "tercile_bounds": [float(lo), float(hi)], "mean_all": float(ms.mean()),
                     "chosen_mean_surprisal": float(-np.mean([c["mean"] for c in chosen])), "chosen_suffix_logp": float(np.mean([c["logps"][-1] for c in chosen]))}
        gen_ctx[ph] = chosen
        items.append((ph, pids, fam_of[ph]["prefix_len"], "primary", "generic_medium", [c["text"] for c in chosen]))
        logging.info(f"generic contexts {ph}: {len(cands)} cands, tercile [{lo:.2f},{hi:.2f}], chosen {len(chosen)} ({time.time()-t0:.0f}s)")
    # surprisal-matched nulls (after each primary's natural contexts)
    null_cands = {nc: tok.encode(" " + nc, add_special_tokens=False) for nc in NULL_CANDS}
    null_cands = {nc: ids for nc, ids in null_cands.items() if len(ids) == 2}
    phrase_suffix_lp = {}; null_lp = {}
    for ph in PRIMARY:
        cs = nat_ctx[ph][:N_FIT]; pids = members[ph]["ids"]
        xs = [model.encode(c, max_length=SEQ) for c in cs[:10]]          # matching uses the first 10 contexts (cost); fitting uses all N_FIT
        phrase_suffix_lp[ph] = float(np.mean([phrase_logps(x, pids)[-1] for x in xs]))
        null_lp[ph] = {nc: float(np.mean([phrase_logps(x, nids)[-1] for x in xs])) for nc, nids in null_cands.items()}
        ranked = sorted(null_lp[ph].items(), key=lambda kv: abs(kv[1] - phrase_suffix_lp[ph]))
        for nc, lp in ranked[:NULLS_PER]:
            items.append((f"null:{nc}@{ph}", null_cands[nc], 1, f"null_for:{ph}", "natural", cs))
            logging.info(f"null for {ph} (suffix logp {phrase_suffix_lp[ph]:+.2f}): '{nc}' {lp:+.2f} {'MATCHED' if abs(lp-phrase_suffix_lp[ph])<=0.3 else 'closest-only'}")
    jids = [tok.encode(JUNK[0], add_special_tokens=False)[0], tok.encode(JUNK[1], add_special_tokens=False)[0]]
    items.append(("null:junk", jids, 1, "null_junk", "natural", nat_ctx["New Zealand"][:N_FIT]))
    json.dump({"phrase_suffix_logp_natural": phrase_suffix_lp, "null_suffix_logp": null_lp, "generic": surpr}, open(f"{RES}/surprisal.json", "w"), indent=1)
    json.dump([{"name": n, "ids": i, "tokens": [tok.decode([t]) for t in i], "prefix_len": k, "kind": kd, "regime": r, "n_contexts": len(cs)} for n, i, k, kd, r, cs in items], open(f"{RES}/items.json", "w"), indent=1)
    json.dump({f"{n}__{r}": cs for n, i, k, kd, r, cs in items} | {"generic_meta": {ph: [{k: v for k, v in c.items() if k != "text"} for c in gen_ctx[ph]] for ph in PRIMARY}}, open(f"{RES}/contexts.json", "w"), indent=1)
    return items

if os.path.exists(f"{RES}/items.json") and os.path.exists(f"{RES}/contexts.json") and os.environ.get("REBUILD_CONTEXTS", "0") != "1":
    saved_items = json.load(open(f"{RES}/items.json")); saved_ctx = json.load(open(f"{RES}/contexts.json"))
    items = [(it["name"], it["ids"], it["prefix_len"], it["kind"], it["regime"], saved_ctx[f"{it['name']}__{it['regime']}"]) for it in saved_items]
    logging.info(f"resumed {len(items)} items from saved items.json/contexts.json")
else:
    items = build_items()
logging.info(f"{len(items)} items, {sum(len(x[-1]) for x in items)} contexts, {sum(len(x[-1])*len(x[1])*len(OBJ) for x in items)} backwards")

# ---------------------------------------------------------------- gradients (checkpoint per item, resume)
def safe(s): return re.sub(r"[^A-Za-z0-9]+", "_", s)
t0 = time.time()
for n, (name, pids, k, kind, regime, cs) in enumerate(items):
    path = f"{GR}/{safe(name)}__{regime}.pt"
    if os.path.exists(path): logging.info(f"skip {name} [{regime}] (exists)"); continue
    recs = []
    for ci, text in enumerate(cs):
        x = model.encode(text, max_length=SEQ)
        if x.shape[1] < SKIP + 4: continue
        out, lps, tprime = pj.per_token_multi(x, pids, OBJ, q_of=q_of)
        rec = {"ctx": ci, "tprime": int(tprime), "logps": lps, "obj": {}}
        for o in OBJ:
            rec["obj"][o] = {l: {"at": torch.stack([out[o][i][l][0] for i in range(len(pids))]).half(),
                                 "mean": torch.stack([out[o][i][l][1] for i in range(len(pids))]).half()} for l in LAYERS}
        recs.append(rec)
    torch.save({"name": name, "ids": pids, "prefix_len": k, "kind": kind, "regime": regime, "layers": LAYERS, "objectives": list(OBJ), "recs": recs}, path)
    logging.info(f"[{n+1}/{len(items)}] {name} [{regime}] n={len(recs)} ({time.time()-t0:.0f}s)")
logging.info("done")
