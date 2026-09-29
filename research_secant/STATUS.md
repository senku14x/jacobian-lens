# STATUS — research_secant (what is running, why, and what we expect)

Updated 2026-09-29. Branch `secant` of `/workspace/ekko-lens`. Layout and conventions are in `README.md`.

## The big picture in three sentences

Your spec's central tool is a **matrix-free J-lens**: compute "J̄ times a vector" directly from forward passes on
ordinary text, instead of using the stored 5120×5120 J matrix. Once that tool reproduces the released lens exactly,
the *definition* of the lens can be changed at no cost, with no refit. Examples: count only effects 9–16 tokens ahead,
count only effects inside this sequence's horizon, or use chat text instead of pile text. Today's runs are (000)
proving the tool is exact, and (001) using it for the cheapest decisive test in your spec: whether the J-lens misses
planned rhyme words only because its definition is dominated by short-range effects.

Nothing is being fitted. Every quantity is a forward-pass average over host documents.

---

## 000 — Is the matrix-free lens exactly J_CB? DONE, all gates pass

- **Design:** `designs/000-matrix-free-lens.md`, committed before the run (04c69ba).
- **Report:** `results/000-matrix-free-lens/report.md`.

**Why.** Every later quantity (horizon gap, lag buckets, secant lens) is built on this tool. If it didn't reproduce the
released lens, every gap downstream would include a definitional mismatch. Your spec called this "the single most
important check".

**What it does.** For each test vector v (random, a real activation, or a difference of activations) at layers
40/48/56:
- add +εv and −εv at every valid position of a host document;
- run the rest of the network in true fp32;
- difference the summed block-62 outputs, and average over documents;
- compare against J_CB·v.

**What we expected (pre-registered).**
- ε and ε/2 agree (90% prior).
- It matches J_CB at cos ≥ 0.99 if we guess the right 25 fit documents (60% prior, since the checkpoint doesn't
  record which docs were used).

**What we found.**
- Precision gate: pass (cos 0.9999996+).
- **It reproduces J_CB at cos 0.997 / 0.998 / 0.999** (L40/48/56), with norm ratio ≈ 1.00, on the first 25 pile-10k
  docs with no BOS. That also identifies J_CB's undocumented fit corpus.
- On 31 fresh documents it gets 0.97–0.995. That's the sampling floor for any later comparison.
- Side result: the spec's original plan (finite differences at ε = 1e-3 under RHUT's bf16-matmul regime) gives
  cos 0.80. It would have been ~20% wrong in direction, which is why the true-fp32 upper stack is needed.

---

## 001 — Lag-bucket lens on planned rhymes (your spec II.5): RUNNING

- **Started** 2026-09-29 ~23:55; about 30 s per item, so ~50 min for 100 items.
- **Design:** `designs/001-lag-bucket-poetry.md`, committed before the run (b39d380).
- **Script:** `scripts/001_lag_bucket_poetry.py`.
- **Output** (live, resumable): `results/001-lag-bucket-poetry/ranks.jsonl`. **Log:** `outputs/logs/001.log`.

**Why.**
- On WorkspaceBench's poetry family, the J-lens almost never shows the rhyme word the model has planned (pass
  ≈ .07). The prompt is "A rhyming couplet: …route ahead,⏎ And told his crew to follow where he'd", with target `led`,
  read at the ⏎.
- J̄ adds up a position's influence on *every* later token, with weights that heavily favour the next few tokens. The
  rhyme word comes 8–12 tokens later.
- **Hypothesis (yours):** the plan *is* in the activation, and the lens definition drowns it in short-range
  (next-token and format) effects. If so, a lens that counts only influence 9–16 tokens ahead should rank the rhyme
  word much higher.
- **The alternative:** the planned rhyme isn't linearly readable in the vocabulary direction at any lag, so no
  re-definition of J helps. The fix would then have to be something else, or the plan isn't there.

**What it does.** For each of the 100 couplets, at the ⏎, at layers 40/44/48/52/56:
- take the activation h;
- on 32 fresh pile documents, add ±εh at every position, **with a random ±1 sign per position**. The signs let one
  forward pair separate the effect at each lag δ: the effect at t'−δ is recovered by multiplying by the sign at t'−δ;
  everything else averages out;
- sum the lag effects into buckets: B0 = same token, B1 = 1–3 ahead, B2 = 4–8, **B3 = 9–16**, B4 = 17–32;
- also compute the **ordinary full J̄h on the same documents (FULL_U)**. This is the fair comparator: same hosts, same
  code, only the lag window differs;
- for each readout, record the rank of the rhyme token in the full 248k vocabulary (0 = top).

Also recorded alongside:
- J_CB (released, n=25);
- J_NP (Neuronpedia, n=1000, WorkspaceBench's J);
- the logit lens;
- a sign-randomised FULL (a noise reference);
- ranks from two halves of the host set (a noise check);
- the top-10 tokens per readout, for qualitative samples.

There are no LLM judges and no API cost.

**What we expect (pre-registered).**
- **Your prediction:** B3 ranks the rhyme word well above J̄.
- **My prior** that it passes the success rule: 25%. phrase_J found lag-specific directions differ from J rows and are
  noisier, and a rhyme plan may not push the word at a consistent lag in generic pile text.
- **Success:** B3 (or B2) top-10 hit rate beats FULL_U by ≥ 0.10 (McNemar p < 0.05), and the median best rank
  improves ≥ 2×.
- **Kill:** no bucket improves the median best rank over FULL_U by ≥ 1.5×.
- **In between:** inconclusive; add sign draws before claiming anything.
- **Guardrail:** if a bucket "wins" by filling its top-10 with the same generic tokens on every item, that is not a
  win. We check the overlap.

**First two items (not interpretable yet, n=2).** Log line labels say "L48" but actually show the *last* layer, L56.
That's a cosmetic bug in the log line only; the data file is correct.

| item | B3 rank | FULL_U | J_CB |
|---|---|---|---|
| `led` | 194,306 | 844 | 1,014 |
| `game` | 51,720 | 4,235 | 5,128 |

So far this points toward the kill, but two items at one layer means nothing yet.

**What each outcome changes.**
- **Success:** planned content is hidden by the lens's lag weighting. Lag- or horizon-matched readouts become a
  standard option, and G_hz in Part I becomes more important.
- **Kill:** re-weighting J̄ by lag doesn't surface the plan. That removes one class of "J++" readers (lag/horizon
  re-weighting) for this content. It says nothing yet about whether the plan is present (the probe ceiling, spec II.7,
  would answer that).

---

## Queued (not started; each needs your go after I show the spec)

1. **Part I Stage 1, revised:** span-level P / A / curvature / P^int×P^ψ on RHUT's 12 two-hop items, plus the horizon
   gap G_hz.
2. **II.1:** write directions (filter vs pattern vs Fisher) on the ekko M2 set.

## Where things are

| what | path |
|---|---|
| review of your spec | `docs/01_spec_review_2026-09-29.md` |
| designs, written before runs | `designs/NNN-*.md` |
| reports, after runs | `results/NNN-*/report.md` |
| code | `scripts/NNN_*.py`, shared `scripts/lib/{engine,mfl}.py` |
| logs, large outputs (gitignored) | `outputs/` |
| environment | `/workspace/ekko-lens/.venv-secant` (torch 2.14 cu126, transformers 5.17), `requirements.txt` |
