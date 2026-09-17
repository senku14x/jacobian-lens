# research_phraseJ — multi-token extensions of the Jacobian lens on Qwen3.6-27B

Branch `phrase_J` of `senku14x/jacobian-lens`. Everything lives under `research_phraseJ/`. Read `CLAUDE.md` (working
agreement) and `STATUS.md` (current state) first; `docs/06_plan_decisions.md` is the decision log.

## Where things stand (2026-09-17)

| # | experiment | result |
|---|---|---|
| 001 | template geometry | late-layer multi-token identity is linearly present (probe AUC 0.94–1.00 at L52–62); emission-trained directions transfer to latent states |
| 002 | phrase-J exact compat | scalar backward reproduces the released J estimator exactly at the fit's graph shape; early-layer discrepancy is bf16 batch shape only; log-prob phrase objectives saturate |
| 003a | phrase objective screen | `lin` (J-compatible linear target) is the reliable object; no readout gain over J-sum; constituent-J ceiling = full probe except the San family |
| 003a-controls | symmetry / San-likeness / mid-band | matched Phrase-J ≤ J-sum except San; mid-band probe-vs-J gap is generic early-J; San is the only late residual; `lin` converges at L52+ |
| 004 | causal geometry | J-const is the better write direction at matched norm and damage; no method flips over two layers; **San: Phrase-J beats J-const at every dose and position set** |

Bottom line: Phrase-J is closed as a general method (read and write). The surviving object is San-type concepts, where
phrase information lies outside constituent-token J coordinates and the phrase-conditioned gradient reads and writes
where token atoms do not. Reports: `research_artifacts/NNN-slug/report.md`.

## Rebuilding the environment

Hardware used: one A100-80GB (the fresh J fit needs `dim_batch ≤ 4`; a batch-1 128-token graph from L8 peaks at 59 GB).
Software: Python 3.13, torch 2.11 (cu128), transformers 5.16, `datasets`, `huggingface_hub`, `safetensors`, `matplotlib`.

```bash
git clone --branch phrase_J https://github.com/senku14x/jacobian-lens.git && cd jacobian-lens
hf auth login          # once; needs read access to the private cache repo
bash research_phraseJ/scripts/setup_env.sh          # installs, downloads model + lenses + pile-10k + cache, verifies
```

What the script fetches and where the code expects it:

| item | source | path | size |
|---|---|---|---|
| model | `Qwen/Qwen3.6-27B` | `/content/models/Qwen3.6-27B` | 52 GB |
| released J / R / template lenses + passages | `camilablank/workspace-lenses` (`qwen3.6-27b/*`) | `/content/lenses/qwen3.6-27b/{j-lens,r-lens,template-lens}` | 24 GB |
| corpus | `NeelNanda/pile-10k` | HF datasets cache | small |
| regenerable outputs | `senku21x/phraseJ-cache` (private dataset) | `research_phraseJ/outputs/` | 7.5 GB |

`setup_env.sh --no-cache` skips the cache; then run `001b_sigma_and_null.py` (6 min) and `001d_capture.py` (15 min)
before anything in 003a/004. `001/unembed.pt` is never cached (a copy of model weights; 001d regenerates it).

Verification is `scripts/000_env_verify.py`: lens shapes and provenance, model load, the smoke test (`spider` rank 4 at
L40, rank 8 at L46 on the web-spinner prompt), and one phrase-J backward with its peak memory.

## Reproducing the experiments (run order and cost on the A100)

| step | script | needs | time |
|---|---|---|---|
| 001 | `001a_build_families.py` → `001b_sigma_and_null.py` → `001c_contexts.py` → `001d_capture.py` → `001e_analysis.py` → `001f_transfer.py` | model, lenses, pile | ~40 min |
| 002 | `002_phraseJ_compat.py` → `002b_same_shape_check.py` | 001 results (committed) | 86 min + 4 min (the fresh 3-sequence J fit is 71 min) |
| 003a | `003a_fit.py` → `003a_eval.py` → `003a_posthoc.py` | `outputs/001/{sigma,null_acts,acts}.pt` | ~80 min + 5 min |
| 003a-controls | `003a_controls_cpu.py`, `003a_controls_gpu.py` | 003a gradients, 001 activations | 20 min + 5 min |
| 004 | `004_first_order.py` (gate) → `004_swaps.py` → `004_eval.py` → `004b_allpos.py` | 003a + 003a-controls gradients | 15 + 55 + 1 + 13 min |

Every script is resumable where it is long (`003a_fit.py` checkpoints per item and reuses saved context lists). Logs
go to `/content/logs/` by convention. All numbers in the reports are reproducible from the committed code at the commit
named in each report.

## Layout

```
research_phraseJ/
  CLAUDE.md  STATUS.md  README.md
  docs/                 notes and the decision log
  scripts/              NNN_*.py runnable experiments; lib/ shared code (edit_harness, phrase_objective, borrowed ekko harness)
  results/NNN-slug/     analysis.md + the JSON it was computed from
  plots/NNN-slug/       figures
  research_artifacts/NNN-slug/   design.md (before the run) and report.md (after)
  outputs/              gitignored; restored from the HF cache
```
