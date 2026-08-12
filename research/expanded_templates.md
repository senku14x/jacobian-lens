# Broadened bank (B1) — template design

**Why.** A0 showed the original D4 family spans only **117 dimensions** of 5120 at ε=0.2
(participation ratio 57.6); the pooled D4+D1 fit set spans 615. Every off-family transfer number is
therefore identifiability-limited, and any secant result is provisional until the natural-delta
family is materially wider. B1 widens it.

**Constraints held fixed from the original bank** (so results stay comparable):
- minimal-pair structure: one template, two arguments, token-aligned, differing in one contiguous
  block; δ = h_ℓ(x′)[p] − h_ℓ(x)[p]
- 96-token `NeelNanda/pile-10k` prefix, so perturbed positions sit deep in a ~104-token context and
  on-distribution for operators fitted with `skip_first=4` on 128-token pile text
- same ε grid {0.01, 0.05, 0.2, 1.0} × median‖h‖ plus native magnitude, antithetic, summed and self
  targets
- layers 16 / 31 / 46; split by base prompt

**What changes.** 4 categories → **10**, 16 templates → **40**, 4 args → **6 per category**.
Arguments are filtered at build time to those that are single-token *in this tokenizer* and that
keep the pair length-aligned; anything failing is dropped and counted.

**No leakage from frozen data.** None of these categories, arguments, or templates are drawn from
`data/evaluations/`. They are authored here. `data/experiments/flexible-generalization.json`
supplies the original four categories only.

## Categories

| category | arguments (pre-filter) |
|---|---|
| countries | France, Canada, China, Egypt, Japan, Brazil |
| months | January, February, March, April, June, July |
| animals | dog, cat, horse, sheep, mouse, bear |
| numbers | three, four, five, six, seven, eight |
| colors | red, blue, green, yellow, black, purple |
| elements | iron, gold, silver, copper, carbon, oxygen |
| professions | doctor, teacher, lawyer, farmer, pilot, nurse |
| sports | tennis, soccer, golf, boxing, chess, hockey |
| body | hand, foot, heart, brain, liver, lung |
| instruments | piano, guitar, violin, drums, flute, trumpet |

Four templates per category (40 total). Templates are ordinary factual or descriptive frames; the
continuation need not be *correct* — the bank measures transport, not accuracy — but natural frames
keep the perturbed activations on-distribution.

## Expected effect on identifiability

Distinct D4 directions rise from 246 (calibrate) to roughly 40 templates × 6 args × 5 alternatives
× 3 positions ÷ 2 ≈ 1800. **Sanity gate: if the D4 effective rank at ε=0.2 does not rise materially
above 117, the family is still too narrow and every downstream gate inherits that caveat — this is
recorded rather than worked around.**
