# Readable Here, Used There (senku14x/Readable-Here-Used-There, branch `extension`)

Commit d4197b0, last activity 2026-09-24 (question-turn ladder frozen). Clone: `/workspace/repos/rhut` (2.4 GB; raw
arrays mirrored to the private HF dataset `senku21x/tcsif-outputs`). The user's MATS application project. Canonical
docs: `STATUS.md` (canonical H3 statement at the top), `context/evidence_ledger.md` (every load-bearing finding with its
level and strongest alternative, including withdrawals), `context/h1_stopping_point.md` (eight H1 verdicts),
`context/full_project_writeup_2026-09-11.md` (871 lines, **not read in full**), and the H3 ladder report
`H3/results/reports/two_hop_question_turn_ladder.md`.

**Setup [REPO].**
- Models: Qwen3.6-27B (hybrid) primary, Qwen3-32B (dense) as a pure-attention replication.
- Instruments: J-lens (Neuronpedia J_NP) primary, a plain-sentence residual axis (RESID_P), and model logits.
  J_CB/R_CB on a subset. Agreement across instruments is treated as robustness, not independent evidence.
- Organism: content introduced (quoted word / tagged words / category / number pair / two-hop clue) → a teacher-forced
  unrelated **carrier** sentence → a later question.
- Interventions are applied separately at **source**, **carrier** and **question** positions, with matched controls.
  Windows L48–50 / L51–59. Every design spec carries dated amendments registered before the forwards they govern.

## H1 — readout phenomena (eight verdicts, 2026-09-07) [REPO]
1. Natural modulation: "maintain" raises readability over the mention control. Mostly priming at L48–50; exceeds a bare
   mention at L51–59.
2. Source causality: source-span replacement from block 36 moves the readout (≈0.5 / ≈0.95 of the natural contrast).
3. The candidate "state" coordinate ĝ is a **uniform readout gain, not a controller** (deflationary).
4. A native pointed−control direction is a *selective* gain that transfers to held-out identities (cos 0.85). A pointer
   contrast is not an address.
5. No J-only artifact: every effect also shows on the plain-sentence axis and the logits.
6. Scope: broad within the tagged copy-bridge organism; untested without it.
7. **Selection** (strongest): a tagged instruction selects which of three sources is carried (amplify + suppress). The
   pointer state is localized in the instruction residuals (tag-dominant, blocks 36–48) and causally controls transfer.
   It reaches category members never shown (member readability ≠ member representation).
8. Transformation: computed sums transfer (a same-sum sibling leaves the sum readout invariant). Early codebook
   availability precomputes F(X) while X stays readable (additive).

## H3 — is readable content behaviourally used? [REPO]
- **Internal vs behavioural dissociation:** a transplant redirecting the readout ~0.76 moves the answer only ~0.2.
- **Position asymmetry:**
  - Carrier-only donor installs the donor value at the readout but moves the answer 0.02–0.27 of the source-donor
    ceiling.
  - Random writes are inert. The effect is real, content-specific, and small.
- **Redundancy, not non-use:** with the source absent (placeholder clue / hidden operands), transplanted carrier
  states are consumed (28/32 sums, 28–32/40 two-hop on the scored candidate set).
- **Consumer dependence:**
  - Clamping the sum-word naming plane at the question lowers the report margin 38.8%; a norm-matched orthogonal clamp
    lowers it 0.2%.
  - Parity is unmoved.
  - Coordinate swaps along sum-word directions flip report but not parity (answer-token steering).
- **Paper's intermediate swap decomposed** (bridgeswap): 0.41 of ceiling at all positions. By position: question turn
  0.20, clue span 0.09, carrier 0.07. The answer-word swap is larger overall (0.54).
- **Question-turn ladder, frozen 2026-09-24** (35 items × 4 carriers, fp32, sequence endpoint; share of the
  question-turn effect +11.16 nats):
  - The question-turn intervention = 0.46 of the natural clue swap.
  - Removing the best 25–64 vocabulary-J directions keeps 0.61, vs 0.95–0.96 for random. Installing the best 25 gives
    0.39, vs 0.13 for 25 fitted functional-Jacobian directions. But the answer's own 2-direction plane installs 0.37,
    so **a privilege for non-answer vocabulary directions is not established**.
  - The answer is already carried at the question turn: the answer plane (0.37) > the intermediate plane (0.19) in
    35/35 items.
  - **A late route outside the tested J-derived span:** 0.41 survives with that span held clean at every answer
    position.
    - Cutting the last layer's read of the question-turn positions leaves 0.04.
    - The last layer reads the eight template tokens ending the question turn.
    - Head 22 dominates a linearized (non-causal) attribution.
  - **Bottom line [REPO]:** "compatible with a causally important verbalizable interface, not with the tested
    J-vocabulary span as the dominant causal bottleneck for this intervention."
- **Replication on Qwen3-32B (dense):** everything reproduces qualitatively; the carrier share is smaller (0.02–0.04).
  - The boundary account of the arithmetic-vs-fact gap is disconfirmed.
  - The released Qwen3-32B lens is a partial fit checkpoint (n=80).
- **Adversarial "forge" study:** a bounded, damage-free block-35 write can move both J and R readouts (transfers 9/16
  pairs) without the plain-sentence probe or behaviour moving. So **lens readouts can be dissociated from probes and
  behaviour by optimisation.** J vs R selectivity is untested.
- **Not established:** H2 (the maintenance mechanism; recurrent state parity not started), the read-site component,
  cross-domain generality without the copy bridge. The bf16 coordinate-write gate (ρ∈[0.9,1.1], κ≥0.99) is not met
  by any CoordClamp/CoordSwap row. The fp32 repair was used from Amendment 5 on.

## [MINE] takeaways
- The strongest caution for "better lens" work: **readability at a position is a weak proxy for causal leverage
  there.** A lens that reads better is not thereby a better causal handle. Evaluate causal use at the positions where
  leverage lives (the question turn / decision positions), not only where content is readable.
- Coordinate swaps whose directions name the emitted answer measure answer steering. Any swap benchmark must separate
  intermediates from answers. This applies to the phrase-lens MT multihop and to the paper's own swaps.
- The forge result means a lens monitor can in principle be fooled by bounded writes that don't change behaviour. That
  is relevant to "use the lens as a monitor" claims.
- Infrastructure worth reusing: `common/scripts/hooks{,_fp32}.py` (SpanWriter, StateMove, CoordSwap/Clamp with
  realized-write gates), `instrument_gates.py`, and the per-run manifests pinning model/lens hashes.
