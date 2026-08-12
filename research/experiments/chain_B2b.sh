#!/usr/bin/env bash
# B2b -- finish the broadened-bank gate re-read after the base_ids fix:
#   1. make both banks self-describing (verified by re-measurement)
#   2. A0 re-verify (manifest-driven counts, exact-dup check)
#   3. A2 Stein + A5 in-span on the broadened bank, ranks from ITS spectrum
set -u
cd /home/ubuntu/jacobian-lens
PY=/home/ubuntu/.venvs/mechinterp/bin/python
export HF_HOME=/home/ubuntu/cot-oracle/hf_home
L=research/outputs

echo "=== B2b-patch: bank base_ids ($(date -u +%H:%M:%S)) ==="
$PY research/experiments/B2_patch_bank_bases.py > $L/B2_patch.log 2>&1
rc=$?
echo "=== B2b-patch exit=$rc ==="
if [ $rc -ne 0 ]; then echo "HARD-STOP: bank patch failed verification"; exit 1; fi

echo "=== B2b-A0: re-verify broadened bank ($(date -u +%H:%M:%S)) ==="
EKKO_BANK=research/outputs/B1_bank EKKO_OUT=$L/B2_A0 \
  $PY research/experiments/A0_verify_bank.py > $L/B2_A0.log 2>&1
echo "=== B2b-A0 exit=$? ==="

echo "=== B2b-A2/A5: Stein + in-span, broadened bank ($(date -u +%H:%M:%S)) ==="
EKKO_BANK=research/outputs/B1_bank EKKO_OPS=$L/B2_A1_A6 EKKO_OUT=$L/B2_A2_A5 \
  EKKO_RANKS=555,1911 EKKO_SG_SITES=120 \
  $PY research/experiments/A2_A5_model.py > $L/B2_A2_A5.log 2>&1
echo "=== B2b-A2/A5 exit=$? ==="

echo "=== B2b chain complete ($(date -u +%H:%M:%S)) ==="
