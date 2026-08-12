#!/usr/bin/env bash
# B2 -- re-read every Phase-A gate on the BROADENED bank (B1). The gates read
# these numbers, not the raw-bank ones.
set -u
cd /home/ubuntu/jacobian-lens
PY=/home/ubuntu/.venvs/mechinterp/bin/python
export EKKO_BANK=research/outputs/B1_bank
export HF_HOME=/home/ubuntu/cot-oracle/hf_home
L=research/outputs

echo "=== B2-A0: verify broadened bank ($(date -u +%H:%M:%S)) ==="
EKKO_OUT=$L/B2_A0 $PY research/experiments/A0_verify_bank.py > $L/B2_A0.log 2>&1
echo "=== B2-A0 exit=$? ==="

echo "=== B2-A1: per-eps fits on broadened bank ($(date -u +%H:%M:%S)) ==="
EKKO_OUT=$L/B2_A1_A6 $PY research/experiments/A1_A6_fits.py > $L/B2_A1_A6.log 2>&1
echo "=== B2-A1 exit=$? ==="

echo "=== B2-A2/A5: Stein + in-span on broadened bank ($(date -u +%H:%M:%S)) ==="
EKKO_OPS=$L/B2_A1_A6 EKKO_OUT=$L/B2_A2_A5 $PY research/experiments/A2_A5_model.py > $L/B2_A2_A5.log 2>&1
echo "=== B2-A2/A5 exit=$? ==="

echo "=== B2 chain complete ($(date -u +%H:%M:%S)) ==="
