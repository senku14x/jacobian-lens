#!/usr/bin/env bash
# Remaining model-side work, serialized (one 27B resident at a time).
#   1. B4 control re-check at M=16 -- T_IG must read ~1.0 or the framework is wrong
#   2. B4 full, 3 layers
#   3. A4 M2 re-run on the BROADENED-bank operators
set -u
cd /home/ubuntu/jacobian-lens
PY=/home/ubuntu/.venvs/mechinterp/bin/python
export HF_HOME=/home/ubuntu/cot-oracle/hf_home
L=research/outputs

echo "=== [1/3] B4 control re-check (L31, 8 sites, M_IG=16) $(date -u +%H:%M:%S) ==="
EKKO_LAYERS=31 EKKO_SITES=8 EKKO_K=4 EKKO_M_IG=16 EKKO_OUT=$L/B4_ctrl \
  $PY research/experiments/B4_smoothing_mechanism.py > $L/B4_ctrl.log 2>&1
echo "=== ctrl exit=$? ==="
grep -E "^     T_IG" $L/B4_ctrl.log || true

echo "=== [2/3] B4 full, 3 layers $(date -u +%H:%M:%S) ==="
EKKO_OUT=$L/B4 $PY research/experiments/B4_smoothing_mechanism.py > $L/B4.log 2>&1
echo "=== B4 full exit=$? ==="

echo "=== [3/3] A4 M2 on broadened operators $(date -u +%H:%M:%S) ==="
EKKO_OPS=$L/B2_A1_A6 EKKO_OUT=$L/B2_A4_m2 \
  $PY research/experiments/A4_m2_ablation.py > $L/B2_A4_m2.log 2>&1
echo "=== A4 exit=$? ==="

echo "=== ALL COMPLETE $(date -u +%H:%M:%S) ==="
