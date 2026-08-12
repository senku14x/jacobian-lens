#!/usr/bin/env bash
set -u
cd /home/ubuntu/jacobian-lens
PY=/home/ubuntu/.venvs/mechinterp/bin/python
export HF_HOME=/home/ubuntu/cot-oracle/hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=research/outputs
while pgrep -f "D2_fit_jown.py" >/dev/null; do sleep 60; done
echo "=== D2 done; D3 smoke (4 sites) $(date -u +%H:%M:%S) ==="
EKKO_SITES=4 EKKO_OUT=$L/D3_smoke $PY research/experiments/D3_block_rules.py > $L/D3_smoke.log 2>&1
echo "=== D3 smoke exit=$? ==="
if [ -s $L/D3_smoke/qwen3.6-27b.json ]; then
  echo "=== D3 full (24 sites) $(date -u +%H:%M:%S) ==="
  EKKO_OUT=$L/D3 $PY research/experiments/D3_block_rules.py > $L/D3.log 2>&1
  echo "=== D3 full exit=$? ==="
fi
echo "=== chain_D complete $(date -u +%H:%M:%S) ==="
