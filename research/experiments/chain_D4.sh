#!/usr/bin/env bash
set -u
cd /home/ubuntu/jacobian-lens
PY=/home/ubuntu/.venvs/mechinterp/bin/python
export HF_HOME=/home/ubuntu/cot-oracle/hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=research/outputs
while pgrep -f "D2_fit_jown.py" >/dev/null; do sleep 60; done
echo "=== D2 done; D4 analysis (no model) $(date -u +%H:%M:%S) ==="
$PY research/experiments/D4_jown_analysis.py > $L/D4.log 2>&1
echo "=== D4 exit=$? ==="
echo "=== chain_D4 complete $(date -u +%H:%M:%S) ==="
