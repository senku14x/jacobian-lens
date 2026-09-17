#!/bin/bash
# Rebuild the research_phraseJ environment on a fresh runtime (verified 2026-09-17 on an A100-80GB Colab-style box).
# Usage: bash research_phraseJ/scripts/setup_env.sh [--no-cache]
# Requires: `hf auth login` already done (read access to the private cache repo; write if you plan to upload).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
NOCACHE=0; [ "${1:-}" = "--no-cache" ] && NOCACHE=1

echo "== 1/5 python package"
pip install -q -e . && python -m pytest tests -q

echo "== 2/5 model (52 GB) and released lenses (24 GB): backgrounded, waits at the end"
mkdir -p /content/models /content/lenses /content/logs
hf download Qwen/Qwen3.6-27B --local-dir /content/models/Qwen3.6-27B > /content/logs/model_download.log 2>&1 &
PID_MODEL=$!
hf download camilablank/workspace-lenses --include "qwen3.6-27b/*" --local-dir /content/lenses > /content/logs/lens_download.log 2>&1 &
PID_LENS=$!

echo "== 3/5 dataset"
python -c "from datasets import load_dataset; ds=load_dataset('NeelNanda/pile-10k', split='train'); print('pile-10k', len(ds))"

if [ $NOCACHE -eq 0 ]; then
  echo "== 4/5 regenerable outputs from the private cache (about 7.5 GB) -> research_phraseJ/outputs/"
  hf download senku21x/phraseJ-cache --repo-type dataset --local-dir research_phraseJ/outputs > /content/logs/cache_download.log 2>&1 &
  PID_CACHE=$!
else
  echo "== 4/5 cache skipped (--no-cache); rerun 001b (6 min) and 001d (15 min) before 003a/004"
fi

wait $PID_MODEL; wait $PID_LENS; [ $NOCACHE -eq 0 ] && wait $PID_CACHE
du -sh /content/models/Qwen3.6-27B /content/lenses research_phraseJ/outputs 2>/dev/null || true

echo "== 5/5 verify (loads the model, reproduces the smoke test, runs one phrase-J backward)"
cd research_phraseJ && python scripts/000_env_verify.py
echo "environment ready"
