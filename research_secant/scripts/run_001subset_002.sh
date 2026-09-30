#!/bin/bash
# 2026-09-30 (user go): 001 v2 on the Amendment-4 subset (45 remaining items), then 002 full run.
# Sequential: one GPU. Logs in outputs/logs/. Analyses are run separately, after both finish.
set -e
cd "$(dirname "$0")/.."
PY=/workspace/ekko-lens/.venv-secant/bin/python
export HF_HOME=/workspace/.hf_home
$PY scripts/001_lag_bucket_poetry_v2.py --items_file results/001-lag-bucket-poetry/v2/subset50.json > outputs/logs/001_v2_subset.log 2>&1
$PY scripts/002_future_words.py > outputs/logs/002.log 2>&1
echo CHAIN_DONE >> outputs/logs/002.log
