#!/usr/bin/env bash
# Full pipeline: data -> blocking -> cross-encoder scores -> matching -> output/*.tsv
# Usage: bash run_pipeline.sh [--data-dir <folder with train/ and test/>] [--work-dir <folder>] [--out-dir <folder>]
set -euo pipefail
cd "$(dirname "$0")"
python src/step1_dense_blocker.py "$@"     # GPU
python src/step2_cross_encoder.py "$@"     # GPU
python src/step3_matcher.py "$@"           # CPU
