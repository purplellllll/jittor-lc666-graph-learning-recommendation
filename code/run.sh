#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPLEMENTARY_RESULT_ZIP="${COMPLEMENTARY_RESULT_ZIP:-data/complementary_result.zip}"

bash code/run_primary.sh "$@"
python code/ensemble.py \
  --primary_result outputs/primary_v69.zip \
  --complementary_result "$COMPLEMENTARY_RESULT_ZIP" \
  --output_dir outputs/high_score \
  --result_zip outputs/result.zip
