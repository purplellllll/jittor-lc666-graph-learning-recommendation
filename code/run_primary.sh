#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python code/check_environment.py
python code/main.py \
  --preset final \
  --data_dir data/track1 \
  --zip_path data/track1_data.zip \
  --output_dir outputs/primary \
  --result_zip outputs/primary_v69.zip \
  "$@"
