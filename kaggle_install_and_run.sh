#!/usr/bin/env bash
set -euxo pipefail

# Run this in a Kaggle Notebook with GPU and Internet enabled.
# Put gcn.py in /kaggle/working before running. If data/cora.pkl is missing,
# gcn.py will download the official starter package automatically.

cd /kaggle/working

python -m pip install -q --upgrade pip
python -m pip install -q \
  numpy==1.24.0 \
  scipy==1.15.1 \
  networkx==3.4.2 \
  tqdm==4.66.4 \
  cupy-cuda12x==13.3.0
python -m pip install -q git+https://github.com/Jittor/jittor.git

if [ ! -d JittorGeometric ]; then
  git clone --depth 1 https://github.com/AlgRUC/JittorGeometric.git
fi
python -m pip install -q -e JittorGeometric

python gcn.py \
  --runs 16 \
  --epochs 400 \
  --patience 120 \
  --hidden-dim 256 \
  --dropout 0.8 \
  --final-train-mask train_val

ls -lh result.json result.zip
