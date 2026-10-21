#!/usr/bin/env bash
# Domain pretraining on a rented GPU. From the repository root, inside tmux so a dropped
# SSH connection does not stop it:
#
#     tmux new -s pretrain
#     HF_REPO=StephanAkkerman/FinTwitRoBERTa-large bash scripts/pretrain_remote.sh
#
# Running it again resumes from the last checkpoint. With HF_REPO set (and logged in with
# `hf auth login` or HF_TOKEN), the model is uploaded as a private repo when done, so
# nothing is lost when the rental ends.
set -euo pipefail

HF_REPO=${HF_REPO:-}
TRIAL_STEPS=${TRIAL_STEPS:-50}

# Use the image's PyTorch, which matches the machine's CUDA driver. RoBERTa does not use
# flash-attn, and building it from source takes most of an hour.
grep -vE '^(torch|flash-attn)' requirements.txt | pip install -q -r /dev/stdin

# Fail now rather than after hours of training
if [ -n "$HF_REPO" ]; then
    hf auth whoami
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
free -g | head -2

# Short trial first: shows speed and memory, and cleans and caches the data
if [ ! -f output/.trial_done ]; then
    python ModernFinTwitBERT/model.py --mode pretrain --max-steps "$TRIAL_STEPS"
    mkdir -p output && touch output/.trial_done
fi

python ModernFinTwitBERT/model.py --mode pretrain

if [ -n "$HF_REPO" ]; then
    MODEL_DIR=output/$(python -c "import json; print(json.load(open('config.json'))['pretrain']['model_name'])")
    hf upload "$HF_REPO" "$MODEL_DIR" . --private
fi
