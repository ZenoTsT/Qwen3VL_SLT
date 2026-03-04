#!/bin/bash
set -euo pipefail

# esempio: 4 GPU
accelerate launch --config_file accelerate_config.yaml \
  train.py \
  --json /path/to/dataset.json \
  --root_dir /optional/prefix \
  --split train \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 1e-4 \
  --fps 25.0 \
  --max_steps 200