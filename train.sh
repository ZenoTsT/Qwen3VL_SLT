#!/bin/bash
#SBATCH --job-name=train_phoenix
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=tesi_ztesta
#SBATCH --partition=all_usr_prod
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G

set -euo pipefail

echo "START JOB $(date)"
echo "NODE: $(hostname)"

# -------------------------
# Environment
# -------------------------

module load anaconda3/2023.09-0-none-none
source activate flash_test

# working directory
cd /homes/ztesta/Qwen3VL_SLT || exit 1

# optional but recommended on clusters
export TMPDIR=/work/tesi_ztesta/tmp
mkdir -p $TMPDIR

# HuggingFace cache (optional)
export HF_HOME=/work/tesi_ztesta/hf_cache

# -------------------------
# Training
# -------------------------

srun accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    src/train.py \
    --json /homes/ztesta/Qwen3VL_SLT/data/phoenix_dataset.json \
    --root_dir /work/tesi_ztesta/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --batch_size 1 \
    --grad_accum 16 \
    --epochs 30 \
    --num_workers 4 \
    --output_dir /work/tesi_ztesta/qwen3vl_checkpoints_phoenix/same_but_r16 \
    --orig_fps 25 \
    --target_fps 8 \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --lora_r 16 \
    --lora_dropout 0.15 \
    --lora_scope joint \


echo "END JOB $(date)"