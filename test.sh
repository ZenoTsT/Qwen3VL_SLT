#!/bin/bash
#SBATCH --job-name=test_phoenix
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
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
source activate qwen3vl_env

# working directory
cd /homes/ztesta/Qwen3VL_SLT || exit 1

# tmp
export TMPDIR=/work/tesi_ztesta/tmp
mkdir -p $TMPDIR

# HuggingFace cache
export HF_HOME=/work/tesi_ztesta/hf_cache

# -------------------------
# Testing / Inference
# -------------------------

srun accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    src/test.py \
    --json /homes/ztesta/Qwen3VL_SLT/data/phoenix_dataset.json \
    --root_dir /work/tesi_ztesta/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --ckpt_dir /work/tesi_ztesta/qwen3vl_checkpoints_phoenix/best \
    --orig_fps 25 \
    --target_fps 12 \
    --max_new_tokens 128 \
    --num_beams 1

echo "END JOB $(date)"