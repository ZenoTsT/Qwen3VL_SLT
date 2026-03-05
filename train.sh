#!/bin/bash
#SBATCH --job-name=train_phoenix
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
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
source activate qwen3vl_env

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
    --json /homes/ztesta/Qwen3VL_SLT/data/phoenix_dataset.json
    --root_dir /work/tesi_ztesta/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 5 \
    --num_workers 8 \
    --output_dir /work/tesi_ztesta/qwen3vl_checkpoints_phoenix

echo "END JOB $(date)"