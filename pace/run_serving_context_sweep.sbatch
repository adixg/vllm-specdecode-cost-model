#!/bin/bash
#SBATCH --job-name=serving-ctx-sweep
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

MODEL_ID="openbmb/MiniCPM5-2B"
DRAFT_ID="openbmb/MiniCPM5-2B-DSpark"
SPEC_TOKENS=7
CALIBRATION_OUT="results/minicpm_h100_dspark_k_spec7_context_sweep.json"
SWEEP_OUT="results/h100_exact_context_sweep_spec7.json"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
GPU_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ "${GPU_CAPABILITY%%.*}" != "9" ]]; then
    echo "ERROR: need Hopper (compute capability 9.x), got $GPU_CAPABILITY" >&2
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/conda-envs/cs6222

export HF_HOME="${HF_HOME:-$HOME/scratch/hf}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/env.sh"
mkdir -p results logs

echo "=== versions ==="
python -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__)"

# Calibrate k once in this DSpark engine, then run exact-token homogeneous
# batches at every input length. Mode order and context order are both
# counterbalanced; every request is forced to generate exactly 128 tokens.
echo "=== exact-token serving context sweep, spec=7 ($(date +%T)) ==="
python bench/serving_ab.py \
    --model "$MODEL_ID" \
    --draft "$DRAFT_ID" \
    --num-speculative-tokens "$SPEC_TOKENS" \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.85 \
    --calibrate-k \
    --calibration-token-counts 64 128 256 384 512 \
    --calibration-context-lens 0 1024 2048 3072 4096 \
    --calibration-replays 25 \
    --calibration-warmup 5 \
    --calibration-out "$CALIBRATION_OUT" \
    --num-prompts 64 \
    --max-tokens 128 \
    --prompt-tokens 64 512 1024 2048 3000 \
    --exact-prompt-sweep \
    --warmup-rounds 2 \
    --rounds 20 \
    --out "$SWEEP_OUT"

echo "=== done ($(date +%T)) ==="
ls -lh "$CALIBRATION_OUT" "$SWEEP_OUT"
