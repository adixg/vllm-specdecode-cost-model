#!/bin/bash
#SBATCH --job-name=dspark-profile
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Conda environment
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/scratch/conda-envs/cs6222"

# Environment variables
export HF_HOME="${HF_HOME:-$HOME/scratch/hf}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export TOKENIZERS_PARALLELISM=false

export HF_HUB_OFFLINE=1

# Verify environment
python -c "import torch, vllm; print('vLLM:', vllm.__version__)"
nvidia-smi

# Run experiment
mkdir -p results

srun python -u experiment.py \
    --model openbmb/MiniCPM5-2B \
    --draft openbmb/MiniCPM5-2B-DSpark \
    --max-num-seqs 32 \
    --max-model-len 16384 \
    --num-tokens 8 16 32 \
    --baseline-context 8192 \
    --train-contexts 512 4096 \
    --test-contexts 1024 2048 12288 \
    --warmup 3 \
    --repeats 20 \
    --out "results/profile_drift_${SLURM_JOB_ID}.json"

