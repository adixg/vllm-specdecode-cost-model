#!/bin/bash
#SBATCH --job-name=logits-probe
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"


module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/scratch/conda-envs/cs6222"

export HF_HOME="${HF_HOME:-$HOME/scratch/hf}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

python -c "import torch, vllm; print('vLLM:', vllm.__version__)"
nvidia-smi
mkdir -p results

srun python -u logits_probe.py \
    --model openbmb/MiniCPM5-2B \
    --draft openbmb/MiniCPM5-2B-DSpark \
    --concurrency 8 32 64 128 \
    --output-tokens 256 \
    --rounds 3 \
    --out "results/logits_probe_${SLURM_JOB_ID}.json"
