#!/bin/bash
#SBATCH --job-name=slot-grid
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-3
#SBATCH --output=slurm-%A_%a.out

# One engine per request count R (max_num_seqs = R).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

REQS=(8 16 32 64)
R=${REQS[$SLURM_ARRAY_TASK_ID]}

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

srun python -u grid.py \
    --model openbmb/MiniCPM5-2B \
    --draft openbmb/MiniCPM5-2B-DSpark \
    --num-reqs "$R" \
    --tokens-per-req 1 2 3 4 5 6 7 8 \
    --contexts 0 1024 4096 8192 \
    --max-model-len 16384 \
    --warmup 3 \
    --repeats 20 \
    --out "results/slot_grid_R${R}_${SLURM_ARRAY_JOB_ID}.json"
