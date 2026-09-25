#!/bin/bash
#SBATCH --job-name=step-trace
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1          # Hopper only: adaptive verification needs FA3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Walltime note: importing torch + vLLM from Lustre with a cold page cache took
# 7.5 minutes on the first attempt, which blew a 10-minute limit before the
# engine had even started. Budget ~8 min imports + ~3 min engine init + the run.

# --- 1. Refuse to run on the wrong hardware -------------------------------
echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ "${CAP%%.*}" != "9" ]]; then
    echo "ERROR: compute capability $CAP is not Hopper (9.x)." >&2
    echo "Adaptive verification needs FA3. Resubmit with --gres=gpu:h100:1" >&2
    exit 1
fi

# --- 2. Environment -------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/conda-envs/cs6222

export HF_HOME="${HF_HOME:-$HOME/scratch/hf}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/env.sh"

mkdir -p results logs

echo "=== versions ==="
python -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__)"

# --- 3. Fail early if the DSpark checkpoints are not cached ---------------
for repo in models--openbmb--MiniCPM5-2B models--openbmb--MiniCPM5-2B-DSpark; do
    if [[ ! -d "$HF_HOME/hub/$repo" ]]; then
        echo "ERROR: $repo not in $HF_HOME/hub." >&2
        echo "Download it on the LOGIN node first (compute nodes are offline):" >&2
        echo "  hf download ${repo//models--/}" >&2
        exit 1
    fi
done

# --- 4. Trace every real generation step ----------------------------------
# No --attention-backend here: on Hopper vLLM picks FLASH_ATTN at FA3, which
# is what production uses and what adaptive verification requires.
echo "=== step trace ==="
python bench/step_trace.py \
    --model openbmb/MiniCPM5-2B \
    --draft openbmb/MiniCPM5-2B-DSpark \
    --num-speculative-tokens 3 \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --num-prompts 64 \
    --max-tokens 128 \
    --prompt-tokens 64 512 2048 \
    --out results/h100_step_trace.json

echo "=== done ==="
ls -la results/
