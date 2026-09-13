#!/bin/bash
#SBATCH --job-name=dspark-cost
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1          # H100 only: adaptive verification needs FA3,
                                   # which vLLM enables only on compute cap 9.x.
                                   # A100/L40S/V100 will NOT work.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out    # %x = job name, %j = job id
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Sanity check: are we actually on a Hopper GPU?
#    If this prints anything other than 9.0, stop - the run would fail later
#    in a much more confusing way.
# ---------------------------------------------------------------------------
echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ "${CAP%%.*}" != "9" ]]; then
    echo "ERROR: compute capability $CAP is not Hopper (9.x)." >&2
    echo "Adaptive verification requires FA3. Request --gres=gpu:h100:1" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Environment
#    HF_HUB_OFFLINE=1 because compute nodes usually have no internet: models
#    must already be in $HF_HOME, downloaded from the login node.
# ---------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"

# `conda activate` does not work in a non-interactive batch shell until the
# conda shell hook is sourced -- without this you get
#   CommandNotFoundError: Your shell has not been properly configured
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/conda-envs/cs6222

export HF_HOME="${HF_HOME:-$HOME/scratch/hf}"
export HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=1   # adaptive verification / DSpark are V2-only

echo "=== versions ==="
python -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__)"

mkdir -p results logs

# ---------------------------------------------------------------------------
# 2. E1 - does step cost depend on context at fixed token count?
#    Same experiment as the laptop run, but on a real GPU with FlashAttention 3
#    and a KV cache big enough that blocks do not alias.
# ---------------------------------------------------------------------------
echo "=== E1: token x context sweep ==="
python bench/verify_cost_probe.py \
    --model "$MODEL" \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --num-tokens 16 32 64 128 256 512 \
    --context-lens 0 256 512 1024 2048 4096 \
    --replays 15 \
    --out results/h100_e1.json

# ---------------------------------------------------------------------------
# 3. E5 - same question in the decode shape, inside the cudagraph regime.
#    num_reqs = min(num_tokens, max_num_seqs), so a large --max-num-seqs with
#    token counts equal to the captured sizes sweeps REQUEST COUNT.
# ---------------------------------------------------------------------------
echo "=== E5: request-count sweep ==="
python bench/verify_cost_probe.py \
    --model "$MODEL" \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --num-tokens 1 2 4 8 16 24 32 40 48 56 64 \
    --context-lens 0 256 512 1024 2048 4096 \
    --replays 35 \
    --out results/h100_e5.json

# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------
python bench/plot_drift.py --in results/h100_e1.json
python bench/plot_drift.py --in results/h100_e5.json

echo "=== done ==="
ls -la results/
