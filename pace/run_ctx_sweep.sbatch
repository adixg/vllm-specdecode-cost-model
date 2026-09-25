#!/bin/bash
#SBATCH --job-name=ctx-sweep
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Tests the sharpest prediction of the E6 hypothesis.
#
# vLLM profiles its cost table assuming every request carries
# VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN (8192) tokens of context.
# Every context measured so far has been AT OR BELOW that, so every error
# observed has been over-prediction. The hypothesis says the sign must FLIP
# above 8192: with more context than assumed, vLLM should UNDER-predict.
#
# Under-prediction is the dangerous direction - it makes the scheduler
# over-commit verification and miss latency targets, rather than merely
# being conservative.
#
# KV budget: the H100 run had a 567,584-token cache. 16 requests x 32,768
# context = 524,288, which fits without block aliasing. Going wider would
# re-introduce the artifact that contaminated the laptop measurements.

echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ "${CAP%%.*}" != "9" ]]; then
    echo "ERROR: compute capability $CAP is not Hopper (9.x)." >&2
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/conda-envs/cs6222

export HF_HOME="${HF_HOME:-$HOME/scratch/hf}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/env.sh"

mkdir -p results logs
python -c "import vllm; print('vllm', vllm.__version__)"

# Contexts straddle the 8192 profiling point: three below, three above.
# max-num-seqs 16 keeps every token count inside the cudagraph capture range,
# so all cells are in the graphed regime where the KV term is exposed.
echo "=== context sweep straddling the 8192 profiling point ==="
python bench/verify_cost_probe.py \
    --model "${MODEL:-Qwen/Qwen3-1.7B}" \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --num-tokens 1 2 4 8 16 \
    --context-lens 1024 4096 8192 16384 24576 32000 \
    --replays 25 \
    --out results/h100_ctx_straddle.json

python bench/plot_drift.py --in results/h100_ctx_straddle.json || true

echo "=== done ==="
ls -la results/
