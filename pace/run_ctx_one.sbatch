#!/bin/bash
#SBATCH --job-name=ctx-one
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j-ctx${CTX}.out
#SBATCH --error=logs/%x-%j-ctx${CTX}.err

set -uo pipefail

# One context length per job. run_serving_context_sweep.sbatch does all five in
# a single 75-minute allocation, which never scheduled: on ice-gpu a long H100
# request can sit for days, while a short one backfills into gaps. Five 25-min
# jobs queue independently and in practice finish sooner than one long job
# starts.
#
#   for CTX in 64 512 1024 2048 3000; do
#       sbatch --export=ALL,CTX=$CTX pace/run_ctx_one.sbatch
#   done
#
# Each job recalibrates k in-engine. That repeats ~2 minutes of work five times
# rather than sharing it once, which is the price of being schedulable - and it
# has a real benefit: five independent calibrations show whether k is stable
# across runs, which a single shared calibration cannot.

: "${CTX:?set CTX, e.g. sbatch --export=ALL,CTX=2048 pace/run_ctx_one.sbatch}"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
[[ "${CAP%%.*}" == "9" ]] || { echo "ERROR: need Hopper (9.x), got $CAP" >&2; exit 1; }

cd "$SLURM_SUBMIT_DIR"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/conda-envs/cs6222

export HF_HOME="${HF_HOME:-$HOME/scratch/hf}"
export HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=1
mkdir -p results logs

echo "=== exact-token A/B at ctx=$CTX, spec=7 ($(date +%T)) ==="
python bench/serving_ab.py \
    --model openbmb/MiniCPM5-2B \
    --draft openbmb/MiniCPM5-2B-DSpark \
    --num-speculative-tokens 7 \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.85 \
    --calibrate-k \
    --calibration-token-counts 64 128 256 384 512 \
    --calibration-context-lens 0 1024 2048 3072 4096 \
    --calibration-replays 25 \
    --calibration-warmup 5 \
    --calibration-out "results/k_spec7_ctx${CTX}.json" \
    --num-prompts 64 \
    --max-tokens 128 \
    --prompt-tokens "$CTX" \
    --exact-prompt-sweep \
    --warmup-rounds 2 \
    --rounds 20 \
    --out "results/ab_spec7_ctx${CTX}.json"

echo "=== done ($(date +%T)) ==="
