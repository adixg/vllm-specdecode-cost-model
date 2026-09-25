#!/bin/bash
#SBATCH --job-name=ab-longctx
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail

# The one untested regime.
#
# Every A/B so far ran with per-request contexts well below the 8192 the cost
# table assumes, so vLLM over-predicted, which inflates the fixed cost, which
# pushes the argmax to the ceiling - accidentally the right answer, since
# marginal cost per draft token is tiny. The correction pulled the budget off
# the ceiling and cost 0.37%.
#
# Above 8192 the sign flips (E8): vLLM under-predicts, by up to 203%. There the
# correction pushes the budget UP, not down. If it does not help there either,
# the negative result is complete and holds in both directions.
#
# 20 minutes, per the queue: short jobs backfill, long ones wait.

echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
[[ "${CAP%%.*}" == "9" ]] || { echo "ERROR: need Hopper (9.x), got $CAP" >&2; exit 1; }

cd "$SLURM_SUBMIT_DIR"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/conda-envs/cs6222

export HF_HOME="${HF_HOME:-$HOME/scratch/hf}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/env.sh"
mkdir -p results logs

# k is MODEL-SPECIFIC: KV bytes per token over bandwidth. 0.0137 us/token was
# measured for MiniCPM5-2B on this H100. Do not reuse the Qwen3-1.7B value.
python bench/serving_ab.py \
    --model openbmb/MiniCPM5-2B \
    --draft openbmb/MiniCPM5-2B-DSpark \
    --num-speculative-tokens 7 \
    --max-model-len 16384 \
    --max-num-seqs 32 \
    --k 1.3702082114757356e-05 \
    --num-prompts 32 \
    --max-tokens 96 \
    --prompt-tokens 9000 11000 13000 \
    --rounds 12 \
    --out results/h100_ab_longctx.json
