#!/bin/bash
#SBATCH --job-name=ceiling-ab
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail

# Does adaptive verification earn anything on this workload?
#
# Arm A is stock vLLM: argmax(estimated_accepted / cost), the real policy.
# Arm B ignores the cost model entirely and always takes every draft token
# on offer - the "ceiling" policy.
#
# We already know stock sits at its ceiling in 92-99.8% of decode steps from
# ctx=512 upward. If the two arms tie, the cost model is doing no useful work
# on this workload and the whole adaptive machinery could be replaced by a
# constant. If arm B loses, the ~5% of steps where the policy does deviate are
# earning their keep, and the feature is justified.
#
# ctx=64 is deliberately included: it is the one length with real headroom
# (stock at ceiling only 58% of the time), so it is where the two policies
# differ most.
#
#   sbatch --export=ALL,CTX=64 pace/run_ceiling_ab.sbatch
#   sbatch --export=ALL,CTX=1024 pace/run_ceiling_ab.sbatch
#
# This run also records per-round acceptance counts in both arms, so the
# acceptance distribution and any drift over the course of generation come
# out of the same job.
#
# k is passed as a constant rather than recalibrated: the ceiling arm never
# consults the cost model, so k only affects diagnostic fields. Skipping
# calibration buys back ~2 minutes of a 25 minute allocation.

: "${CTX:?set CTX, e.g. sbatch --export=ALL,CTX=64 pace/run_ceiling_ab.sbatch}"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
case "$cap" in
    9.*) echo "compute capability $cap - FA3 available" ;;
    *)   echo "compute capability $cap is not 9.x; DSpark adaptive "\
              "verification needs FlashAttention 3. Exiting." >&2; exit 1 ;;
esac

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/conda-envs/cs6222

export HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=1
mkdir -p results logs

echo "=== stock vs always-ceiling at ctx=$CTX, spec=7 ($(date +%T)) ==="
python bench/serving_ab.py \
    --model openbmb/MiniCPM5-2B \
    --draft openbmb/MiniCPM5-2B-DSpark \
    --num-speculative-tokens 7 \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --arm-b ceiling \
    --k 1.37e-5 \
    --num-prompts 64 \
    --max-tokens 128 \
    --prompt-tokens "$CTX" \
    --exact-prompt-sweep \
    --warmup-rounds 2 \
    --rounds 20 \
    --out "results/ceiling_ab_ctx${CTX}.json"

echo "=== done ($(date +%T)) ==="
