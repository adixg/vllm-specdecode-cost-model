#!/bin/bash
#SBATCH --job-name=ceiling-seqs
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

# Does adaptive verification start earning its keep at higher concurrency?
#
# At 64 sequences, always-ceiling beat the stock policy by +1.08% median
# (20/20 paired rounds) at ctx=64, and tied at ctx=1024. But 64 sequences is
# the regime where draft tokens are nearly free: the GPU is memory-bound with
# spare compute, so verifying everything costs almost nothing. Adaptive
# verification is motivated by the opposite regime, where drafts compete with
# real tokens for compute and rejected ones are wasted work.
#
# So this sweeps concurrency at the one context length with decision headroom.
# Either always-ceiling keeps winning - in which case the policy is not paying
# for itself anywhere reachable here - or it stops winning somewhere, which
# locates the crossover. Both are results.
#
#   for S in 16 128 256; do
#       sbatch --export=ALL,SEQS=$S pace/run_ceiling_seqs.sbatch
#   done
#
# SEQS=64 is already measured in results/ceiling_ab_ctx64.json.
# ROUNDS defaults to 20; lower it if a large SEQS run runs out of walltime,
# since each round's work scales with SEQS.

: "${SEQS:?set SEQS, e.g. sbatch --export=ALL,SEQS=256 pace/run_ceiling_seqs.sbatch}"
ROUNDS="${ROUNDS:-20}"

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

source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/env.sh"
mkdir -p results logs

require_hf_repo openbmb/MiniCPM5-2B
require_hf_repo openbmb/MiniCPM5-2B-DSpark

# num-prompts must equal max-num-seqs: the exact-prompt sweep requires it so
# every measured step is one full-concurrency batch.
echo "=== stock vs always-ceiling at seqs=$SEQS, ctx=64, spec=7 ($(date +%T)) ==="
python bench/serving_ab.py \
    --model openbmb/MiniCPM5-2B \
    --draft openbmb/MiniCPM5-2B-DSpark \
    --num-speculative-tokens 7 \
    --max-model-len 4096 \
    --max-num-seqs "$SEQS" \
    --arm-b ceiling \
    --collect-acceptance \
    --k 1.37e-5 \
    --num-prompts "$SEQS" \
    --max-tokens 128 \
    --prompt-tokens 64 \
    --exact-prompt-sweep \
    --warmup-rounds 2 \
    --rounds "$ROUNDS" \
    --out "results/ceiling_seqs${SEQS}.json"
status=$?

echo "=== done ($(date +%T)), exit status $status ==="
exit $status
