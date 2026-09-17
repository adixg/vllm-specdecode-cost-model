#!/bin/bash
#SBATCH --job-name=saturation
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail

# The step trace found the draft budget pinned at its maximum on every step:
# 192 of 192, for all 30 steps. If that holds generally then adaptive
# verification never actually adapts, the cost model feeds a saturated argmax,
# and its accuracy is irrelevant by construction - which would explain the 0.7%
# ceiling measured by the serving A/B far better than "the effect is small".
#
# This varies the two things most likely to pull the budget off the ceiling:
# a longer draft block (more slots, lower survival at the tail) and higher
# concurrency (more KV, so a larger fixed cost for marginal tokens to compete
# against).

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

echo "=== warming imports ($(date +%T)) ==="
python -c "import torch, vllm; print('vllm', vllm.__version__)"

STATUS=0
for SPEC in 3 7; do
  for SEQS in 16 64; do
    echo
    echo "=== spec=$SPEC max_num_seqs=$SEQS ($(date +%T)) ==="
    python bench/step_trace.py \
        --model openbmb/MiniCPM5-2B \
        --draft openbmb/MiniCPM5-2B-DSpark \
        --num-speculative-tokens "$SPEC" \
        --max-model-len 4096 \
        --max-num-seqs "$SEQS" \
        --num-prompts 48 \
        --max-tokens 96 \
        --prompt-tokens 64 512 1024 2048 3000 \
        --out "results/sat_spec${SPEC}_seqs${SEQS}.json" \
        || { echo "FAILED spec=$SPEC seqs=$SEQS"; STATUS=1; }
  done
done

echo
echo "=== done ($(date +%T)) exit=$STATUS ==="
ls -la results/sat_*.json 2>/dev/null
exit $STATUS
