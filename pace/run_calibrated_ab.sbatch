#!/bin/bash
#SBATCH --job-name=calibrated-ab
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Override at submission time if a spec-3 replication is wanted:
#   sbatch --export=ALL,SPEC_TOKENS=3 pace/run_calibrated_ab.sbatch
SPEC_TOKENS="${SPEC_TOKENS:-7}"
MODEL_ID="openbmb/MiniCPM5-2B"
DRAFT_ID="openbmb/MiniCPM5-2B-DSpark"
CALIBRATION_OUT="results/minicpm_h100_dspark_k_spec${SPEC_TOKENS}.json"
AB_OUT="results/h100_calibrated_ab_spec${SPEC_TOKENS}.json"

case "$SPEC_TOKENS" in
    7) CALIBRATION_TOKENS=(64 128 256 384 512) ;;
    3) CALIBRATION_TOKENS=(64 128 192 256) ;;
    *)
        echo "ERROR: this experiment is defined for SPEC_TOKENS=3 or 7" >&2
        exit 2
        ;;
esac

echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
GPU_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ "${GPU_CAPABILITY%%.*}" != "9" ]]; then
    echo "ERROR: need Hopper (compute capability 9.x), got $GPU_CAPABILITY" >&2
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/conda-envs/cs6222

export HF_HOME="${HF_HOME:-$HOME/scratch/hf}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/env.sh"
mkdir -p results logs

echo "=== versions ==="
python -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__)"

# One process creates one DSpark engine. It first calibrates k using that live
# engine's captured graphs, then freezes k and runs the A/B without reloading or
# changing execution regimes. The intended/default DSpark block size is seven.
echo "=== in-engine calibration + A/B, spec=$SPEC_TOKENS ($(date +%T)) ==="
python bench/serving_ab.py \
    --model "$MODEL_ID" \
    --draft "$DRAFT_ID" \
    --num-speculative-tokens "$SPEC_TOKENS" \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.85 \
    --calibrate-k \
    --calibration-token-counts "${CALIBRATION_TOKENS[@]}" \
    --calibration-context-lens 0 1024 2048 3072 4096 \
    --calibration-replays 25 \
    --calibration-warmup 5 \
    --calibration-out "$CALIBRATION_OUT" \
    --num-prompts 64 \
    --max-tokens 128 \
    --prompt-tokens 64 512 1024 2048 3000 \
    --warmup-rounds 2 \
    --rounds 20 \
    --out "$AB_OUT"

echo "=== done ($(date +%T)) ==="
ls -lh "$CALIBRATION_OUT" "$AB_OUT"
