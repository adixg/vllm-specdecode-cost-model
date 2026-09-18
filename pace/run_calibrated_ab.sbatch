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
PROBE_OUT="results/minicpm_h100_k_probe.json"
AB_OUT="results/h100_calibrated_ab_spec${SPEC_TOKENS}.json"

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
export HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=1
mkdir -p results logs

echo "=== versions ==="
python -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__)"

# Fit MiniCPM's own context coefficient on this GPU. All token shapes use 64
# requests and remain inside the CUDA-graph capture range. serving_ab.py fits
# one shared slope while allowing a separate intercept for each token shape.
echo "=== calibrating MiniCPM k ($(date +%T)) ==="
python bench/verify_cost_probe.py \
    --model "$MODEL_ID" \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.85 \
    --num-tokens 128 256 384 512 \
    --context-lens 0 1024 4096 \
    --replays 25 \
    --warmup 5 \
    --out "$PROBE_OUT"

# The intended DSpark block size is seven. Modes are paired and run AB/BA on
# alternating rounds. The output includes every chosen budget, ceiling status,
# correction offset, and whether the selected verification cost was clamped.
echo "=== calibrated A/B, spec=$SPEC_TOKENS ($(date +%T)) ==="
python bench/serving_ab.py \
    --model "$MODEL_ID" \
    --draft "$DRAFT_ID" \
    --num-speculative-tokens "$SPEC_TOKENS" \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.85 \
    --k-probe "$PROBE_OUT" \
    --num-prompts 64 \
    --max-tokens 128 \
    --prompt-tokens 64 512 1024 2048 3000 \
    --warmup-rounds 2 \
    --rounds 20 \
    --out "$AB_OUT"

echo "=== done ($(date +%T)) ==="
ls -lh "$PROBE_OUT" "$AB_OUT"
