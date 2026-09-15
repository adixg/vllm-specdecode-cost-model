#!/bin/bash
#SBATCH --job-name=dspark-all
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail          # NOT -e: one failing experiment must not lose the others

# Runs both remaining experiments in one allocation.
#
# Why merged: importing torch + vLLM from Lustre with a cold page cache costs
# ~7.5 minutes. Two separate jobs pay that twice and queue twice. Within one
# job the second Python start finds those modules already in page cache, so it
# is far cheaper.
#
# Ordering: the step trace runs first because it is the more valuable and the
# less proven of the two. The context sweep reuses verify_cost_probe.py, which
# has already completed successfully on this cluster.

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
export HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=1

mkdir -p results logs

echo "=== warming the import cache ($(date +%T)) ==="
python -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__)"
echo "=== imports ready ($(date +%T)) ==="

for repo in models--openbmb--MiniCPM5-2B models--openbmb--MiniCPM5-2B-DSpark; do
    [[ -d "$HF_HOME/hub/$repo" ]] || { echo "ERROR: $repo not cached; download on the login node" >&2; exit 1; }
done

STATUS=0

# --- A. real generation steps: vLLM's own prediction vs its own measurement --
echo
echo "=== A. step trace ($(date +%T)) ==="
python bench/step_trace.py \
    --model openbmb/MiniCPM5-2B \
    --draft openbmb/MiniCPM5-2B-DSpark \
    --num-speculative-tokens 3 \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --num-prompts 64 \
    --max-tokens 128 \
    --prompt-tokens 64 512 2048 \
    --out results/h100_step_trace.json || { echo "STEP TRACE FAILED"; STATUS=1; }

# --- B. does the error change sign above the 8192 profiling point? ----------
# Contexts straddle 8192: three below, three above. 16 reqs x 32768 = 524,288
# KV tokens, inside the ~567k cache, so no block aliasing.
echo
echo "=== B. context sweep across the profiling point ($(date +%T)) ==="
python bench/verify_cost_probe.py \
    --model "${MODEL:-Qwen/Qwen3-1.7B}" \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --num-tokens 1 2 4 8 16 \
    --context-lens 1024 4096 8192 16384 24576 32000 \
    --replays 25 \
    --out results/h100_ctx_straddle.json || { echo "CTX SWEEP FAILED"; STATUS=1; }

python bench/plot_drift.py --in results/h100_ctx_straddle.json || true

echo
echo "=== done ($(date +%T)) exit=$STATUS ==="
ls -la results/
exit $STATUS
