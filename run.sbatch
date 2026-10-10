#!/usr/bin/env bash
#SBATCH --job-name=dspark-profile-drift
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

# Submit with your own PACE account and QoS, e.g.:
# sbatch -A <YOUR_PACE_ACCOUNT> -q inferno run.sbatch
# Use your cluster's allowed QoS; don't blindly copy the example.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Use a previously prepared vLLM environment (install once, not per job).
# You can override: VENV_PATH=/path/to/venv sbatch ...
VENV_PATH="${VENV_PATH:-$PWD/.venv}"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Missing $VENV_PATH/bin/activate; set VENV_PATH to your vLLM venv" >&2
  exit 2
fi
source "$VENV_PATH/bin/activate"
python -c 'import vllm; print("vLLM", vllm.__version__)'
nvidia-smi

export VLLM_USE_V2_MODEL_RUNNER=1
# Required for shipping the trusted local Python callable into the vLLM worker.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
mkdir -p results
srun python -u experiment.py \
  --model "$MODEL" \
  --max-num-seqs 8 \
  --max-model-len 16384 \
  --num-tokens 64 128 256 \
  --baseline-context 8192 \
  --train-contexts 512 4096 \
  --test-contexts 1024 2048 12288 \
  --warmup 3 \
  --repeats 20 \
  --out "results/profile_drift_${SLURM_JOB_ID:-manual}.json"

