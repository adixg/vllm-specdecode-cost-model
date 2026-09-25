# Shared environment for every PACE job. Source it from an sbatch script:
#
#     source "$SLURM_SUBMIT_DIR/pace/env.sh"
#
# (Not to be confused with env.sh at the repo root, which is for the local
# WSL setup and is irrelevant here.)
#
# Why this file exists: HF_HOME used to be set only in an interactive login
# shell and reached jobs via `sbatch --export=ALL`. Jobs 5922739 and 5922740
# were submitted from a shell where it was not set, and with HF_HUB_OFFLINE=1
# vLLM looked in the default cache, found nothing, and could not download.
# Nothing a job needs should depend on the shell it was submitted from.

# Compute nodes have no internet, so models must already be cached.
export HF_HOME="${HF_HOME:-$HOME/scratch/hf}"
export HF_HUB_OFFLINE=1

# DSpark and adaptive verification exist only in the V2 model runner.
export VLLM_USE_V2_MODEL_RUNNER=1

if [[ ! -d "$HF_HOME/hub" ]]; then
    echo "error: no HF cache at $HF_HOME/hub" >&2
    echo "       set HF_HOME to wherever the models live, or download them" >&2
    echo "       on a login node (compute nodes have no internet)." >&2
    exit 1
fi

# require_hf_repo org/name  -> fail now, not eight minutes into the job.
require_hf_repo() {
    local id="$1" dir="$HF_HOME/hub/models--${1/\//--}"
    if [[ ! -d "$dir" ]]; then
        echo "error: $id is not cached at $dir" >&2
        echo "       on a login node, run:" >&2
        echo "         export HF_HOME=$HF_HOME" >&2
        echo "         HF_HUB_OFFLINE=0 hf download $id" >&2
        exit 1
    fi
}

echo "HF cache: $HF_HOME/hub"
