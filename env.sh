# Source before running vLLM in this project:  source env.sh
export VIRTUAL_ENV="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# WSL2: vLLM disables pinned memory on WSL by default, but the v1 model runner
# requires UVA buffers. Without this: "RuntimeError: UVA is not available".
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

# No system CUDA toolkit; point at the shim over the pip-installed cu13 tree.
# Needed by FlashInfer JIT and vllm.third_party.deep_gemm.
export CUDA_HOME="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/cuda-shim"
