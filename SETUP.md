# vLLM setup on WSL2 + RTX 4060 Laptop

Working environment for the speculative-decoding experiments. Everything here was
needed to get `vllm==0.28.0` to start at all on this machine; none of it is optional.

## Hardware / platform baseline

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB (sm_89) |
| Windows driver | **616.64** (CUDA 13.0) |
| Platform | WSL2, kernel 6.6.87.2-microsoft-standard-WSL2 |
| Python | 3.12 (system Python is 3.14 — too new for vLLM wheels) |
| vLLM / torch | 0.28.0 / 2.13.0+cu130 |

### The driver is a hard requirement

vLLM 0.28.0 pins `torch==2.13.0`, and **torch 2.11.0 and later publish CUDA 13
wheels only** — there is no cu128 build to fall back to. CUDA 13 needs driver
**r580+**. The original driver here was 560.94 (CUDA 12.6), which produced:

    RuntimeError: The NVIDIA driver on your system is too old (found version 12060)

Under WSL2 there is no Linux-side driver to install: `/usr/lib/wsl/lib/libcuda.so`
is injected by the Windows host driver. Fix it on Windows, then `wsl --shutdown`.

If you ever must stay on a CUDA 12 driver, the last vLLM release on torch 2.10.0
(CUDA 12) is **0.19.1**. Everything from 0.20.0 up is CUDA 13.

## Running

    source env.sh

`env.sh` sets the two variables below. Both are required on every run.

### `VLLM_WSL2_ENABLE_PIN_MEMORY=1`

Without it: `RuntimeError: UVA is not available`.

`vllm/platforms/cuda.py:290` detects WSL and, on kernels >= 4.19.121, reports
pinned memory as *supported but disabled by default*, returning
`envs.VLLM_WSL2_ENABLE_PIN_MEMORY` (default `0`). vLLM 0.28's v1 model runner
allocates UVA buffers unconditionally, so startup dies. The comment at
`envs.py:2052` says it directly: "Set to 1 when pinned memory or UVA is required
(e.g. CPU offloading or v2 model runner)."

### `CUDA_HOME=<repo>/cuda-shim`

There is no system CUDA toolkit and none is needed — do **not**
`apt install nvidia-cuda-toolkit`, which is CUDA 12.x on Ubuntu 24.04 and would
reintroduce a major-version mismatch. The toolkit ships as pip packages under
`.venv/lib/python3.12/site-packages/nvidia/cu13`.

Two consumers need `CUDA_HOME`:
- **FlashInfer** JIT-compiles its sampling kernels on first use.
- **`vllm.third_party.deep_gemm`**, which otherwise logs
  `Module ... was found but failed to import / AssertionError` from
  `_find_cuda_home()`. (Harmless for bf16 models — those are FP8 kernels — but it
  is a real failed import, not noise.)

## `cuda-shim/` — why it exists

FlashInfer links with `-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib64/stubs -lcudart -lcuda`.
The pip layout does not match that in three ways:

1. it has `lib/`, not `lib64/`
2. it ships `libcudart.so.13` but no `libcudart.so` dev symlink
3. it has no `libcuda.so` at all — under WSL that lives in `/usr/lib/wsl/lib/`

Without the shim: `ld: cannot find -lcudart` / `ld: cannot find -lcuda`.

`cuda-shim/` is symlinks only, no copied binaries. It deliberately sits outside
`site-packages` so that `uv pip install` cannot wipe it. Rebuild it with
`./rebuild-cuda-shim.sh` after any change to the `nvidia-cuda-*` packages.

## CRITICAL: the CUDA component pins are load-bearing and unprotected

These six packages must all agree on **13.0**:

    nvidia-cuda-nvcc     13.0.88     nvidia-cuda-cccl     13.0.85
    nvidia-cuda-crt      13.0.88     nvidia-cuda-runtime  13.0.96
    nvidia-nvvm          13.0.88     nvidia-cuda-nvrtc    13.0.88

`nvidia-cuda-runtime` is hard-pinned to `13.0.96` by the `cuda-toolkit 13.0.3.0`
metapackage, but **`tilelang` requires `nvidia-cuda-nvcc>=13.0.48` with no upper
bound** and `humming-kernels` requires it unpinned. A fresh resolve therefore pulls
nvcc/cccl/crt/nvvm up to 13.3.x while runtime stays at 13.0.96, and FlashInfer
breaks in two distinct ways:

- **13.3 nvcc + 13.0 headers** ->
  `error "CUDA compiler and CUDA toolkit headers are incompatible"`
  from `cuda_toolkit.h:41`, which checks `_CCCL_CUDACC_EQUAL(CUDART_VERSION/1000, ...)`.
  Installed headers define `CUDART_VERSION 13000` -> CTK 13.0; nvcc reported 13.3.
- **13.3 nvvm/crt + 13.0 ptxas** ->
  `ptxas fatal: Unsupported .version 9.3; current version is '9.0'`.
  Downgrading nvcc alone is NOT enough: `nvidia-nvvm` and `nvidia-cuda-crt` are
  unpinned dependencies *of nvcc*, and the front-end keeps emitting PTX ISA 9.3
  for a 9.0 assembler.

**Any `uv pip install` in this venv can silently undo this.** After installing
anything, re-check:

    uv pip list | grep -E 'nvidia-(cuda-(nvcc|crt|runtime|nvrtc|cccl)|nvvm)'

and if versions drift, restore with:

    uv pip install 'nvidia-cuda-nvcc==13.0.88' 'nvidia-cuda-cccl==13.0.85' \
                   'nvidia-nvvm==13.0.88' 'nvidia-cuda-crt==13.0.88'

`requirements-lock.txt` is a full `uv pip freeze` of the known-good state.

## Verifying a good environment

    source env.sh
    python -c "
    from vllm.utils.import_utils import _has_module
    from vllm.utils.platform_utils import is_uva_available
    import vllm.envs as e, torch
    print('cuda      ', torch.cuda.is_available())
    print('uva       ', is_uva_available())
    print('deep_gemm ', _has_module('vllm.third_party.deep_gemm'))
    print('fi sampler', e.VLLM_USE_FLASHINFER_SAMPLER)
    "

All four must be `True`. Known-good smoke test result:

    Using FLASH_ATTN attention backend
    GPU KV cache size: 256,608 tokens
    OUTPUT>>> ' Paris. It is the largest city in Europe...'

First FlashInfer run stalls for a few minutes compiling into `~/.cache/flashinfer`.

## Note for the benchmarks

vLLM selects **FLASH_ATTN**, not FlashInfer, for attention; `VLLM_USE_FLASHINFER_SAMPLER`
only affects top-k/top-p sampling, and the speculative-decoding experiments run at
temperature 0 (argmax). So none of the FlashInfer work above changes throughput —
it buys a correctly aligned toolchain, not speed.

Do not change the attention backend partway through a batch-size sweep; treat it as
a separate variable or the crossover point becomes uninterpretable.
