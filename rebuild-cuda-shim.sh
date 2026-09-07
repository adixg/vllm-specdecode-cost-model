#!/usr/bin/env bash
# Rebuilds cuda-shim/ — a CUDA_HOME layout that FlashInfer's linker accepts,
# built from the pip-installed cu13 tree. See SETUP.md.
# Re-run after any change to the nvidia-cuda-* packages.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CU="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13"
SHIM="$ROOT/cuda-shim"

[ -d "$CU" ] || { echo "error: $CU not found (is the venv installed?)" >&2; exit 1; }
[ -e "$CU/lib/libcudart.so.13" ] || { echo "error: libcudart.so.13 missing" >&2; exit 1; }
[ -e /usr/lib/wsl/lib/libcuda.so ] || { echo "error: WSL libcuda.so missing" >&2; exit 1; }

rm -rf "$SHIM"
mkdir -p "$SHIM/lib64/stubs"
ln -sfn "$CU/bin" "$SHIM/bin"
ln -sfn "$CU/include" "$SHIM/include"
ln -sfn "$CU/nvvm" "$SHIM/nvvm"
for f in "$CU"/lib/*; do ln -sfn "$f" "$SHIM/lib64/$(basename "$f")"; done
# dev symlink: pip ships only the versioned .so, but ld needs -lcudart
ln -sfn "$CU/lib/libcudart.so.13" "$SHIM/lib64/libcudart.so"
# under WSL the driver library is not in the pip tree
ln -sfn /usr/lib/wsl/lib/libcuda.so "$SHIM/lib64/stubs/libcuda.so"
ln -sfn "$SHIM/lib64" "$SHIM/lib"
echo "rebuilt $SHIM"
