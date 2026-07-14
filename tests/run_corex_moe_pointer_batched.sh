#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COREX_ROOT=${COREX_ROOT:-/usr/local/corex-3.2.3}
TORCH_ROOT=${TORCH_ROOT:-${COREX_ROOT}/lib64/python3/dist-packages/torch}
OUT_DIR=${1:-/tmp/corex-moe-pointer-batched}
GPU=${GPU:-0}

mkdir -p "${OUT_DIR}"
export LD_LIBRARY_PATH="${TORCH_ROOT}/lib:${COREX_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

"${COREX_ROOT}/bin/clang++" \
    -std=c++17 -O3 -shared -fPIC \
    --cuda-path="${COREX_ROOT}" \
    --cuda-gpu-arch=ivcore10 \
    --no-cuda-version-check \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DTORCH_EXTENSION_NAME=corex_moe_pointer_batched \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -I"${TORCH_ROOT}/include" \
    -I"${TORCH_ROOT}/include/torch/csrc/api/include" \
    -I"${TORCH_ROOT}/include/TH" \
    -I"${TORCH_ROOT}/include/THC" \
    -I/usr/local/include/python3.10 \
    "${ROOT_DIR}/tests/corex_moe_pointer_batched_ext.cu" \
    -L"${TORCH_ROOT}/lib" \
    -L"${COREX_ROOT}/lib64" \
    -Wl,-rpath,"${TORCH_ROOT}/lib" \
    -Wl,-rpath,"${COREX_ROOT}/lib64" \
    -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch \
    -lc10_cuda -lc10 -lcublas -lcudart \
    -o "${OUT_DIR}/corex_moe_pointer_batched.so"

CUDA_VISIBLE_DEVICES="${GPU}" python3 \
    "${ROOT_DIR}/tests/bench_moe_pointer_batched.py" \
    --extension "${OUT_DIR}/corex_moe_pointer_batched.so" \
    --device cuda:0 \
    --out "${OUT_DIR}/result.json"
