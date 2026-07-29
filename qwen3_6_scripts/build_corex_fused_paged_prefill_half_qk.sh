#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${1:?usage: build_corex_fused_paged_prefill_half_qk.sh OUTPUT_DIR}
COREX_ROOT=${COREX_ROOT:-/usr/local/corex-3.2.3}
TORCH_ROOT=${TORCH_ROOT:-${COREX_ROOT}/lib64/python3/dist-packages/torch}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT=${OUTPUT_DIR}/corex_fused_paged_prefill_half_qk.so

mkdir -p "$OUTPUT_DIR"
"${COREX_ROOT}/bin/clang++" \
    -std=c++17 -O3 -shared -fPIC \
    --cuda-path="${COREX_ROOT}" --cuda-gpu-arch=ivcore10 \
    --no-cuda-version-check -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DBI100_HALF_INPUT_QK=1 \
    -DTORCH_EXTENSION_NAME=corex_fused_paged_prefill \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -I"${TORCH_ROOT}/include" \
    -I"${TORCH_ROOT}/include/torch/csrc/api/include" \
    -I"${TORCH_ROOT}/include/TH" -I"${TORCH_ROOT}/include/THC" \
    -I/usr/local/include/python3.10 \
    "${SCRIPT_DIR}/corex_fused_paged_prefill_half_qk.cu" \
    -L"${TORCH_ROOT}/lib" -L"${COREX_ROOT}/lib64" \
    -Wl,-rpath,"${TORCH_ROOT}/lib" -Wl,-rpath,"${COREX_ROOT}/lib64" \
    -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch \
    -lc10_cuda -lc10 -lcublas -lcudart -o "${OUTPUT}"

test -s "${OUTPUT}"
printf '[ok] CoreX half-QK split4 extension %s\n' "${OUTPUT}"
