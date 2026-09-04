#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${1:?usage: build_corex_fused_paged_prefill_variant_runtime.sh OUTPUT_DIR VARIANT}
VARIANT=${2:?usage: build_corex_fused_paged_prefill_variant_runtime.sh OUTPUT_DIR VARIANT}
COREX_ROOT=${COREX_ROOT:-/usr/local/corex-3.2.3}
TORCH_ROOT=${TORCH_ROOT:-${COREX_ROOT}/lib64/python3/dist-packages/torch}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${VARIANT}" in
    m1_109_fp32_qk)
        SOURCE=${SCRIPT_DIR}/corex_fused_paged_prefill_split4.cu
        ;;
    m1_162_fp16_qk)
        SOURCE=${SCRIPT_DIR}/corex_fused_paged_prefill_fp16_qk.cu
        ;;
    *)
        printf 'unsupported fused-prefill variant: %s\n' "${VARIANT}" >&2
        exit 2
        ;;
esac

mkdir -p "${OUTPUT_DIR}"
OUTPUT=${OUTPUT_DIR}/corex_fused_paged_prefill_${VARIANT}.so

"${COREX_ROOT}/bin/clang++" \
    -std=c++17 -O3 -shared -fPIC \
    --cuda-path="${COREX_ROOT}" --cuda-gpu-arch=ivcore10 \
    --no-cuda-version-check -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DTORCH_EXTENSION_NAME=corex_fused_paged_prefill \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -I"${TORCH_ROOT}/include" \
    -I"${TORCH_ROOT}/include/torch/csrc/api/include" \
    -I"${TORCH_ROOT}/include/TH" -I"${TORCH_ROOT}/include/THC" \
    -I/usr/local/include/python3.10 \
    "${SOURCE}" \
    -L"${TORCH_ROOT}/lib" -L"${COREX_ROOT}/lib64" \
    -Wl,-rpath,"${TORCH_ROOT}/lib" -Wl,-rpath,"${COREX_ROOT}/lib64" \
    -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch \
    -lc10_cuda -lc10 -lcublas -lcudart -o "${OUTPUT}"

test -s "${OUTPUT}"
printf '[ok] CoreX fused-prefill variant=%s extension=%s\n' \
    "${VARIANT}" "${OUTPUT}"
