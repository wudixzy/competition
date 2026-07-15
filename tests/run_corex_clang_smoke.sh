#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COREX_ROOT=${COREX_ROOT:-/usr/local/corex-3.2.3}
TORCH_ROOT=${TORCH_ROOT:-${COREX_ROOT}/lib64/python3/dist-packages/torch}
OUT_DIR=${1:-/tmp/corex-clang-smoke}
GPU=${GPU:-0}

mkdir -p "${OUT_DIR}"
export LD_LIBRARY_PATH="${TORCH_ROOT}/lib:${COREX_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

"${COREX_ROOT}/bin/clang++" \
    -std=c++17 -O2 \
    --cuda-path="${COREX_ROOT}" \
    --cuda-gpu-arch=ivcore10 \
    --no-cuda-version-check \
    "${ROOT_DIR}/tests/corex_extension_smoke.cu" \
    -L"${COREX_ROOT}/lib64" \
    -Wl,-rpath,"${COREX_ROOT}/lib64" \
    -lcudart \
    -o "${OUT_DIR}/corex_extension_smoke"

CUDA_VISIBLE_DEVICES="${GPU}" "${OUT_DIR}/corex_extension_smoke"

"${COREX_ROOT}/bin/clang++" \
    -std=c++17 -O2 -shared -fPIC \
    --cuda-path="${COREX_ROOT}" \
    --cuda-gpu-arch=ivcore10 \
    --no-cuda-version-check \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DTORCH_EXTENSION_NAME=corex_torch_smoke \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -I"${TORCH_ROOT}/include" \
    -I"${TORCH_ROOT}/include/torch/csrc/api/include" \
    -I"${TORCH_ROOT}/include/TH" \
    -I"${TORCH_ROOT}/include/THC" \
    -I/usr/local/include/python3.10 \
    "${ROOT_DIR}/tests/corex_torch_extension_smoke.cu" \
    -L"${TORCH_ROOT}/lib" \
    -L"${COREX_ROOT}/lib64" \
    -Wl,-rpath,"${TORCH_ROOT}/lib" \
    -Wl,-rpath,"${COREX_ROOT}/lib64" \
    -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch \
    -lc10_cuda -lc10 -lcudart \
    -o "${OUT_DIR}/corex_torch_smoke.so"

CUDA_VISIBLE_DEVICES="${GPU}" \
COREX_TORCH_SMOKE_PATH="${OUT_DIR}/corex_torch_smoke.so" \
python3 - <<'PY'
import importlib.util
import os

import torch

path = os.environ["COREX_TORCH_SMOKE_PATH"]
spec = importlib.util.spec_from_file_location("corex_torch_smoke", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
value = torch.arange(256, device="cuda", dtype=torch.float32)
output = module.add_one(value)
torch.cuda.synchronize()
assert torch.equal(output, value + 1)
assert torch.equal(value, torch.arange(256, device="cuda", dtype=torch.float32))
print("COREX_TORCH_EXTENSION_SMOKE_OK", output[-1].item())
PY
