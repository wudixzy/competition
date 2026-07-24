#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
    echo "usage: $0 BASE MODEL_PATH RUNTIME_CONTRACT LABEL RUNTIME_IDENTITY INSTANCE OUT" >&2
    exit 2
fi

BASE=$1
MODEL_PATH=$2
RUNTIME_CONTRACT=$3
LABEL=$4
RUNTIME_IDENTITY=$5
INSTANCE=$6
OUT=$7
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REVISION=$(git -C "$ROOT" rev-parse HEAD)
RUN_ID="${LABEL}-functional-$(date -u +%Y%m%dT%H%M%SZ)-$$"

python3 "$ROOT/tests/quality_gate_api.py" \
    --base "$BASE" \
    --model llm \
    --endpoint-mode direct \
    --allow-bare-engine-n2-skip \
    --tier extended \
    --max-model-len 262144 \
    --truncation-tokens 32768 \
    --label "$LABEL" \
    --source-revision "$REVISION" \
    --runtime-identity "$RUNTIME_IDENTITY" \
    --runtime-contract "$RUNTIME_CONTRACT" \
    --instance "$INSTANCE" \
    --gpu-count 4 \
    --tensor-parallel-size 4 \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$MODEL_PATH" \
    --run-id "$RUN_ID" \
    --out "$OUT"
