#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
    echo "usage: $0 BASE MODEL_PATH RUNTIME_CONTRACT TRACE_LOG LABEL RUNTIME_IDENTITY INSTANCE OUT" >&2
    exit 2
fi

BASE=$1
MODEL_PATH=$2
RUNTIME_CONTRACT=$3
TRACE_LOG=$4
LABEL=$5
RUNTIME_IDENTITY=$6
INSTANCE=$7
OUT=$8
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REVISION=$(git -C "$ROOT" rev-parse HEAD)
RUN_ID="${LABEL}-long-context-$(date -u +%Y%m%dT%H%M%SZ)-$$"

python3 "$ROOT/tests/long_context_quality_api.py" \
    --base "$BASE" \
    --model-path "$MODEL_PATH" \
    --served-model-name llm \
    --chat-template-kwargs-mode direct \
    --tier extended \
    --cache-trace-file "$TRACE_LOG" \
    --label "$LABEL" \
    --source-revision "$REVISION" \
    --runtime-identity "$RUNTIME_IDENTITY" \
    --runtime-contract "$RUNTIME_CONTRACT" \
    --instance "$INSTANCE" \
    --gpu-count 4 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --run-id "$RUN_ID" \
    --fresh-service-attested \
    --out "$OUT"
