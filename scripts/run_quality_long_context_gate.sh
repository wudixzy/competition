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
CASE_ARGS=()

if [[ -n "${BI100_LONG_CONTEXT_CASES:-}" ]]; then
    IFS=',' read -r -a REQUESTED_CASES <<< "$BI100_LONG_CONTEXT_CASES"
    declare -A SEEN_CASES=()
    for case_id in "${REQUESTED_CASES[@]}"; do
        case "$case_id" in
            short_basic_recall|4k_cold_warm_recall|32k_partial_branch|\
32k_multimodal_isolation|65k_multiturn_large_tools|65k_long_tool_result|\
65k_interleaved_sessions|131k_cold_warm_recall|131k_reasoning_recall|\
235k_agent_large_output_budget|235k_partial_branch|near_262k_capacity) ;;
            *)
                echo "unknown BI100_LONG_CONTEXT_CASES id: $case_id" >&2
                exit 2
                ;;
        esac
        if [[ -n "${SEEN_CASES[$case_id]:-}" ]]; then
            echo "duplicate BI100_LONG_CONTEXT_CASES id: $case_id" >&2
            exit 2
        fi
        SEEN_CASES[$case_id]=1
        CASE_ARGS+=(--case "$case_id")
    done
fi

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
    "${CASE_ARGS[@]}" \
    --out "$OUT"
