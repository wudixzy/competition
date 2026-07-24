#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"
AB_DIR=${M1_49_AB_DIR:-$ROOT/bench_runs/m1_49/hybrid_kv_ab}
RUN_ROOT=${M1_49_LONG_RUN_ROOT:-$ROOT/bench_runs/m1_49/full_attention_long}
RESUME_FROM=${M1_49_LONG_RESUME_FROM:-}
MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
RUN_ID=${M1_49_LONG_RUN_ID:-m149-full-attention-long-20260722}
ACTIVE_PID=""
ACTIVE_PGID=""

export LD_LIBRARY_PATH="/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}"
export HOST=0.0.0.0
export PORT=8000
export MODEL_PATH
export ENABLE_CUSTOM_IPC=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export BI100_MOE_COREX_DIRECT_ROUTED=1
export BI100_GDN_COREX_PACKED_DECODE=1
export BI100_CPU_KV_OFFLOAD=0
export BI100_GDN_CACHE_POLICY=admission64
export BI100_GDN_RESTORE_MODE=direct
export BI100_CACHE_TRACE=0
export BI100_ATTN_COREX_FUSED_PREFILL=0
export BI100_KV_EVICTION_POLICY=lru
export BI100_HYBRID_KV_ACCOUNTING=full_attention
export BI100_RUNTIME_WORKDIR=/tmp/m1-49-long-runtime
export BI100_PROFILE=0
export BI100_PROFILE_INCLUDE_STARTUP=0
export BI100_PAGED_ATTN_DIAGNOSTICS=0
export BI100_GDN_ALLOW_NAN_ZERO=0
export BI100_GDN_FINITE_CHECK=0
unset NUM_GPU_BLOCKS_OVERRIDE BI100_MOE_COREX_THREE_BUCKET

if [[ -z "${BI100_RUNTIME_SITE_PACKAGES:-}" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/vllm" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/transformers" ]]; then
    echo "M1-49 requires an atomic bare-host runtime overlay" >&2
    exit 3
fi
export BI100_RUNTIME_SITE_PACKAGES
export PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages:${PYTHONPATH:-}"

require_zero_rc() {
    local path=$1
    if [[ ! -f "$path" ]] || [[ $(<"$path") != 0 ]]; then
        echo "required successful gate is missing: $path" >&2
        exit 3
    fi
}

require_nonzero_rc() {
    local path=$1
    local value=""
    if [[ -f "$path" ]]; then
        IFS= read -r value < "$path" || true
    fi
    if [[ ! "$value" =~ ^[0-9]+$ ]] || [[ "$value" == 0 ]]; then
        echo "required failed gate is missing: $path" >&2
        exit 3
    fi
}

require_zero_rc "$AB_DIR/overall.rc"
require_zero_rc "$AB_DIR/cleanup.rc"
require_zero_rc "$AB_DIR/comparison.rc"
require_zero_rc "$AB_DIR/preflight_comparison_final.rc"
if [[ ! -f "$AB_DIR/comparison.json" ]]; then
    echo "M1-49 A/B comparison is missing: $AB_DIR/comparison.json" >&2
    exit 3
fi
python3 - "$AB_DIR/comparison.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("schema") != "bi100-hybrid-kv-accounting-ab-v1":
    raise SystemExit("M1-49 A/B comparison schema is invalid")
if report.get("version") != 1 or report.get("qualified") is not True:
    raise SystemExit("M1-49 A/B comparison is not qualified")
candidate = (report.get("startup") or {}).get("candidate") or {}
if candidate.get("mode") != "full_attention":
    raise SystemExit("M1-49 A/B candidate is not full_attention")
if candidate.get("observed_attention_layers") != 10:
    raise SystemExit("M1-49 A/B candidate did not allocate ten KV layers")
PY

SMOKE_REPORT="$RUN_ROOT/smoke.json"
MULTIMODAL_REPORT="$RUN_ROOT/multimodal_qualification.json"
if [[ -n "$RESUME_FROM" ]]; then
    if [[ ! -d "$RESUME_FROM" ]]; then
        echo "M1-49 resume source is missing: $RESUME_FROM" >&2
        exit 3
    fi
    for gate in \
            cleanup startup startup_gate runtime_contract smoke \
            multimodal_prefix multimodal_qualification \
            long_131k_api long_131k_safe long_235k_api long_235k_safe; do
        require_zero_rc "$RESUME_FROM/${gate}.rc"
    done
    require_nonzero_rc "$RESUME_FROM/overall.rc"
    require_nonzero_rc "$RESUME_FROM/long_262k_api.rc"
    for evidence in \
            startup_gate.json smoke.json multimodal_qualification.json \
            long_131k_exact/long_context_summary.json \
            long_235k_warm_repeat/long_context_summary.json server.log; do
        if [[ ! -f "$RESUME_FROM/$evidence" ]]; then
            echo "M1-49 resume evidence is missing: $RESUME_FROM/$evidence" >&2
            exit 3
        fi
    done
    if grep -Eiq 'CUDA error|SIGSEGV|Fatal Python error|out of memory|worker process.*died|Gloo.*failed|AssertionError' \
            "$RESUME_FROM/server.log"; then
        echo "M1-49 resume source contains a fatal server signature" >&2
        exit 3
    fi
    python3 - "$AB_DIR/comparison.json" "$RESUME_FROM/startup_gate.json" <<'PY'
import json
import sys

ab = json.load(open(sys.argv[1], encoding="utf-8"))
source = json.load(open(sys.argv[2], encoding="utf-8"))
candidate = (ab.get("startup") or {}).get("candidate") or {}
if source.get("qualified") is not True:
    raise SystemExit("M1-49 resume startup is not qualified")
for field in (
    "mode",
    "observed_attention_layers",
    "runtime_contract_invariant_sha256",
):
    if source.get(field) != candidate.get(field):
        raise SystemExit(f"M1-49 resume startup differs from A/B in {field}")
PY
    SMOKE_REPORT="$RESUME_FROM/smoke.json"
    MULTIMODAL_REPORT="$RESUME_FROM/multimodal_qualification.json"
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model directory is missing: $MODEL_PATH" >&2
    exit 4
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "M1-49 long-context output already exists: $RUN_ROOT" >&2
    exit 5
fi
mkdir -p "$RUN_ROOT"
if [[ -n "$RESUME_FROM" ]]; then
    printf '%s\n' "$RESUME_FROM" > "$RUN_ROOT/resumed_from.txt"
fi

health() {
    python3 -c 'import urllib.request; urllib.request.urlopen(
        "http://127.0.0.1:8000/health", timeout=5).read()' >/dev/null 2>&1
}

port_is_free() {
    python3 - <<'PY'
import socket

sock = socket.socket()
try:
    sock.bind(("127.0.0.1", 8000))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

wait_for_port_free() {
    for _ in $(seq 1 180); do
        port_is_free && return 0
        sleep 1
    done
    echo "port 8000 remained busy; refusing to mix service lifetimes" >&2
    return 1
}

stop_service() {
    if [[ -n "$ACTIVE_PGID" ]]; then
        bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID" || return $?
    elif [[ -n "$ACTIVE_PID" ]]; then
        kill -TERM "$ACTIVE_PID" 2>/dev/null || true
        echo "service PID $ACTIVE_PID has no verified process group" >&2
        return 2
    fi
    ACTIVE_PID=""
    ACTIVE_PGID=""
}

finish() {
    local rc=$?
    local cleanup_rc=0
    trap - EXIT
    set +e
    stop_service
    cleanup_rc=$?
    if [[ $cleanup_rc -eq 0 ]]; then
        wait_for_port_free
        cleanup_rc=$?
    fi
    if [[ $cleanup_rc -ne 0 ]]; then
        echo "M1-49 long-context service cleanup failed" >&2
        rc=1
    fi
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/cleanup.rc"
    printf '%s\n' "$rc" > "$RUN_ROOT/overall.rc"
    exit "$rc"
}
trap finish EXIT

run_preflight() {
    local label=$1
    set +e
    timeout --signal=TERM --kill-after=10s 150s \
        python3 "$ROOT/tests/bi100_preflight.py" \
        --gpus 0,1,2,3 --timeout-s 25 --matmul-size 1024 \
        --json-out "$RUN_ROOT/preflight_${label}.json" \
        > "$RUN_ROOT/preflight_${label}.stdout" \
        2> "$RUN_ROOT/preflight_${label}.stderr"
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$RUN_ROOT/preflight_${label}.rc"
    if [[ $rc -ne 0 ]]; then
        echo "M1-49 four-GPU preflight failed at $label" >&2
        return "$rc"
    fi
}

run_gate() {
    local name=$1
    local timeout_s=$2
    shift 2
    set +e
    timeout --signal=TERM --kill-after=30s "${timeout_s}s" "$@" \
        > "$RUN_ROOT/${name}.stdout" 2> "$RUN_ROOT/${name}.stderr"
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$RUN_ROOT/${name}.rc"
    if [[ $rc -ne 0 ]]; then
        echo "M1-49 long-context gate failed: $name rc=$rc" >&2
        return "$rc"
    fi
    health
}

fatal_scan() {
    if grep -Eiq 'CUDA error|SIGSEGV|Fatal Python error|out of memory|worker process.*died|Gloo.*failed|AssertionError' \
            "$RUN_ROOT/server.log"; then
        grep -Ein 'CUDA error|SIGSEGV|Fatal Python error|out of memory|worker process.*died|Gloo.*failed|AssertionError' \
            "$RUN_ROOT/server.log" > "$RUN_ROOT/fatal_scan.txt" || true
        printf '%s\n' 1 > "$RUN_ROOT/fatal_scan.rc"
        return 1
    fi
    : > "$RUN_ROOT/fatal_scan.txt"
    printf '%s\n' 0 > "$RUN_ROOT/fatal_scan.rc"
}

run_preflight before_long
wait_for_port_free
setsid "$ROOT/launch_service" > "$RUN_ROOT/server.log" 2>&1 < /dev/null &
ACTIVE_PID=$!
ACTIVE_PGID=$ACTIVE_PID
printf '%s\n' "$ACTIVE_PID" > "$RUN_ROOT/server.pid"
OBSERVED_PGID=""
for _ in $(seq 1 20); do
    OBSERVED_PGID=$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null | tr -d ' ')
    [[ -n "$OBSERVED_PGID" ]] && break
    sleep 1
done
if [[ -z "$OBSERVED_PGID" || "$OBSERVED_PGID" != "$ACTIVE_PGID" ]]; then
    echo "service did not enter an isolated process group" >&2
    exit 1
fi
printf '%s\n' "$ACTIVE_PGID" > "$RUN_ROOT/server.pgid"

for _ in $(seq 1 360); do
    if health; then
        printf '%s\n' 0 > "$RUN_ROOT/startup.rc"
        break
    fi
    if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
        printf '%s\n' 1 > "$RUN_ROOT/startup.rc"
        tail -120 "$RUN_ROOT/server.log" >&2 || true
        exit 1
    fi
    sleep 10
done
require_zero_rc "$RUN_ROOT/startup.rc"

grep -Fq '[BI100] fixed evaluator contract;' "$RUN_ROOT/server.log"
grep -Fq '[BI100] fixed kernels; moe_direct=1 gdn_packed=1' "$RUN_ROOT/server.log"
grep -Fq '[BI100] GDN cache; policy=admission64 restore=direct' "$RUN_ROOT/server.log"
grep -Fq '[BI100] M1-49 runtime contract; accounting=full_attention' \
    "$RUN_ROOT/server.log"
printf '%s\n' 0 > "$RUN_ROOT/runtime_contract.rc"

run_gate startup_gate 180 \
    python3 "$ROOT/tests/hybrid_kv_startup_gate.py" \
    "$RUN_ROOT/server.log" --mode full_attention --model-path "$MODEL_PATH" \
    --max-model-len 262144 --block-size 16 --tensor-parallel-size 4 \
    --out "$RUN_ROOT/startup_gate.json"

mkdir -p "$RUN_ROOT/long_131k_exact" "$RUN_ROOT/long_235k_warm_repeat"
if [[ -z "$RESUME_FROM" ]]; then
    run_gate smoke 900 \
        python3 "$ROOT/tests/smoke_api.py" \
        --base http://127.0.0.1:8000 --mode quick \
        --json-out "$RUN_ROOT/smoke.json"

    run_gate multimodal_prefix 1800 \
        python3 "$ROOT/tests/multimodal_prefix_isolation_api.py" \
        --base http://127.0.0.1:8000 \
        --run-id "${RUN_ID}-multimodal" \
        --json-out "$RUN_ROOT/multimodal_prefix.json"
    run_gate multimodal_qualification 60 \
        python3 "$ROOT/tests/qualify_multimodal_prefix_isolation.py" \
        --source "$RUN_ROOT/multimodal_prefix.json" \
        --out "$RUN_ROOT/multimodal_qualification.json"

    run_gate long_131k_api 6000 \
        python3 "$ROOT/tests/long_context_api.py" \
        --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
        --target-prompt-tokens 131000 --max-tokens 256 \
        --min-completion-tokens 256 --max-model-len 262144 \
        --min-cached-tokens 130992 --max-first-cached-tokens 0 \
        --equivalence-mode exact \
        --timeout-s 2400 --run-id "${RUN_ID}-131k" \
        --output-dir "$RUN_ROOT/long_131k_exact"
    LONG_131K_INPUT="$RUN_ROOT/long_131k_exact/long_context_summary.json"

    run_gate long_235k_api 12600 \
        python3 "$ROOT/tests/long_context_api.py" \
        --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
        --target-prompt-tokens 235000 --max-tokens 1000 \
        --min-completion-tokens 1000 --max-model-len 262144 \
        --min-cached-tokens 234992 --max-first-cached-tokens 0 \
        --equivalence-mode warm-repeat \
        --timeout-s 4200 --run-id "${RUN_ID}-235k" \
        --output-dir "$RUN_ROOT/long_235k_warm_repeat"
    LONG_235K_INPUT="$RUN_ROOT/long_235k_warm_repeat/long_context_summary.json"
else
    LONG_131K_INPUT="$RESUME_FROM/long_131k_exact/long_context_summary.json"
    LONG_235K_INPUT="$RESUME_FROM/long_235k_warm_repeat/long_context_summary.json"
fi

run_gate long_131k_safe 60 \
    python3 "$ROOT/tests/qualify_long_context_summary.py" \
    --input "$LONG_131K_INPUT" \
    --out "$RUN_ROOT/long_131k_exact/safe_gate.json" \
    --target-prompt-tokens 131000 --max-tokens 256 \
    --min-cached-tokens 130992 --max-first-cached-tokens 0 \
    --min-completion-tokens 256 \
    --equivalence-mode exact

run_gate long_235k_safe 60 \
    python3 "$ROOT/tests/qualify_long_context_summary.py" \
    --input "$LONG_235K_INPUT" \
    --out "$RUN_ROOT/long_235k_warm_repeat/safe_gate.json" \
    --target-prompt-tokens 235000 --max-tokens 1000 \
    --min-cached-tokens 234992 --max-first-cached-tokens 0 \
    --min-completion-tokens 1000 \
    --equivalence-mode warm-repeat

mkdir -p "$RUN_ROOT/long_262k_capacity"
run_gate long_262k_api 10000 \
    python3 "$ROOT/tests/long_context_api.py" \
    --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
    --target-prompt-tokens 262000 --max-tokens 16 \
    --min-completion-tokens 16 --max-model-len 262144 \
    --min-cached-tokens 261984 --max-first-cached-tokens 32 \
    --equivalence-mode exact \
    --timeout-s 4800 --run-id "${RUN_ID}-262k" \
    --output-dir "$RUN_ROOT/long_262k_capacity"
run_gate long_262k_safe 60 \
    python3 "$ROOT/tests/qualify_long_context_summary.py" \
    --input "$RUN_ROOT/long_262k_capacity/long_context_summary.json" \
    --out "$RUN_ROOT/long_262k_capacity/safe_gate.json" \
    --target-prompt-tokens 262000 --max-tokens 16 \
    --min-cached-tokens 261984 --max-first-cached-tokens 32 \
    --min-completion-tokens 16 \
    --equivalence-mode exact

fatal_scan
stop_service
wait_for_port_free
run_preflight after_long
set +e
python3 "$ROOT/tests/compare_bi100_preflights.py" \
    --preflight "before_long=$RUN_ROOT/preflight_before_long.json" \
    --preflight "after_long=$RUN_ROOT/preflight_after_long.json" \
    --max-free-memory-drop-bytes 1073741824 \
    --out "$RUN_ROOT/preflight_comparison.json" \
    > "$RUN_ROOT/preflight_comparison.stdout" \
    2> "$RUN_ROOT/preflight_comparison.stderr"
preflight_comparison_rc=$?
set -e
printf '%s\n' "$preflight_comparison_rc" \
    > "$RUN_ROOT/preflight_comparison.rc"
if [[ $preflight_comparison_rc -ne 0 ]]; then
    echo "M1-49 long-context GPU preflight comparison failed" >&2
    exit "$preflight_comparison_rc"
fi

set +e
python3 "$ROOT/tests/qualify_m1_49_long_context.py" \
    --ab "$AB_DIR/comparison.json" \
    --startup "$RUN_ROOT/startup_gate.json" \
    --preflight "$RUN_ROOT/preflight_comparison.json" \
    --smoke "$SMOKE_REPORT" \
    --multimodal "$MULTIMODAL_REPORT" \
    --long-131k "$RUN_ROOT/long_131k_exact/safe_gate.json" \
    --long-235k "$RUN_ROOT/long_235k_warm_repeat/safe_gate.json" \
    --long-262k "$RUN_ROOT/long_262k_capacity/safe_gate.json" \
    --out "$RUN_ROOT/qualification.json" \
    > "$RUN_ROOT/qualification.stdout" \
    2> "$RUN_ROOT/qualification.stderr"
qualification_rc=$?
set -e
printf '%s\n' "$qualification_rc" > "$RUN_ROOT/qualification.rc"
if [[ $qualification_rc -ne 0 ]]; then
    echo "M1-49 long-context qualification failed" >&2
    exit "$qualification_rc"
fi

echo "M1-49 full-attention long-context gates passed"
