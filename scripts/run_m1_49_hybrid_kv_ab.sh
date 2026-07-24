#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"
RUN_ROOT=${M1_49_RUN_ROOT:-$ROOT/bench_runs/m1_49/hybrid_kv_ab}
MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
RUN_ID=${M1_49_RUN_ID:-m149-hybrid-kv-fixed-20260721}
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
export BI100_RUNTIME_WORKDIR=/tmp/m1-49-runtime
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

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model directory is missing: $MODEL_PATH" >&2
    exit 4
fi
if [[ -e "$RUN_ROOT/legacy40" || -e "$RUN_ROOT/full_attention" ]]; then
    echo "M1-49 output already exists; refusing to mix service lifetimes: $RUN_ROOT" >&2
    exit 5
fi
mkdir -p "$RUN_ROOT"

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
        wait "$ACTIVE_PID" 2>/dev/null || true
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
    if [[ $cleanup_rc -ne 0 ]]; then
        echo "M1-49 service cleanup failed" >&2
        rc=1
    fi
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/cleanup.rc"
    printf '%s\n' "$rc" > "$RUN_ROOT/overall.rc"
    exit "$rc"
}
trap finish EXIT

fatal_scan() {
    local log=$1
    local output=$2
    if grep -Eiq 'CUDA error|SIGSEGV|Fatal Python error|out of memory|worker process.*died|Gloo.*failed|AssertionError' "$log"; then
        grep -Ein 'CUDA error|SIGSEGV|Fatal Python error|out of memory|worker process.*died|Gloo.*failed|AssertionError' \
            "$log" > "$output" || true
        return 1
    fi
    : > "$output"
}

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

compare_preflights() {
    local label=$1
    shift
    local args=()
    local stage
    for stage in "$@"; do
        args+=(--preflight "$stage=$RUN_ROOT/preflight_${stage}.json")
    done
    set +e
    python3 "$ROOT/tests/compare_bi100_preflights.py" \
        "${args[@]}" --out "$RUN_ROOT/preflight_comparison_${label}.json" \
        > "$RUN_ROOT/preflight_comparison_${label}.stdout" \
        2> "$RUN_ROOT/preflight_comparison_${label}.stderr"
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$RUN_ROOT/preflight_comparison_${label}.rc"
    if [[ $rc -ne 0 ]]; then
        echo "M1-49 GPU preflight comparison failed at $label" >&2
        return "$rc"
    fi
}

run_arm() {
    local mode=$1
    local pressure_mode=$2
    local output=$3
    mkdir -p "$output"
    wait_for_port_free

    BI100_HYBRID_KV_ACCOUNTING="$mode" \
        setsid "$ROOT/launch_service" \
        > "$output/server.log" 2>&1 < /dev/null &
    ACTIVE_PID=$!
    printf '%s\n' "$ACTIVE_PID" > "$output/server.pid"
    for _ in $(seq 1 20); do
        ACTIVE_PGID=$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null | tr -d ' ')
        [[ -n "$ACTIVE_PGID" ]] && break
        sleep 1
    done
    if [[ -z "$ACTIVE_PGID" || "$ACTIVE_PGID" != "$ACTIVE_PID" ]]; then
        echo "service did not enter an isolated process group" >&2
        return 1
    fi
    printf '%s\n' "$ACTIVE_PGID" > "$output/server.pgid"

    for _ in $(seq 1 360); do
        if health; then
            printf '%s\n' 0 > "$output/startup.rc"
            break
        fi
        if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
            printf '%s\n' 1 > "$output/startup.rc"
            tail -120 "$output/server.log" >&2 || true
            return 1
        fi
        sleep 10
    done
    if [[ ! -f "$output/startup.rc" || $(<"$output/startup.rc") != 0 ]]; then
        echo "M1-49 $mode service did not become healthy" >&2
        return 1
    fi

    grep -Fq '[BI100] fixed evaluator contract;' "$output/server.log"
    grep -Fq '[BI100] fixed kernels; moe_direct=1 gdn_packed=1' "$output/server.log"
    grep -Fq '[BI100] GDN cache; policy=admission64 restore=direct' "$output/server.log"

    set +e
    BI100_HYBRID_KV_ACCOUNTING="$mode" \
        timeout --signal=TERM --kill-after=20s 180s \
        python3 "$ROOT/tests/hybrid_kv_startup_gate.py" \
        "$output/server.log" --mode "$mode" --model-path "$MODEL_PATH" \
        --max-model-len 262144 --block-size 16 --tensor-parallel-size 4 \
        --out "$output/startup_gate.json" \
        > "$output/startup_gate.stdout" 2> "$output/startup_gate.stderr"
    local startup_gate_rc=$?
    set -e
    printf '%s\n' "$startup_gate_rc" > "$output/startup_gate.rc"
    if [[ $startup_gate_rc -ne 0 ]]; then
        return "$startup_gate_rc"
    fi

    set +e
    timeout --signal=TERM --kill-after=30s 7200s \
        python3 "$ROOT/tests/cpu_kv_offload_pressure_api.py" \
        --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
        --target-prompt-tokens 65536 --pressure-prompt-tokens 135040 \
        --pressure-count 2 --max-tokens 8 --timeout-s 900 \
        --run-id "$RUN_ID" --mode "$pressure_mode" --block-size 16 \
        --min-candidate-cached 65504 --max-control-cached 16 \
        --json-out "$output/pressure.json" \
        > "$output/pressure.stdout" 2> "$output/pressure.stderr"
    local pressure_rc=$?
    set -e
    printf '%s\n' "$pressure_rc" > "$output/pressure.rc"
    if [[ $pressure_rc -ne 0 ]]; then
        return "$pressure_rc"
    fi

    if fatal_scan "$output/server.log" "$output/fatal_scan.txt"; then
        printf '%s\n' 0 > "$output/fatal_scan.rc"
    else
        printf '%s\n' 1 > "$output/fatal_scan.rc"
        return 1
    fi
    stop_service
    wait_for_port_free
}

run_preflight before_legacy
run_arm legacy40 control "$RUN_ROOT/legacy40"
run_preflight after_legacy
compare_preflights after_legacy before_legacy after_legacy
run_arm full_attention candidate "$RUN_ROOT/full_attention"
run_preflight after_candidate
compare_preflights final before_legacy after_legacy after_candidate

set +e
python3 "$ROOT/tests/compare_hybrid_kv_accounting_ab.py" \
    --legacy-startup "$RUN_ROOT/legacy40/startup_gate.json" \
    --candidate-startup "$RUN_ROOT/full_attention/startup_gate.json" \
    --legacy-pressure "$RUN_ROOT/legacy40/pressure.json" \
    --candidate-pressure "$RUN_ROOT/full_attention/pressure.json" \
    --out "$RUN_ROOT/comparison.json" \
    > "$RUN_ROOT/comparison.stdout" 2> "$RUN_ROOT/comparison.stderr"
comparison_rc=$?
set -e
printf '%s\n' "$comparison_rc" > "$RUN_ROOT/comparison.rc"
printf '%s\n' "$comparison_rc" > "$RUN_ROOT/overall.rc"
if [[ $comparison_rc -ne 0 ]]; then
    echo "M1-49 hybrid KV A/B failed its fixed gate" >&2
    exit "$comparison_rc"
fi

echo "M1-49 hybrid KV startup and pressure A/B passed"
