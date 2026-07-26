#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

RUN_ROOT=${M1_58_RUN_ROOT:-$ROOT/bench_runs/m1_58/block_major_kv_ab}
MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
RUN_ID=m158-block-major-fixed-20260726
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
export BI100_CPU_KV_OFFLOAD=1
export BI100_GDN_CACHE_POLICY=admission64
export BI100_GDN_RESTORE_MODE=direct
export BI100_CACHE_TRACE=0
export BI100_BLOCK_MAJOR_CPU_KV_TRACE=0
export BI100_ATTN_COREX_FUSED_PREFILL=0
export BI100_KV_EVICTION_POLICY=lru
export BI100_HYBRID_KV_ACCOUNTING=full_attention
export BI100_RUNTIME_WORKDIR=/tmp/m1-58-runtime
export BI100_PROFILE=0
export BI100_PROFILE_INCLUDE_STARTUP=0
export BI100_PAGED_ATTN_DIAGNOSTICS=0
export BI100_GDN_ALLOW_NAN_ZERO=0
export BI100_GDN_FINITE_CHECK=0
unset CUDA_VISIBLE_DEVICES NUM_GPU_BLOCKS_OVERRIDE BI100_MOE_COREX_THREE_BUCKET

if [[ -z "${BI100_RUNTIME_SITE_PACKAGES:-}" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/vllm" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/transformers" ]]; then
    echo "M1-58 requires an atomic bare-host runtime overlay" >&2
    exit 3
fi
if [[ -z "${RUNTIME_INSTALL_REPORT:-}" \
        || ! -f "$RUNTIME_INSTALL_REPORT" ]]; then
    echo "M1-58 requires RUNTIME_INSTALL_REPORT" >&2
    exit 4
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model directory is missing: $MODEL_PATH" >&2
    exit 5
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
    echo "M1-58 output already exists; refusing to overwrite: $RUN_ROOT" >&2
    exit 6
fi
mkdir -p "$RUN_ROOT"

export BI100_RUNTIME_SITE_PACKAGES
export PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages:${PYTHONPATH:-}"

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
        echo "M1-58 service cleanup failed" >&2
        rc=1
    fi
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/cleanup.rc"
    printf '%s\n' "$rc" > "$RUN_ROOT/overall.rc"
    exit "$rc"
}
trap finish EXIT

run_offline_gate() {
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
        echo "M1-58 offline gate failed: $name rc=$rc" >&2
        return "$rc"
    fi
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
        echo "M1-58 four-GPU preflight failed at $label" >&2
        return "$rc"
    fi
}

fatal_scan() {
    local log=$1
    local output=$2
    local pattern='CUDA error|SIGSEGV|Fatal Python error|out of memory|OOM|worker process.*died|worker.*lost|Gloo.*failed|Connection reset by peer|NCCL.*error|AssertionError|Traceback \(most recent call last\)'
    if grep -Eiq "$pattern" "$log"; then
        grep -Ein "$pattern" "$log" > "$output" || true
        return 1
    fi
    : > "$output"
}

run_arm() {
    local label=$1
    local selector=$2
    local output=$RUN_ROOT/$label
    mkdir -p "$output"
    wait_for_port_free

    BI100_BLOCK_MAJOR_CPU_KV="$selector" \
        setsid "$ROOT/launch_service" \
        > "$output/server.log" 2>&1 < /dev/null &
    ACTIVE_PID=$!
    ACTIVE_PGID=$ACTIVE_PID
    printf '%s\n' "$ACTIVE_PID" > "$output/server.pid"

    local observed_pgid=""
    for _ in $(seq 1 20); do
        observed_pgid=$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null | tr -d ' ')
        [[ -n "$observed_pgid" ]] && break
        sleep 1
    done
    if [[ -z "$observed_pgid" || "$observed_pgid" != "$ACTIVE_PGID" ]]; then
        echo "$label service did not enter an isolated process group" >&2
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
        echo "M1-58 $label service did not become healthy" >&2
        return 1
    fi

    grep -Fq '[BI100] fixed evaluator contract;' "$output/server.log"
    grep -Fq '[BI100] fixed kernels; moe_direct=1 gdn_packed=1' \
        "$output/server.log"
    grep -Fq '[BI100] GDN cache; policy=admission64 restore=direct' \
        "$output/server.log"
    grep -Fq \
        "[BI100] M1-49 runtime contract; accounting=full_attention" \
        "$output/server.log"
    grep -Fq "cpu_kv_offload=1 block_major_cpu_kv=$selector" \
        "$output/server.log"

    if [[ "$selector" == "1" ]]; then
        grep -Fq '[BI100 BLOCK KV] capacity reserve' "$output/server.log"
        grep -Fq '[BI100 BLOCK KV] enabled' "$output/server.log"
    elif grep -Fq '[BI100 BLOCK KV]' "$output/server.log"; then
        echo "control emitted block-major runtime markers" >&2
        return 1
    fi

    set +e
    BI100_BLOCK_MAJOR_CPU_KV="$selector" \
        timeout --signal=TERM --kill-after=20s 180s \
        python3 "$ROOT/tests/hybrid_kv_startup_gate.py" \
        "$output/server.log" --mode full_attention \
        --model-path "$MODEL_PATH" --max-model-len 262144 \
        --block-size 16 --tensor-parallel-size 4 \
        --expected-cpu-kv-offload 1 \
        --expected-block-major-cpu-kv "$selector" \
        --expected-block-major-cpu-kv-trace 0 \
        --out "$output/startup_gate.json" \
        > "$output/startup_gate.stdout" \
        2> "$output/startup_gate.stderr"
    local startup_gate_rc=$?
    set -e
    printf '%s\n' "$startup_gate_rc" > "$output/startup_gate.rc"
    if [[ $startup_gate_rc -ne 0 ]]; then
        return "$startup_gate_rc"
    fi

    set +e
    timeout --signal=TERM --kill-after=60s 18000s \
        python3 "$ROOT/tests/cpu_kv_offload_pressure_api.py" \
        --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
        --target-prompt-tokens 65536 \
        --pressure-prompt-tokens 135040 --pressure-count 9 \
        --max-tokens 8 --timeout-s 900 --run-id "$RUN_ID" \
        --mode candidate --block-size 16 \
        --min-candidate-cached 65504 --max-control-cached 16 \
        --json-out "$output/pressure.json" \
        > "$output/pressure.stdout" 2> "$output/pressure.stderr"
    local pressure_rc=$?
    set -e
    printf '%s\n' "$pressure_rc" > "$output/pressure.rc"
    if [[ $pressure_rc -ne 0 ]]; then
        return "$pressure_rc"
    fi
    health

    if fatal_scan "$output/server.log" "$output/fatal_scan.txt"; then
        printf '%s\n' 0 > "$output/fatal_scan.rc"
    else
        printf '%s\n' 1 > "$output/fatal_scan.rc"
        return 1
    fi

    stop_service
    wait_for_port_free
}

run_offline_gate runtime_identity 90 \
    python3 "$ROOT/tests/verify_bare_host_runtime_identity.py" \
    --source-root "$ROOT" \
    --runtime-site-packages "$BI100_RUNTIME_SITE_PACKAGES" \
    --runtime-install "$RUNTIME_INSTALL_REPORT" \
    --out "$RUN_ROOT/runtime_identity.json"

run_preflight before_control
run_arm control 0
run_preflight after_control
run_arm candidate 1
run_preflight after_candidate

run_offline_gate preflight_comparison 120 \
    python3 "$ROOT/tests/compare_bi100_preflights.py" \
    --preflight \
    "before_control=$RUN_ROOT/preflight_before_control.json" \
    --preflight \
    "after_control=$RUN_ROOT/preflight_after_control.json" \
    --preflight \
    "after_candidate=$RUN_ROOT/preflight_after_candidate.json" \
    --max-free-memory-drop-bytes 268435456 \
    --out "$RUN_ROOT/preflight_comparison.json"

run_offline_gate comparison 120 \
    python3 "$ROOT/tests/compare_m1_58_block_major_ab.py" \
    --control-startup "$RUN_ROOT/control/startup_gate.json" \
    --candidate-startup "$RUN_ROOT/candidate/startup_gate.json" \
    --control-pressure "$RUN_ROOT/control/pressure.json" \
    --candidate-pressure "$RUN_ROOT/candidate/pressure.json" \
    --runtime-identity "$RUN_ROOT/runtime_identity.json" \
    --preflight-comparison "$RUN_ROOT/preflight_comparison.json" \
    --out "$RUN_ROOT/comparison.json"

printf '%s\n' 0 > "$RUN_ROOT/overall.rc"
echo "M1-58 fixed TP4 block-major CacheEngine A/B passed"
