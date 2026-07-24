#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"
M1_49_LONG_DIR=${M1_49_LONG_DIR:-$ROOT/bench_runs/m1_49/full_attention_long}
RUN_ROOT=${M1_48_RUN_ROOT:-$ROOT/bench_runs/m1_48/prefill_path_profile}
MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
RUN_ID=${M1_48_RUN_ID:-m148-post-m149-prefill-profile-20260722}
PROFILE_FILTER='model.*,layer.*,full_attn.*,xformers.*,paged_attn.*,moe.*,gdn_prefix.*'
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
export BI100_RUNTIME_WORKDIR=/tmp/m1-48-profile-runtime
export BI100_PROFILE_MODE=event
export BI100_PROFILE_INCLUDE_STARTUP=0
export BI100_PROFILE_FILTER="$PROFILE_FILTER"
export BI100_PAGED_ATTN_DIAGNOSTICS=0
export BI100_GDN_ALLOW_NAN_ZERO=0
export BI100_GDN_FINITE_CHECK=0
unset NUM_GPU_BLOCKS_OVERRIDE BI100_MOE_COREX_THREE_BUCKET

require_zero_rc() {
    local path=$1
    if [[ ! -f "$path" ]] || [[ $(<"$path") != 0 ]]; then
        echo "required successful gate is missing: $path" >&2
        exit 3
    fi
}

require_zero_rc "$M1_49_LONG_DIR/overall.rc"
require_zero_rc "$M1_49_LONG_DIR/cleanup.rc"
require_zero_rc "$M1_49_LONG_DIR/qualification.rc"
M1_49_QUALIFICATION="$M1_49_LONG_DIR/qualification.json"
if [[ ! -f "$M1_49_QUALIFICATION" ]]; then
    echo "M1-49 qualification report is missing" >&2
    exit 3
fi
python3 - "$M1_49_QUALIFICATION" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("schema") != "bi100-m1-49-long-context-qualification-v1":
    raise SystemExit("M1-49 prerequisite schema is invalid")
if report.get("version") != 1 or report.get("qualified") is not True:
    raise SystemExit("M1-49 prerequisite is not qualified")
if report.get("scope") != "hybrid-kv-capacity-correctness-not-prefill-speed":
    raise SystemExit("M1-49 prerequisite scope is invalid")
PY

if [[ -z "${BI100_RUNTIME_SITE_PACKAGES:-}" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/vllm" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/transformers" ]]; then
    echo "M1-48 requires an atomic bare-host runtime overlay" >&2
    exit 3
fi
export BI100_RUNTIME_SITE_PACKAGES
RUNTIME_ROOT=$(dirname "$BI100_RUNTIME_SITE_PACKAGES")
RUNTIME_INSTALL_REPORT=${BI100_RUNTIME_INSTALL_REPORT:-$RUNTIME_ROOT/install.json}
if [[ ! -f "$RUNTIME_INSTALL_REPORT" ]]; then
    echo "M1-48 runtime install report is missing" >&2
    exit 3
fi
python3 - "$RUNTIME_INSTALL_REPORT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
required = {"vllm_model", "bi100_profile", "paged_attention", "xformers_backend"}
files = report.get("files") or {}
if (report.get("schema") != "bi100-bare-host-runtime-install-v2"
        or report.get("version") != 2
        or report.get("qualified") is not True
        or report.get("startup_profile_guard_patch") is not True
        or not required.issubset(files)):
    raise SystemExit("M1-48 runtime install report is not qualified")
for name in required:
    row = files[name]
    if (row.get("same") is not True
            or row.get("source_sha256") != row.get("installed_sha256")):
        raise SystemExit(f"M1-48 runtime identity failed: {name}")
PY
export PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages:${PYTHONPATH:-}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model directory is missing: $MODEL_PATH" >&2
    exit 4
fi
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
        . ':(exclude)bench_runs/**')" ]]; then
    echo "M1-48 refuses a dirty source tree" >&2
    exit 4
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "M1-48 output already exists: $RUN_ROOT" >&2
    exit 5
fi
mkdir -p "$RUN_ROOT"
git -C "$ROOT" rev-parse HEAD > "$RUN_ROOT/source_revision.txt"
python3 "$ROOT/tests/verify_m1_48_runtime_identity.py" \
    --source-root "$ROOT" \
    --runtime-site-packages "$BI100_RUNTIME_SITE_PACKAGES" \
    --runtime-install "$RUNTIME_INSTALL_REPORT" \
    --out "$RUN_ROOT/runtime_identity.json" \
    > "$RUN_ROOT/runtime_identity.stdout" \
    2> "$RUN_ROOT/runtime_identity.stderr"
printf '%s\n' 0 > "$RUN_ROOT/runtime_identity.rc"

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
        echo "M1-48 service cleanup failed" >&2
        rm -f "$RUN_ROOT/qualification.json"
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
        echo "M1-48 four-GPU preflight failed at $label" >&2
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
        echo "M1-48 gate failed: $name rc=$rc" >&2
        return "$rc"
    fi
}

verify_prequalification_cleanup() {
    local rc=0
    set +e
    stop_service
    rc=$?
    if [[ $rc -eq 0 ]]; then
        wait_for_port_free
        rc=$?
    fi
    set -e
    printf '%s\n' "$rc" > "$RUN_ROOT/prequalification_cleanup.rc"
    if [[ $rc -ne 0 ]]; then
        echo "M1-48 pre-qualification cleanup failed" >&2
        return "$rc"
    fi
}

fatal_scan() {
    local arm=$1
    local log="$RUN_ROOT/$arm/server.log"
    local output="$RUN_ROOT/$arm/fatal_scan.txt"
    if grep -Eiq 'CUDA error|SIGSEGV|Fatal Python error|out of memory|worker process.*died|Gloo.*failed|AssertionError' "$log"; then
        grep -Ein 'CUDA error|SIGSEGV|Fatal Python error|out of memory|worker process.*died|Gloo.*failed|AssertionError' \
            "$log" > "$output" || true
        printf '%s\n' 1 > "$RUN_ROOT/$arm/fatal_scan.rc"
        return 1
    fi
    : > "$output"
    printf '%s\n' 0 > "$RUN_ROOT/$arm/fatal_scan.rc"
}

run_arm() {
    local arm=$1
    local profile_enabled=$2
    local output="$RUN_ROOT/$arm"
    mkdir -p "$output"
    wait_for_port_free

    BI100_PROFILE="$profile_enabled" setsid "$ROOT/launch_service" \
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
        echo "M1-48 $arm service did not enter an isolated process group" >&2
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
    require_zero_rc "$output/startup.rc"

    grep -Fq '[BI100] fixed evaluator contract;' "$output/server.log"
    grep -Fq '[BI100] fixed kernels; moe_direct=1 gdn_packed=1' "$output/server.log"
    grep -Fq '[BI100] GDN cache; policy=admission64 restore=direct' "$output/server.log"
    grep -Fq '[BI100] M1-49 runtime contract; accounting=full_attention' \
        "$output/server.log"
    grep -Fq "[BI100] M1-48 profile contract; enabled=$profile_enabled mode=event include_startup=0 filter=$PROFILE_FILTER" \
        "$output/server.log"
    printf '%s\n' 0 > "$output/runtime_contract.rc"

    run_gate "${arm}_startup_gate" 300 \
        python3 "$ROOT/tests/hybrid_kv_startup_gate.py" \
        "$output/server.log" --mode full_attention --model-path "$MODEL_PATH" \
        --max-model-len 262144 --block-size 16 --tensor-parallel-size 4 \
        --out "$output/startup_gate.json"

    run_gate "${arm}_service" 12600 \
        python3 "$ROOT/tests/measure_m1_48_prefill_service.py" \
        --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
        --target-prompt-tokens 235000 --max-model-len 262144 \
        --timeout-s 4200 --run-id "$RUN_ID" --mode "$arm" \
        --out "$output/service.json"
    health
    fatal_scan "$arm"
    stop_service
    wait_for_port_free

    if [[ "$profile_enabled" == 0 ]]; then
        if grep -Fq '[BI100_PROFILE_EVENT]' "$output/server.log"; then
            echo "M1-48 control arm emitted profile events" >&2
            return 1
        fi
    elif ! grep -Fq '[BI100_PROFILE_EVENT]' "$output/server.log"; then
        echo "M1-48 profile arm emitted no profile events" >&2
        return 1
    fi
}

run_preflight before_control
run_arm control 0
run_preflight after_control
run_arm profile 1
run_preflight after_profile

run_gate preflight_comparison 120 \
    python3 "$ROOT/tests/compare_bi100_preflights.py" \
    --preflight "before_control=$RUN_ROOT/preflight_before_control.json" \
    --preflight "after_control=$RUN_ROOT/preflight_after_control.json" \
    --preflight "after_profile=$RUN_ROOT/preflight_after_profile.json" \
    --max-free-memory-drop-bytes 1073741824 \
    --out "$RUN_ROOT/preflight_comparison.json"

run_gate profile_summary 300 \
    python3 "$ROOT/tests/summarize_prefill_path_profile.py" \
    "$RUN_ROOT/profile/server.log" \
    --expected-prefill-tokens 235000 --expected-processes 4 \
    --expected-chunk-size 8192 --block-size 16 \
    --control-service "$RUN_ROOT/control/service.json" \
    --profile-service "$RUN_ROOT/profile/service.json" \
    --out "$RUN_ROOT/profile_summary.json"

verify_prequalification_cleanup

run_gate qualification 300 \
    python3 "$ROOT/tests/qualify_m1_48_prefill_profile.py" \
    --m1-49 "$M1_49_QUALIFICATION" \
    --runtime-install "$RUNTIME_INSTALL_REPORT" \
    --runtime-identity "$RUN_ROOT/runtime_identity.json" \
    --preflight "$RUN_ROOT/preflight_comparison.json" \
    --control-startup "$RUN_ROOT/control/startup_gate.json" \
    --profile-startup "$RUN_ROOT/profile/startup_gate.json" \
    --control-service "$RUN_ROOT/control/service.json" \
    --profile-service "$RUN_ROOT/profile/service.json" \
    --profile-summary "$RUN_ROOT/profile_summary.json" \
    --prequalification-cleanup \
        "$RUN_ROOT/prequalification_cleanup.rc" \
    --source-revision "$RUN_ROOT/source_revision.txt" \
    --control-log "$RUN_ROOT/control/server.log" \
    --profile-log "$RUN_ROOT/profile/server.log" \
    --out "$RUN_ROOT/qualification.json"

echo "M1-48 post-M1-49 prefill path profile qualified"
