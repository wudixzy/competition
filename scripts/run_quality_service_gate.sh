#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

if [[ $# -ne 8 ]]; then
    echo "usage: $0 SUITE POLICY RESTORE_MODE FUSED_PREFILL KV_EVICTION LABEL INSTANCE RUN_ROOT" >&2
    exit 2
fi

SUITE=$1
POLICY=$2
RESTORE_MODE=$3
FUSED_PREFILL=$4
KV_EVICTION=$5
LABEL=$6
INSTANCE=$7
RUN_ROOT=$8
MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
KERNEL_PROFILE=${BI100_QUALITY_KERNEL_PROFILE:-submission}
ACTIVE_PID=""
ACTIVE_PGID=""
SERVICE_STARTED=0
BEFORE_PREFLIGHT_PASSED=0

case "$SUITE" in
    functional|long-context) ;;
    *) echo "SUITE must be functional or long-context" >&2; exit 2 ;;
esac
case "$POLICY" in
    fine32|admission64) ;;
    *) echo "POLICY must be fine32 or admission64" >&2; exit 2 ;;
esac
case "$RESTORE_MODE" in
    direct|aligned) ;;
    *) echo "RESTORE_MODE must be direct or aligned" >&2; exit 2 ;;
esac
case "$FUSED_PREFILL" in
    0|1) ;;
    *) echo "FUSED_PREFILL must be 0 or 1" >&2; exit 2 ;;
esac
case "$KV_EVICTION" in
    lru|frequency) ;;
    *) echo "KV_EVICTION must be lru or frequency" >&2; exit 2 ;;
esac
case "$KERNEL_PROFILE" in
    submission)
        MOE_DIRECT=1
        GDN_PACKED=1
        GDN_COMBINED_QK=0
        ;;
    strict-reference)
        MOE_DIRECT=0
        GDN_PACKED=0
        GDN_COMBINED_QK=0
        ;;
    strict-reference-combined-qk)
        MOE_DIRECT=0
        GDN_PACKED=0
        GDN_COMBINED_QK=1
        ;;
    *)
        echo "BI100_QUALITY_KERNEL_PROFILE is invalid" >&2
        exit 2
        ;;
esac

RUN_ROOT=$(python3 - "$RUN_ROOT" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)
case "$RUN_ROOT/" in
    "$ROOT/"*)
        echo "quality run output must stay outside the source repository" >&2
        exit 2
        ;;
esac
if [[ "$RUN_ROOT" != /tmp/* ]]; then
    echo "quality run output must use a private /tmp path" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "quality run output already exists: $RUN_ROOT" >&2
    exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model path is missing: $MODEL_PATH" >&2
    exit 3
fi
MODEL_PATH=$(python3 - "$MODEL_PATH" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=True))
PY
)
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
        . ':(exclude)bench_runs/**')" ]]; then
    echo "quality gate refuses a dirty source tree" >&2
    exit 3
fi
if pgrep -f 'vllm\.entrypoints\.openai\.api_server' >/dev/null 2>&1; then
    echo "an API server process is already running" >&2
    exit 3
fi

if [[ -z "${BI100_RUNTIME_SITE_PACKAGES:-}" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/vllm" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/transformers" ]]; then
    echo "quality gate requires an atomic runtime overlay" >&2
    exit 3
fi
BI100_RUNTIME_SITE_PACKAGES=$(python3 - "$BI100_RUNTIME_SITE_PACKAGES" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=True))
PY
)
export BI100_RUNTIME_SITE_PACKAGES
RUNTIME_ROOT=$(dirname "$BI100_RUNTIME_SITE_PACKAGES")
RUNTIME_INSTALL=${BI100_RUNTIME_INSTALL_REPORT:-$RUNTIME_ROOT/install.json}
if [[ ! -f "$RUNTIME_INSTALL" ]]; then
    echo "runtime install report is missing: $RUNTIME_INSTALL" >&2
    exit 3
fi

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
export PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH"
export LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
export HOST=0.0.0.0
export PORT=8000
export MODEL_PATH
export ENABLE_CUSTOM_IPC=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export BI100_ALLOW_PREFIX_GUARD_CAP=0
export BI100_ATTN_COREX_HEAD_RMS_NORM=1
export BI100_ATTN_COREX_PAGED_GATHER=1
export BI100_MOE_COREX_DIRECT_ROUTED="$MOE_DIRECT"
export BI100_MOE_COREX_EXACT_REDUCE=1
export BI100_MOE_COREX_WEIGHT_GATHER=1
export BI100_MOE_FUSED_ACTIVATION=1
export BI100_GDN_COREX_PACKED_DECODE="$GDN_PACKED"
export BI100_GDN_COMBINED_QK_NORM="$GDN_COMBINED_QK"
export BI100_GDN_COREX_BETA_DECAY=1
export BI100_GDN_COREX_CAUSAL_CONV=1
export BI100_GDN_COREX_GATED_NORM=1
export BI100_GDN_COREX_QK_MAP=1
export BI100_DNN_CHUNK=4096
export BI100_EXECUTOR_STARTUP_DEBUG=1
export BI100_FORCE_PAGED_ATTN_V2=0
export BI100_CPU_KV_OFFLOAD=0
export BI100_GDN_CACHE_POLICY="$POLICY"
export BI100_GDN_RESTORE_MODE="$RESTORE_MODE"
export BI100_CACHE_TRACE=1
export BI100_ATTN_COREX_FUSED_PREFILL="$FUSED_PREFILL"
export BI100_KV_EVICTION_POLICY="$KV_EVICTION"
export BI100_HYBRID_KV_ACCOUNTING=full_attention
export BI100_PROFILE=0
export BI100_PROFILE_INCLUDE_STARTUP=0
export BI100_PAGED_ATTN_DIAGNOSTICS=0
export BI100_PREFIX_BLOCKS_PER_TILE=32
export BI100_PREFIX_DTYPE=float16
export BI100_PREFIX_MODEL_FINGERPRINT=Qwen3.6-35B-A3B
export BI100_PREFIX_TP_SIZE=4
export BI100_GDN_ALLOW_NAN_ZERO=0
export BI100_GDN_FINITE_CHECK=0
export BI100_PROFILE_FILTER=""
export BI100_PROFILE_MODE=sync
export BI100_PYTORCH_DECODE_THRESHOLD=32768
export BI100_RUNTIME_WORKDIR="$RUN_ROOT/runtime-workdir"
export BI100_UNSET_CUDA_VISIBLE_DEVICES=1
unset CUDA_VISIBLE_DEVICES NUM_GPU_BLOCKS_OVERRIDE BI100_MOE_COREX_THREE_BUCKET

mkdir -p "$RUN_ROOT"
git -C "$ROOT" rev-parse HEAD > "$RUN_ROOT/source_revision.txt"
git -C "$ROOT" branch --show-current > "$RUN_ROOT/source_branch.txt"

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
    echo "port 8000 remained busy after service cleanup" >&2
    return 1
}

run_preflight() {
    local name=$1
    local rc=0
    if timeout --signal=TERM --kill-after=10s 150s \
            python3 "$ROOT/tests/bi100_preflight.py" \
            --gpus 0,1,2,3 --timeout-s 25 --matmul-size 1024 \
            --json-out "$RUN_ROOT/preflight_${name}.json" \
            > "$RUN_ROOT/preflight_${name}.stdout" \
            2> "$RUN_ROOT/preflight_${name}.stderr"; then
        rc=0
    else
        rc=$?
    fi
    printf '%s\n' "$rc" > "$RUN_ROOT/preflight_${name}.rc"
    return "$rc"
}

stop_service() {
    if [[ -n "$ACTIVE_PGID" ]]; then
        # Allow TP4 workers and collective runtimes a full minute to unwind.
        # SIGKILL is only used for live members after this grace period.
        bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID" 60 20
    elif [[ -n "$ACTIVE_PID" ]]; then
        echo "service PID lacks a verified process group" >&2
        return 2
    fi
    ACTIVE_PID=""
    ACTIVE_PGID=""
}

scan_fatal_log() {
    if [[ ! -f "$RUN_ROOT/server.log" ]]; then
        printf '%s\n' 1 > "$RUN_ROOT/fatal_scan.rc"
        echo "service log is missing" > "$RUN_ROOT/fatal_scan.txt"
        return 1
    fi
    local pattern
    pattern='CUDA error|SIGSEGV|Fatal Python error|out of memory|AssertionError|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|TimeoutError|engine iteration timed out|watchdog.*tim(e|ed) out'
    if grep -Eiq "$pattern" \
            "$RUN_ROOT/server.log"; then
        grep -Ein "$pattern" \
            "$RUN_ROOT/server.log" > "$RUN_ROOT/fatal_scan.txt" || true
        printf '%s\n' 1 > "$RUN_ROOT/fatal_scan.rc"
        return 1
    fi
    : > "$RUN_ROOT/fatal_scan.txt"
    printf '%s\n' 0 > "$RUN_ROOT/fatal_scan.rc"
}

run_service_postflight() {
    local rc=0
    if python3 "$ROOT/tests/service_postflight_gate.py" \
            --gpus 0,1,2,3 \
            --out "$RUN_ROOT/service_postflight.json" \
            > "$RUN_ROOT/service_postflight.stdout" \
            2> "$RUN_ROOT/service_postflight.stderr"; then
        rc=0
    else
        rc=$?
    fi
    printf '%s\n' "$rc" > "$RUN_ROOT/service_postflight.rc"
    return "$rc"
}

scan_runner_timeouts() {
    local file
    local value
    local found=0
    : > "$RUN_ROOT/timeout_scan.txt"
    for file in startup.rc quality.rc agent_workload.rc; do
        [[ -f "$RUN_ROOT/$file" ]] || continue
        value=$(tr -d '[:space:]' < "$RUN_ROOT/$file")
        case "$value" in
            124|137)
                printf '%s=%s\n' "$file" "$value" \
                    >> "$RUN_ROOT/timeout_scan.txt"
                found=1
                ;;
        esac
    done
    printf '%s\n' "$found" > "$RUN_ROOT/timeout_scan.rc"
    return "$found"
}

write_status() {
    local final_rc=$1
    python3 - "$RUN_ROOT" "$SUITE" "$POLICY" "$RESTORE_MODE" \
            "$FUSED_PREFILL" "$KV_EVICTION" "$LABEL" "$INSTANCE" \
            "$KERNEL_PROFILE" "$final_rc" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def read_rc(name):
    path = root / name
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return int(value) if value.isdigit() else None

report = {
    "schema": "bi100-quality-service-gate-status-v1",
    "version": 1,
    "suite": sys.argv[2],
    "optimization": {
        "gdn_cache_policy": sys.argv[3],
        "gdn_restore_mode": sys.argv[4],
        "fused_prefill": sys.argv[5],
        "kv_eviction_policy": sys.argv[6],
        "kernel_profile": sys.argv[9],
    },
    "label": sys.argv[7],
    "instance": sys.argv[8],
    "overall_rc": int(sys.argv[10]),
    "source_revision": (root / "source_revision.txt").read_text(
        encoding="utf-8").strip(),
    "source_branch": (root / "source_branch.txt").read_text(
        encoding="utf-8").strip(),
    "gates": {
        "runtime_identity": read_rc("runtime_identity.rc"),
        "runtime_contract": read_rc("runtime_contract.rc"),
        "prefix_allocator": read_rc("prefix_allocator.rc"),
        "gdn_action_broadcast": read_rc("gdn_action_broadcast.rc"),
        "preflight_before": read_rc("preflight_before.rc"),
        "startup": read_rc("startup.rc"),
        "startup_contract": read_rc("startup_contract.rc"),
        "quality": read_rc("quality.rc"),
        "agent_workload": read_rc("agent_workload.rc"),
        "cleanup": read_rc("cleanup.rc"),
        "service_postflight": read_rc("service_postflight.rc"),
        "fatal_scan": read_rc("fatal_scan.rc"),
        "timeout_scan": read_rc("timeout_scan.rc"),
        "preflight_after": read_rc("preflight_after.rc"),
        "preflight_comparison": read_rc("preflight_comparison.rc"),
    },
    "privacy": {
        "raw_service_log_outside_repository": True,
        "contains_credentials": False,
    },
}
contract = root / "runtime_contract.json"
quality = root / "quality_report.json"
agent = root / "agent_workload.json"
report["artifacts"] = {
    "runtime_contract_sha256": (
        hashlib.sha256(contract.read_bytes()).hexdigest()
        if contract.is_file() else None),
    "quality_report_sha256": (
        hashlib.sha256(quality.read_bytes()).hexdigest()
        if quality.is_file() else None),
    "agent_workload_sha256": (
        hashlib.sha256(agent.read_bytes()).hexdigest()
        if agent.is_file() else None),
}
(root / "status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

finish() {
    local primary_rc=$?
    local cleanup_rc=0
    local service_postflight_rc=0
    local fatal_rc=0
    local timeout_rc=0
    local after_rc=0
    local comparison_rc=0
    local final_rc=$primary_rc
    trap - EXIT
    set +e

    if [[ "$SERVICE_STARTED" == 1 ]]; then
        stop_service
        cleanup_rc=$?
        if [[ $cleanup_rc -eq 0 ]]; then
            wait_for_port_free
            cleanup_rc=$?
        fi
        scan_fatal_log
        fatal_rc=$?
    fi
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/cleanup.rc"
    run_service_postflight
    service_postflight_rc=$?
    scan_runner_timeouts
    timeout_rc=$?

    if [[ "$BEFORE_PREFLIGHT_PASSED" == 1 ]]; then
        run_preflight after
        after_rc=$?
        if [[ $after_rc -eq 0 ]]; then
            python3 "$ROOT/tests/compare_bi100_preflights.py" \
                --preflight "before=$RUN_ROOT/preflight_before.json" \
                --preflight "after=$RUN_ROOT/preflight_after.json" \
                --max-free-memory-drop-bytes 1073741824 \
                --out "$RUN_ROOT/preflight_comparison.json" \
                > "$RUN_ROOT/preflight_comparison.stdout" \
                2> "$RUN_ROOT/preflight_comparison.stderr"
            comparison_rc=$?
        else
            comparison_rc=1
        fi
        printf '%s\n' "$comparison_rc" > "$RUN_ROOT/preflight_comparison.rc"
    fi

    if [[ $cleanup_rc -ne 0 || $service_postflight_rc -ne 0 \
            || $fatal_rc -ne 0 || $timeout_rc -ne 0 \
            || $after_rc -ne 0 || $comparison_rc -ne 0 ]]; then
        final_rc=1
    fi
    printf '%s\n' "$final_rc" > "$RUN_ROOT/overall.rc"
    write_status "$final_rc"
    exit "$final_rc"
}
trap finish EXIT

set +e
python3 "$ROOT/tests/verify_bare_host_runtime_identity.py" \
    --source-root "$ROOT" \
    --runtime-site-packages "$BI100_RUNTIME_SITE_PACKAGES" \
    --runtime-install "$RUNTIME_INSTALL" \
    --out "$RUN_ROOT/runtime_identity.json" \
    > "$RUN_ROOT/runtime_identity.stdout" \
    2> "$RUN_ROOT/runtime_identity.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/runtime_identity.rc"
[[ $rc -eq 0 ]]

set +e
python3 "$ROOT/tests/build_quality_runtime_contract.py" \
    --source-root "$ROOT" \
    --runtime-site-packages "$BI100_RUNTIME_SITE_PACKAGES" \
    --runtime-install "$RUNTIME_INSTALL" \
    --model-path "$MODEL_PATH" \
    --instance "$INSTANCE" \
    --optimization-label "$LABEL" \
    --gdn-cache-policy "$POLICY" \
    --gdn-restore-mode "$RESTORE_MODE" \
    --fused-prefill "$FUSED_PREFILL" \
    --kv-eviction-policy "$KV_EVICTION" \
    --kernel-profile "$KERNEL_PROFILE" \
    --out "$RUN_ROOT/runtime_contract.json" \
    > "$RUN_ROOT/runtime_contract.stdout" \
    2> "$RUN_ROOT/runtime_contract.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/runtime_contract.rc"
[[ $rc -eq 0 ]]
RUNTIME_IDENTITY=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["runtime_identity"])' \
    "$RUN_ROOT/runtime_contract.json")

set +e
python3 "$ROOT/tests/prefix_namespace_fork_gate.py" \
    --out "$RUN_ROOT/prefix_allocator.json" \
    > "$RUN_ROOT/prefix_allocator.stdout" \
    2> "$RUN_ROOT/prefix_allocator.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/prefix_allocator.rc"
[[ $rc -eq 0 ]]

set +e
python3 "$ROOT/tests/gdn_action_broadcast_gate.py" \
    --out "$RUN_ROOT/gdn_action_broadcast.json" \
    > "$RUN_ROOT/gdn_action_broadcast.stdout" \
    2> "$RUN_ROOT/gdn_action_broadcast.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/gdn_action_broadcast.rc"
[[ $rc -eq 0 ]]

run_preflight before
BEFORE_PREFLIGHT_PASSED=1
port_is_free

setsid "$ROOT/launch_service" > "$RUN_ROOT/server.log" 2>&1 < /dev/null &
ACTIVE_PID=$!
ACTIVE_PGID=$ACTIVE_PID
SERVICE_STARTED=1
printf '%s\n' "$ACTIVE_PID" > "$RUN_ROOT/server.pid"
for _ in $(seq 1 20); do
    observed_pgid=$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null | tr -d ' ')
    [[ -n "$observed_pgid" ]] && break
    sleep 1
done
if [[ -z "${observed_pgid:-}" || "$observed_pgid" != "$ACTIVE_PGID" ]]; then
    echo "service did not enter an isolated process group" >&2
    exit 1
fi
printf '%s\n' "$ACTIVE_PGID" > "$RUN_ROOT/server.pgid"

startup_rc=1
for _ in $(seq 1 360); do
    if health; then
        startup_rc=0
        break
    fi
    if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
        tail -120 "$RUN_ROOT/server.log" >&2 || true
        break
    fi
    sleep 10
done
printf '%s\n' "$startup_rc" > "$RUN_ROOT/startup.rc"
[[ $startup_rc -eq 0 ]]

set +e
python3 "$ROOT/tests/hybrid_kv_startup_gate.py" \
    "$RUN_ROOT/server.log" \
    --mode full_attention \
    --model-path "$MODEL_PATH" \
    --max-model-len 262144 \
    --block-size 16 \
    --tensor-parallel-size 4 \
    --expected-cache-trace 1 \
    --expected-gdn-cache-policy "$POLICY" \
    --expected-gdn-restore-mode "$RESTORE_MODE" \
    --expected-fused-prefill "$FUSED_PREFILL" \
    --expected-kv-eviction-policy "$KV_EVICTION" \
    --expected-kernel-profile "$KERNEL_PROFILE" \
    --out "$RUN_ROOT/startup_contract.json" \
    > "$RUN_ROOT/startup_contract.stdout" \
    2> "$RUN_ROOT/startup_contract.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/startup_contract.rc"
[[ $rc -eq 0 ]]

set +e
if [[ "$SUITE" == functional ]]; then
    timeout --signal=TERM --kill-after=30s 21600s \
        "$ROOT/scripts/run_quality_functional_gate.sh" \
        http://127.0.0.1:8000 "$MODEL_PATH" \
        "$RUN_ROOT/runtime_contract.json" "$LABEL" "$RUNTIME_IDENTITY" \
        "$INSTANCE" "$RUN_ROOT/quality_report.json" \
        > "$RUN_ROOT/quality.stdout" 2> "$RUN_ROOT/quality.stderr"
    quality_rc=$?
    printf '%s\n' "$quality_rc" > "$RUN_ROOT/quality.rc"

    timeout --signal=TERM --kill-after=30s 3600s \
        python3 "$ROOT/tests/agent_workload_matrix.py" \
        --base http://127.0.0.1:8000 \
        --source-revision "$(git -C "$ROOT" rev-parse HEAD)" \
        --runtime-identity "$RUNTIME_IDENTITY" \
        --runtime-contract "$RUN_ROOT/runtime_contract.json" \
        --instance "$INSTANCE" \
        --label "$LABEL" \
        --run-id "${LABEL}-agent-workload-$(date -u +%Y%m%dT%H%M%SZ)-$$" \
        --out "$RUN_ROOT/agent_workload.json" \
        > "$RUN_ROOT/agent_workload.stdout" \
        2> "$RUN_ROOT/agent_workload.stderr"
    agent_rc=$?
    printf '%s\n' "$agent_rc" > "$RUN_ROOT/agent_workload.rc"
    rc=0
    if [[ $quality_rc -ne 0 || $agent_rc -ne 0 ]]; then
        rc=1
    fi
else
    timeout --signal=TERM --kill-after=30s 43200s \
        "$ROOT/scripts/run_quality_long_context_gate.sh" \
        http://127.0.0.1:8000 "$MODEL_PATH" \
        "$RUN_ROOT/runtime_contract.json" "$RUN_ROOT/server.log" \
        "$LABEL" "$RUNTIME_IDENTITY" "$INSTANCE" \
        "$RUN_ROOT/quality_report.json" \
        > "$RUN_ROOT/quality.stdout" 2> "$RUN_ROOT/quality.stderr"
    rc=$?
    printf '%s\n' "$rc" > "$RUN_ROOT/quality.rc"
fi
set -e
[[ $rc -eq 0 ]]
