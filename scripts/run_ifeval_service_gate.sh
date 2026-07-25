#!/bin/bash
set -Eeuo pipefail

EVALUATOR_ROOT=$(cd "$(dirname "$0")/.." && pwd)

if [[ $# -ne 11 ]]; then
    echo "usage: $0 RUNTIME_ROOT RUNTIME_SITE_PACKAGES IFEVAL_ENV EXPECTED_RUNTIME_REVISION POLICY RESTORE_MODE FUSED_PREFILL KV_EVICTION LABEL INSTANCE RUN_ROOT" >&2
    exit 2
fi

RUNTIME_ROOT=$(realpath "$1")
RUNTIME_SITE_PACKAGES=$(realpath "$2")
IFEVAL_ENV=$(realpath "$3")
EXPECTED_RUNTIME_REVISION=$4
POLICY=$5
RESTORE_MODE=$6
FUSED_PREFILL=$7
KV_EVICTION=$8
LABEL=$9
INSTANCE=${10}
RUN_ROOT=$(realpath -m "${11}")
MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
ACTIVE_PID=""
ACTIVE_PGID=""
SERVICE_STARTED=0
BEFORE_PREFLIGHT_PASSED=0

source "$RUNTIME_ROOT/scripts/lib/process_group.sh"

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
if [[ "$RUN_ROOT" != /tmp/* ]]; then
    echo "IFEval output must use a private /tmp path" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "IFEval output already exists: $RUN_ROOT" >&2
    exit 2
fi
if [[ ! -d "$MODEL_PATH" || ! -d "$RUNTIME_SITE_PACKAGES/vllm" \
        || ! -d "$RUNTIME_SITE_PACKAGES/transformers" \
        || ! -f "$IFEVAL_ENV/install.json" \
        || ! -d "$IFEVAL_ENV/site-packages" \
        || ! -d "$IFEVAL_ENV/nltk_data" ]]; then
    echo "runtime, model, or offline IFEval environment is incomplete" >&2
    exit 3
fi
if [[ "$(git -C "$RUNTIME_ROOT" rev-parse HEAD)" \
        != "$EXPECTED_RUNTIME_REVISION" ]]; then
    echo "runtime source revision differs" >&2
    exit 3
fi
if [[ -n "$(git -C "$RUNTIME_ROOT" status --porcelain --untracked-files=all)" \
        || -n "$(git -C "$EVALUATOR_ROOT" status --porcelain --untracked-files=all \
            -- . ':(exclude)bench_runs/**')" ]]; then
    echo "runtime and evaluator source trees must be clean" >&2
    exit 3
fi
if pgrep -f 'vllm\.entrypoints\.openai\.api_server' >/dev/null 2>&1; then
    echo "an API server process is already running" >&2
    exit 3
fi

RUNTIME_INSTALL=$(dirname "$RUNTIME_SITE_PACKAGES")/install.json
if [[ ! -f "$RUNTIME_INSTALL" ]]; then
    echo "runtime install report is missing" >&2
    exit 3
fi

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
export PYTHONPATH="$RUNTIME_ROOT/tests:$RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH"
IFEVAL_PYTHONPATH="$EVALUATOR_ROOT/tests:$EVALUATOR_ROOT/quality/external/google_ifeval:$IFEVAL_ENV/site-packages:$RUNTIME_ROOT/tests:$RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH"
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
export BI100_MOE_COREX_DIRECT_ROUTED=1
export BI100_MOE_COREX_EXACT_REDUCE=1
export BI100_MOE_COREX_WEIGHT_GATHER=1
export BI100_MOE_FUSED_ACTIVATION=1
export BI100_GDN_COREX_PACKED_DECODE=1
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
export BI100_RUNTIME_SITE_PACKAGES="$RUNTIME_SITE_PACKAGES"
export BI100_UNSET_CUDA_VISIBLE_DEVICES=1
unset CUDA_VISIBLE_DEVICES NUM_GPU_BLOCKS_OVERRIDE BI100_MOE_COREX_THREE_BUCKET

mkdir -p "$RUN_ROOT"
git -C "$RUNTIME_ROOT" rev-parse HEAD > "$RUN_ROOT/runtime_source_revision.txt"
git -C "$EVALUATOR_ROOT" rev-parse HEAD > "$RUN_ROOT/evaluator_source_revision.txt"

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
    return 1
}

run_preflight() {
    local name=$1
    local rc=0
    timeout --signal=TERM --kill-after=10s 150s \
        python3 "$RUNTIME_ROOT/tests/bi100_preflight.py" \
        --gpus 0,1,2,3 --timeout-s 25 --matmul-size 1024 \
        --json-out "$RUN_ROOT/preflight_${name}.json" \
        > "$RUN_ROOT/preflight_${name}.stdout" \
        2> "$RUN_ROOT/preflight_${name}.stderr" || rc=$?
    printf '%s\n' "$rc" > "$RUN_ROOT/preflight_${name}.rc"
    return "$rc"
}

stop_service() {
    if [[ -n "$ACTIVE_PGID" ]]; then
        bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID"
    elif [[ -n "$ACTIVE_PID" ]]; then
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

read_rc() {
    local path=$1
    if [[ -f "$path" ]]; then
        tr -d '\n' < "$path"
    else
        printf null
    fi
}

write_status() {
    local final_rc=$1
    python3 - "$RUN_ROOT" "$final_rc" "$POLICY" "$RESTORE_MODE" \
            "$FUSED_PREFILL" "$KV_EVICTION" "$LABEL" "$INSTANCE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def rc(name):
    path = root / f"{name}.rc"
    return int(path.read_text().strip()) if path.is_file() else None

def digest(name):
    path = root / name
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

value = {
    "schema": "bi100-ifeval-service-gate-status-v1",
    "version": 1,
    "overall_rc": int(sys.argv[2]),
    "optimization": {
        "gdn_cache_policy": sys.argv[3],
        "gdn_restore_mode": sys.argv[4],
        "fused_prefill": sys.argv[5],
        "kv_eviction_policy": sys.argv[6],
    },
    "label": sys.argv[7],
    "instance": sys.argv[8],
    "runtime_source_revision": (root / "runtime_source_revision.txt").read_text().strip(),
    "evaluator_source_revision": (root / "evaluator_source_revision.txt").read_text().strip(),
    "gates": {name: rc(name) for name in (
        "runtime_identity", "runtime_contract", "prefix_allocator",
        "gdn_action_broadcast", "preflight_before", "startup",
        "startup_contract", "ifeval", "cleanup", "fatal_scan",
        "preflight_after", "preflight_comparison")},
    "artifacts": {name: digest(name) for name in (
        "runtime_identity.json", "runtime_contract.json",
        "startup_contract.json", "ifeval_report.json", "fatal_scan.txt",
        "preflight_before.json", "preflight_after.json",
        "preflight_comparison.json")},
    "privacy": {
        "raw_service_log_outside_repository": True,
        "raw_checkpoint_deleted_by_runner": not (root / "ifeval.checkpoint.json").exists(),
        "contains_credentials": False,
    },
}
(root / "status.json").write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

finish() {
    local primary_rc=$?
    local cleanup_rc=0
    local fatal_rc=0
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
    if [[ "$BEFORE_PREFLIGHT_PASSED" == 1 ]]; then
        run_preflight after
        after_rc=$?
        if [[ $after_rc -eq 0 ]]; then
            python3 "$RUNTIME_ROOT/tests/compare_bi100_preflights.py" \
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
    if [[ $cleanup_rc -ne 0 || $fatal_rc -ne 0 || $after_rc -ne 0 \
            || $comparison_rc -ne 0 ]]; then
        final_rc=1
    fi
    printf '%s\n' "$final_rc" > "$RUN_ROOT/overall.rc"
    write_status "$final_rc"
    exit "$final_rc"
}
trap finish EXIT

set +e
python3 "$RUNTIME_ROOT/tests/verify_bare_host_runtime_identity.py" \
    --source-root "$RUNTIME_ROOT" \
    --runtime-site-packages "$RUNTIME_SITE_PACKAGES" \
    --runtime-install "$RUNTIME_INSTALL" \
    --out "$RUN_ROOT/runtime_identity.json" \
    > "$RUN_ROOT/runtime_identity.stdout" \
    2> "$RUN_ROOT/runtime_identity.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/runtime_identity.rc"
[[ $rc -eq 0 ]]

set +e
python3 "$RUNTIME_ROOT/tests/build_quality_runtime_contract.py" \
    --source-root "$RUNTIME_ROOT" \
    --runtime-site-packages "$RUNTIME_SITE_PACKAGES" \
    --runtime-install "$RUNTIME_INSTALL" \
    --model-path "$MODEL_PATH" \
    --instance "$INSTANCE" \
    --optimization-label "$LABEL" \
    --gdn-cache-policy "$POLICY" \
    --gdn-restore-mode "$RESTORE_MODE" \
    --fused-prefill "$FUSED_PREFILL" \
    --kv-eviction-policy "$KV_EVICTION" \
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

for gate in prefix_allocator gdn_action_broadcast; do
    script=prefix_namespace_fork_gate.py
    [[ "$gate" == gdn_action_broadcast ]] && script=gdn_action_broadcast_gate.py
    set +e
    python3 "$RUNTIME_ROOT/tests/$script" --out "$RUN_ROOT/$gate.json" \
        > "$RUN_ROOT/$gate.stdout" 2> "$RUN_ROOT/$gate.stderr"
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$RUN_ROOT/$gate.rc"
    [[ $rc -eq 0 ]]
done

run_preflight before
BEFORE_PREFLIGHT_PASSED=1
port_is_free

setsid "$RUNTIME_ROOT/launch_service" > "$RUN_ROOT/server.log" 2>&1 < /dev/null &
ACTIVE_PID=$!
ACTIVE_PGID=$ACTIVE_PID
SERVICE_STARTED=1
printf '%s\n' "$ACTIVE_PID" > "$RUN_ROOT/server.pid"
for _ in $(seq 1 20); do
    observed_pgid=$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null | tr -d ' ')
    [[ -n "$observed_pgid" ]] && break
    sleep 1
done
[[ -n "${observed_pgid:-}" && "$observed_pgid" == "$ACTIVE_PGID" ]]
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
python3 "$RUNTIME_ROOT/tests/hybrid_kv_startup_gate.py" \
    "$RUN_ROOT/server.log" --mode full_attention --model-path "$MODEL_PATH" \
    --max-model-len 262144 --block-size 16 --tensor-parallel-size 4 \
    --expected-cache-trace 1 --expected-gdn-cache-policy "$POLICY" \
    --expected-gdn-restore-mode "$RESTORE_MODE" \
    --expected-fused-prefill "$FUSED_PREFILL" \
    --expected-kv-eviction-policy "$KV_EVICTION" \
    --out "$RUN_ROOT/startup_contract.json" \
    > "$RUN_ROOT/startup_contract.stdout" \
    2> "$RUN_ROOT/startup_contract.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/startup_contract.rc"
[[ $rc -eq 0 ]]

set +e
timeout --signal=TERM --kill-after=30s 43200s \
    env PYTHONPATH="$IFEVAL_PYTHONPATH" NLTK_DATA="$IFEVAL_ENV/nltk_data" \
    python3 "$EVALUATOR_ROOT/tests/ifeval_quality_api.py" \
    --base http://127.0.0.1:8000 --model llm \
    --out "$RUN_ROOT/ifeval_report.json" \
    --checkpoint "$RUN_ROOT/ifeval.checkpoint.json" \
    --source-revision "$EXPECTED_RUNTIME_REVISION" \
    --runtime-identity "$RUNTIME_IDENTITY" \
    --runtime-overlay-sha256 "$(python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["runtime_overlay_sha256"])' \
        "$RUN_ROOT/runtime_contract.json")" \
    --runtime-contract "$RUN_ROOT/runtime_contract.json" \
    --instance "$INSTANCE" --model-path "$MODEL_PATH" \
    --tokenizer-path "$MODEL_PATH" --gdn-cache-policy "$POLICY" \
    --gdn-restore-mode "$RESTORE_MODE" --fused-prefill "$FUSED_PREFILL" \
    --kv-eviction-policy "$KV_EVICTION" \
    > "$RUN_ROOT/ifeval.stdout" 2> "$RUN_ROOT/ifeval.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/ifeval.rc"
[[ $rc -eq 0 ]]
