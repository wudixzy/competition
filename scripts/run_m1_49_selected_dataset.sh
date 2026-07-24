#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"
M1_49_LONG_DIR=${M1_49_LONG_DIR:-$ROOT/bench_runs/m1_49/full_attention_long}
RUN_ROOT=${M1_49_SELECTED_RUN_ROOT:-$ROOT/bench_runs/m1_49/selected_dataset}
MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
DATASET=${M1_49_SELECTED_DATASET:-$ROOT/chat_dataset_v0.json}
RUN_ID=${M1_49_SELECTED_RUN_ID:-m149-selected-dataset-20260724}
EXPECTED_DATASET_SHA256=dac6afc77621b51dbc09cfa046c008a1e51a779bb771edcb27cb6a686f8884c8
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
export BI100_RUNTIME_WORKDIR=/tmp/m1-49-selected-runtime
export BI100_PROFILE=0
export BI100_PROFILE_INCLUDE_STARTUP=0
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
    echo "M1-49 long-context qualification is missing" >&2
    exit 3
fi
python3 - "$M1_49_QUALIFICATION" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if (report.get("schema") != "bi100-m1-49-long-context-qualification-v1"
        or report.get("version") != 1
        or report.get("qualified") is not True
        or report.get("scope")
        != "hybrid-kv-capacity-correctness-not-prefill-speed"):
    raise SystemExit("M1-49 long-context prerequisite is invalid")
PY

if [[ -z "${BI100_RUNTIME_SITE_PACKAGES:-}" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/vllm" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/transformers" ]]; then
    echo "selected replay requires an atomic bare-host runtime overlay" >&2
    exit 3
fi
export BI100_RUNTIME_SITE_PACKAGES
export PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages:${PYTHONPATH:-}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model directory is missing: $MODEL_PATH" >&2
    exit 4
fi
if [[ ! -f "$DATASET" ]]; then
    echo "selected dataset is missing: $DATASET" >&2
    exit 4
fi
python3 - "$DATASET" "$EXPECTED_DATASET_SHA256" <<'PY'
import hashlib
import json
import sys

payload = open(sys.argv[1], "rb").read()
if hashlib.sha256(payload).hexdigest() != sys.argv[2]:
    raise SystemExit("selected dataset SHA-256 differs from the fixed gate")
data = json.loads(payload)
if [len(row.get("user_questions", [])) for row in data] != [4, 4, 3, 2]:
    raise SystemExit("selected dataset shape differs from the fixed gate")
PY
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
        . ':(exclude)bench_runs/**')" ]]; then
    echo "selected replay refuses a dirty source tree" >&2
    exit 4
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "selected replay output already exists: $RUN_ROOT" >&2
    exit 5
fi
mkdir -p "$RUN_ROOT"
git -C "$ROOT" rev-parse HEAD > "$RUN_ROOT/source_revision.txt"

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
        echo "selected replay service cleanup failed" >&2
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
        echo "selected replay four-GPU preflight failed at $label" >&2
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
        echo "selected replay gate failed: $name rc=$rc" >&2
        return "$rc"
    fi
    health
}

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
        echo "selected replay offline gate failed: $name rc=$rc" >&2
        return "$rc"
    fi
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

run_preflight before_selected
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
grep -Fq '[BI100] GDN cache; policy=admission64 restore=direct' \
    "$RUN_ROOT/server.log"
grep -Fq '[BI100] M1-49 runtime contract; accounting=full_attention' \
    "$RUN_ROOT/server.log"
printf '%s\n' 0 > "$RUN_ROOT/runtime_contract.rc"

run_gate startup_gate 180 \
    python3 "$ROOT/tests/hybrid_kv_startup_gate.py" \
    "$RUN_ROOT/server.log" --mode full_attention --model-path "$MODEL_PATH" \
    --max-model-len 262144 --block-size 16 --tensor-parallel-size 4 \
    --out "$RUN_ROOT/startup_gate.json"
python3 - "$M1_49_QUALIFICATION" "$RUN_ROOT/startup_gate.json" <<'PY'
import json
import sys

prior = json.load(open(sys.argv[1], encoding="utf-8"))
current = json.load(open(sys.argv[2], encoding="utf-8"))
expected = prior.get("candidate_startup") or {}
if (current.get("qualified") is not True
        or current.get("mode") != expected.get("mode")
        or current.get("observed_attention_layers")
        != expected.get("attention_layers")
        or current.get("runtime_contract_sha256")
        != expected.get("runtime_contract_sha256")):
    raise SystemExit("selected replay startup differs from M1-49 qualification")
PY
printf '%s\n' 0 > "$RUN_ROOT/startup_match.rc"

run_gate replay 12600 \
    python3 "$ROOT/scripts/replay_selected_dataset.py" \
    --dataset "$DATASET" --label "$RUN_ID" \
    --base http://127.0.0.1:8000 --max-tokens 256 \
    --seed 20260713 --timeout-s 900 \
    --out "$RUN_ROOT/replay.json"

fatal_scan
stop_service
wait_for_port_free
run_preflight after_selected
run_offline_gate preflight_comparison 120 \
    python3 "$ROOT/tests/compare_bi100_preflights.py" \
    --preflight "before_selected=$RUN_ROOT/preflight_before_selected.json" \
    --preflight "after_selected=$RUN_ROOT/preflight_after_selected.json" \
    --max-free-memory-drop-bytes 1073741824 \
    --out "$RUN_ROOT/preflight_comparison.json"
run_offline_gate qualification 60 \
    python3 "$ROOT/tests/qualify_selected_dataset_replay.py" \
    --source "$RUN_ROOT/replay.json" \
    --out "$RUN_ROOT/qualification.json"

echo "M1-49 selected dataset replay qualified"
