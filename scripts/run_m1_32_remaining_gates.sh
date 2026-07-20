#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_ROOT=${M1_32_RUN_ROOT:-$ROOT/bench_runs/m1_32}
MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
START_AT=${M1_32_START_AT:-fine}
FIXED_DIR="$RUN_ROOT/fine32_direct_fixed"
ACTIVE_PID=""

export PYTHONPATH="$ROOT/tests:/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}"

if [[ "$START_AT" != fine && "$START_AT" != aligned ]]; then
    echo "M1_32_START_AT must be fine or aligned" >&2
    exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model directory is missing: $MODEL_PATH" >&2
    exit 2
fi
if [[ ! -f "$FIXED_DIR/matrix.rc" ]] || [[ $(<"$FIXED_DIR/matrix.rc") != 0 ]]; then
    echo "fixed matrix is incomplete or failed: $FIXED_DIR/matrix.rc" >&2
    exit 3
fi

mkdir -p "$RUN_ROOT"
python3 "$ROOT/scripts/summarize_dataset_shaped_matrix.py" \
    "$FIXED_DIR" --out "$FIXED_DIR/summary.json" \
    > "$FIXED_DIR/summary.stdout"
python3 - "$FIXED_DIR/summary.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
validation = report["validation"]
required = (
    "complete_matrix",
    "token_count_match",
    "target_within_one_block",
    "cold_warm_pair_salts_match",
)
failed = [name for name in required if not validation.get(name)]
if validation.get("success_rate", 0.0) < 0.99:
    failed.append("success_rate_at_least_99pct")
if failed:
    raise SystemExit("fixed matrix validation failed: " + ", ".join(failed))
PY

health() {
    python3 -c 'import urllib.request; urllib.request.urlopen(
        "http://127.0.0.1:8000/health", timeout=5).read()' >/dev/null 2>&1
}

stop_service() {
    if [[ -n "$ACTIVE_PID" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
        kill -TERM "$ACTIVE_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$ACTIVE_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$ACTIVE_PID" 2>/dev/null; then
            kill -KILL "$ACTIVE_PID" 2>/dev/null || true
        fi
        wait "$ACTIVE_PID" 2>/dev/null || true
    fi
    ACTIVE_PID=""
}

stop_stale_services() {
    local pids
    pids=$(pgrep -f 'vllm\.entrypoints\.openai\.api_server' || true)
    if [[ -z "$pids" ]]; then
        return 0
    fi
    kill -TERM $pids 2>/dev/null || true
    for _ in $(seq 1 60); do
        pids=$(pgrep -f 'vllm\.entrypoints\.openai\.api_server' || true)
        [[ -z "$pids" ]] && return 0
        sleep 1
    done
    kill -KILL $pids 2>/dev/null || true
}

start_service() {
    local label=$1
    local policy=$2
    local mode=$3
    local output_dir="$RUN_ROOT/$label"

    stop_service
    stop_stale_services
    mkdir -p "$output_dir"
    BI100_GDN_CACHE_POLICY="$policy" \
    BI100_GDN_RESTORE_MODE="$mode" \
    MODEL_PATH="$MODEL_PATH" \
        nohup "$ROOT/launch_service" \
        > "$output_dir/server.log" 2>&1 &
    ACTIVE_PID=$!
    printf '%s\n' "$ACTIVE_PID" > "$output_dir/server.pid"

    for _ in $(seq 1 240); do
        if health; then
            printf '%s\n' 0 > "$output_dir/startup.rc"
            return 0
        fi
        if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
            printf '%s\n' 1 > "$output_dir/startup.rc"
            tail -100 "$output_dir/server.log" >&2 || true
            return 1
        fi
        sleep 10
    done
    printf '%s\n' 124 > "$output_dir/startup.rc"
    echo "service startup timed out: $label" >&2
    return 124
}

run_gate() {
    local output_dir=$1
    local name=$2
    local timeout_s=$3
    shift 3

    mkdir -p "$output_dir"
    set +e
    timeout --signal=TERM --kill-after=20s "${timeout_s}s" "$@" \
        > "$output_dir/$name.stdout" 2> "$output_dir/$name.stderr"
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$output_dir/$name.rc"
    if [[ $rc -ne 0 ]]; then
        echo "gate failed: $name rc=$rc" >&2
        return "$rc"
    fi
}

finish() {
    local rc=$?
    stop_service
    if [[ $rc -ne 0 ]]; then
        printf '%s\n' "$rc" > "$RUN_ROOT/remaining_gates.rc"
    fi
}
trap finish EXIT

if [[ "$START_AT" == fine ]]; then
    FINE_DIR="$RUN_ROOT/fine32_direct_long"
    start_service fine32_direct_long fine32 direct
    run_gate "$FINE_DIR" smoke 300 \
        python3 "$ROOT/tests/smoke_api.py" \
        --base http://127.0.0.1:8000 --mode quick \
        --json-out "$FINE_DIR/smoke.json"
    run_gate "$FINE_DIR" long_131k_exact 4000 \
        python3 "$ROOT/tests/long_context_api.py" \
        --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
        --target-prompt-tokens 131000 --max-tokens 256 \
        --max-model-len 262144 --min-cached-tokens 130992 \
        --equivalence-mode exact --timeout-s 1800 \
        --run-id m1_32_fine_131k \
        --output-dir "$FINE_DIR/long_131k_exact"
    run_gate "$FINE_DIR" long_235k_warm_repeat 6000 \
        python3 "$ROOT/tests/long_context_api.py" \
        --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
        --target-prompt-tokens 235000 --max-tokens 256 \
        --max-model-len 262144 --min-cached-tokens 234992 \
        --equivalence-mode warm-repeat --timeout-s 1800 \
        --run-id m1_32_fine_235k \
        --output-dir "$FINE_DIR/long_235k_warm_repeat"
fi

ALIGNED_DIR="$RUN_ROOT/admission64_aligned_gate"
start_service admission64_aligned_gate admission64 aligned
run_gate "$ALIGNED_DIR" smoke 300 \
    python3 "$ROOT/tests/smoke_api.py" \
    --base http://127.0.0.1:8000 --mode quick \
    --json-out "$ALIGNED_DIR/smoke.json"
run_gate "$ALIGNED_DIR" pressure 5400 \
    python3 "$ROOT/tests/prefix_cache_stress.py" \
    --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
    --eviction-count 17 --timeout-s 600 \
    --run-id m1_32_admission64_aligned \
    --json-out "$ALIGNED_DIR/pressure.json"
run_gate "$ALIGNED_DIR" long_235k_exact 7800 \
    python3 "$ROOT/tests/long_context_api.py" \
    --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
    --target-prompt-tokens 235000 --max-tokens 1000 \
    --max-model-len 262144 --min-cached-tokens 229376 \
    --equivalence-mode exact --timeout-s 3600 \
    --run-id m1_32_aligned_235k \
    --output-dir "$ALIGNED_DIR/long_235k_exact"

printf '%s\n' 0 > "$RUN_ROOT/remaining_gates.rc"
echo "M1-32 remaining runtime gates passed"
