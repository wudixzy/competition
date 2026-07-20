#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_ROOT=${M1_34_RUN_ROOT:-$ROOT/bench_runs/m1_34}
MATRIX_DIR=${M1_34_MATRIX_DIR:-$RUN_ROOT/admission64_direct_guard2_fixed}
OUTPUT_DIR=${M1_34_POST_DIR:-$RUN_ROOT/admission64_direct_guard2_post}
MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
ACTIVE_PID=""

export PYTHONPATH="$ROOT/tests:/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}"

require_zero_rc() {
    local path=$1
    if [[ ! -f "$path" ]] || [[ $(<"$path") != 0 ]]; then
        echo "required successful gate is missing: $path" >&2
        exit 3
    fi
}

require_zero_rc "$MATRIX_DIR/qualification.rc"
require_zero_rc "$MATRIX_DIR/startup_capacity.rc"
require_zero_rc "$MATRIX_DIR/compare.rc"
if [[ ! -f "$MATRIX_DIR/comparison.json" ]]; then
    echo "matrix comparison is missing: $MATRIX_DIR/comparison.json" >&2
    exit 3
fi
python3 - "$MATRIX_DIR/comparison.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if not report.get("stage_qualified"):
    raise SystemExit("matrix comparison did not qualify the cache stage")
PY
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model directory is missing: $MODEL_PATH" >&2
    exit 3
fi
if compgen -G "$OUTPUT_DIR/long_*/*.json" >/dev/null; then
    echo "post-matrix directory already contains long-context results: $OUTPUT_DIR" >&2
    exit 4
fi

mkdir -p "$OUTPUT_DIR"

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
    for _ in $(seq 1 120); do
        port_is_free && return 0
        sleep 1
    done
    echo "port 8000 remained busy; refusing to mix service lifetimes" >&2
    return 1
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

finish() {
    local rc=$?
    trap - EXIT
    set +e
    stop_service
    if [[ $rc -ne 0 ]]; then
        printf '%s\n' "$rc" > "$OUTPUT_DIR/post_matrix.rc"
    elif [[ ! -f "$OUTPUT_DIR/post_matrix.rc" ]]; then
        printf '%s\n' 97 > "$OUTPUT_DIR/post_matrix.rc"
        rc=97
    fi
    exit "$rc"
}
trap finish EXIT

run_gate() {
    local name=$1
    local timeout_s=$2
    shift 2

    mkdir -p "$OUTPUT_DIR/$name"
    set +e
    timeout --signal=TERM --kill-after=20s "${timeout_s}s" "$@" \
        > "$OUTPUT_DIR/$name.stdout" 2> "$OUTPUT_DIR/$name.stderr"
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$OUTPUT_DIR/$name.rc"
    if [[ $rc -ne 0 ]]; then
        echo "M1-34 post-matrix gate failed: $name rc=$rc" >&2
        return "$rc"
    fi
    health
}

wait_for_port_free
BI100_GDN_CACHE_POLICY=admission64 \
BI100_GDN_RESTORE_MODE=direct \
MODEL_PATH="$MODEL_PATH" \
    nohup "$ROOT/launch_service" \
    > "$OUTPUT_DIR/server.log" 2>&1 < /dev/null &
ACTIVE_PID=$!
printf '%s\n' "$ACTIVE_PID" > "$OUTPUT_DIR/server.pid"

for _ in $(seq 1 240); do
    if health; then
        printf '%s\n' 0 > "$OUTPUT_DIR/startup.rc"
        break
    fi
    if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
        printf '%s\n' 1 > "$OUTPUT_DIR/startup.rc"
        tail -100 "$OUTPUT_DIR/server.log" >&2 || true
        exit 1
    fi
    sleep 10
done
require_zero_rc "$OUTPUT_DIR/startup.rc"

grep -Fq '[BI100] fixed kernels; moe_direct=1 gdn_packed=1' \
    "$OUTPUT_DIR/server.log"
grep -Fq '[BI100] GDN cache; policy=admission64 restore=direct' \
    "$OUTPUT_DIR/server.log"
printf '%s\n' 0 > "$OUTPUT_DIR/runtime_contract.rc"
run_gate startup_capacity 30 \
    python3 "$ROOT/scripts/check_startup_capacity.py" \
    "$OUTPUT_DIR/server.log" --max-model-len 262144 --block-size 16 \
    --out "$OUTPUT_DIR/startup_capacity.json"

run_gate smoke 300 \
    python3 "$ROOT/tests/smoke_api.py" \
    --base http://127.0.0.1:8000 --mode quick \
    --json-out "$OUTPUT_DIR/smoke.json"
run_gate long_131k_exact 5400 \
    python3 "$ROOT/tests/long_context_api.py" \
    --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
    --target-prompt-tokens 131000 --max-tokens 256 \
    --min-completion-tokens 256 --max-model-len 262144 \
    --min-cached-tokens 130992 --equivalence-mode exact \
    --timeout-s 2400 --run-id m1_34_direct_guard2_131k \
    --output-dir "$OUTPUT_DIR/long_131k_exact"
run_gate long_235k_exact 7800 \
    python3 "$ROOT/tests/long_context_api.py" \
    --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
    --target-prompt-tokens 235000 --max-tokens 1000 \
    --min-completion-tokens 1000 --max-model-len 262144 \
    --min-cached-tokens 234992 --equivalence-mode exact \
    --timeout-s 3600 --run-id m1_34_direct_guard2_235k \
    --output-dir "$OUTPUT_DIR/long_235k_exact"
run_gate long_262k_capacity 9000 \
    python3 "$ROOT/tests/long_context_api.py" \
    --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
    --target-prompt-tokens 262000 --max-tokens 16 \
    --min-completion-tokens 16 --max-model-len 262144 \
    --min-cached-tokens 261984 --equivalence-mode exact \
    --timeout-s 4200 --run-id m1_34_direct_guard2_262k \
    --output-dir "$OUTPUT_DIR/long_262k_capacity"

if grep -Eiq 'CUDA error|SIGSEGV|Fatal Python error|out of memory|worker process.*died|Gloo.*failed|AssertionError' \
        "$OUTPUT_DIR/server.log"; then
    grep -Ein 'CUDA error|SIGSEGV|Fatal Python error|out of memory|worker process.*died|Gloo.*failed|AssertionError' \
        "$OUTPUT_DIR/server.log" > "$OUTPUT_DIR/fatal_scan.txt" || true
    printf '%s\n' 1 > "$OUTPUT_DIR/fatal_scan.rc"
    exit 1
fi
: > "$OUTPUT_DIR/fatal_scan.txt"
printf '%s\n' 0 > "$OUTPUT_DIR/fatal_scan.rc"
printf '%s\n' 0 > "$OUTPUT_DIR/post_matrix.rc"
echo "M1-34 direct long-context and 256K capacity gates passed"
