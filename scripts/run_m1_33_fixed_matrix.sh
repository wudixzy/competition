#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_ROOT=${M1_33_RUN_ROOT:-$ROOT/bench_runs/m1_33}
GATES_DIR=${M1_33_GATES_DIR:-$RUN_ROOT/admission64_chunk64_gate}
BASELINE_DIR=${M1_33_BASELINE_DIR:-$ROOT/bench_runs/m1_32/fine32_direct_fixed}
CANDIDATE_DIR=${M1_33_MATRIX_DIR:-$RUN_ROOT/admission64_chunk64_fixed}
MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
SALT_NAMESPACE=${M1_33_SALT_NAMESPACE:-m1_32_ab}
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

require_zero_rc "$RUN_ROOT/runtime_preflight/runtime.rc"
require_zero_rc "$RUN_ROOT/operator_exactness/operator.rc"
require_zero_rc "$GATES_DIR/pressure.rc"
require_zero_rc "$GATES_DIR/long_235k_exact.rc"
require_zero_rc "$RUN_ROOT/remaining_gates.rc"
require_zero_rc "$BASELINE_DIR/matrix.rc"
if [[ ! -f "$BASELINE_DIR/summary.json" ]]; then
    echo "baseline summary is missing: $BASELINE_DIR/summary.json" >&2
    exit 3
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model directory is missing: $MODEL_PATH" >&2
    exit 3
fi
if compgen -G "$CANDIDATE_DIR/requests/*.json" >/dev/null; then
    echo "candidate directory already contains request results: $CANDIDATE_DIR" >&2
    exit 4
fi

mkdir -p "$CANDIDATE_DIR"

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
        printf '%s\n' "$rc" > "$CANDIDATE_DIR/qualification.rc"
    elif [[ ! -f "$CANDIDATE_DIR/qualification.rc" ]]; then
        printf '%s\n' 97 > "$CANDIDATE_DIR/qualification.rc"
        rc=97
    fi
    exit "$rc"
}
trap finish EXIT

run_gate() {
    local name=$1
    local timeout_s=$2
    shift 2

    set +e
    timeout --signal=TERM --kill-after=20s "${timeout_s}s" "$@" \
        > "$CANDIDATE_DIR/$name.stdout" \
        2> "$CANDIDATE_DIR/$name.stderr"
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$CANDIDATE_DIR/$name.rc"
    if [[ $rc -ne 0 ]]; then
        echo "M1-33 matrix gate failed: $name rc=$rc" >&2
        return "$rc"
    fi
}

wait_for_port_free
BI100_GDN_CACHE_POLICY=admission64 \
BI100_GDN_RESTORE_MODE=chunk64 \
MODEL_PATH="$MODEL_PATH" \
    nohup "$ROOT/launch_service" \
    > "$CANDIDATE_DIR/server.log" 2>&1 < /dev/null &
ACTIVE_PID=$!
printf '%s\n' "$ACTIVE_PID" > "$CANDIDATE_DIR/server.pid"

for _ in $(seq 1 240); do
    if health; then
        printf '%s\n' 0 > "$CANDIDATE_DIR/startup.rc"
        break
    fi
    if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
        printf '%s\n' 1 > "$CANDIDATE_DIR/startup.rc"
        tail -100 "$CANDIDATE_DIR/server.log" >&2 || true
        exit 1
    fi
    sleep 10
done
require_zero_rc "$CANDIDATE_DIR/startup.rc"

grep -Fq '[BI100] fixed kernels; moe_direct=1 gdn_packed=1' \
    "$CANDIDATE_DIR/server.log"
grep -Fq '[BI100] GDN cache; policy=admission64 restore=chunk64' \
    "$CANDIDATE_DIR/server.log"
printf '%s\n' 0 > "$CANDIDATE_DIR/runtime_contract.rc"

run_gate smoke 300 \
    python3 "$ROOT/tests/smoke_api.py" \
    --base http://127.0.0.1:8000 --mode quick \
    --json-out "$CANDIDATE_DIR/smoke.json"
run_gate matrix 7200 \
    bash "$ROOT/scripts/run_dataset_shaped_matrix.sh" \
    "$CANDIDATE_DIR" admission64_chunk64_fixed "$SALT_NAMESPACE"
run_gate summarize 120 \
    python3 "$ROOT/scripts/summarize_dataset_shaped_matrix.py" \
    "$CANDIDATE_DIR" --out "$CANDIDATE_DIR/summary.json"
run_gate compare 120 \
    python3 "$ROOT/scripts/compare_dataset_shaped_policies.py" \
    "$BASELINE_DIR/summary.json" "$CANDIDATE_DIR/summary.json" \
    --out "$CANDIDATE_DIR/comparison.json"

printf '%s\n' 0 > "$CANDIDATE_DIR/qualification.rc"
echo "M1-33 fixed-kernel matrix gate passed"
