#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_ROOT=${M1_34_RUN_ROOT:-$ROOT/bench_runs/m1_34/direct_suffix_probe}
MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
ACTIVE_PID=""

export PYTHONPATH="$ROOT/tests:/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model directory is missing: $MODEL_PATH" >&2
    exit 2
fi
for target_mod in 1 2; do
    if [[ -e "$RUN_ROOT/suffix_mod${target_mod}/pressure.rc" ]]; then
        echo "probe output already exists for suffix_mod${target_mod}" >&2
        exit 4
    fi
done
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
    for _ in $(seq 1 120); do
        port_is_free && return 0
        sleep 1
    done
    echo "port 8000 remained busy" >&2
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
    if [[ $rc -ne 0 && ! -f "$RUN_ROOT/probe.rc" ]]; then
        printf '%s\n' "$rc" > "$RUN_ROOT/probe.rc"
    fi
    exit "$rc"
}
trap finish EXIT

start_service() {
    local output_dir=$1
    stop_service
    wait_for_port_free
    BI100_GDN_CACHE_POLICY=admission64 \
    BI100_GDN_RESTORE_MODE=direct \
    MODEL_PATH="$MODEL_PATH" \
        nohup "$ROOT/launch_service" \
        > "$output_dir/server.log" 2>&1 < /dev/null &
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
    return 124
}

for target_mod in 1 2; do
    output_dir="$RUN_ROOT/suffix_mod${target_mod}"
    mkdir -p "$output_dir"
    if ! start_service "$output_dir"; then
        echo "M1-34 service startup failed for suffix_mod${target_mod}" >&2
        exit 1
    fi
    grep -Fq '[BI100] fixed kernels; moe_direct=1 gdn_packed=1' \
        "$output_dir/server.log"
    grep -Fq '[BI100] GDN cache; policy=admission64 restore=direct' \
        "$output_dir/server.log"
    printf '%s\n' 0 > "$output_dir/runtime_contract.rc"

    set +e
    timeout --signal=TERM --kill-after=20s 5400s \
        python3 "$ROOT/tests/prefix_cache_stress.py" \
        --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
        --eviction-count 17 --eviction-target-mod "$target_mod" \
        --timeout-s 600 --run-id "m1_34_direct_suffix_${target_mod}" \
        --json-out "$output_dir/pressure.json" \
        > "$output_dir/pressure.stdout" 2> "$output_dir/pressure.stderr"
    pressure_rc=$?
    set -e
    printf '%s\n' "$pressure_rc" > "$output_dir/pressure.rc"

    if grep -Eiq 'CUDA error|SIGSEGV|Fatal Python error|out of memory|worker process.*died|Gloo.*failed' \
            "$output_dir/server.log"; then
        grep -Ein 'CUDA error|SIGSEGV|Fatal Python error|out of memory|worker process.*died|Gloo.*failed' \
            "$output_dir/server.log" > "$output_dir/fatal_scan.txt" || true
        printf '%s\n' 1 > "$output_dir/fatal_scan.rc"
    else
        : > "$output_dir/fatal_scan.txt"
        printf '%s\n' 0 > "$output_dir/fatal_scan.rc"
    fi
    stop_service
done

python3 - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rcs = {
    suffix: int((root / f"suffix_mod{suffix}" / "pressure.rc").read_text())
    for suffix in (1, 2)
}
if rcs == {1: 1, 2: 0}:
    classification = "single_token_specific"
elif rcs == {1: 0, 2: 0}:
    classification = "direct_currently_stable"
elif rcs[1] != 0 and rcs[2] != 0:
    classification = "broader_direct_failure"
else:
    classification = "inverted_or_inconclusive"
report = {"pressure_rc": rcs, "classification": classification}
(root / "classification.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
PY

printf '%s\n' 0 > "$RUN_ROOT/probe.rc"
echo "M1-34 direct suffix probe completed"
