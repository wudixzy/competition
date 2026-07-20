#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_ROOT=${M1_33_RUN_ROOT:-$ROOT/bench_runs/m1_33}
BASELINE_DIR=${M1_33_BASELINE_DIR:-$ROOT/bench_runs/m1_32/fine32_direct_fixed}
MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
OPERATOR_DIR="$RUN_ROOT/operator_exactness"

export PYTHONPATH="$ROOT/tests:/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$OPERATOR_DIR"
set +e
timeout --signal=TERM --kill-after=20s 900s \
    python3 "$ROOT/tests/gdn_split_exactness.py" \
    --device cuda:0 --out "$OPERATOR_DIR/result.json" \
    > "$OPERATOR_DIR/stdout.log" 2> "$OPERATOR_DIR/stderr.log"
operator_rc=$?
set -e
printf '%s\n' "$operator_rc" > "$OPERATOR_DIR/operator.rc"
if [[ $operator_rc -ne 0 ]]; then
    echo "M1-33 native-chunk exactness failed: rc=$operator_rc" >&2
    exit "$operator_rc"
fi

exec env \
    M1_32_RUN_ROOT="$RUN_ROOT" \
    M1_32_FIXED_DIR="$BASELINE_DIR" \
    M1_32_START_AT=aligned \
    M1_32_FALLBACK_LABEL=admission64_chunk64_gate \
    M1_32_FALLBACK_POLICY=admission64 \
    M1_32_FALLBACK_MODE=chunk64 \
    M1_32_FALLBACK_MIN_CACHED=234944 \
    M1_32_FALLBACK_PRESSURE_RUN_ID=m1_33_chunk64_pressure \
    M1_32_FALLBACK_RUN_ID=m1_33_chunk64_235k \
    MODEL_PATH="$MODEL_PATH" \
    bash "$ROOT/scripts/run_m1_32_remaining_gates.sh"
