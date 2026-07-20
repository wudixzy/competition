#!/bin/bash
set -u -o pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 OUTPUT_DIR LABEL" >&2
    exit 2
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR=$1
LABEL=$2
MODEL_PATH=/root/public-storage/models/Qwen/Qwen3.6-35B-A3B
PROFILE_SCRIPT="$ROOT/bench_runs/m1_32/profile_dataset_shaped_prompt.py"
BASE=http://127.0.0.1:8000
COREX_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages

mkdir -p "$OUTPUT_DIR/requests"
: > "$OUTPUT_DIR/exit_codes.txt"

health() {
    python3 -c 'import urllib.request; urllib.request.urlopen(
        "http://127.0.0.1:8000/health", timeout=10).read()'
}

if [[ ! -f "$PROFILE_SCRIPT" ]]; then
    echo "missing profile script: $PROFILE_SCRIPT" >&2
    exit 2
fi

for target in 4096 7800 16000; do
    for pair in 1 2 3; do
        salt="${LABEL}_${target}_${pair}_$(date +%s%N)"
        for phase in cold warm; do
            out="$OUTPUT_DIR/requests/${target}_pair${pair}_${phase}.json"
            PYTHONPATH="$ROOT/tests:$COREX_PYTHONPATH:${PYTHONPATH:-}" \
                timeout --signal=TERM --kill-after=20s 900s \
                python3 "$PROFILE_SCRIPT" \
                --model "$MODEL_PATH" \
                --base "$BASE" \
                --target-tokens "$target" \
                --max-tokens 64 \
                --tools 29 \
                --timeout-s 900 \
                --stream \
                --prompt-salt "$salt" \
                --out "$out" \
                "$ROOT/qwen3_6_scripts/qwen3_5.py" \
                "$ROOT/docs/HANDOFF_SUMMARY.md" \
                "$ROOT/tests/bench_perf.py" \
                > "$OUTPUT_DIR/requests/${target}_pair${pair}_${phase}.stdout" \
                2> "$OUTPUT_DIR/requests/${target}_pair${pair}_${phase}.stderr"
            rc=$?
            printf '%s %s\n' "$rc" "$out" >> "$OUTPUT_DIR/exit_codes.txt"
            if [[ $rc -ne 0 ]]; then
                echo "request failed: target=$target pair=$pair phase=$phase rc=$rc" >&2
                exit "$rc"
            fi
            if ! health; then
                echo "service health failed after target=$target pair=$pair phase=$phase" >&2
                exit 90
            fi
        done
    done
done
