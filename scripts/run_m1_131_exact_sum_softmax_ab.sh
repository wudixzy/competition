#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
COREX_LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
COREX_PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

if [[ ${1:-} == "--cell" ]]; then
    if [[ $# -ne 11 ]]; then
        echo "invalid internal M1-131 cell invocation" >&2
        exit 2
    fi
    INSTANCE=$2
    GPU=$3
    CASE_NAME=$4
    CONTROL_EXTENSION=$5
    CONTROL_SHA=$6
    CANDIDATE_EXTENSION=$7
    CANDIDATE_SHA=$8
    RUN_ROOT=$9
    SOURCE_REVISION=${10}
    RUNTIME_IDENTITY=${11}

    set +e
    timeout --foreground --signal=TERM --kill-after=60s 3600s \
        env CUDA_VISIBLE_DEVICES="$GPU" \
        PYTHONPATH="$ROOT:$SYSTEM_PYTHONPATH" \
        LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
        python3 "$ROOT/tests/bench_m1_131_exact_sum_softmax.py" \
        --case "$CASE_NAME" \
        --control-extension "$CONTROL_EXTENSION" \
        --candidate-extension "$CANDIDATE_EXTENSION" \
        --expected-control-sha256 "$CONTROL_SHA" \
        --expected-candidate-sha256 "$CANDIDATE_SHA" \
        --source-commit "$SOURCE_REVISION" \
        --runtime-identity "$RUNTIME_IDENTITY" \
        --instance "$INSTANCE" \
        --visible-physical-gpu "$GPU" \
        --output "$RUN_ROOT/${CASE_NAME}.json"
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$RUN_ROOT/${CASE_NAME}.rc"
    exit "$rc"
fi

if [[ $# -ne 4 ]]; then
    echo "usage: $0 INSTANCE CONTROL_EXTENSION CANDIDATE_EXTENSION RUN_ROOT" >&2
    exit 2
fi

INSTANCE=$1
CONTROL_EXTENSION=$(realpath "$2")
CANDIDATE_EXTENSION=$(realpath "$3")
RUN_ROOT=$(python3 - "$4" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)

if [[ "$RUN_ROOT" != /tmp/* ]]; then
    echo "M1-131 output must use a private /tmp path" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
    echo "M1-131 output already exists: $RUN_ROOT" >&2
    exit 2
fi
if [[ ! -s "$CONTROL_EXTENSION" || ! -s "$CANDIDATE_EXTENSION" ]]; then
    echo "both extension artifacts must exist and be non-empty" >&2
    exit 2
fi

CONTROL_SHA=$(sha256sum "$CONTROL_EXTENSION" | cut -d' ' -f1)
CANDIDATE_SHA=$(sha256sum "$CANDIDATE_EXTENSION" | cut -d' ' -f1)
if [[ "$CONTROL_SHA" == "$CANDIDATE_SHA" ]]; then
    echo "control and candidate extension artifacts are identical" >&2
    exit 2
fi

if [[ -n $(git -C "$ROOT" status --porcelain --untracked-files=all) ]]; then
    echo "M1-131 requires a clean committed source tree" >&2
    exit 2
fi
SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
RUNTIME_IDENTITY=corex-3.2.3-m1-131-exact-sum
CASES=(
    production_dense_q8176
    production_65k_q8176
    production_128k_q8176
    production_235k_q5616
)
PIDS=()
BEFORE_PREFLIGHT_PASSED=0

mkdir -p "$RUN_ROOT"
printf '%s\n' "$SOURCE_REVISION" > "$RUN_ROOT/source_revision.txt"
printf '%s\n' "$CONTROL_SHA" > "$RUN_ROOT/control_extension_sha256.txt"
printf '%s\n' "$CANDIDATE_SHA" > "$RUN_ROOT/candidate_extension_sha256.txt"

finish() {
    local primary_rc=$?
    local cleanup_rc=0
    local fatal_rc=0
    local find_rc=0
    local postflight_rc=0
    local preflight_rc=0
    local preflight_comparison_rc=0
    local artifact
    local grep_rc
    local pid
    trap - EXIT INT TERM
    set +e

    for pid in "${PIDS[@]}"; do
        bi100_stop_process_group "$pid" "$pid" 60 20 || cleanup_rc=1
    done
    if [[ $cleanup_rc -ne 0 ]]; then
        primary_rc=1
    fi

    : > "$RUN_ROOT/fatal_scan.txt"
    find "$RUN_ROOT" -type f \
        \( -name '*.stdout' -o -name '*.stderr' \) -print0 \
        > "$RUN_ROOT/.fatal_scan_files"
    find_rc=$?
    printf '%s\n' "$find_rc" > "$RUN_ROOT/fatal_scan_find.rc"
    if [[ $find_rc -ne 0 ]]; then
        printf 'reason=find_error\n' >> "$RUN_ROOT/fatal_scan.txt"
        fatal_rc=1
    else
        while IFS= read -r -d '' artifact; do
            grep -Eiq \
                    'CUDA error|illegal memory access|SIGSEGV|Fatal Python error|out of memory|device-side assert|CoreX.*(failed|fatal)|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|Timeout(Error|Expired)' \
                    "$artifact"
            grep_rc=$?
            if [[ $grep_rc -eq 0 ]]; then
                printf 'file=%s reason=matched\n' \
                    "$artifact" >> "$RUN_ROOT/fatal_scan.txt"
                fatal_rc=1
            elif [[ $grep_rc -ne 1 ]]; then
                printf 'file=%s reason=read_error\n' \
                    "$artifact" >> "$RUN_ROOT/fatal_scan.txt"
                fatal_rc=1
            fi
        done < "$RUN_ROOT/.fatal_scan_files"
    fi
    rm -f "$RUN_ROOT/.fatal_scan_files"
    printf '%s\n' "$fatal_rc" > "$RUN_ROOT/fatal_scan.rc"

    timeout --signal=TERM --kill-after=70s 240s \
        env PYTHONPATH="$ROOT" \
        python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus 0,1,2,3 --settle-timeout-s 90 \
        --clean-samples 3 --sample-interval-s 2 \
        --out "$RUN_ROOT/final_postflight.json" \
        > "$RUN_ROOT/final_postflight.stdout" \
        2> "$RUN_ROOT/final_postflight.stderr"
    postflight_rc=$?
    printf '%s\n' "$postflight_rc" > "$RUN_ROOT/final_postflight.rc"

    if [[ "$BEFORE_PREFLIGHT_PASSED" == 1 \
            && $cleanup_rc -eq 0 && $postflight_rc -eq 0 ]]; then
        timeout --signal=TERM --kill-after=70s 480s \
            env PYTHONPATH="$ROOT:$SYSTEM_PYTHONPATH" \
            LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
            python3 "$ROOT/tests/bi100_preflight.py" \
            --gpus 0,1,2,3 --timeout-s 25 --matmul-size 1024 \
            --json-out "$RUN_ROOT/preflight_after.json" \
            > "$RUN_ROOT/preflight_after.stdout" \
            2> "$RUN_ROOT/preflight_after.stderr"
        preflight_rc=$?
        printf '%s\n' "$preflight_rc" > "$RUN_ROOT/preflight_after.rc"
        if [[ $preflight_rc -eq 0 ]]; then
            python3 "$ROOT/tests/compare_bi100_preflights.py" \
                --preflight "before=$RUN_ROOT/preflight_before.json" \
                --preflight "after=$RUN_ROOT/preflight_after.json" \
                --expected-gpus 0,1,2,3 \
                --max-free-memory-drop-bytes 1073741824 \
                --out "$RUN_ROOT/preflight_comparison.json" \
                > "$RUN_ROOT/preflight_comparison.stdout" \
                2> "$RUN_ROOT/preflight_comparison.stderr"
            preflight_comparison_rc=$?
        else
            preflight_comparison_rc=1
        fi
        printf '%s\n' "$preflight_comparison_rc" \
            > "$RUN_ROOT/preflight_comparison.rc"
    fi

    if [[ $fatal_rc -ne 0 || $postflight_rc -ne 0 \
            || $preflight_rc -ne 0 \
            || $preflight_comparison_rc -ne 0 ]]; then
        primary_rc=1
    fi
    exit "$primary_rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

timeout --signal=TERM --kill-after=70s 240s \
    env PYTHONPATH="$ROOT" \
    python3 "$ROOT/tests/service_postflight_gate.py" \
    --gpus 0,1,2,3 --settle-timeout-s 90 \
    --clean-samples 3 --sample-interval-s 2 \
    --out "$RUN_ROOT/postflight_before.json" \
    > "$RUN_ROOT/postflight_before.stdout" \
    2> "$RUN_ROOT/postflight_before.stderr"
printf '%s\n' 0 > "$RUN_ROOT/postflight_before.rc"

timeout --signal=TERM --kill-after=70s 480s \
    env PYTHONPATH="$ROOT:$SYSTEM_PYTHONPATH" \
    LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
    python3 "$ROOT/tests/bi100_preflight.py" \
    --gpus 0,1,2,3 --timeout-s 25 --matmul-size 1024 \
    --json-out "$RUN_ROOT/preflight_before.json" \
    > "$RUN_ROOT/preflight_before.stdout" \
    2> "$RUN_ROOT/preflight_before.stderr"
printf '%s\n' 0 > "$RUN_ROOT/preflight_before.rc"
BEFORE_PREFLIGHT_PASSED=1

for gpu in 0 1 2 3; do
    setsid "$0" --cell \
        "$INSTANCE" "$gpu" "${CASES[$gpu]}" \
        "$CONTROL_EXTENSION" "$CONTROL_SHA" \
        "$CANDIDATE_EXTENSION" "$CANDIDATE_SHA" \
        "$RUN_ROOT" "$SOURCE_REVISION" "$RUNTIME_IDENTITY" \
        > "$RUN_ROOT/${CASES[$gpu]}.stdout" \
        2> "$RUN_ROOT/${CASES[$gpu]}.stderr" &
    PIDS+=("$!")
done

cell_rc=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || cell_rc=1
done
if [[ $cell_rc -ne 0 ]]; then
    echo "one or more M1-131 component cells failed" >&2
    exit "$cell_rc"
fi
post_wait_cleanup_rc=0
for pid in "${PIDS[@]}"; do
    bi100_stop_process_group "$pid" "$pid" 60 20 \
        || post_wait_cleanup_rc=1
done
if [[ $post_wait_cleanup_rc -ne 0 ]]; then
    echo "one or more M1-131 cell process groups failed cleanup" >&2
    exit "$post_wait_cleanup_rc"
fi
PIDS=()

comparison_args=()
for case_name in "${CASES[@]}"; do
    comparison_args+=(--cell "$RUN_ROOT/${case_name}.json")
done
set +e
python3 "$ROOT/tests/compare_m1_131_exact_sum_softmax.py" \
    "${comparison_args[@]}" \
    --out "$RUN_ROOT/comparison.json" \
    > "$RUN_ROOT/comparison.stdout" \
    2> "$RUN_ROOT/comparison.stderr"
comparison_rc=$?
set -e
printf '%s\n' "$comparison_rc" > "$RUN_ROOT/comparison.rc"
exit "$comparison_rc"
