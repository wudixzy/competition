#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

if [[ $# -ne 2 ]]; then
    echo "usage: $0 INSTANCE RUN_ROOT" >&2
    exit 2
fi

INSTANCE=$1
RUN_ROOT=$2
PREFIX_GPU=${PREFIX_GPU:-0}
WMMA_GPU=${WMMA_GPU:-1}
EXPERIMENT_TIMEOUT_S=7200
CURRENT_STAGE=argument_validation
FINALIZED=0
FATAL_PATTERN='CUDA error|illegal memory access|SIGSEGV|Fatal Python error|out of memory|device-side assert|CoreX.*(failed|fatal)|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|engine iteration timed out|watchdog.*tim(e|ed) out'

declare -A CHILD_PID=()
declare -A CHILD_PGID=()
declare -A CHILD_STARTTIME=()
declare -A CHILD_TOKEN=()
declare -A CHILD_IDENTITY=()
declare -A CHILD_DONE=()

if [[ ! "$INSTANCE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    echo "INSTANCE must be a short non-sensitive label" >&2
    exit 2
fi
for gpu in "$PREFIX_GPU" "$WMMA_GPU"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
        echo "oracle GPU indices must be non-negative integers" >&2
        exit 2
    fi
done
if [[ "$PREFIX_GPU" == "$WMMA_GPU" ]]; then
    echo "prefix and WMMA oracles require distinct physical GPUs" >&2
    exit 2
fi

RUN_ROOT=$(python3 - "$RUN_ROOT" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)
if [[ "$RUN_ROOT/" == "$ROOT/"* ]]; then
    echo "RUN_ROOT must stay outside the source repository" >&2
    exit 3
fi
if [[ "$RUN_ROOT" != /tmp/* ]]; then
    echo "RUN_ROOT must use a private /tmp path" >&2
    exit 3
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "RUN_ROOT already exists: $RUN_ROOT" >&2
    exit 3
fi
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
        . ':(exclude)bench_runs/**')" ]]; then
    echo "M1-103 refuses a dirty source tree" >&2
    exit 3
fi

mkdir -p "$RUN_ROOT/prefix" "$RUN_ROOT/wmma" "$RUN_ROOT/wmma_build"
SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
SOURCE_BRANCH=$(git -C "$ROOT" branch --show-current)
printf '%s\n' "$SOURCE_REVISION" > "$RUN_ROOT/source_revision.txt"
printf '%s\n' "$SOURCE_BRANCH" > "$RUN_ROOT/source_branch.txt"
printf '%s\n' "$INSTANCE" > "$RUN_ROOT/instance.txt"
printf '%s\n' "$PREFIX_GPU" > "$RUN_ROOT/prefix_gpu.txt"
printf '%s\n' "$WMMA_GPU" > "$RUN_ROOT/wmma_gpu.txt"

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
COREX_LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
COREX_PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
GPU_SET="$PREFIX_GPU,$WMMA_GPU"

write_stage() {
    printf '%s\n' "$CURRENT_STAGE" > "$RUN_ROOT/stage.txt"
}

run_preflight() {
    local output=$1
    timeout --signal=TERM --kill-after=90s 360s \
        env -u CUDA_VISIBLE_DEVICES \
        PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
        LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
        python3 "$ROOT/tests/bi100_preflight.py" \
        --gpus "$GPU_SET" --timeout-s 25 --matmul-size 1024 \
        --json-out "$output.json" \
        > "$output.stdout" 2> "$output.stderr"
}

run_postflight() {
    local output=$1
    timeout --signal=TERM --kill-after=90s 300s \
        env -u CUDA_VISIBLE_DEVICES PYTHONPATH="$ROOT/tests" \
        python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus "$GPU_SET" --settle-timeout-s 90 \
        --clean-samples 3 --sample-interval-s 2 \
        --out "$output.json" \
        > "$output.stdout" 2> "$output.stderr"
}

read_process_identity() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys

value = (Path("/proc") / sys.argv[1] / "stat").read_text(encoding="ascii")
tail = value[value.rfind(")") + 2:].split()
print(tail[0], tail[19])
PY
}

child_is_live() {
    local label=$1
    local observed
    [[ -n "${CHILD_PID[$label]:-}" \
        && -n "${CHILD_STARTTIME[$label]:-}" ]] || return 1
    observed=$(read_process_identity "${CHILD_PID[$label]}" \
        2>/dev/null) || return 1
    local state
    local starttime
    read -r state starttime <<< "$observed"
    [[ "$state" != Z && "$starttime" == "${CHILD_STARTTIME[$label]}" ]]
}

attest_child() {
    local label=$1
    local identity=${CHILD_IDENTITY[$label]}
    local pid=${CHILD_PID[$label]}
    local starttime=${CHILD_STARTTIME[$label]}
    local observed
    local pgid
    local token
    observed=$(python3 - "$identity" "$pid" "$starttime" <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pid = int(sys.argv[2])
starttime = int(sys.argv[3])
token = value.get("session_token")
if (
    value.get("schema") != "bi100-process-session-v1"
    or value.get("version") != 1
    or value.get("pid") != pid
    or value.get("pgid") != pid
    or value.get("sid") != pid
    or value.get("starttime_ticks") != starttime
    or not isinstance(token, str)
    or len(token) != 32
    or any(character not in "0123456789abcdef" for character in token)
):
    raise SystemExit(1)
print(value["pgid"], token)
PY
    ) || return 1
    read -r pgid token <<< "$observed"
    CHILD_PGID[$label]=$pgid
    CHILD_TOKEN[$label]=$token
    [[ "${CHILD_PGID[$label]}" == "$pid" \
        && "${CHILD_TOKEN[$label]}" =~ ^[0-9a-f]{32}$ ]]
}

start_child() {
    local label=$1
    local gpu=$2
    shift 2
    local identity="$RUN_ROOT/${label}_identity.json"
    (
        exec python3 "$ROOT/scripts/exec_bi100_session.py" \
            "$identity" -- \
            env CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
            LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
            "$@"
    ) > "$RUN_ROOT/${label}.stdout" \
      2> "$RUN_ROOT/${label}.stderr" &
    CHILD_PID[$label]=$!
    CHILD_IDENTITY[$label]=$identity
    CHILD_DONE[$label]=0
    CHILD_STARTTIME[$label]=""
    for _ in $(seq 1 40); do
        local observed
        observed=$(read_process_identity "${CHILD_PID[$label]}" \
            2>/dev/null || true)
        if [[ -n "$observed" ]]; then
            local state
            local starttime
            read -r state starttime <<< "$observed"
            CHILD_STARTTIME[$label]=$starttime
            break
        fi
        kill -0 "${CHILD_PID[$label]}" 2>/dev/null || break
        sleep 0.1
    done
    if [[ -z "${CHILD_STARTTIME[$label]}" ]]; then
        wait "${CHILD_PID[$label]}" 2>/dev/null || true
        return 125
    fi
    for _ in $(seq 1 40); do
        if [[ -s "$identity" ]] && attest_child "$label"; then
            return 0
        fi
        child_is_live "$label" || break
        sleep 0.25
    done
    return 125
}

stop_child() {
    local label=$1
    local rc=0
    [[ "${CHILD_DONE[$label]:-1}" == 0 ]] || return 0
    if child_is_live "$label"; then
        if [[ -n "${CHILD_PGID[$label]:-}" \
                && -n "${CHILD_TOKEN[$label]:-}" ]]; then
            bi100_stop_process_group \
                "${CHILD_PGID[$label]}" "${CHILD_PID[$label]}" 60 20 \
                "${CHILD_STARTTIME[$label]}" "${CHILD_TOKEN[$label]}" \
                || rc=$?
        else
            kill -TERM "${CHILD_PID[$label]}" 2>/dev/null || true
            for _ in $(seq 1 60); do
                child_is_live "$label" || break
                sleep 1
            done
            if child_is_live "$label"; then
                kill -KILL "${CHILD_PID[$label]}" 2>/dev/null || true
                for _ in $(seq 1 20); do
                    child_is_live "$label" || break
                    sleep 1
                done
            fi
            child_is_live "$label" && rc=1
        fi
    fi
    wait "${CHILD_PID[$label]}" 2>/dev/null || true
    CHILD_DONE[$label]=1
    return "$rc"
}

stop_all_children() {
    local rc=0
    local label
    for label in prefix wmma; do
        if [[ "${CHILD_DONE[$label]:-1}" == 0 ]]; then
            stop_child "$label" || rc=1
        fi
    done
    return "$rc"
}

wait_for_children() {
    local deadline=$((SECONDS + EXPERIMENT_TIMEOUT_S))
    local remaining=2
    local label
    while ((remaining > 0)); do
        for label in prefix wmma; do
            [[ "${CHILD_DONE[$label]}" == 0 ]] || continue
            if ! child_is_live "$label"; then
                set +e
                wait "${CHILD_PID[$label]}"
                local rc=$?
                set -e
                printf '%s\n' "$rc" > "$RUN_ROOT/${label}.rc"
                CHILD_DONE[$label]=1
                remaining=$((remaining - 1))
            fi
        done
        ((remaining == 0)) && break
        if ((SECONDS >= deadline)); then
            for label in prefix wmma; do
                [[ "${CHILD_DONE[$label]}" == 0 ]] || continue
                stop_child "$label" || true
                printf '%s\n' 124 > "$RUN_ROOT/${label}.rc"
                remaining=$((remaining - 1))
            done
            return 124
        fi
        sleep 2
    done
}

recover_recorded_children() {
    local identities=()
    local label
    for label in prefix wmma; do
        if [[ -f "$RUN_ROOT/${label}_identity.json" ]]; then
            identities+=(--identity "$RUN_ROOT/${label}_identity.json")
        fi
    done
    python3 "$ROOT/scripts/cleanup_recorded_bi100_sessions.py" \
        "${identities[@]}" --out "$RUN_ROOT/recovery.json" \
        > "$RUN_ROOT/recovery.stdout" 2> "$RUN_ROOT/recovery.stderr"
}

qualify_recovery() {
    python3 "$ROOT/tests/qualify_recorded_session_cleanup.py" \
        "$RUN_ROOT/recovery.json" \
        --expected-identity "$RUN_ROOT/prefix_identity.json" \
        --expected-identity "$RUN_ROOT/wmma_identity.json" \
        --out "$RUN_ROOT/recovery_clean.json" \
        > "$RUN_ROOT/recovery_clean.stdout" \
        2> "$RUN_ROOT/recovery_clean.stderr"
}

compare_preflights() {
    python3 "$ROOT/tests/compare_bi100_preflights.py" \
        --preflight "before=$RUN_ROOT/preflight_before.json" \
        --preflight "after=$RUN_ROOT/preflight_after.json" \
        --expected-gpus "$GPU_SET" \
        --max-free-memory-drop-bytes 1073741824 \
        --out "$RUN_ROOT/preflight_comparison.json" \
        > "$RUN_ROOT/preflight_comparison.stdout" \
        2> "$RUN_ROOT/preflight_comparison.stderr"
}

scan_fatal_logs() {
    local file
    local found=0
    : > "$RUN_ROOT/fatal_scan.txt"
    while IFS= read -r -d '' file; do
        if grep -Eiq "$FATAL_PATTERN" "$file" 2>/dev/null; then
            printf 'file=%s\n' "$file" >> "$RUN_ROOT/fatal_scan.txt"
            grep -Ein "$FATAL_PATTERN" "$file" \
                >> "$RUN_ROOT/fatal_scan.txt" 2>/dev/null || true
            found=1
        fi
    done < <(find "$RUN_ROOT" -type f \
        \( -name '*.log' -o -name '*.stdout' -o -name '*.stderr' \) \
        -print0)
    return "$found"
}

scan_timeout_rcs() {
    local file
    local value
    local found=0
    : > "$RUN_ROOT/timeout_scan.txt"
    while IFS= read -r -d '' file; do
        value=$(tr -d '[:space:]' < "$file")
        if [[ ! "$value" =~ ^[0-9]+$ ]]; then
            printf '%s=malformed:%s\n' "$file" "$value" \
                >> "$RUN_ROOT/timeout_scan.txt"
            found=1
            continue
        fi
        case "$value" in
            124|137|143)
                printf '%s=%s\n' "$file" "$value" \
                    >> "$RUN_ROOT/timeout_scan.txt"
                found=1
                ;;
        esac
    done < <(find "$RUN_ROOT" -type f -name '*.rc' -print0)
    return "$found"
}

finish() {
    local primary_rc=$?
    local final_rc=$primary_rc
    local cleanup_rc=0
    local recovery_rc=0
    local recovery_clean_rc=0
    local postflight_rc=0
    local preflight_rc=0
    local comparison_rc=0
    local fatal_rc=0
    local timeout_rc=0
    local status_rc=0
    if [[ "$FINALIZED" == 1 ]]; then
        exit "$primary_rc"
    fi
    trap - EXIT
    trap '' TERM INT
    set +e

    stop_all_children
    cleanup_rc=$?
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/child_cleanup.rc"
    recover_recorded_children
    recovery_rc=$?
    printf '%s\n' "$recovery_rc" > "$RUN_ROOT/recovery.rc"
    qualify_recovery
    recovery_clean_rc=$?
    printf '%s\n' "$recovery_clean_rc" > "$RUN_ROOT/recovery_clean.rc"
    run_postflight "$RUN_ROOT/postflight"
    postflight_rc=$?
    printf '%s\n' "$postflight_rc" > "$RUN_ROOT/postflight.rc"
    run_preflight "$RUN_ROOT/preflight_after"
    preflight_rc=$?
    printf '%s\n' "$preflight_rc" > "$RUN_ROOT/preflight_after.rc"
    compare_preflights
    comparison_rc=$?
    printf '%s\n' "$comparison_rc" > "$RUN_ROOT/preflight_comparison.rc"
    scan_fatal_logs
    fatal_rc=$?
    printf '%s\n' "$fatal_rc" > "$RUN_ROOT/fatal_scan.rc"
    scan_timeout_rcs
    timeout_rc=$?
    printf '%s\n' "$timeout_rc" > "$RUN_ROOT/timeout_scan.rc"

    if [[ $cleanup_rc -ne 0 || $recovery_rc -ne 0 \
            || $recovery_clean_rc -ne 0 || $postflight_rc -ne 0 \
            || $preflight_rc -ne 0 || $comparison_rc -ne 0 \
            || $fatal_rc -ne 0 || $timeout_rc -ne 0 ]]; then
        final_rc=1
    fi
    python3 "$ROOT/tests/qualify_m1_103_legacy_oracle_queue.py" \
        --root "$RUN_ROOT" \
        --expected-source-revision "$SOURCE_REVISION" \
        --expected-prefix-gpu "$PREFIX_GPU" \
        --expected-wmma-gpu "$WMMA_GPU" \
        --runner-returncode "$final_rc" \
        --out "$RUN_ROOT/status.json" \
        > "$RUN_ROOT/status.stdout" 2> "$RUN_ROOT/status.stderr"
    status_rc=$?
    printf '%s\n' "$status_rc" > "$RUN_ROOT/status.rc"
    if [[ $status_rc -ne 0 ]]; then
        final_rc=1
    fi
    FINALIZED=1
    exit "$final_rc"
}
trap 'exit 143' TERM
trap 'exit 130' INT
trap finish EXIT

CURRENT_STAGE=postflight_before
write_stage
set +e
run_postflight "$RUN_ROOT/postflight_before"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/postflight_before.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=preflight_before
write_stage
set +e
run_preflight "$RUN_ROOT/preflight_before"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/preflight_before.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=build_wmma
write_stage
set +e
timeout --signal=TERM --kill-after=60s 600s \
    env LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
    bash "$ROOT/tests/build_corex_attention_wmma_qk_probe.sh" \
    "$RUN_ROOT/wmma_build" \
    > "$RUN_ROOT/wmma_build.stdout" \
    2> "$RUN_ROOT/wmma_build.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/wmma_build.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=parallel_oracles
write_stage
start_child prefix "$PREFIX_GPU" \
    python3 "$ROOT/tests/bench_prefix_cold_chunk_high_precision.py" \
    --device cuda:0 --out "$RUN_ROOT/prefix/report.json"
start_child wmma "$WMMA_GPU" \
    python3 "$ROOT/tests/bench_m1_101_wmma_qk_high_precision.py" \
    --extension "$RUN_ROOT/wmma_build/corex_attention_wmma_qk_probe.so" \
    --device cuda:0 --out "$RUN_ROOT/wmma/report.json"
wait_for_children

CURRENT_STAGE=completed
write_stage
exit 0
