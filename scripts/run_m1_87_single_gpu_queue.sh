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
GPU_INDEX=${GPU_INDEX:-0}
DIAGNOSTIC_PORT=${DIAGNOSTIC_PORT:-8040}
MULTI_IMAGE_PORT=${MULTI_IMAGE_PORT:-8050}
STARTUP_TIMEOUT_S=${STARTUP_TIMEOUT_S:-900}
MODEL_PATH=${MODEL_PATH:-/root/shared-storage/models/Qwen/Qwen3.6-35B-A3B-diagnostic-4L-real}
SOURCE_MODEL_PATH=${SOURCE_MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
BI100_RUNTIME_SITE_PACKAGES=${BI100_RUNTIME_SITE_PACKAGES:-}
ACTIVE_CHILD_PID=""
ACTIVE_CHILD_PGID=""
ACTIVE_CHILD_STARTTIME=""
ACTIVE_CHILD_SESSION_TOKEN=""
ACTIVE_CHILD_IDENTITY=""
CURRENT_STAGE=argument_validation
FINALIZED=0
FATAL_PATTERN='CUDA error|SIGSEGV|Fatal Python error|out of memory|device-side assert|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|Timeout(Error|Expired)|engine iteration timed out|watchdog.*tim(e|ed) out|scheduler requested a missing GDN prefix state|non-finite GatedDeltaNet'

if [[ ! "$INSTANCE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    echo "INSTANCE must be a short non-sensitive label" >&2
    exit 2
fi
if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
    echo "GPU_INDEX must be a non-negative integer" >&2
    exit 2
fi
for port in "$DIAGNOSTIC_PORT" "$MULTI_IMAGE_PORT"; do
    if [[ ! "$port" =~ ^[0-9]+$ || "$port" -lt 1 || "$port" -gt 65535 ]]; then
        echo "queue ports must be between 1 and 65535" >&2
        exit 2
    fi
done
if [[ "$DIAGNOSTIC_PORT" == "$MULTI_IMAGE_PORT" ]]; then
    echo "diagnostic and multi-image ports must differ" >&2
    exit 2
fi
if [[ ! "$STARTUP_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]]; then
    echo "STARTUP_TIMEOUT_S must be a positive integer" >&2
    exit 2
fi
if [[ -z "$BI100_RUNTIME_SITE_PACKAGES" ]]; then
    echo "BI100_RUNTIME_SITE_PACKAGES is required" >&2
    exit 3
fi

RUN_ROOT=$(python3 - "$RUN_ROOT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)
case "$RUN_ROOT/" in
    "$ROOT/"*)
        echo "RUN_ROOT must stay outside the source repository" >&2
        exit 3
        ;;
esac
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
    echo "M1-87 queue refuses a dirty source tree" >&2
    exit 3
fi
if pgrep -f 'vllm\.entrypoints\.openai\.api_server' >/dev/null 2>&1; then
    echo "an API server is already running" >&2
    exit 3
fi

MODEL_PATH=$(python3 - "$MODEL_PATH" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=True))
PY
)
SOURCE_MODEL_PATH=$(python3 - "$SOURCE_MODEL_PATH" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=True))
PY
)
BI100_RUNTIME_SITE_PACKAGES=$(python3 - \
        "$BI100_RUNTIME_SITE_PACKAGES" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=True))
PY
)
RUNTIME_INSTALL=$(dirname "$BI100_RUNTIME_SITE_PACKAGES")/install.json
if [[ ! -f "$RUNTIME_INSTALL" ]]; then
    echo "runtime install report is missing: $RUNTIME_INSTALL" >&2
    exit 3
fi

mkdir -p "$RUN_ROOT"
SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
SOURCE_BRANCH=$(git -C "$ROOT" branch --show-current)
printf '%s\n' "$SOURCE_REVISION" > "$RUN_ROOT/source_revision.txt"
printf '%s\n' "$SOURCE_BRANCH" > "$RUN_ROOT/source_branch.txt"
printf '%s\n' "$GPU_INDEX" > "$RUN_ROOT/physical_gpu.txt"

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
COREX_LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
COREX_PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

run_postflight() {
    local output=$1
    timeout --signal=TERM --kill-after=70s 240s \
        env -u CUDA_VISIBLE_DEVICES PYTHONPATH="$ROOT/tests" \
        python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus "$GPU_INDEX" \
        --settle-timeout-s 90 --clean-samples 3 \
        --sample-interval-s 2 \
        --out "$output.json" \
        > "$output.stdout" 2> "$output.stderr"
}

run_preflight() {
    local output=$1
    timeout --signal=TERM --kill-after=90s 240s \
        env -u CUDA_VISIBLE_DEVICES \
        PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
        LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
        python3 "$ROOT/tests/bi100_preflight.py" \
        --gpus "$GPU_INDEX" --timeout-s 25 --matmul-size 1024 \
        --json-out "$output.json" \
        > "$output.stdout" 2> "$output.stderr"
}

read_process_starttime() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys

value = (Path("/proc") / sys.argv[1] / "stat").read_text(encoding="ascii")
tail = value[value.rfind(")") + 2:].split()
print(tail[19])
PY
}

active_child_is_same() {
    local observed
    [[ -n "$ACTIVE_CHILD_PID" && -n "$ACTIVE_CHILD_STARTTIME" ]] || return 1
    observed=$(read_process_starttime "$ACTIVE_CHILD_PID" 2>/dev/null) \
        || return 1
    [[ "$observed" == "$ACTIVE_CHILD_STARTTIME" ]]
}

stop_active_child() {
    local rc=0
    if [[ -z "$ACTIVE_CHILD_PID" ]]; then
        return 0
    fi
    if [[ -n "$ACTIVE_CHILD_PGID" ]]; then
        bi100_stop_process_group \
            "$ACTIVE_CHILD_PGID" "$ACTIVE_CHILD_PID" 60 20 \
            "$ACTIVE_CHILD_STARTTIME" \
            "$ACTIVE_CHILD_SESSION_TOKEN" || rc=$?
    else
        if active_child_is_same; then
            kill -TERM "$ACTIVE_CHILD_PID" 2>/dev/null || true
            for _ in $(seq 1 60); do
                active_child_is_same || break
                sleep 1
            done
        fi
        if active_child_is_same; then
            kill -KILL "$ACTIVE_CHILD_PID" 2>/dev/null || true
            for _ in $(seq 1 20); do
                active_child_is_same || break
                sleep 1
            done
        fi
    fi
    wait "$ACTIVE_CHILD_PID" 2>/dev/null || true
    if active_child_is_same; then
        echo "queue child survived scoped cleanup" >&2
        rc=1
    fi
    if [[ $rc -eq 0 ]]; then
        ACTIVE_CHILD_PID=""
        ACTIVE_CHILD_PGID=""
        ACTIVE_CHILD_STARTTIME=""
        ACTIVE_CHILD_SESSION_TOKEN=""
        ACTIVE_CHILD_IDENTITY=""
    fi
    return "$rc"
}

run_child() {
    local label=$1
    local identity="$RUN_ROOT/${label}_child_identity.json"
    local identity_ok=0
    local observed_pgid=""
    local observed_token=""
    local observed_identity=""
    shift
    set +e
    (
        exec python3 "$ROOT/scripts/exec_bi100_session.py" \
            "$identity" -- "$@"
    ) > "$RUN_ROOT/${label}_runner.stdout" \
      2> "$RUN_ROOT/${label}_runner.stderr" &
    ACTIVE_CHILD_PID=$!
    ACTIVE_CHILD_STARTTIME=""
    for _ in $(seq 1 20); do
        ACTIVE_CHILD_STARTTIME=$(
            read_process_starttime "$ACTIVE_CHILD_PID" 2>/dev/null || true)
        [[ -n "$ACTIVE_CHILD_STARTTIME" ]] && break
        kill -0 "$ACTIVE_CHILD_PID" 2>/dev/null || break
        sleep 0.1
    done
    if [[ -z "$ACTIVE_CHILD_STARTTIME" ]]; then
        wait "$ACTIVE_CHILD_PID" 2>/dev/null || true
        ACTIVE_CHILD_PID=""
        ACTIVE_CHILD_IDENTITY=""
        set -e
        return 125
    fi
    ACTIVE_CHILD_IDENTITY=$identity
    for _ in $(seq 1 20); do
        if [[ -s "$identity" ]]; then
            if observed_identity=$(python3 - "$identity" \
                    "$ACTIVE_CHILD_PID" "$ACTIVE_CHILD_STARTTIME" <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_pid = int(sys.argv[2])
expected_starttime = int(sys.argv[3])
token = value.get("session_token")
if (
    value.get("schema") != "bi100-process-session-v1"
    or value.get("version") != 1
    or value.get("pid") != expected_pid
    or value.get("pgid") != expected_pid
    or value.get("sid") != expected_pid
    or value.get("starttime_ticks") != expected_starttime
    or not isinstance(token, str)
    or len(token) != 32
    or any(character not in "0123456789abcdef" for character in token)
):
    raise SystemExit(1)
print(value["pgid"], token)
PY
            ); then
                read -r observed_pgid observed_token <<< "$observed_identity"
                if [[ "$observed_pgid" == "$ACTIVE_CHILD_PID" \
                        && "$observed_token" =~ ^[0-9a-f]{32}$ ]]; then
                    identity_ok=1
                    ACTIVE_CHILD_PGID=$observed_pgid
                    ACTIVE_CHILD_SESSION_TOKEN=$observed_token
                    break
                fi
            fi
        fi
        active_child_is_same || break
        sleep 1
    done
    if [[ "$identity_ok" != 1 ]]; then
        stop_active_child
        set -e
        return 125
    fi
    wait "$ACTIVE_CHILD_PID"
    local rc=$?
    ACTIVE_CHILD_PID=""
    ACTIVE_CHILD_PGID=""
    ACTIVE_CHILD_STARTTIME=""
    ACTIVE_CHILD_SESSION_TOKEN=""
    ACTIVE_CHILD_IDENTITY=""
    set -e
    return "$rc"
}

recover_recorded_services() {
    local identities=()
    local path
    for path in \
        "$RUN_ROOT/m1_84_child_identity.json" \
        "$RUN_ROOT/m1_86_child_identity.json" \
        "$RUN_ROOT/m1_84/process_group_identity.json" \
        "$RUN_ROOT/m1_86/control/process_group_identity.json" \
        "$RUN_ROOT/m1_86/candidate/process_group_identity.json"; do
        if [[ -f "$path" ]]; then
            identities+=(--identity "$path")
        fi
    done
    python3 "$ROOT/scripts/cleanup_recorded_bi100_sessions.py" \
        "${identities[@]}" \
        --out "$RUN_ROOT/service_recovery.json" \
        > "$RUN_ROOT/service_recovery.stdout" \
        2> "$RUN_ROOT/service_recovery.stderr"
}

run_interstage_audit() {
    local postflight_rc=0
    local preflight_rc=0
    unset CUDA_VISIBLE_DEVICES
    set +e
    run_postflight "$RUN_ROOT/interstage_postflight"
    postflight_rc=$?
    run_preflight "$RUN_ROOT/interstage_preflight"
    preflight_rc=$?
    set -e
    printf '%s\n' "$postflight_rc" > "$RUN_ROOT/interstage_postflight.rc"
    printf '%s\n' "$preflight_rc" > "$RUN_ROOT/interstage_preflight.rc"
    [[ $postflight_rc -eq 0 && $preflight_rc -eq 0 ]]
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
    local postflight_rc=0
    local preflight_rc=0
    local fatal_rc=0
    local timeout_rc=0
    local qualifier_rc=0
    if [[ "$FINALIZED" == 1 ]]; then
        exit "$primary_rc"
    fi
    trap - EXIT
    trap '' TERM INT
    set +e

    stop_active_child
    cleanup_rc=$?
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/child_cleanup.rc"
    recover_recorded_services
    recovery_rc=$?
    printf '%s\n' "$recovery_rc" > "$RUN_ROOT/service_recovery.rc"
    unset CUDA_VISIBLE_DEVICES

    run_postflight "$RUN_ROOT/final_postflight"
    postflight_rc=$?
    printf '%s\n' "$postflight_rc" > "$RUN_ROOT/final_postflight.rc"
    run_preflight "$RUN_ROOT/final_preflight"
    preflight_rc=$?
    printf '%s\n' "$preflight_rc" > "$RUN_ROOT/final_preflight.rc"

    scan_fatal_logs
    fatal_rc=$?
    printf '%s\n' "$fatal_rc" > "$RUN_ROOT/fatal_scan.rc"
    scan_timeout_rcs
    timeout_rc=$?
    printf '%s\n' "$timeout_rc" > "$RUN_ROOT/timeout_scan.rc"

    if [[ $cleanup_rc -ne 0 || $recovery_rc -ne 0 \
            || $postflight_rc -ne 0 \
            || $preflight_rc -ne 0 || $fatal_rc -ne 0 \
            || $timeout_rc -ne 0 ]]; then
        final_rc=1
    fi
    python3 "$ROOT/tests/qualify_m1_87_single_gpu_queue.py" \
        --root "$RUN_ROOT" \
        --expected-source-revision "$SOURCE_REVISION" \
        --expected-gpu "$GPU_INDEX" \
        --runner-returncode "$final_rc" \
        --out "$RUN_ROOT/queue_status.json" \
        > "$RUN_ROOT/queue_status.stdout" \
        2> "$RUN_ROOT/queue_status.stderr"
    qualifier_rc=$?
    printf '%s\n' "$qualifier_rc" > "$RUN_ROOT/queue_status.rc"
    if [[ $qualifier_rc -ne 0 ]]; then
        final_rc=1
    fi
    FINALIZED=1
    exit "$final_rc"
}
trap 'exit 143' TERM
trap 'exit 130' INT
trap finish EXIT

CURRENT_STAGE=m1_89_overlay_identity
set +e
timeout --signal=TERM --kill-after=10s 120s \
    env -u CUDA_VISIBLE_DEVICES \
    python3 "$ROOT/tests/verify_bare_host_runtime_identity.py" \
    --source-root "$ROOT" \
    --runtime-site-packages "$BI100_RUNTIME_SITE_PACKAGES" \
    --runtime-install "$RUNTIME_INSTALL" \
    --out "$RUN_ROOT/m1_89_runtime_overlay_identity.json" \
    > "$RUN_ROOT/m1_89_runtime_overlay_identity.stdout" \
    2> "$RUN_ROOT/m1_89_runtime_overlay_identity.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/m1_89_overlay_identity.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=m1_89_runtime_gate
set +e
(
    cd /tmp
    timeout --signal=TERM --kill-after=10s 120s \
        env -u CUDA_VISIBLE_DEVICES \
        PYTHONPATH="$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH" \
        LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
        python3 "$ROOT/tests/qwen36_cache_namespace_runtime_gate.py" \
        --runtime-site-packages "$BI100_RUNTIME_SITE_PACKAGES" \
        --source-revision "$SOURCE_REVISION" \
        --out "$RUN_ROOT/m1_89_cache_namespace_runtime_gate.json"
) > "$RUN_ROOT/m1_89_cache_namespace_runtime_gate.stdout" \
  2> "$RUN_ROOT/m1_89_cache_namespace_runtime_gate.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/m1_89_runtime_gate.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=m1_84
set +e
run_child m1_84 \
    env BI100_RUNTIME_SITE_PACKAGES="$BI100_RUNTIME_SITE_PACKAGES" \
    SOURCE_MODEL_PATH="$SOURCE_MODEL_PATH" \
    PORT="$DIAGNOSTIC_PORT" STARTUP_TIMEOUT_S="$STARTUP_TIMEOUT_S" \
    bash "$ROOT/scripts/run_qwen36_diagnostic_gate.sh" \
    "$MODEL_PATH" 1 "$GPU_INDEX" "$INSTANCE-m184" "$RUN_ROOT/m1_84"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/m1_84.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=interstage_audit
run_interstage_audit

CURRENT_STAGE=m1_86
set +e
run_child m1_86 \
    env BI100_RUNTIME_SITE_PACKAGES="$BI100_RUNTIME_SITE_PACKAGES" \
    MODEL_PATH="$MODEL_PATH" SOURCE_MODEL_PATH="$SOURCE_MODEL_PATH" \
    GPU_INDEX="$GPU_INDEX" PORT="$MULTI_IMAGE_PORT" \
    STARTUP_TIMEOUT_S="$STARTUP_TIMEOUT_S" \
    bash "$ROOT/scripts/run_m1_86_multi_image_ab.sh" \
    "$INSTANCE-m186" "$RUN_ROOT/m1_86"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/m1_86.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=completed
exit 0
