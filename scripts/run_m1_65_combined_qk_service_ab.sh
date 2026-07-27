#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

if [[ $# -ne 2 ]]; then
    echo "usage: $0 INSTANCE RUN_ROOT" >&2
    exit 2
fi

INSTANCE=$1
RUN_ROOT=$(python3 - "$2" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)
if [[ "$RUN_ROOT/" == "$ROOT/"* ]]; then
    echo "A/B output must stay outside the source repository" >&2
    exit 2
fi
if [[ "$RUN_ROOT" != /tmp/* ]]; then
    echo "A/B output must use a private /tmp path" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "A/B output already exists: $RUN_ROOT" >&2
    exit 2
fi
if [[ -z "${BI100_RUNTIME_SITE_PACKAGES:-}" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/vllm" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/transformers" ]]; then
    echo "A/B requires one immutable runtime overlay" >&2
    exit 3
fi
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
        . ':(exclude)bench_runs/**')" ]]; then
    echo "A/B refuses a dirty source tree" >&2
    exit 3
fi

mkdir -p "$RUN_ROOT"
git -C "$ROOT" rev-parse HEAD > "$RUN_ROOT/source_revision.txt"
git -C "$ROOT" branch --show-current > "$RUN_ROOT/source_branch.txt"
ACTIVE_CHILD_PID=""
ACTIVE_CHILD_PGID=""
CHILD_TERM_GRACE_S=900
CHILD_KILL_GRACE_S=30

write_status() {
    local final_rc=$1
    python3 - "$RUN_ROOT" "$INSTANCE" "$final_rc" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def read_rc(path):
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return int(value) if value.isdigit() else None

report = {
    "schema": "bi100-gdn-combined-qk-service-ab-runner-v1",
    "version": 1,
    "source_revision": (root / "source_revision.txt").read_text(
        encoding="utf-8").strip(),
    "source_branch": (root / "source_branch.txt").read_text(
        encoding="utf-8").strip(),
    "instance": sys.argv[2],
    "returncode": int(sys.argv[3]),
    "gates": {
        "control": read_rc(root / "control.rc"),
        "candidate": read_rc(root / "candidate.rc"),
        "comparison": read_rc(root / "comparison.rc"),
        "orchestrator_cleanup": read_rc(
            root / "orchestrator_cleanup.rc"),
        "orchestrator_postflight": read_rc(
            root / "orchestrator_postflight.rc"),
        "orchestrator_preflight_after": read_rc(
            root / "orchestrator_preflight_after.rc"),
        "orchestrator_fatal_scan": read_rc(
            root / "orchestrator_fatal_scan.rc"),
        "orchestrator_timeout_scan": read_rc(
            root / "orchestrator_timeout_scan.rc"),
    },
    "production_promotion_authorized": False,
}
(root / "runner_status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8")
PY
}

stop_active_child() {
    local rc=0
    if [[ -z "$ACTIVE_CHILD_PID" || -z "$ACTIVE_CHILD_PGID" ]]; then
        return 0
    fi
    # The child runs its own 60-second TP4 shutdown and full postflight. The
    # outer grace covers that path, including the bounded four-GPU recheck.
    bi100_stop_process_group \
        "$ACTIVE_CHILD_PGID" "$ACTIVE_CHILD_PID" \
        "$CHILD_TERM_GRACE_S" "$CHILD_KILL_GRACE_S" || rc=$?
    ACTIVE_CHILD_PID=""
    ACTIVE_CHILD_PGID=""
    return "$rc"
}

run_orchestrator_postflight() {
    python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus 0,1,2,3 \
        --settle-timeout-s 30 --clean-samples 3 \
        --sample-interval-s 1 \
        --out "$RUN_ROOT/orchestrator_postflight.json" \
        > "$RUN_ROOT/orchestrator_postflight.stdout" \
        2> "$RUN_ROOT/orchestrator_postflight.stderr"
}

run_orchestrator_preflight() {
    timeout --signal=TERM --kill-after=70s 480s \
        python3 "$ROOT/tests/bi100_preflight.py" \
        --gpus 0,1,2,3 --timeout-s 25 --matmul-size 1024 \
        --json-out "$RUN_ROOT/orchestrator_preflight_after.json" \
        > "$RUN_ROOT/orchestrator_preflight_after.stdout" \
        2> "$RUN_ROOT/orchestrator_preflight_after.stderr"
}

scan_orchestrator_fatal_logs() {
    local file
    local found=0
    local pattern
    pattern='CUDA error|SIGSEGV|Fatal Python error|out of memory|AssertionError|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|TimeoutError|engine iteration timed out|watchdog.*tim(e|ed) out'
    : > "$RUN_ROOT/orchestrator_fatal_scan.txt"
    while IFS= read -r -d '' file; do
        if grep -Eiq "$pattern" "$file"; then
            printf '%s\n' "file=$file" \
                >> "$RUN_ROOT/orchestrator_fatal_scan.txt"
            grep -Ein "$pattern" "$file" \
                >> "$RUN_ROOT/orchestrator_fatal_scan.txt" || true
            found=1
        fi
    done < <(find "$RUN_ROOT" -type f -name server.log -print0)
    return "$found"
}

scan_orchestrator_timeouts() {
    local file
    local found=0
    local value
    : > "$RUN_ROOT/orchestrator_timeout_scan.txt"
    while IFS= read -r -d '' file; do
        value=$(tr -d '[:space:]' < "$file")
        case "$value" in
            124|137)
                printf '%s=%s\n' "$file" "$value" \
                    >> "$RUN_ROOT/orchestrator_timeout_scan.txt"
                found=1
                ;;
        esac
    done < <(find "$RUN_ROOT" -type f \
        \( -name startup.rc -o -name quality.rc \
        -o -name agent_workload.rc \) -print0)
    return "$found"
}

finish() {
    local rc=$?
    local cleanup_rc=0
    local postflight_rc=0
    local preflight_rc=0
    local fatal_rc=0
    local timeout_rc=0
    trap - EXIT TERM INT
    set +e
    stop_active_child
    cleanup_rc=$?
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/orchestrator_cleanup.rc"
    run_orchestrator_postflight
    postflight_rc=$?
    printf '%s\n' "$postflight_rc" \
        > "$RUN_ROOT/orchestrator_postflight.rc"
    run_orchestrator_preflight
    preflight_rc=$?
    printf '%s\n' "$preflight_rc" \
        > "$RUN_ROOT/orchestrator_preflight_after.rc"
    scan_orchestrator_fatal_logs
    fatal_rc=$?
    printf '%s\n' "$fatal_rc" > "$RUN_ROOT/orchestrator_fatal_scan.rc"
    scan_orchestrator_timeouts
    timeout_rc=$?
    printf '%s\n' "$timeout_rc" > "$RUN_ROOT/orchestrator_timeout_scan.rc"
    if [[ $cleanup_rc -ne 0 || $postflight_rc -ne 0 \
            || $preflight_rc -ne 0 || $fatal_rc -ne 0 \
            || $timeout_rc -ne 0 ]]; then
        rc=1
    fi
    write_status "$rc"
    exit "$rc"
}
trap 'exit 143' TERM
trap 'exit 130' INT
trap finish EXIT

run_arm() {
    local profile=$1
    local label=$2
    local output=$3
    local observed_pgid=""
    setsid env BI100_QUALITY_KERNEL_PROFILE="$profile" \
        "$ROOT/scripts/run_quality_service_gate.sh" \
        decode fine32 direct 0 lru "$label" "$INSTANCE" "$output" &
    ACTIVE_CHILD_PID=$!
    ACTIVE_CHILD_PGID=$ACTIVE_CHILD_PID
    for _ in $(seq 1 20); do
        observed_pgid=$(ps -o pgid= -p "$ACTIVE_CHILD_PID" 2>/dev/null \
            | tr -d ' ')
        [[ -n "$observed_pgid" ]] && break
        sleep 1
    done
    if [[ -z "${observed_pgid:-}" \
            || "$observed_pgid" != "$ACTIVE_CHILD_PGID" ]]; then
        echo "A/B arm did not enter an isolated process group" >&2
        return 1
    fi
    set +e
    wait "$ACTIVE_CHILD_PID"
    local rc=$?
    set -e
    ACTIVE_CHILD_PID=""
    ACTIVE_CHILD_PGID=""
    return "$rc"
}

set +e
run_arm strict-reference m1-65-control "$RUN_ROOT/control"
control_rc=$?
set -e
printf '%s\n' "$control_rc" > "$RUN_ROOT/control.rc"
[[ $control_rc -eq 0 ]]

set +e
run_arm strict-reference-combined-qk m1-65-candidate \
    "$RUN_ROOT/candidate"
candidate_rc=$?
set -e
printf '%s\n' "$candidate_rc" > "$RUN_ROOT/candidate.rc"
[[ $candidate_rc -eq 0 ]]

set +e
python3 "$ROOT/tests/compare_gdn_combined_qk_service_ab.py" \
    "$RUN_ROOT/control/quality_report.json" \
    "$RUN_ROOT/candidate/quality_report.json" \
    --out "$RUN_ROOT/comparison.json" \
    > "$RUN_ROOT/comparison.stdout" \
    2> "$RUN_ROOT/comparison.stderr"
comparison_rc=$?
set -e
printf '%s\n' "$comparison_rc" > "$RUN_ROOT/comparison.rc"
[[ $comparison_rc -eq 0 ]]
