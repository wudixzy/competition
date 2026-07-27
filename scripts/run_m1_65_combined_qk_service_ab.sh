#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)

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
        "orchestrator_postflight": read_rc(
            root / "orchestrator_postflight.rc"),
    },
    "production_promotion_authorized": False,
}
(root / "runner_status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8")
PY
}

stop_active_child() {
    local waited=0
    if [[ -z "$ACTIVE_CHILD_PID" || -z "$ACTIVE_CHILD_PGID" ]]; then
        return 0
    fi
    if kill -0 "$ACTIVE_CHILD_PID" 2>/dev/null; then
        kill -TERM -- "-$ACTIVE_CHILD_PGID" 2>/dev/null || true
        while kill -0 "$ACTIVE_CHILD_PID" 2>/dev/null && ((waited < 120)); do
            sleep 1
            waited=$((waited + 1))
        done
        if kill -0 "$ACTIVE_CHILD_PID" 2>/dev/null; then
            kill -KILL -- "-$ACTIVE_CHILD_PGID" 2>/dev/null || true
        fi
    fi
    wait "$ACTIVE_CHILD_PID" 2>/dev/null || true
    ACTIVE_CHILD_PID=""
    ACTIVE_CHILD_PGID=""
}

finish() {
    local rc=$?
    local postflight_rc=0
    trap - EXIT TERM INT
    set +e
    stop_active_child
    python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus 0,1,2,3 \
        --out "$RUN_ROOT/orchestrator_postflight.json" \
        > "$RUN_ROOT/orchestrator_postflight.stdout" \
        2> "$RUN_ROOT/orchestrator_postflight.stderr"
    postflight_rc=$?
    printf '%s\n' "$postflight_rc" \
        > "$RUN_ROOT/orchestrator_postflight.rc"
    if [[ $postflight_rc -ne 0 ]]; then
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
