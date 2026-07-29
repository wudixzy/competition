#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
COREX_LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
COREX_PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

if [[ $# -ne 3 ]]; then
    echo "usage: $0 INSTANCE EXTENSION RUN_ROOT" >&2
    exit 2
fi

INSTANCE=$1
EXTENSION=$(realpath "$2")
RUN_ROOT=$(python3 - "$3" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)

if [[ "$RUN_ROOT" != /tmp/* ]]; then
    echo "M1-127 output must use a private /tmp path" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
    echo "M1-127 output already exists: $RUN_ROOT" >&2
    exit 2
fi
if [[ ! -s "$EXTENSION" ]]; then
    echo "M1-127 extension must exist and be non-empty" >&2
    exit 2
fi
if [[ -n $(git -C "$ROOT" status --porcelain) ]]; then
    echo "M1-127 source worktree must be clean" >&2
    exit 2
fi

SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
EXTENSION_SHA=$(sha256sum "$EXTENSION" | cut -d' ' -f1)
CASES=(q8176 q5616)
GPUS=(0 1)
PIDS=()
STARTTIMES=()
IDENTITIES=()
BEFORE_PREFLIGHT_PASSED=0

mkdir -p "$RUN_ROOT"
printf '%s\n' "$SOURCE_REVISION" > "$RUN_ROOT/source_revision.txt"
printf '%s\n' "$EXTENSION_SHA" > "$RUN_ROOT/extension_sha256.txt"

read_process_starttime() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys

value = (Path("/proc") / sys.argv[1] / "stat").read_text(encoding="ascii")
tail = value[value.rfind(")") + 2:].split()
print(tail[19])
PY
}

stop_session() {
    local index=$1
    local pid=${PIDS[$index]}
    local starttime=${STARTTIMES[$index]}
    local identity=${IDENTITIES[$index]}
    local observed=""
    local pgid=""
    local leader=""
    local identity_start=""
    local token=""

    if [[ -s "$identity" ]]; then
        read -r pgid leader identity_start token < <(
            python3 - "$identity" "$pid" "$starttime" <<'PY'
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
):
    raise SystemExit(1)
print(
    value["pgid"],
    value["pid"],
    value["starttime_ticks"],
    token,
)
PY
        ) || return 1
        bi100_stop_process_group \
            "$pgid" "$leader" 60 20 "$identity_start" "$token"
        return $?
    fi

    observed=$(read_process_starttime "$pid" 2>/dev/null || true)
    if [[ "$observed" != "$starttime" ]]; then
        return 0
    fi
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
        observed=$(read_process_starttime "$pid" 2>/dev/null || true)
        [[ "$observed" != "$starttime" ]] && return 0
        sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
        observed=$(read_process_starttime "$pid" 2>/dev/null || true)
        [[ "$observed" != "$starttime" ]] && return 0
        sleep 1
    done
    return 1
}

finish() {
    local primary_rc=$?
    local cleanup_rc=0
    local fatal_rc=0
    local timeout_rc=0
    local postflight_rc=0
    local preflight_rc=0
    local comparison_rc=0
    local artifact=""
    local index=0

    trap - EXIT INT TERM
    set +e
    for index in "${!PIDS[@]}"; do
        stop_session "$index" || cleanup_rc=1
        wait "${PIDS[$index]}" 2>/dev/null || true
    done
    if [[ $cleanup_rc -ne 0 ]]; then
        primary_rc=1
    fi

    : > "$RUN_ROOT/fatal_scan.txt"
    while IFS= read -r -d '' artifact; do
        if grep -Eiq \
                'CUDA error|illegal memory access|SIGSEGV|Fatal Python error|out of memory|device-side assert|CoreX.*(failed|fatal)|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|Timeout(Error|Expired)' \
                "$artifact"; then
            printf 'file=%s\n' "$artifact" >> "$RUN_ROOT/fatal_scan.txt"
            fatal_rc=1
        fi
    done < <(find "$RUN_ROOT" -type f \
        \( -name '*.stdout' -o -name '*.stderr' \) -print0)
    printf '%s\n' "$fatal_rc" > "$RUN_ROOT/fatal_scan.rc"

    : > "$RUN_ROOT/timeout_scan.txt"
    while IFS= read -r -d '' artifact; do
        if grep -Eiq \
                'timed out|Timeout(Error|Expired)|command terminated by signal' \
                "$artifact"; then
            printf 'file=%s\n' "$artifact" >> "$RUN_ROOT/timeout_scan.txt"
            timeout_rc=1
        fi
    done < <(find "$RUN_ROOT" -type f \
        \( -name '*.stdout' -o -name '*.stderr' \) -print0)
    for artifact in "$RUN_ROOT"/q*.rc; do
        [[ -f "$artifact" ]] || continue
        if grep -Eq '^(124|137|143)$' "$artifact"; then
            printf 'file=%s\n' "$artifact" >> "$RUN_ROOT/timeout_scan.txt"
            timeout_rc=1
        fi
    done
    printf '%s\n' "$timeout_rc" > "$RUN_ROOT/timeout_scan.rc"

    timeout --signal=TERM --kill-after=70s 240s \
        env PYTHONPATH="$ROOT/tests" \
        python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus 0,1,2,3 --settle-timeout-s 90 \
        --clean-samples 3 --sample-interval-s 2 \
        --out "$RUN_ROOT/postflight_after.json" \
        > "$RUN_ROOT/postflight_after.stdout" \
        2> "$RUN_ROOT/postflight_after.stderr"
    postflight_rc=$?
    printf '%s\n' "$postflight_rc" > "$RUN_ROOT/postflight_after.rc"

    if [[ "$BEFORE_PREFLIGHT_PASSED" == 1 \
            && $cleanup_rc -eq 0 && $postflight_rc -eq 0 ]]; then
        timeout --signal=TERM --kill-after=70s 480s \
            env PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
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
            comparison_rc=$?
        else
            comparison_rc=1
        fi
        printf '%s\n' "$comparison_rc" > "$RUN_ROOT/preflight_comparison.rc"
    fi

    if [[ $fatal_rc -ne 0 || $timeout_rc -ne 0 || $postflight_rc -ne 0 \
            || $preflight_rc -ne 0 || $comparison_rc -ne 0 ]]; then
        primary_rc=1
    fi
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/cleanup.rc"
    printf '%s\n' "$primary_rc" > "$RUN_ROOT/overall.rc"
    exit "$primary_rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

timeout --signal=TERM --kill-after=70s 240s \
    env PYTHONPATH="$ROOT/tests" \
    python3 "$ROOT/tests/service_postflight_gate.py" \
    --gpus 0,1,2,3 --settle-timeout-s 90 \
    --clean-samples 3 --sample-interval-s 2 \
    --out "$RUN_ROOT/postflight_before.json" \
    > "$RUN_ROOT/postflight_before.stdout" \
    2> "$RUN_ROOT/postflight_before.stderr"
printf '%s\n' 0 > "$RUN_ROOT/postflight_before.rc"

timeout --signal=TERM --kill-after=70s 480s \
    env PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
    LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
    python3 "$ROOT/tests/bi100_preflight.py" \
    --gpus 0,1,2,3 --timeout-s 25 --matmul-size 1024 \
    --json-out "$RUN_ROOT/preflight_before.json" \
    > "$RUN_ROOT/preflight_before.stdout" \
    2> "$RUN_ROOT/preflight_before.stderr"
printf '%s\n' 0 > "$RUN_ROOT/preflight_before.rc"
BEFORE_PREFLIGHT_PASSED=1

for index in "${!CASES[@]}"; do
    case_name=${CASES[$index]}
    gpu=${GPUS[$index]}
    identity="$RUN_ROOT/${case_name}_session.json"
    python3 "$ROOT/scripts/exec_bi100_session.py" "$identity" -- \
        timeout --foreground --signal=TERM --kill-after=60s 900s \
        env CUDA_VISIBLE_DEVICES="$gpu" \
        PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
        LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
        python3 "$ROOT/tests/bench_m1_127_half_input_qk.py" \
        --case "$case_name" \
        --extension "$EXTENSION" \
        --expected-extension-sha256 "$EXTENSION_SHA" \
        --source-revision "$SOURCE_REVISION" \
        --instance "$INSTANCE" \
        --visible-physical-gpu "$gpu" \
        --out "$RUN_ROOT/${case_name}.json" \
        > "$RUN_ROOT/${case_name}.stdout" \
        2> "$RUN_ROOT/${case_name}.stderr" &
    pid=$!
    PIDS+=("$pid")
    STARTTIMES+=("$(read_process_starttime "$pid")")
    IDENTITIES+=("$identity")
done

run_rc=0
for index in "${!PIDS[@]}"; do
    case_rc=0
    wait "${PIDS[$index]}" || case_rc=$?
    printf '%s\n' "$case_rc" > "$RUN_ROOT/${CASES[$index]}.rc"
    [[ $case_rc -eq 0 ]] || run_rc=1
done
PIDS=()
STARTTIMES=()
IDENTITIES=()
if [[ $run_rc -ne 0 ]]; then
    echo "one or more M1-127 capability cells failed" >&2
    exit 1
fi

python3 - "$RUN_ROOT" "$SOURCE_REVISION" "$EXTENSION_SHA" \
        "$INSTANCE" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
source_revision, extension_sha, instance = sys.argv[2:5]
expected = (("q8176", 0), ("q5616", 1))
rows = []
reasons = []
for case_name, gpu in expected:
    report = json.loads(
        (root / f"{case_name}.json").read_text(encoding="utf-8"))
    if report.get("source_revision") != source_revision:
        reasons.append(f"{case_name}: source revision differs")
    if report.get("instance") != instance:
        reasons.append(f"{case_name}: instance differs")
    if report.get("case") != case_name:
        reasons.append(f"{case_name}: case identity differs")
    if report.get("visible_physical_gpu") != gpu:
        reasons.append(f"{case_name}: physical GPU identity differs")
    if report.get("extension", {}).get("sha256") != extension_sha:
        reasons.append(f"{case_name}: extension identity differs")
    if report.get("qualified") is not True:
        reasons.append(f"{case_name}: capability gate failed")
    decision = report.get("decision") or {}
    if (
        decision.get("full_pipeline_integration_authorized") is not True
        or decision.get("tp4_service_authorized") is not False
        or decision.get("main_or_yaml_change_authorized") is not False
    ):
        reasons.append(f"{case_name}: decision contract differs")
    rows.append({
        "case": case_name,
        "gpu": gpu,
        "qualified": report.get("qualified"),
        "speedup": (report.get("timing") or {}).get(
            "control_over_candidate_speedup"),
        "reason_count": len(report.get("reasons") or []),
    })

qualified = not reasons and len(rows) == len(expected)
comparison = {
    "schema": "bi100-m1-127-half-input-qk-comparison-v1",
    "version": 1,
    "qualified": qualified,
    "source_revision": source_revision,
    "instance": instance,
    "extension_sha256": extension_sha,
    "rows": rows,
    "reasons": reasons,
    "decision": {
        "full_pipeline_integration_authorized": qualified,
        "tp4_service_authorized": False,
        "main_or_yaml_change_authorized": False,
    },
    "privacy": {
        "contains_raw_tensors": False,
        "contains_model_outputs": False,
        "contains_credentials": False,
    },
}
(root / "comparison.json").write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    "qualified": qualified,
    "rows": rows,
    "reasons": reasons,
}, sort_keys=True))
raise SystemExit(0 if qualified else 1)
PY
