#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
COREX_LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
COREX_PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

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

if [[ "$RUN_ROOT" != /tmp/* ]]; then
    echo "M1-130 output must use a private /tmp path" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
    echo "M1-130 output already exists: $RUN_ROOT" >&2
    exit 2
fi
if [[ -n $(git -C "$ROOT" status --porcelain) ]]; then
    echo "M1-130 source worktree must be clean" >&2
    exit 2
fi

SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
CASES=(q8176 q5616)
GPUS=(0 1)
PIDS=()
STARTTIMES=()
IDENTITIES=()
BEFORE_PREFLIGHT_PASSED=0

mkdir -p "$RUN_ROOT"
printf '%s\n' "$SOURCE_REVISION" > "$RUN_ROOT/source_revision.txt"

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
        python3 "$ROOT/tests/bench_m1_130_cublas_concurrency.py" \
        --case "$case_name" \
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
printf '%s\n' "$run_rc" > "$RUN_ROOT/cells_execution.rc"

python3 - "$RUN_ROOT" "$SOURCE_REVISION" "$INSTANCE" <<'PY'
import json
import math
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
source_revision, instance = sys.argv[2:4]
expected = (("q8176", 0), ("q5616", 1))
EXPECTED_QUERY_TOKENS = {"q8176": 8176, "q5616": 5616}
MIN_MEDIAN_SPEEDUP = 1.10
MIN_CELL_SPEEDUP = 1.05
RELATIVE_L2_LIMIT = 1e-7
MAX_ABS_LIMIT = 1e-5
WARMUPS = 5
TRIALS = 20
rows = []
reasons = []
speedups = []


def valid_number(value, *, positive=False):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return False
    return float(value) > 0 if positive else float(value) >= 0


for case_name, gpu in expected:
    rc_path = root / f"{case_name}.rc"
    try:
        execution_rc = int(rc_path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        execution_rc = None
        reasons.append(f"{case_name}: execution return code is unavailable")
    if execution_rc != 0:
        reasons.append(f"{case_name}: execution return code is {execution_rc}")

    report_path = root / f"{case_name}.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        reasons.append(
            f"{case_name}: report is unavailable or invalid "
            f"({type(error).__name__})")
        rows.append({
            "case": case_name,
            "gpu": gpu,
            "execution_rc": execution_rc,
            "cell_qualified": False,
            "speedup": None,
        })
        continue
    if report.get("source_revision") != source_revision:
        reasons.append(f"{case_name}: source revision differs")
    if report.get("instance") != instance:
        reasons.append(f"{case_name}: instance differs")
    if report.get("case") != case_name:
        reasons.append(f"{case_name}: case identity differs")
    if report.get("visible_physical_gpu") != gpu:
        reasons.append(f"{case_name}: physical GPU identity differs")
    if report.get("schema") != "bi100-m1-130-cublas-concurrency-cell-v1":
        reasons.append(f"{case_name}: schema differs")
    expected_shape = {
        "heads": 4,
        "query_tokens": EXPECTED_QUERY_TOKENS[case_name],
        "key_tokens": 512,
        "head_dim": 256,
        "dtype": "float32",
    }
    if report.get("shape") != expected_shape:
        reasons.append(f"{case_name}: fixed shape or dtype differs")
    if report.get("seed") != 20260729:
        reasons.append(f"{case_name}: seed differs")
    thresholds = report.get("thresholds") or {}
    if thresholds != {
        "relative_l2": RELATIVE_L2_LIMIT,
        "max_abs": MAX_ABS_LIMIT,
        "minimum_cell_speedup": MIN_CELL_SPEEDUP,
    }:
        reasons.append(f"{case_name}: threshold contract differs")
    numerical = report.get("numerical") or {}
    if numerical.get("finite") is not True:
        reasons.append(f"{case_name}: non-finite output")
    for label, key in (
        ("qk", "qk_concurrent_vs_sequential"),
        ("pv", "pv_concurrent_vs_sequential"),
    ):
        metrics = numerical.get(key) or {}
        relative_l2 = metrics.get("relative_l2")
        max_abs = metrics.get("max_abs")
        if (
            not valid_number(relative_l2)
            or float(relative_l2) > RELATIVE_L2_LIMIT
        ):
            reasons.append(f"{case_name}: {label} relative-L2 gate failed")
        if (
            not valid_number(max_abs)
            or float(max_abs) > MAX_ABS_LIMIT
        ):
            reasons.append(
                f"{case_name}: {label} maximum-absolute-error gate failed")
    if report.get("qualified") is not True:
        reasons.extend(
            f"{case_name}: {reason}"
            for reason in (report.get("reasons") or ["cell gate failed"])
        )
    elif report.get("reasons") != []:
        reasons.append(f"{case_name}: qualified report contains reasons")
    decision = report.get("decision") or {}
    if (
        decision.get("double_buffer_pipeline_authorized") is not False
        or decision.get("tp4_service_authorized") is not False
        or decision.get("main_or_yaml_change_authorized") is not False
    ):
        reasons.append(f"{case_name}: decision contract differs")
    privacy = report.get("privacy") or {}
    if privacy != {
        "contains_raw_tensors": False,
        "contains_model_outputs": False,
        "contains_credentials": False,
    }:
        reasons.append(f"{case_name}: privacy contract differs")
    timing = report.get("timing") or {}
    if timing.get("warmups") != WARMUPS or timing.get("trials") != TRIALS:
        reasons.append(f"{case_name}: timing trial contract differs")
    for timing_name in ("qk_only", "pv_only", "sequential", "concurrent"):
        summary = timing.get(timing_name) or {}
        for field in (
            "minimum_ms", "p10_ms", "median_ms", "p90_ms", "maximum_ms",
        ):
            if not valid_number(summary.get(field), positive=True):
                reasons.append(
                    f"{case_name}: {timing_name}.{field} is invalid")
    sequential_ms = (
        (timing.get("sequential") or {}).get("median_ms"))
    concurrent_ms = (
        (timing.get("concurrent") or {}).get("median_ms"))
    reported_speedup = timing.get("sequential_over_concurrent_speedup")
    if not (
        valid_number(sequential_ms, positive=True)
        and valid_number(concurrent_ms, positive=True)
        and valid_number(reported_speedup, positive=True)
    ):
        reasons.append(f"{case_name}: speedup is invalid")
        speedup = 0.0
    else:
        speedup = float(sequential_ms) / float(concurrent_ms)
        if not math.isclose(
            speedup,
            float(reported_speedup),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            reasons.append(f"{case_name}: reported speedup is inconsistent")
    speedups.append(speedup)
    if speedup < MIN_CELL_SPEEDUP:
        reasons.append(
            f"{case_name}: speedup {speedup:.6f} is below "
            f"{MIN_CELL_SPEEDUP:.2f}x")
    rows.append({
        "case": case_name,
        "gpu": gpu,
        "execution_rc": execution_rc,
        "cell_qualified": report.get("qualified"),
        "speedup": speedup,
        "qk_only_ms": (
            ((report.get("timing") or {}).get("qk_only") or {})
            .get("median_ms")
        ),
        "pv_only_ms": (
            ((report.get("timing") or {}).get("pv_only") or {})
            .get("median_ms")
        ),
        "sequential_ms": (
            ((report.get("timing") or {}).get("sequential") or {})
            .get("median_ms")
        ),
        "concurrent_ms": (
            ((report.get("timing") or {}).get("concurrent") or {})
            .get("median_ms")
        ),
        "overlap_efficiency": (
            (report.get("timing") or {}).get("overlap_efficiency")
        ),
    })

median_speedup = statistics.median(speedups) if speedups else 0.0
if median_speedup < MIN_MEDIAN_SPEEDUP:
    reasons.append(
        f"median speedup {median_speedup:.6f} is below "
        f"{MIN_MEDIAN_SPEEDUP:.2f}x")
qualified = (
    not reasons
    and len(rows) == len(expected)
    and len(speedups) == len(expected)
)
comparison = {
    "schema": "bi100-m1-130-cublas-concurrency-comparison-v1",
    "version": 1,
    "qualified": qualified,
    "source_revision": source_revision,
    "instance": instance,
    "rows": rows,
    "aggregate": {
        "median_speedup": median_speedup,
        "minimum_speedup": min(speedups),
    },
    "thresholds": {
        "minimum_median_speedup": MIN_MEDIAN_SPEEDUP,
        "minimum_cell_speedup": MIN_CELL_SPEEDUP,
    },
    "reasons": reasons,
    "decision": {
        "double_buffer_pipeline_authorized": qualified,
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
    "median_speedup": median_speedup,
    "rows": rows,
    "reasons": reasons,
}, sort_keys=True))
raise SystemExit(0 if qualified else 1)
PY
