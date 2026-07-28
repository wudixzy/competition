#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
COREX_LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
COREX_PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

if [[ ${1:-} == "--cell-pair" ]]; then
    if [[ $# -ne 10 ]]; then
        echo "invalid internal M1-109 cell-pair invocation" >&2
        exit 2
    fi
    INSTANCE=$2
    GPU=$3
    CASE_NAME=$4
    OLD_EXTENSION=$5
    OLD_SHA=$6
    NEW_EXTENSION=$7
    NEW_SHA=$8
    RUN_ROOT=$9
    SOURCE_REVISION=${10}

    run_internal_cell() {
        local label=$1
        local extension=$2
        local digest=$3
        timeout --foreground --signal=TERM --kill-after=60s 3600s \
            env CUDA_VISIBLE_DEVICES="$GPU" \
            PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
            LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
            python3 "$ROOT/tests/bench_m1_55_production_prefill.py" \
            --case "$CASE_NAME" \
            --extension "$extension" \
            --expected-extension-sha256 "$digest" \
            --source-commit "$SOURCE_REVISION" \
            --runtime-identity "corex-3.2.3-$label" \
            --instance "$INSTANCE" \
            --visible-physical-gpu "$GPU" \
            --output "$RUN_ROOT/${CASE_NAME}_${label}.json"
    }

    if (( GPU % 2 == 0 )); then
        run_internal_cell old "$OLD_EXTENSION" "$OLD_SHA"
        run_internal_cell new "$NEW_EXTENSION" "$NEW_SHA"
    else
        run_internal_cell new "$NEW_EXTENSION" "$NEW_SHA"
        run_internal_cell old "$OLD_EXTENSION" "$OLD_SHA"
    fi
    exit 0
fi

if [[ $# -ne 4 ]]; then
    echo "usage: $0 INSTANCE OLD_EXTENSION NEW_EXTENSION RUN_ROOT" >&2
    exit 2
fi

INSTANCE=$1
OLD_EXTENSION=$(realpath "$2")
NEW_EXTENSION=$(realpath "$3")
RUN_ROOT=$(python3 - "$4" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)

if [[ "$RUN_ROOT" != /tmp/* ]]; then
    echo "M1-109 output must use a private /tmp path" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
    echo "M1-109 output already exists: $RUN_ROOT" >&2
    exit 2
fi
if [[ ! -s "$OLD_EXTENSION" || ! -s "$NEW_EXTENSION" ]]; then
    echo "both extension artifacts must exist and be non-empty" >&2
    exit 2
fi

OLD_SHA=$(sha256sum "$OLD_EXTENSION" | cut -d' ' -f1)
NEW_SHA=$(sha256sum "$NEW_EXTENSION" | cut -d' ' -f1)
if [[ "$OLD_SHA" == "$NEW_SHA" ]]; then
    echo "old and new extension artifacts are identical" >&2
    exit 2
fi

SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
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
printf '%s\n' "$OLD_SHA" > "$RUN_ROOT/old_extension_sha256.txt"
printf '%s\n' "$NEW_SHA" > "$RUN_ROOT/new_extension_sha256.txt"

finish() {
    local primary_rc=$?
    local cleanup_rc=0
    local fatal_rc=0
    local postflight_rc=0
    local preflight_rc=0
    local comparison_rc=0
    local artifact
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

    timeout --signal=TERM --kill-after=70s 240s \
        env PYTHONPATH="$ROOT/tests" \
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
        printf '%s\n' "$comparison_rc" \
            > "$RUN_ROOT/preflight_comparison.rc"
    fi
    if [[ $fatal_rc -ne 0 || $postflight_rc -ne 0 \
            || $preflight_rc -ne 0 || $comparison_rc -ne 0 ]]; then
        primary_rc=1
    fi
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

for gpu in 0 1 2 3; do
    setsid "$0" --cell-pair \
        "$INSTANCE" "$gpu" "${CASES[$gpu]}" \
        "$OLD_EXTENSION" "$OLD_SHA" "$NEW_EXTENSION" "$NEW_SHA" \
        "$RUN_ROOT" "$SOURCE_REVISION" \
        > "$RUN_ROOT/${CASES[$gpu]}.stdout" \
        2> "$RUN_ROOT/${CASES[$gpu]}.stderr" &
    PIDS+=("$!")
done

rc=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || rc=1
done
PIDS=()
if [[ $rc -ne 0 ]]; then
    echo "one or more M1-109 component pairs failed" >&2
    exit "$rc"
fi

python3 - "$RUN_ROOT" "$OLD_SHA" "$NEW_SHA" "${CASES[@]}" <<'PY'
import json
import math
import statistics
from pathlib import Path
import sys

root = Path(sys.argv[1])
old_sha, new_sha = sys.argv[2:4]
cases = sys.argv[4:]
rows = []
reasons = []
speedups = []

for case_name in cases:
    old = json.loads(
        (root / f"{case_name}_old.json").read_text(encoding="utf-8"))
    new = json.loads(
        (root / f"{case_name}_new.json").read_text(encoding="utf-8"))
    if old.get("case") != case_name or new.get("case") != case_name:
        reasons.append(f"{case_name}: case identity mismatch")
        continue
    if old.get("extension", {}).get("sha256") != old_sha:
        reasons.append(f"{case_name}: old extension identity mismatch")
    if new.get("extension", {}).get("sha256") != new_sha:
        reasons.append(f"{case_name}: new extension identity mismatch")
    numerical = new.get("numerical", {})
    finite = numerical.get("finite") is True
    output_l2 = numerical.get("output_relative_l2")
    lse_l2 = numerical.get("lse_relative_l2")
    max_abs = numerical.get("output_max_abs")
    if not (
        finite
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in (output_l2, lse_l2, max_abs)
        )
        and output_l2 <= 1e-5
        and lse_l2 <= 1e-5
        and max_abs <= 1e-3
    ):
        reasons.append(f"{case_name}: candidate numerical gate failed")
    old_ms = old.get("timings", {}).get("candidate", {}).get("cuda_median_ms")
    new_ms = new.get("timings", {}).get("candidate", {}).get("cuda_median_ms")
    if not (
        isinstance(old_ms, (int, float))
        and isinstance(new_ms, (int, float))
        and old_ms > 0
        and new_ms > 0
        and math.isfinite(float(old_ms))
        and math.isfinite(float(new_ms))
    ):
        reasons.append(f"{case_name}: invalid extension timing")
        continue
    speedup = old_ms / new_ms
    speedups.append(speedup)
    rows.append({
        "case": case_name,
        "old_extension_ms": old_ms,
        "new_extension_ms": new_ms,
        "old_over_new_speedup": speedup,
        "new_output_relative_l2": output_l2,
        "new_lse_relative_l2": lse_l2,
        "new_output_max_abs": max_abs,
    })

if len(speedups) != len(cases):
    reasons.append("not all fixed production cases produced timings")
else:
    median_speedup = statistics.median(speedups)
    positive_cases = sum(value > 1.0 for value in speedups)
    if median_speedup < 1.10:
        reasons.append(
            f"median old/new speedup {median_speedup:.6f} is below 1.10x")
    if positive_cases < 3:
        reasons.append("new extension must improve at least three cases")
    if min(speedups) < 1.0 / 1.02:
        reasons.append("a production case regressed by more than 2%")

report = {
    "schema": "bi100-m1-109-fused-softmax-component-ab-v1",
    "qualified": not reasons,
    "thresholds": {
        "maximum_relative_l2": 1e-5,
        "maximum_output_abs": 1e-3,
        "minimum_median_old_over_new_speedup": 1.10,
        "minimum_positive_cases": 3,
        "maximum_single_case_regression": 0.02,
    },
    "old_extension_sha256": old_sha,
    "new_extension_sha256": new_sha,
    "rows": rows,
    "median_old_over_new_speedup": (
        statistics.median(speedups) if speedups else None),
    "positive_cases": sum(value > 1.0 for value in speedups),
    "reasons": reasons,
    "decision": {
        "tp4_service_experiment_authorized": not reasons,
        "main_or_yaml_change_authorized": False,
        "official_score_claim_authorized": False,
    },
}
(root / "comparison.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    "qualified": report["qualified"],
    "median_old_over_new_speedup": report[
        "median_old_over_new_speedup"],
    "positive_cases": report["positive_cases"],
    "reasons": reasons,
}, sort_keys=True))
raise SystemExit(0 if report["qualified"] else 1)
PY
