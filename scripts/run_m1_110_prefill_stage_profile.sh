#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
COREX_LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
COREX_PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

if [[ ${1:-} == "--cell" ]]; then
    if [[ $# -ne 10 ]]; then
        echo "invalid internal M1-110 cell invocation" >&2
        exit 2
    fi
    INSTANCE=$2
    GPU=$3
    CASE_NAME=$4
    PRODUCTION_EXTENSION=$5
    PRODUCTION_SHA=$6
    PROFILE_EXTENSION=$7
    PROFILE_SHA=$8
    RUN_ROOT=$9
    SOURCE_REVISION=${10}

    timeout --foreground --signal=TERM --kill-after=60s 3600s \
        env CUDA_VISIBLE_DEVICES="$GPU" \
        PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
        LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
        python3 "$ROOT/tests/profile_m1_110_fused_prefill_stages.py" \
        --case "$CASE_NAME" \
        --production-extension "$PRODUCTION_EXTENSION" \
        --profile-extension "$PROFILE_EXTENSION" \
        --expected-production-sha256 "$PRODUCTION_SHA" \
        --expected-profile-sha256 "$PROFILE_SHA" \
        --source-commit "$SOURCE_REVISION" \
        --instance "$INSTANCE" \
        --visible-physical-gpu "$GPU" \
        --output "$RUN_ROOT/${CASE_NAME}.json"
    exit 0
fi

if [[ $# -ne 4 ]]; then
    echo "usage: $0 INSTANCE PRODUCTION_EXTENSION PROFILE_EXTENSION RUN_ROOT" >&2
    exit 2
fi

INSTANCE=$1
PRODUCTION_EXTENSION=$(realpath "$2")
PROFILE_EXTENSION=$(realpath "$3")
RUN_ROOT=$(python3 - "$4" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)

if [[ "$RUN_ROOT" != /tmp/* ]]; then
    echo "M1-110 output must use a private /tmp path" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
    echo "M1-110 output already exists: $RUN_ROOT" >&2
    exit 2
fi
if [[ ! -s "$PRODUCTION_EXTENSION" || ! -s "$PROFILE_EXTENSION" ]]; then
    echo "both extension artifacts must exist and be non-empty" >&2
    exit 2
fi

PRODUCTION_SHA=$(sha256sum "$PRODUCTION_EXTENSION" | cut -d' ' -f1)
PROFILE_SHA=$(sha256sum "$PROFILE_EXTENSION" | cut -d' ' -f1)
if [[ "$PRODUCTION_SHA" == "$PROFILE_SHA" ]]; then
    echo "production and profile artifacts are identical" >&2
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
printf '%s\n' "$PRODUCTION_SHA" \
    > "$RUN_ROOT/production_extension_sha256.txt"
printf '%s\n' "$PROFILE_SHA" > "$RUN_ROOT/profile_extension_sha256.txt"

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
    setsid "$0" --cell \
        "$INSTANCE" "$gpu" "${CASES[$gpu]}" \
        "$PRODUCTION_EXTENSION" "$PRODUCTION_SHA" \
        "$PROFILE_EXTENSION" "$PROFILE_SHA" \
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
    echo "one or more M1-110 profile cells failed" >&2
    exit "$rc"
fi

python3 - "$RUN_ROOT" "$PRODUCTION_SHA" "$PROFILE_SHA" \
        "${CASES[@]}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
production_sha, profile_sha = sys.argv[2:4]
cases = sys.argv[4:]
rows = []
reasons = []

for case_name in cases:
    cell = json.loads(
        (root / f"{case_name}.json").read_text(encoding="utf-8"))
    if cell.get("case") != case_name:
        reasons.append(f"{case_name}: case identity mismatch")
        continue
    if cell.get("schema") != "bi100-m1-110-fused-prefill-stage-profile-v1":
        reasons.append(f"{case_name}: schema mismatch")
    if (
        cell.get("production_extension", {}).get("sha256")
        != production_sha
    ):
        reasons.append(f"{case_name}: production identity mismatch")
    if cell.get("profile_extension", {}).get("sha256") != profile_sha:
        reasons.append(f"{case_name}: profile identity mismatch")
    if cell.get("qualified") is not True or cell.get("reasons") != []:
        reasons.append(f"{case_name}: profile cell did not qualify")
    ranked = cell.get("ranked_stages")
    shares = cell.get("stage_share")
    if (
        not isinstance(ranked, list)
        or not ranked
        or not isinstance(shares, dict)
        or ranked[0] not in shares
    ):
        reasons.append(f"{case_name}: stage ranking is invalid")
        continue
    rows.append({
        "case": case_name,
        "dominant_stage": ranked[0],
        "dominant_stage_share": shares[ranked[0]],
        "stage_median_ms": cell.get("stage_median_ms"),
        "stage_share": shares,
        "production_cuda_median_ms": cell.get(
            "production_timing", {}).get("cuda_median_ms"),
        "profile_build_cuda_median_ms": cell.get(
            "profile_build_forward_timing", {}).get("cuda_median_ms"),
        "event_total_median_ms": cell.get("event_total_median_ms"),
        "representative_runtime_delta": cell.get(
            "representative_runtime_delta"),
        "event_perturbation": cell.get("event_perturbation"),
    })

if len(rows) != len(cases):
    reasons.append("not all fixed production cases produced profile rows")

report = {
    "schema": "bi100-m1-110-fused-prefill-stage-profile-matrix-v1",
    "qualified": not reasons,
    "production_extension_sha256": production_sha,
    "profile_extension_sha256": profile_sha,
    "rows": rows,
    "reasons": reasons,
    "decision": {
        "deeper_fusion_design_selection_authorized": not reasons,
        "tp4_service_experiment_authorized": False,
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
    "dominant_stages": {
        row["case"]: row["dominant_stage"] for row in rows
    },
    "reasons": reasons,
}, sort_keys=True))
raise SystemExit(0 if report["qualified"] else 1)
PY
