#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SELF="$ROOT/scripts/run_m1_111_query_tiled_production_retest.sh"

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
COREX_LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
COREX_PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

if [[ ${1:-} == "--cell" ]]; then
    if [[ $# -ne 11 ]]; then
        echo "invalid internal M1-111 cell invocation" >&2
        exit 2
    fi
    INSTANCE=$2
    GPU=$3
    CASES_CSV=$4
    ORDER=$5
    BASELINE_EXTENSION=$6
    BASELINE_SHA=$7
    CANDIDATE_EXTENSION=$8
    CANDIDATE_SHA=$9
    RUN_ROOT=${10}
    SOURCE_REVISION=${11}
    IFS=',' read -r -a CELL_CASES <<< "$CASES_CSV"

    for case_name in "${CELL_CASES[@]}"; do
        timeout --foreground --signal=TERM --kill-after=60s 3600s \
            env CUDA_VISIBLE_DEVICES="$GPU" \
            PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
            LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
            python3 \
            "$ROOT/tests/bench_m1_111_query_tiled_production_retest.py" \
            --case "$case_name" \
            --baseline-extension "$BASELINE_EXTENSION" \
            --candidate-extension "$CANDIDATE_EXTENSION" \
            --expected-baseline-sha256 "$BASELINE_SHA" \
            --expected-candidate-sha256 "$CANDIDATE_SHA" \
            --source-commit "$SOURCE_REVISION" \
            --instance "$INSTANCE" \
            --visible-physical-gpu "$GPU" \
            --order "$ORDER" \
            --output "$RUN_ROOT/${case_name}.json"
    done
    exit 0
fi

if [[ $# -ne 4 ]]; then
    echo "usage: $0 INSTANCE BASELINE_EXTENSION CANDIDATE_EXTENSION RUN_ROOT" >&2
    exit 2
fi

INSTANCE=$1
BASELINE_EXTENSION=$(realpath "$2")
CANDIDATE_EXTENSION=$(realpath "$3")
RUN_ROOT=$(python3 - "$4" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)

if [[ "$RUN_ROOT" != /tmp/* ]]; then
    echo "M1-111 output must use a private /tmp path" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
    echo "M1-111 output already exists: $RUN_ROOT" >&2
    exit 2
fi
if [[ ! -s "$BASELINE_EXTENSION" || ! -s "$CANDIDATE_EXTENSION" ]]; then
    echo "both extension artifacts must exist and be non-empty" >&2
    exit 2
fi
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
        . ':(exclude)bench_runs/**')" ]]; then
    echo "M1-111 retest refuses a dirty source tree" >&2
    exit 2
fi

BASELINE_SHA=$(sha256sum "$BASELINE_EXTENSION" | cut -d' ' -f1)
CANDIDATE_SHA=$(sha256sum "$CANDIDATE_EXTENSION" | cut -d' ' -f1)
if [[ "$BASELINE_SHA" == "$CANDIDATE_SHA" ]]; then
    echo "baseline and candidate artifacts are identical" >&2
    exit 2
fi

SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
CASE_GROUPS=(
    production_dense_q8176,production_32k_q8176
    production_65k_q8176
    production_128k_q8176
    production_235k_q5616,boundary_262k_q8192
)
ORDERS=(
    baseline,candidate
    candidate,baseline
    baseline,candidate
    candidate,baseline
)
ALL_CASES=(
    production_dense_q8176
    production_32k_q8176
    production_65k_q8176
    production_128k_q8176
    production_235k_q5616
    boundary_262k_q8192
)
PIDS=()
IDENTITIES=(
    "$RUN_ROOT/gpu0_identity.json"
    "$RUN_ROOT/gpu1_identity.json"
    "$RUN_ROOT/gpu2_identity.json"
    "$RUN_ROOT/gpu3_identity.json"
)
BEFORE_PREFLIGHT_PASSED=0

mkdir -p "$RUN_ROOT"
printf '%s\n' "$SOURCE_REVISION" > "$RUN_ROOT/source_revision.txt"
git -C "$ROOT" branch --show-current > "$RUN_ROOT/source_branch.txt"
printf '%s\n' "$BASELINE_SHA" > "$RUN_ROOT/baseline_extension_sha256.txt"
printf '%s\n' "$CANDIDATE_SHA" \
    > "$RUN_ROOT/candidate_extension_sha256.txt"

finish() {
    local primary_rc=$?
    local cleanup_rc=0
    local recovery_rc=0
    local recovery_clean_rc=0
    local fatal_rc=0
    local timeout_rc=0
    local postflight_rc=0
    local preflight_rc=0
    local comparison_rc=0
    local artifact
    local identity_args=()
    local qualification_args=()
    local identity
    local pid
    trap - EXIT INT TERM
    set +e

    for identity in "${IDENTITIES[@]}"; do
        identity_args+=(--identity "$identity")
        qualification_args+=(--expected-identity "$identity")
    done
    python3 "$ROOT/scripts/cleanup_recorded_bi100_sessions.py" \
        "${identity_args[@]}" \
        --out "$RUN_ROOT/session_recovery.json" \
        > "$RUN_ROOT/session_recovery.stdout" \
        2> "$RUN_ROOT/session_recovery.stderr"
    recovery_rc=$?
    printf '%s\n' "$recovery_rc" > "$RUN_ROOT/session_recovery.rc"
    python3 "$ROOT/tests/qualify_recorded_session_cleanup.py" \
        "$RUN_ROOT/session_recovery.json" \
        "${qualification_args[@]}" \
        --out "$RUN_ROOT/session_recovery_clean.json" \
        > "$RUN_ROOT/session_recovery_clean.stdout" \
        2> "$RUN_ROOT/session_recovery_clean.stderr"
    recovery_clean_rc=$?
    printf '%s\n' "$recovery_clean_rc" \
        > "$RUN_ROOT/session_recovery_clean.rc"
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/cleanup.rc"

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
            && $cleanup_rc -eq 0 && $recovery_rc -eq 0 \
            && $recovery_clean_rc -eq 0 && $postflight_rc -eq 0 ]]; then
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
        value=$(tr -d '[:space:]' < "$artifact")
        if [[ ! "$value" =~ ^[0-9]+$ ]]; then
            printf '%s=malformed:%s\n' "$artifact" "$value" \
                >> "$RUN_ROOT/timeout_scan.txt"
            timeout_rc=1
            continue
        fi
        case "$value" in
            124|137|143)
                printf '%s=%s\n' "$artifact" "$value" \
                    >> "$RUN_ROOT/timeout_scan.txt"
                timeout_rc=1
                ;;
        esac
    done < <(find "$RUN_ROOT" -type f -name '*.rc' -print0)
    printf '%s\n' "$timeout_rc" > "$RUN_ROOT/timeout_scan.rc"

    if [[ $cleanup_rc -ne 0 || $recovery_rc -ne 0 \
            || $recovery_clean_rc -ne 0 || $fatal_rc -ne 0 \
            || $timeout_rc -ne 0 || $postflight_rc -ne 0 \
            || $preflight_rc -ne 0 || $comparison_rc -ne 0 ]]; then
        primary_rc=1
    fi
    exit "$primary_rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set +e
timeout --signal=TERM --kill-after=70s 240s \
    env PYTHONPATH="$ROOT/tests" \
    python3 "$ROOT/tests/service_postflight_gate.py" \
    --gpus 0,1,2,3 --settle-timeout-s 90 \
    --clean-samples 3 --sample-interval-s 2 \
    --out "$RUN_ROOT/postflight_before.json" \
    > "$RUN_ROOT/postflight_before.stdout" \
    2> "$RUN_ROOT/postflight_before.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/postflight_before.rc"
[[ $rc -eq 0 ]]

set +e
timeout --signal=TERM --kill-after=70s 480s \
    env PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
    LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
    python3 "$ROOT/tests/bi100_preflight.py" \
    --gpus 0,1,2,3 --timeout-s 25 --matmul-size 1024 \
    --json-out "$RUN_ROOT/preflight_before.json" \
    > "$RUN_ROOT/preflight_before.stdout" \
    2> "$RUN_ROOT/preflight_before.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/preflight_before.rc"
[[ $rc -eq 0 ]]
BEFORE_PREFLIGHT_PASSED=1

for gpu in 0 1 2 3; do
    python3 "$ROOT/scripts/exec_bi100_session.py" \
        "${IDENTITIES[$gpu]}" -- \
        "$SELF" --cell \
        "$INSTANCE" "$gpu" "${CASE_GROUPS[$gpu]}" "${ORDERS[$gpu]}" \
        "$BASELINE_EXTENSION" "$BASELINE_SHA" \
        "$CANDIDATE_EXTENSION" "$CANDIDATE_SHA" \
        "$RUN_ROOT" "$SOURCE_REVISION" \
        > "$RUN_ROOT/gpu${gpu}.stdout" \
        2> "$RUN_ROOT/gpu${gpu}.stderr" &
    PIDS+=("$!")
    pid=${PIDS[$gpu]}
    identity_ok=0
    for _ in $(seq 1 50); do
        if [[ -s "${IDENTITIES[$gpu]}" ]] \
                && python3 - "${IDENTITIES[$gpu]}" "$pid" <<'PY'
import json
from pathlib import Path
import sys

identity = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pid = int(sys.argv[2])
stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
fields = stat[stat.rfind(")") + 2:].split()
token = identity.get("session_token")
expected = f"BI100_PROCESS_SESSION_TOKEN={token}".encode("ascii")
environment = (
    Path("/proc") / str(pid) / "environ"
).read_bytes().split(b"\0")
if (
    identity.get("schema") != "bi100-process-session-v1"
    or identity.get("version") != 1
    or identity.get("pid") != pid
    or identity.get("pgid") != pid
    or identity.get("sid") != pid
    or identity.get("starttime_ticks") != int(fields[19])
    or not isinstance(token, str)
    or len(token) != 32
    or any(character not in "0123456789abcdef" for character in token)
    or expected not in environment
):
    raise SystemExit(1)
PY
        then
            identity_ok=1
            break
        fi
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
    done
    if [[ "$identity_ok" != 1 ]]; then
        echo "M1-111 cell $gpu process identity was not attested" >&2
        exit 1
    fi
done

rc=0
for gpu in 0 1 2 3; do
    set +e
    wait "${PIDS[$gpu]}"
    cell_rc=$?
    set -e
    printf '%s\n' "$cell_rc" > "$RUN_ROOT/gpu${gpu}.rc"
    if [[ $cell_rc -ne 0 ]]; then
        rc=1
    fi
done
if [[ $rc -ne 0 ]]; then
    echo "one or more M1-111 retest cells failed to execute" >&2
    exit "$rc"
fi

set +e
python3 - "$RUN_ROOT" "$BASELINE_SHA" "$CANDIDATE_SHA" \
        "${ALL_CASES[@]}" <<'PY'
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
baseline_sha, candidate_sha = sys.argv[2:4]
cases = sys.argv[4:]
rows = []
reasons = []

for case_name in cases:
    cell = json.loads(
        (root / f"{case_name}.json").read_text(encoding="utf-8"))
    if cell.get("schema") != (
        "bi100-m1-111-query-tiled-production-retest-v1"
    ):
        reasons.append(f"{case_name}: schema mismatch")
        continue
    if cell.get("case") != case_name:
        reasons.append(f"{case_name}: case identity mismatch")
    if cell.get("baseline_extension", {}).get("sha256") != baseline_sha:
        reasons.append(f"{case_name}: baseline identity mismatch")
    if cell.get("candidate_extension", {}).get("sha256") != candidate_sha:
        reasons.append(f"{case_name}: candidate identity mismatch")
    if cell.get("qualified") is not True or cell.get("reasons") != []:
        reasons.append(f"{case_name}: fixed gate did not qualify")
    timings = cell.get("timings", {})
    numerical = cell.get("numerical", {}).get(
        "candidate_vs_reference", {})
    rows.append({
        "case": case_name,
        "context_len": cell.get("context_len"),
        "query_len": cell.get("query_len"),
        "total_kv_len": cell.get("total_kv_len"),
        "order": cell.get("order"),
        "baseline_cuda_median_ms": timings.get(
            "baseline", {}).get("cuda_median_ms"),
        "candidate_cuda_median_ms": timings.get(
            "candidate", {}).get("cuda_median_ms"),
        "baseline_over_candidate": timings.get(
            "baseline_over_candidate"),
        "candidate_output_relative_l2": numerical.get(
            "output_relative_l2"),
        "candidate_lse_relative_l2": numerical.get("lse_relative_l2"),
        "candidate_output_max_abs": numerical.get("output_max_abs"),
    })

long_speedups = [
    row["baseline_over_candidate"]
    for row in rows
    if row["case"] in {
        "production_65k_q8176",
        "production_128k_q8176",
        "production_235k_q5616",
        "boundary_262k_q8192",
    }
]
report = {
    "schema": "bi100-m1-111-query-tiled-production-retest-matrix-v1",
    "qualified": not reasons and len(rows) == len(cases),
    "baseline_extension_sha256": baseline_sha,
    "candidate_extension_sha256": candidate_sha,
    "rows": rows,
    "long_speedup_median": (
        statistics.median(long_speedups)
        if len(long_speedups) == 4 else None
    ),
    "reasons": reasons,
    "decision": {
        "runtime_integration_authorized":
            not reasons and len(rows) == len(cases),
        "unchanged_m1_55_route_closed":
            bool(reasons) or len(rows) != len(cases),
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
    "long_speedup_median": report["long_speedup_median"],
    "reasons": reasons,
}, sort_keys=True))
raise SystemExit(0 if report["qualified"] else 1)
PY
comparison_rc=$?
set -e
printf '%s\n' "$comparison_rc" > "$RUN_ROOT/comparison.rc"
exit "$comparison_rc"
