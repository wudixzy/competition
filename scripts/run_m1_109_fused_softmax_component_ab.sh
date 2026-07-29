#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)
COMPONENT_AB_VARIANT=${BI100_COMPONENT_AB_VARIANT:-m1-109-fused-softmax}
case "$COMPONENT_AB_VARIANT" in
    m1-109-fused-softmax|m1-113-group2048) ;;
    *)
        echo "BI100_COMPONENT_AB_VARIANT is invalid" >&2
        exit 2
        ;;
esac
export BI100_COMPONENT_AB_VARIANT="$COMPONENT_AB_VARIANT"

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
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
        . ':(exclude)bench_runs/**')" ]]; then
    echo "M1-109 component A/B refuses a dirty source tree" >&2
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
printf '%s\n' "$OLD_SHA" > "$RUN_ROOT/old_extension_sha256.txt"
printf '%s\n' "$NEW_SHA" > "$RUN_ROOT/new_extension_sha256.txt"

write_status() {
    local final_rc=$1
    python3 - "$RUN_ROOT" "$COMPONENT_AB_VARIANT" "$SOURCE_REVISION" \
            "$OLD_SHA" "$NEW_SHA" "$final_rc" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])


def read_rc(name):
    path = root / name
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    try:
        return int(value)
    except ValueError:
        return None


def digest(name):
    path = root / name
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


report = {
    "schema": "bi100-component-ab-runner-status-v1",
    "version": 1,
    "variant": sys.argv[2],
    "source_revision": sys.argv[3],
    "source_branch": (root / "source_branch.txt").read_text(
        encoding="utf-8").strip(),
    "old_extension_sha256": sys.argv[4],
    "new_extension_sha256": sys.argv[5],
    "returncode": int(sys.argv[6]),
    "gates": {
        "postflight_before": read_rc("postflight_before.rc"),
        "preflight_before": read_rc("preflight_before.rc"),
        "gpu0": read_rc("gpu0.rc"),
        "gpu1": read_rc("gpu1.rc"),
        "gpu2": read_rc("gpu2.rc"),
        "gpu3": read_rc("gpu3.rc"),
        "comparison": read_rc("comparison.rc"),
        "session_recovery": read_rc("session_recovery.rc"),
        "session_recovery_clean": read_rc("session_recovery_clean.rc"),
        "final_postflight": read_rc("final_postflight.rc"),
        "preflight_after": read_rc("preflight_after.rc"),
        "preflight_comparison": read_rc("preflight_comparison.rc"),
        "fatal_scan": read_rc("fatal_scan.rc"),
        "timeout_scan": read_rc("timeout_scan.rc"),
    },
    "artifacts": {
        "comparison_sha256": digest("comparison.json"),
        "session_recovery_sha256": digest("session_recovery.json"),
        "session_recovery_clean_sha256": digest(
            "session_recovery_clean.json"),
        "preflight_comparison_sha256": digest(
            "preflight_comparison.json"),
    },
    "tp4_service_experiment_authorized": False,
    "main_or_yaml_change_authorized": False,
}
(root / "runner_status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

finish() {
    local primary_rc=$?
    local recovery_rc=0
    local recovery_clean_rc=0
    local fatal_rc=0
    local timeout_rc=0
    local postflight_rc=0
    local preflight_rc=0
    local comparison_rc=0
    local artifact
    local identity
    local pid
    local identity_args=()
    local qualification_args=()
    local value
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
            && $recovery_rc -eq 0 && $recovery_clean_rc -eq 0 \
            && $postflight_rc -eq 0 ]]; then
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

    if [[ $recovery_rc -ne 0 || $recovery_clean_rc -ne 0 \
            || $fatal_rc -ne 0 || $timeout_rc -ne 0 \
            || $postflight_rc -ne 0 \
            || $preflight_rc -ne 0 || $comparison_rc -ne 0 ]]; then
        primary_rc=1
    fi
    write_status "$primary_rc"
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
    python3 "$ROOT/scripts/exec_bi100_session.py" \
        "${IDENTITIES[$gpu]}" -- \
        "$0" --cell-pair \
        "$INSTANCE" "$gpu" "${CASES[$gpu]}" \
        "$OLD_EXTENSION" "$OLD_SHA" "$NEW_EXTENSION" "$NEW_SHA" \
        "$RUN_ROOT" "$SOURCE_REVISION" \
        > "$RUN_ROOT/${CASES[$gpu]}.stdout" \
        2> "$RUN_ROOT/${CASES[$gpu]}.stderr" &
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
        echo "M1-109 component cell $gpu identity was not attested" >&2
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
    echo "one or more M1-109 component pairs failed" >&2
    exit "$rc"
fi

set +e
python3 - "$RUN_ROOT" "$OLD_SHA" "$NEW_SHA" \
        "$COMPONENT_AB_VARIANT" "${CASES[@]}" <<'PY'
import json
import math
import statistics
from pathlib import Path
import sys

root = Path(sys.argv[1])
old_sha, new_sha = sys.argv[2:4]
variant = sys.argv[4]
cases = sys.argv[5:]
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
    "schema": (
        "bi100-m1-113-group2048-component-ab-v1"
        if variant == "m1-113-group2048"
        else "bi100-m1-109-fused-softmax-component-ab-v1"
    ),
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
comparison_rc=$?
set -e
printf '%s\n' "$comparison_rc" > "$RUN_ROOT/comparison.rc"
exit "$comparison_rc"
