#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"
source "$ROOT/scripts/lib/private_artifacts.sh"

if [[ $# -ne 2 ]]; then
    echo "usage: $0 INSTANCE RUN_ROOT" >&2
    exit 2
fi

INSTANCE=$1
QUALITY_AB_VARIANT=${BI100_QUALITY_AB_VARIANT:-admission64-policy}
case "$QUALITY_AB_VARIANT" in
    admission64-policy|m1-112-fused-prefill|\
m1-116-fused-prefill-adjudication|\
m1-117-fused-prefill-long-context|\
m1-122-fused-prefill-ifeval|\
m1-137-fused-prefill-ifeval-power149|\
m1-132-fused-prefill-teacher-forced|\
m1-134-teacher-forced-control-repeat) ;;
    *)
        echo "BI100_QUALITY_AB_VARIANT is invalid" \
            >&2
        exit 2
        ;;
esac
IFEVAL_VARIANT=0
if [[ "$QUALITY_AB_VARIANT" == m1-122-fused-prefill-ifeval \
        || "$QUALITY_AB_VARIANT" == \
        m1-137-fused-prefill-ifeval-power149 ]]; then
    IFEVAL_VARIANT=1
fi
FUSED_OUTPUT_DIAGNOSTIC_HMAC_KEY=""
TEACHER_FORCED_HMAC_KEY=""
if [[ "$QUALITY_AB_VARIANT" == \
        m1-116-fused-prefill-adjudication ]]; then
    FUSED_OUTPUT_DIAGNOSTIC_HMAC_KEY=$(
        python3 -c 'import secrets; print(secrets.token_hex(32))')
fi
if [[ "$QUALITY_AB_VARIANT" == \
        m1-132-fused-prefill-teacher-forced \
        || "$QUALITY_AB_VARIANT" == \
        m1-134-teacher-forced-control-repeat ]]; then
    TEACHER_FORCED_HMAC_KEY=$(
        python3 -c 'import secrets; print(secrets.token_hex(32))')
fi
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
if [[ "$IFEVAL_VARIANT" == 1 ]]; then
    if [[ -z "${BI100_IFEVAL_ENV:-}" ]]; then
        echo "BI100_IFEVAL_ENV is required for IFEval A/B" >&2
        exit 3
    fi
    BI100_IFEVAL_ENV=$(python3 - "$BI100_IFEVAL_ENV" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=True))
PY
    )
    if [[ ! -f "$BI100_IFEVAL_ENV/install.json" \
            || ! -d "$BI100_IFEVAL_ENV/site-packages" \
            || ! -d "$BI100_IFEVAL_ENV/nltk_data" ]]; then
        echo "offline IFEval A/B environment is incomplete" >&2
        exit 3
    fi
    if [[ -z "${BI100_IFEVAL_MANIFEST:-}" ]]; then
        echo "BI100_IFEVAL_MANIFEST is required for IFEval A/B" >&2
        exit 3
    fi
    BI100_IFEVAL_MANIFEST=$(python3 - "$BI100_IFEVAL_MANIFEST" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=True))
PY
    )
    export BI100_IFEVAL_ENV BI100_IFEVAL_MANIFEST
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
ACTIVE_CHILD_STARTTIME=""
ACTIVE_CHILD_SESSION_TOKEN=""
ACTIVE_CHILD_IDENTITY=""
CHILD_TERM_GRACE_S=60
CHILD_KILL_GRACE_S=20

write_status() {
    local final_rc=$1
    python3 - "$RUN_ROOT" "$INSTANCE" "$final_rc" \
            "$QUALITY_AB_VARIANT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
variant = sys.argv[4]

def read_rc(path):
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return int(value) if value.isdigit() else None

schemas = {
    "admission64-policy": "bi100-admission64-quality-ab-runner-v2",
    "m1-112-fused-prefill": "bi100-fused-prefill-quality-ab-runner-v1",
    "m1-116-fused-prefill-adjudication": (
        "bi100-fused-prefill-quality-adjudication-ab-runner-v1"),
    "m1-117-fused-prefill-long-context": (
        "bi100-fused-prefill-long-context-ab-runner-v1"),
    "m1-122-fused-prefill-ifeval": (
        "bi100-m1-122-ifeval-fused-prefill-ab-runner-v1"),
    "m1-137-fused-prefill-ifeval-power149": (
        "bi100-m1-137-ifeval-power149-fused-prefill-ab-runner-v1"),
    "m1-132-fused-prefill-teacher-forced": (
        "bi100-m1-132-teacher-forced-fused-prefill-ab-runner-v1"),
    "m1-134-teacher-forced-control-repeat": (
        "bi100-m1-134-teacher-forced-control-repeat-runner-v1"),
}
report = {
    "schema": schemas[variant],
    "version": 2 if variant == "admission64-policy" else 1,
    "source_revision": (root / "source_revision.txt").read_text(
        encoding="utf-8").strip(),
    "source_branch": (root / "source_branch.txt").read_text(
        encoding="utf-8").strip(),
    "instance": sys.argv[2],
    "returncode": int(sys.argv[3]),
    "fixed_order": (
        ["fused-off-a", "fused-off-b"]
        if variant == "m1-134-teacher-forced-control-repeat"
        else (
            ["fine32", "admission64"]
            if variant == "admission64-policy"
            else ["fused-off", "fused-on"]
        )
    ),
    "gates": {
        "control": read_rc(root / "control.rc"),
        "candidate": read_rc(root / "candidate.rc"),
        "quality_comparison": read_rc(root / "quality_comparison.rc"),
        "agent_comparison": read_rc(root / "agent_comparison.rc"),
        "aggregate": read_rc(root / "aggregate.rc"),
        "fused_output_comparison": read_rc(
            root / "fused_output_comparison.rc"),
        "long_context_comparison": read_rc(
            root / "long_context_comparison.rc"),
        "orchestrator_cleanup": read_rc(
            root / "orchestrator_cleanup.rc"),
        "orchestrator_recovery": read_rc(
            root / "orchestrator_recovery.rc"),
        "orchestrator_recovery_clean": read_rc(
            root / "orchestrator_recovery_clean.rc"),
        "orchestrator_postflight": read_rc(
            root / "orchestrator_postflight.rc"),
        "orchestrator_preflight_after": read_rc(
            root / "orchestrator_preflight_after.rc"),
        "orchestrator_fatal_scan": read_rc(
            root / "orchestrator_fatal_scan.rc"),
        "orchestrator_timeout_scan": read_rc(
            root / "orchestrator_timeout_scan.rc"),
    },
    "performance_authorized": False,
    "default_policy_change_authorized": False,
    "production_promotion_authorized": False,
}
if variant in {
    "m1-122-fused-prefill-ifeval",
    "m1-137-fused-prefill-ifeval-power149",
}:
    report["gates"].update({
        "ifeval_paired_noninferiority": read_rc(
            root / "ifeval_paired_noninferiority.rc"),
    })
    if variant == "m1-137-fused-prefill-ifeval-power149":
        report["gates"]["final_qualification"] = read_rc(
            root / "final_qualification.rc")
    report["diagnostics"] = {
        "ifeval_score_comparison": read_rc(
            root / "ifeval_score_comparison.rc"),
        "ifeval_exact_comparison": read_rc(
            root / "ifeval_exact_comparison.rc"),
    }
elif variant in {
    "m1-132-fused-prefill-teacher-forced",
    "m1-134-teacher-forced-control-repeat",
}:
    report["gates"].update({
        "teacher_forced_comparison": read_rc(
            root / "teacher_forced_comparison.rc"),
        "private_observation_cleanup": read_rc(
            root / "private_observation_cleanup.rc"),
    })
artifacts = {}
for name, relative_path in (
    ("control_child_identity", "control_child_identity.json"),
    ("control_service_identity", "control/process_group_identity.json"),
    ("candidate_child_identity", "candidate_child_identity.json"),
    ("candidate_service_identity", "candidate/process_group_identity.json"),
    ("orchestrator_recovery", "orchestrator_recovery.json"),
    ("orchestrator_recovery_clean", "orchestrator_recovery_clean.json"),
    ("fused_output_comparison", "fused_output_comparison.json"),
    ("long_context_comparison", "long_context_comparison.json"),
):
    path = root / relative_path
    artifacts[f"{name}_sha256"] = (
        hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_file() else None
    )
if variant in {
    "m1-122-fused-prefill-ifeval",
    "m1-137-fused-prefill-ifeval-power149",
}:
    for name in (
        "ifeval_score_comparison",
        "ifeval_exact_comparison",
        "ifeval_paired_noninferiority",
        "aggregate",
    ):
        path = root / f"{name}.json"
        artifacts[f"{name}_sha256"] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file() else None
        )
    if variant == "m1-137-fused-prefill-ifeval-power149":
        path = root / "final_qualification.json"
        artifacts["final_qualification_sha256"] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file() else None
        )
elif variant in {
    "m1-132-fused-prefill-teacher-forced",
    "m1-134-teacher-forced-control-repeat",
}:
    path = root / "teacher_forced_comparison.json"
    artifacts["teacher_forced_comparison_sha256"] = (
        hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_file() else None
    )
report["artifacts"] = artifacts
if variant == "m1-137-fused-prefill-ifeval-power149":
    final_qualification = None
    path = root / "final_qualification.json"
    if path.is_file():
        try:
            final_qualification = json.loads(
                path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            final_qualification = None
    final_rescan = (
        final_qualification.get("final_rescan", {})
        if isinstance(final_qualification, dict) else {}
    )
    report["qualified"] = (
        report["returncode"] == 0
        and report["gates"]["final_qualification"] == 0
        and isinstance(final_qualification, dict)
        and final_qualification.get("schema")
        == "bi100-m1-137-ifeval-power149-final-qualification-v1"
        and final_qualification.get("version") == 1
        and final_qualification.get("qualified") is True
        and final_qualification.get("reasons") == []
        and final_rescan.get("fatal_match_file_count") == 0
        and final_rescan.get("malformed_return_code_count") == 0
        and final_rescan.get("timeout_return_code_count") == 0
        and isinstance(final_rescan.get("fatal_input_set_sha256"), str)
        and len(final_rescan["fatal_input_set_sha256"]) == 64
        and isinstance(
            final_rescan.get("return_code_input_set_sha256"), str)
        and len(final_rescan["return_code_input_set_sha256"]) == 64
        and all(
            final_qualification.get(name) is False
            for name in (
                "performance_authorized",
                "default_change_authorized",
                "yaml_change_authorized",
                "main_merge_authorized",
                "production_promotion_authorized",
            )
        )
    )
    report["ifeval_two_point_capability_surface_authorized"] = (
        report["qualified"]
        and final_qualification.get(
            "ifeval_two_point_capability_surface_authorized") is True
    )
(root / "runner_status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8")
PY
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
            "$ACTIVE_CHILD_PGID" "$ACTIVE_CHILD_PID" \
            "$CHILD_TERM_GRACE_S" "$CHILD_KILL_GRACE_S" \
            "$ACTIVE_CHILD_STARTTIME" \
            "$ACTIVE_CHILD_SESSION_TOKEN" || rc=$?
    else
        if active_child_is_same; then
            kill -TERM "$ACTIVE_CHILD_PID" 2>/dev/null || true
            for _ in $(seq 1 "$CHILD_TERM_GRACE_S"); do
                active_child_is_same || break
                sleep 1
            done
        fi
        if active_child_is_same; then
            kill -KILL "$ACTIVE_CHILD_PID" 2>/dev/null || true
            for _ in $(seq 1 "$CHILD_KILL_GRACE_S"); do
                active_child_is_same || break
                sleep 1
            done
        fi
    fi
    wait "$ACTIVE_CHILD_PID" 2>/dev/null || true
    if active_child_is_same; then
        echo "A/B child survived scoped cleanup" >&2
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

recover_recorded_children() {
    local identities=()
    local path
    for path in \
        "$RUN_ROOT/control_child_identity.json" \
        "$RUN_ROOT/control/process_group_identity.json" \
        "$RUN_ROOT/candidate_child_identity.json" \
        "$RUN_ROOT/candidate/process_group_identity.json"; do
        if [[ -f "$path" ]]; then
            identities+=(--identity "$path")
        fi
    done
    python3 "$ROOT/scripts/cleanup_recorded_bi100_sessions.py" \
        "${identities[@]}" \
        --out "$RUN_ROOT/orchestrator_recovery.json" \
        > "$RUN_ROOT/orchestrator_recovery.stdout" \
        2> "$RUN_ROOT/orchestrator_recovery.stderr"
}

qualify_recorded_children() {
    local expected_identities=()
    local path
    for path in \
        "$RUN_ROOT/control_child_identity.json" \
        "$RUN_ROOT/control/process_group_identity.json" \
        "$RUN_ROOT/candidate_child_identity.json" \
        "$RUN_ROOT/candidate/process_group_identity.json"; do
        if [[ -f "$path" ]]; then
            expected_identities+=(--expected-identity "$path")
        fi
    done
    if [[ ${#expected_identities[@]} -eq 0 ]]; then
        echo "no recorded A/B process identities were available" >&2
        return 1
    fi
    python3 "$ROOT/tests/qualify_recorded_session_cleanup.py" \
        "$RUN_ROOT/orchestrator_recovery.json" \
        "${expected_identities[@]}" \
        --out "$RUN_ROOT/orchestrator_recovery_clean.json" \
        > "$RUN_ROOT/orchestrator_recovery_clean.stdout" \
        2> "$RUN_ROOT/orchestrator_recovery_clean.stderr"
}

run_orchestrator_postflight() {
    timeout --signal=TERM --kill-after=70s 240s \
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
    pattern='CUDA error|SIGSEGV|Fatal Python error|out of memory|device-side assert|AssertionError|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|Timeout(Error|Expired)|engine iteration timed out|watchdog.*tim(e|ed) out|scheduler requested a missing GDN prefix state|non-finite GatedDeltaNet'
    : > "$RUN_ROOT/orchestrator_fatal_scan.txt"
    while IFS= read -r -d '' file; do
        if grep -Eiq "$pattern" "$file"; then
            printf '%s\n' "file=$file" \
                >> "$RUN_ROOT/orchestrator_fatal_scan.txt"
            grep -Ein "$pattern" "$file" \
                >> "$RUN_ROOT/orchestrator_fatal_scan.txt" || true
            found=1
        fi
    done < <(find "$RUN_ROOT" -type f \
        \( -name '*.log' -o -name '*.stdout' -o -name '*.stderr' \) \
        -print0)
    return "$found"
}

scan_orchestrator_timeouts() {
    local file
    local found=0
    local value
    : > "$RUN_ROOT/orchestrator_timeout_scan.txt"
    while IFS= read -r -d '' file; do
        value=$(tr -d '[:space:]' < "$file")
        if [[ ! "$value" =~ ^[0-9]+$ ]]; then
            printf '%s=malformed:%s\n' "$file" "$value" \
                >> "$RUN_ROOT/orchestrator_timeout_scan.txt"
            found=1
            continue
        fi
        case "$value" in
            124|137|143)
                printf '%s=%s\n' "$file" "$value" \
                    >> "$RUN_ROOT/orchestrator_timeout_scan.txt"
                found=1
                ;;
        esac
    done < <(find "$RUN_ROOT" -type f -name '*.rc' -print0)
    return "$found"
}

finish() {
    local rc=$?
    local cleanup_rc=0
    local recovery_rc=0
    local recovery_clean_rc=0
    local postflight_rc=0
    local preflight_rc=0
    local fatal_rc=0
    local timeout_rc=0
    local private_observation_cleanup_rc=0
    local final_qualification_rc=0
    trap - EXIT
    trap '' TERM INT
    set +e
    stop_active_child
    cleanup_rc=$?
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/orchestrator_cleanup.rc"
    recover_recorded_children
    recovery_rc=$?
    printf '%s\n' "$recovery_rc" > "$RUN_ROOT/orchestrator_recovery.rc"
    qualify_recorded_children
    recovery_clean_rc=$?
    printf '%s\n' "$recovery_clean_rc" \
        > "$RUN_ROOT/orchestrator_recovery_clean.rc"
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
    if [[ "$QUALITY_AB_VARIANT" == \
            m1-132-fused-prefill-teacher-forced \
            || "$QUALITY_AB_VARIANT" == \
            m1-134-teacher-forced-control-repeat ]]; then
        remove_teacher_forced_observations "$RUN_ROOT"
        private_observation_cleanup_rc=$?
        printf '%s\n' "$private_observation_cleanup_rc" \
            > "$RUN_ROOT/private_observation_cleanup.rc"
    fi
    if [[ "$QUALITY_AB_VARIANT" == \
            m1-137-fused-prefill-ifeval-power149 ]]; then
        python3 "$ROOT/tests/qualify_m1_137_outer_lifecycle.py" \
            --run-root "$RUN_ROOT" \
            --aggregate "$RUN_ROOT/aggregate.json" \
            --out "$RUN_ROOT/final_qualification.json" \
            > "$RUN_ROOT/final_qualification.stdout" \
            2> "$RUN_ROOT/final_qualification.stderr"
        final_qualification_rc=$?
        printf '%s\n' "$final_qualification_rc" \
            > "$RUN_ROOT/final_qualification.rc"
    fi
    if [[ $cleanup_rc -ne 0 || $recovery_rc -ne 0 \
            || $recovery_clean_rc -ne 0 || $postflight_rc -ne 0 \
            || $preflight_rc -ne 0 || $fatal_rc -ne 0 \
            || $timeout_rc -ne 0 \
            || $private_observation_cleanup_rc -ne 0 \
            || $final_qualification_rc -ne 0 ]]; then
        rc=1
    fi
    write_status "$rc"
    exit "$rc"
}
trap 'exit 143' TERM
trap 'exit 130' INT
trap finish EXIT

run_arm() {
    local arm=$1
    local policy=$2
    local label=$3
    local output=$4
    local restore_mode=direct
    local fused_prefill=0
    local runner_name=$policy
    local suite=functional
    if [[ "$QUALITY_AB_VARIANT" != admission64-policy ]]; then
        restore_mode=hybrid64
        runner_name=$arm
        if [[ "$arm" == candidate \
                && "$QUALITY_AB_VARIANT" != \
                m1-134-teacher-forced-control-repeat ]]; then
            fused_prefill=1
        fi
    fi
    if [[ "$QUALITY_AB_VARIANT" == \
            m1-117-fused-prefill-long-context ]]; then
        suite=long-context
    elif [[ "$IFEVAL_VARIANT" == 1 ]]; then
        suite=ifeval
    elif [[ "$QUALITY_AB_VARIANT" == \
            m1-132-fused-prefill-teacher-forced \
            || "$QUALITY_AB_VARIANT" == \
            m1-134-teacher-forced-control-repeat ]]; then
        suite=teacher-forced
    fi
    local identity="$RUN_ROOT/${arm}_child_identity.json"
    local identity_ok=0
    local observed_pgid=""
    local observed_token=""
    local observed_identity=""
    local runner_env=(BI100_QUALITY_KERNEL_PROFILE=submission)
    if [[ "$QUALITY_AB_VARIANT" == \
            m1-116-fused-prefill-adjudication ]]; then
        runner_env+=(
            BI100_RUN_FUSED_OUTPUT_DIAGNOSTIC=1
            BI100_FUSED_OUTPUT_DIAGNOSTIC_RUN_ID=m1-109-pair-1-20260729
            BI100_FUSED_OUTPUT_DIAGNOSTIC_HMAC_KEY="$FUSED_OUTPUT_DIAGNOSTIC_HMAC_KEY"
        )
    elif [[ "$IFEVAL_VARIANT" == 1 ]]; then
        runner_env+=(
            BI100_IFEVAL_ENV="$BI100_IFEVAL_ENV"
            BI100_IFEVAL_MANIFEST="$BI100_IFEVAL_MANIFEST"
        )
    fi
    (
        if [[ "$QUALITY_AB_VARIANT" == \
                m1-132-fused-prefill-teacher-forced \
                || "$QUALITY_AB_VARIANT" == \
                m1-134-teacher-forced-control-repeat ]]; then
            export BI100_TEACHER_FORCED_HMAC_KEY="$TEACHER_FORCED_HMAC_KEY"
        fi
        if [[ "$suite" == long-context ]]; then
            exec python3 "$ROOT/scripts/exec_bi100_session.py" \
                "$identity" -- \
                env "${runner_env[@]}" \
                "$ROOT/scripts/run_quality_service_gate.sh" \
                long-context "$policy" "$restore_mode" \
                "$fused_prefill" lru \
                "$label" "$INSTANCE" "$output"
        fi
        if [[ "$suite" == ifeval ]]; then
            exec python3 "$ROOT/scripts/exec_bi100_session.py" \
                "$identity" -- \
                env "${runner_env[@]}" \
                "$ROOT/scripts/run_quality_service_gate.sh" \
                ifeval "$policy" "$restore_mode" \
                "$fused_prefill" lru \
                "$label" "$INSTANCE" "$output"
        fi
        if [[ "$suite" == teacher-forced ]]; then
            exec python3 "$ROOT/scripts/exec_bi100_session.py" \
                "$identity" -- \
                env "${runner_env[@]}" \
                "$ROOT/scripts/run_quality_service_gate.sh" \
                teacher-forced "$policy" "$restore_mode" \
                "$fused_prefill" lru \
                "$label" "$INSTANCE" "$output"
        fi
        exec python3 "$ROOT/scripts/exec_bi100_session.py" \
            "$identity" -- \
            env "${runner_env[@]}" \
            "$ROOT/scripts/run_quality_service_gate.sh" \
            functional "$policy" "$restore_mode" "$fused_prefill" lru \
            "$label" "$INSTANCE" "$output"
    ) > "$RUN_ROOT/${runner_name}_runner.stdout" \
      2> "$RUN_ROOT/${runner_name}_runner.stderr" &
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
        return 125
    fi
    local rc=0
    if wait "$ACTIVE_CHILD_PID"; then
        rc=0
    else
        rc=$?
    fi
    ACTIVE_CHILD_PID=""
    ACTIVE_CHILD_PGID=""
    ACTIVE_CHILD_STARTTIME=""
    ACTIVE_CHILD_SESSION_TOKEN=""
    ACTIVE_CHILD_IDENTITY=""
    return "$rc"
}

set +e
if [[ "$QUALITY_AB_VARIANT" == m1-112-fused-prefill ]]; then
    run_arm control admission64 m1-112-control-fused-off \
        "$RUN_ROOT/control"
elif [[ "$QUALITY_AB_VARIANT" == \
        m1-116-fused-prefill-adjudication ]]; then
    run_arm control admission64 m1-116-control-fused-off \
        "$RUN_ROOT/control"
elif [[ "$QUALITY_AB_VARIANT" == \
        m1-117-fused-prefill-long-context ]]; then
    run_arm control admission64 m1-117-control-fused-off \
        "$RUN_ROOT/control"
elif [[ "$QUALITY_AB_VARIANT" == \
        m1-122-fused-prefill-ifeval ]]; then
    run_arm control admission64 m1-122-control-fused-off \
        "$RUN_ROOT/control"
elif [[ "$QUALITY_AB_VARIANT" == \
        m1-137-fused-prefill-ifeval-power149 ]]; then
    run_arm control admission64 m1-137-control-fused-off \
        "$RUN_ROOT/control"
elif [[ "$QUALITY_AB_VARIANT" == \
        m1-132-fused-prefill-teacher-forced ]]; then
    run_arm control admission64 m1-132-control-fused-off \
        "$RUN_ROOT/control"
elif [[ "$QUALITY_AB_VARIANT" == \
        m1-134-teacher-forced-control-repeat ]]; then
    run_arm control admission64 m1-134-control-a-fused-off \
        "$RUN_ROOT/control"
else
    run_arm control fine32 m1-85-control-fine32 "$RUN_ROOT/control"
fi
control_rc=$?
set -e
printf '%s\n' "$control_rc" > "$RUN_ROOT/control.rc"
[[ $control_rc -eq 0 ]]

set +e
if [[ "$QUALITY_AB_VARIANT" == m1-112-fused-prefill ]]; then
    run_arm candidate admission64 m1-112-candidate-fused-on \
        "$RUN_ROOT/candidate"
elif [[ "$QUALITY_AB_VARIANT" == \
        m1-116-fused-prefill-adjudication ]]; then
    run_arm candidate admission64 m1-116-candidate-fused-on \
        "$RUN_ROOT/candidate"
elif [[ "$QUALITY_AB_VARIANT" == \
        m1-117-fused-prefill-long-context ]]; then
    run_arm candidate admission64 m1-117-candidate-fused-on \
        "$RUN_ROOT/candidate"
elif [[ "$QUALITY_AB_VARIANT" == \
        m1-122-fused-prefill-ifeval ]]; then
    run_arm candidate admission64 m1-122-candidate-fused-on \
        "$RUN_ROOT/candidate"
elif [[ "$QUALITY_AB_VARIANT" == \
        m1-137-fused-prefill-ifeval-power149 ]]; then
    run_arm candidate admission64 m1-137-candidate-fused-on \
        "$RUN_ROOT/candidate"
elif [[ "$QUALITY_AB_VARIANT" == \
        m1-132-fused-prefill-teacher-forced ]]; then
    run_arm candidate admission64 m1-132-candidate-fused-on \
        "$RUN_ROOT/candidate"
elif [[ "$QUALITY_AB_VARIANT" == \
        m1-134-teacher-forced-control-repeat ]]; then
    run_arm candidate admission64 m1-134-control-b-fused-off \
        "$RUN_ROOT/candidate"
else
    run_arm candidate admission64 m1-85-candidate-admission64 \
        "$RUN_ROOT/candidate"
fi
candidate_rc=$?
set -e
printf '%s\n' "$candidate_rc" > "$RUN_ROOT/candidate.rc"
[[ $candidate_rc -eq 0 ]]

quality_comparison_rc=0
agent_comparison_rc=0
aggregate_rc=0
long_context_comparison_rc=0
ifeval_score_comparison_rc=0
ifeval_exact_comparison_rc=0
ifeval_paired_noninferiority_rc=0
teacher_forced_comparison_rc=0
if [[ "$QUALITY_AB_VARIANT" == \
        m1-117-fused-prefill-long-context ]]; then
    set +e
    python3 "$ROOT/tests/compare_long_context_quality_reports.py" \
        "$RUN_ROOT/control/quality_report.json" \
        "$RUN_ROOT/candidate/quality_report.json" \
        --out "$RUN_ROOT/long_context_comparison.json" \
        > "$RUN_ROOT/long_context_comparison.stdout" \
        2> "$RUN_ROOT/long_context_comparison.stderr"
    long_context_comparison_rc=$?
    set -e
    printf '%s\n' "$long_context_comparison_rc" \
        > "$RUN_ROOT/long_context_comparison.rc"
elif [[ "$QUALITY_AB_VARIANT" == \
        m1-132-fused-prefill-teacher-forced \
        || "$QUALITY_AB_VARIANT" == \
        m1-134-teacher-forced-control-repeat ]]; then
    set +e
    teacher_forced_comparison_mode=candidate
    if [[ "$QUALITY_AB_VARIANT" == \
            m1-134-teacher-forced-control-repeat ]]; then
        teacher_forced_comparison_mode=control-repeat
    fi
    python3 "$ROOT/tests/compare_teacher_forced_logprobs.py" \
        "$RUN_ROOT/control/teacher_forced_observation.json" \
        "$RUN_ROOT/candidate/teacher_forced_observation.json" \
        --contract "$ROOT/quality/layered_quality_gate.v1.json" \
        --comparison-mode "$teacher_forced_comparison_mode" \
        --out "$RUN_ROOT/teacher_forced_comparison.json" \
        > "$RUN_ROOT/teacher_forced_comparison.stdout" \
        2> "$RUN_ROOT/teacher_forced_comparison.stderr"
    teacher_forced_comparison_rc=$?
    set -e
    printf '%s\n' "$teacher_forced_comparison_rc" \
        > "$RUN_ROOT/teacher_forced_comparison.rc"
elif [[ "$IFEVAL_VARIANT" == 1 ]]; then
    set +e
    python3 "$ROOT/tests/compare_ifeval_reports.py" \
        --baseline "$RUN_ROOT/control/ifeval_report.json" \
        --candidate "$RUN_ROOT/candidate/ifeval_report.json" \
        --allowed-switch fused_prefill \
        --out "$RUN_ROOT/ifeval_score_comparison.json" \
        > "$RUN_ROOT/ifeval_score_comparison.stdout" \
        2> "$RUN_ROOT/ifeval_score_comparison.stderr"
    ifeval_score_comparison_rc=$?
    printf '%s\n' "$ifeval_score_comparison_rc" \
        > "$RUN_ROOT/ifeval_score_comparison.rc"

    python3 "$ROOT/tests/compare_ifeval_reports.py" \
        --baseline "$RUN_ROOT/control/ifeval_report.json" \
        --candidate "$RUN_ROOT/candidate/ifeval_report.json" \
        --allowed-switch fused_prefill \
        --require-exact-output \
        --out "$RUN_ROOT/ifeval_exact_comparison.json" \
        > "$RUN_ROOT/ifeval_exact_comparison.stdout" \
        2> "$RUN_ROOT/ifeval_exact_comparison.stderr"
    ifeval_exact_comparison_rc=$?
    printf '%s\n' "$ifeval_exact_comparison_rc" \
        > "$RUN_ROOT/ifeval_exact_comparison.rc"

    layered_contract="$ROOT/quality/layered_quality_gate.v1.json"
    aggregate_script="$ROOT/tests/compare_m1_122_ifeval_service_ab.py"
    if [[ "$QUALITY_AB_VARIANT" == \
            m1-137-fused-prefill-ifeval-power149 ]]; then
        layered_contract="$ROOT/quality/layered_quality_gate.v2.json"
        aggregate_script="$ROOT/tests/compare_m1_137_ifeval_power_ab.py"
    fi
    python3 "$ROOT/tests/compare_ifeval_paired_noninferiority.py" \
        --baseline "$RUN_ROOT/control/ifeval_report.json" \
        --candidate "$RUN_ROOT/candidate/ifeval_report.json" \
        --contract "$layered_contract" \
        --allowed-switch fused_prefill \
        --out "$RUN_ROOT/ifeval_paired_noninferiority.json" \
        > "$RUN_ROOT/ifeval_paired_noninferiority.stdout" \
        2> "$RUN_ROOT/ifeval_paired_noninferiority.stderr"
    ifeval_paired_noninferiority_rc=$?
    printf '%s\n' "$ifeval_paired_noninferiority_rc" \
        > "$RUN_ROOT/ifeval_paired_noninferiority.rc"

    python3 "$aggregate_script" \
        --control-root "$RUN_ROOT/control" \
        --candidate-root "$RUN_ROOT/candidate" \
        --score-comparison "$RUN_ROOT/ifeval_score_comparison.json" \
        --exact-comparison "$RUN_ROOT/ifeval_exact_comparison.json" \
        --paired-noninferiority \
            "$RUN_ROOT/ifeval_paired_noninferiority.json" \
        --out "$RUN_ROOT/aggregate.json" \
        > "$RUN_ROOT/aggregate.stdout" \
        2> "$RUN_ROOT/aggregate.stderr"
    aggregate_rc=$?
    set -e
    printf '%s\n' "$aggregate_rc" > "$RUN_ROOT/aggregate.rc"
else
    set +e
    python3 "$ROOT/tests/compare_quality_gate_reports.py" \
        "$RUN_ROOT/control/quality_report.json" \
        "$RUN_ROOT/candidate/quality_report.json" \
        --out "$RUN_ROOT/quality_comparison.json" \
        > "$RUN_ROOT/quality_comparison.stdout" \
        2> "$RUN_ROOT/quality_comparison.stderr"
    quality_comparison_rc=$?
    set -e
    printf '%s\n' "$quality_comparison_rc" \
        > "$RUN_ROOT/quality_comparison.rc"

    set +e
    python3 "$ROOT/tests/compare_agent_workload_reports.py" \
        "$RUN_ROOT/control/agent_workload.json" \
        "$RUN_ROOT/candidate/agent_workload.json" \
        --out "$RUN_ROOT/agent_comparison.json" \
        > "$RUN_ROOT/agent_comparison.stdout" \
        2> "$RUN_ROOT/agent_comparison.stderr"
    agent_comparison_rc=$?
    set -e
    printf '%s\n' "$agent_comparison_rc" \
        > "$RUN_ROOT/agent_comparison.rc"

    set +e
    if [[ "$QUALITY_AB_VARIANT" != admission64-policy ]]; then
        python3 "$ROOT/tests/compare_fused_prefill_quality_service_ab.py" \
            --control-root "$RUN_ROOT/control" \
            --candidate-root "$RUN_ROOT/candidate" \
            --variant "$QUALITY_AB_VARIANT" \
            --quality-comparison "$RUN_ROOT/quality_comparison.json" \
            --agent-comparison "$RUN_ROOT/agent_comparison.json" \
            --out "$RUN_ROOT/aggregate.json" \
            > "$RUN_ROOT/aggregate.stdout" \
            2> "$RUN_ROOT/aggregate.stderr"
    else
        python3 "$ROOT/tests/compare_admission64_quality_service_ab.py" \
            --control-root "$RUN_ROOT/control" \
            --candidate-root "$RUN_ROOT/candidate" \
            --quality-comparison "$RUN_ROOT/quality_comparison.json" \
            --agent-comparison "$RUN_ROOT/agent_comparison.json" \
            --out "$RUN_ROOT/aggregate.json" \
            > "$RUN_ROOT/aggregate.stdout" \
            2> "$RUN_ROOT/aggregate.stderr"
    fi
    aggregate_rc=$?
    set -e
    printf '%s\n' "$aggregate_rc" > "$RUN_ROOT/aggregate.rc"
fi

fused_output_comparison_rc=0
if [[ "$QUALITY_AB_VARIANT" == \
        m1-116-fused-prefill-adjudication ]]; then
    set +e
    python3 "$ROOT/tests/compare_m1_116_fused_prefill_output.py" \
        "$RUN_ROOT/control/fused_output_diagnostic.json" \
        "$RUN_ROOT/candidate/fused_output_diagnostic.json" \
        --out "$RUN_ROOT/fused_output_comparison.json" \
        > "$RUN_ROOT/fused_output_comparison.stdout" \
        2> "$RUN_ROOT/fused_output_comparison.stderr"
    fused_output_comparison_rc=$?
    set -e
    printf '%s\n' "$fused_output_comparison_rc" \
        > "$RUN_ROOT/fused_output_comparison.rc"
fi

if [[ "$IFEVAL_VARIANT" == 1 ]]; then
    [[ $ifeval_paired_noninferiority_rc -eq 0 ]]
    [[ $aggregate_rc -eq 0 ]]
elif [[ "$QUALITY_AB_VARIANT" == \
        m1-132-fused-prefill-teacher-forced \
        || "$QUALITY_AB_VARIANT" == \
        m1-134-teacher-forced-control-repeat ]]; then
    [[ $teacher_forced_comparison_rc -eq 0 ]]
else
    [[ $quality_comparison_rc -eq 0 ]]
    [[ $agent_comparison_rc -eq 0 ]]
    [[ $aggregate_rc -eq 0 ]]
    [[ $fused_output_comparison_rc -eq 0 ]]
    [[ $long_context_comparison_rc -eq 0 ]]
fi
