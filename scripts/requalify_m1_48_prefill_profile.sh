#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"
SOURCE_RUN=${M1_48_REQUALIFY_SOURCE_RUN:-}
RUN_ROOT=${M1_48_REQUALIFY_RUN_ROOT:-$ROOT/bench_runs/m1_48/prefill_profile_requalified}
M1_49_LONG_DIR=${M1_49_LONG_DIR:-$ROOT/bench_runs/m1_49/full_attention_long}
RUNTIME_ROOT=${BI100_RUNTIME_SITE_PACKAGES:+$(dirname "$BI100_RUNTIME_SITE_PACKAGES")}
RUNTIME_INSTALL_REPORT=${BI100_RUNTIME_INSTALL_REPORT:-${RUNTIME_ROOT:-/missing}/install.json}

require_zero_rc() {
    local path=$1
    if [[ ! -f "$path" ]] || [[ $(<"$path") != 0 ]]; then
        echo "required successful gate is missing: $path" >&2
        exit 3
    fi
}

require_nonzero_rc() {
    local path=$1
    local value=""
    if [[ -f "$path" ]]; then
        IFS= read -r value < "$path" || true
    fi
    if [[ ! "$value" =~ ^[0-9]+$ ]] || [[ "$value" == 0 ]]; then
        echo "required failed gate is missing: $path" >&2
        exit 3
    fi
}

if [[ -z "$SOURCE_RUN" || ! -d "$SOURCE_RUN" ]]; then
    echo "M1-48 failed source run is required" >&2
    exit 3
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "M1-48 requalification output already exists: $RUN_ROOT" >&2
    exit 5
fi
for gate in \
        cleanup runtime_identity preflight_before_control \
        control_startup_gate control_service control/fatal_scan \
        preflight_after_control profile_startup_gate profile_service \
        profile/fatal_scan preflight_after_profile preflight_comparison; do
    require_zero_rc "$SOURCE_RUN/${gate}.rc"
done
require_nonzero_rc "$SOURCE_RUN/profile_summary.rc"
require_nonzero_rc "$SOURCE_RUN/overall.rc"

for evidence in \
        runtime_identity.json preflight_comparison.json source_revision.txt \
        control/server.log control/server.pgid control/startup_gate.json \
        control/service.json profile/server.log profile/server.pgid \
        profile/startup_gate.json profile/service.json profile_summary.json; do
    if [[ ! -f "$SOURCE_RUN/$evidence" ]]; then
        echo "M1-48 source evidence is missing: $SOURCE_RUN/$evidence" >&2
        exit 3
    fi
done
if [[ ! -f "$M1_49_LONG_DIR/qualification.json" \
        || ! -f "$RUNTIME_INSTALL_REPORT" ]]; then
    echo "M1-48 prerequisite qualification or runtime report is missing" >&2
    exit 3
fi

python3 - "$SOURCE_RUN/profile_summary.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    f"rank {rank} paged dispatch differs at offset {offset}"
    for rank in range(4)
    for offset in range(29)
}
reasons = report.get("reasons")
if (report.get("qualified_profile") is not False
        or not isinstance(reasons, list)
        or len(reasons) != 116
        or set(reasons) != expected):
    raise SystemExit(
        "M1-48 source failure is not the exact TP4 query-head contract bug")
PY

for arm in control profile; do
    pgid=$(<"$SOURCE_RUN/$arm/server.pgid")
    live_count=$(bi100_process_group_count "$pgid" live)
    if [[ "$live_count" != 0 ]]; then
        echo "M1-48 source process group still has live members: $pgid" >&2
        exit 3
    fi
done
python3 - <<'PY'
import socket

sock = socket.socket()
try:
    sock.bind(("127.0.0.1", 8000))
finally:
    sock.close()
PY

mkdir -p "$RUN_ROOT"
git -C "$ROOT" rev-parse HEAD > "$RUN_ROOT/requalifier_revision.txt"
printf '%s\n' "$SOURCE_RUN" > "$RUN_ROOT/source_run.txt"

finish() {
    local rc=$?
    trap - EXIT
    printf '%s\n' 0 > "$RUN_ROOT/cleanup.rc"
    printf '%s\n' "$rc" > "$RUN_ROOT/overall.rc"
    exit "$rc"
}
trap finish EXIT

run_gate() {
    local name=$1
    local timeout_s=$2
    shift 2
    set +e
    timeout --signal=TERM --kill-after=30s "${timeout_s}s" "$@" \
        > "$RUN_ROOT/${name}.stdout" 2> "$RUN_ROOT/${name}.stderr"
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$RUN_ROOT/${name}.rc"
    if [[ $rc -ne 0 ]]; then
        echo "M1-48 requalification gate failed: $name rc=$rc" >&2
        return "$rc"
    fi
}

run_gate post_source_preflight 150 \
    python3 "$ROOT/tests/bi100_preflight.py" \
    --gpus 0,1,2,3 --timeout-s 25 --matmul-size 1024 \
    --json-out "$RUN_ROOT/post_source_preflight.json"

run_gate profile_summary 300 \
    python3 "$ROOT/tests/summarize_prefill_path_profile.py" \
    "$SOURCE_RUN/profile/server.log" \
    --expected-prefill-tokens 235000 --expected-processes 4 \
    --expected-chunk-size 8192 --block-size 16 \
    --num-attention-heads 16 \
    --control-service "$SOURCE_RUN/control/service.json" \
    --profile-service "$SOURCE_RUN/profile/service.json" \
    --out "$RUN_ROOT/profile_summary.json"

printf '%s\n' 0 > "$RUN_ROOT/prequalification_cleanup.rc"
run_gate qualification 300 \
    python3 "$ROOT/tests/qualify_m1_48_prefill_profile.py" \
    --m1-49 "$M1_49_LONG_DIR/qualification.json" \
    --runtime-install "$RUNTIME_INSTALL_REPORT" \
    --runtime-identity "$SOURCE_RUN/runtime_identity.json" \
    --preflight "$SOURCE_RUN/preflight_comparison.json" \
    --control-startup "$SOURCE_RUN/control/startup_gate.json" \
    --profile-startup "$SOURCE_RUN/profile/startup_gate.json" \
    --control-service "$SOURCE_RUN/control/service.json" \
    --profile-service "$SOURCE_RUN/profile/service.json" \
    --profile-summary "$RUN_ROOT/profile_summary.json" \
    --prequalification-cleanup "$RUN_ROOT/prequalification_cleanup.rc" \
    --source-revision "$SOURCE_RUN/source_revision.txt" \
    --control-log "$SOURCE_RUN/control/server.log" \
    --profile-log "$SOURCE_RUN/profile/server.log" \
    --out "$RUN_ROOT/qualification.json"

echo "M1-48 prefill profile requalification passed"
