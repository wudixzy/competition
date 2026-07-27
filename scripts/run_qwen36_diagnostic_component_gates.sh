#!/bin/bash
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

usage() {
    cat >&2 <<'EOF'
Usage:
  run_qwen36_diagnostic_component_gates.sh GPU_INDEX INSTANCE RUN_ROOT

GPU_INDEX is one healthy physical BI100 index. BI100_RUNTIME_SITE_PACKAGES
must point at an immutable runtime overlay. RUN_ROOT must be outside the repo.
EOF
}

if [[ $# -ne 3 ]]; then
    usage
    exit 2
fi

GPU_INDEX=$1
INSTANCE=$2
RUN_ROOT=$3
CURRENT_STAGE=argument_validation
ACTIVE_LAUNCHER_PID=""
ACTIVE_PID=""
ACTIVE_PGID=""
ACTIVE_START_TIME=""
ACTIVE_READY_FILE=""
PREFLIGHT_ATTEMPTED=0
POSTFLIGHT_COMPLETE=0
FATAL_PATTERN='CUDA error|SIGSEGV|segmentation fault|Fatal Python error|Traceback \(most recent call last\)|out of memory|device-side assert|illegal memory access|worker process.*died|worker.*(lost|exited unexpectedly)|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|TimeoutError|watchdog.*tim(e|ed) out'

if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
    echo "GPU_INDEX must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "$INSTANCE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    echo "INSTANCE must be a short non-sensitive label" >&2
    exit 2
fi
RUN_ROOT=$(python3 - "$RUN_ROOT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
) || exit 3
case "$RUN_ROOT/" in
    "$ROOT/"*)
        echo "RUN_ROOT must stay outside the source repository" >&2
        exit 3
        ;;
esac
if [[ -e "$RUN_ROOT" ]]; then
    echo "RUN_ROOT already exists: $RUN_ROOT" >&2
    exit 3
fi
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
        . ':(exclude)bench_runs/**')" ]]; then
    echo "component gate refuses a dirty source tree" >&2
    exit 3
fi
if [[ -z "${BI100_RUNTIME_SITE_PACKAGES:-}" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/vllm" ]]; then
    echo "BI100_RUNTIME_SITE_PACKAGES must identify an immutable overlay" >&2
    exit 3
fi
BI100_RUNTIME_SITE_PACKAGES=$(python3 - \
        "$BI100_RUNTIME_SITE_PACKAGES" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=True))
PY
) || exit 3
RUNTIME_INSTALL=$(dirname "$BI100_RUNTIME_SITE_PACKAGES")/install.json
if [[ ! -f "$RUNTIME_INSTALL" ]]; then
    echo "runtime install report is missing: $RUNTIME_INSTALL" >&2
    exit 3
fi
if pgrep -f 'vllm\.entrypoints\.openai\.api_server' >/dev/null 2>&1; then
    echo "an API server is already running" >&2
    exit 3
fi

VLLM_ROOT=$BI100_RUNTIME_SITE_PACKAGES/vllm
EXTENSIONS=(
    "$VLLM_ROOT/corex_moe_direct_routed.so"
    "$VLLM_ROOT/corex_moe_weight_gather.so"
    "$VLLM_ROOT/corex_moe_exact_reduce.so"
    "$VLLM_ROOT/corex_gdn_packed_decode.so"
    "$VLLM_ROOT/corex_gdn_beta_decay.so"
    "$VLLM_ROOT/corex_gdn_qk_map.so"
    "$VLLM_ROOT/corex_paged_kv_gather.so"
    "$VLLM_ROOT/corex_block_major_kv_transfer.so"
)
for extension in "${EXTENSIONS[@]}"; do
    if [[ ! -f "$extension" ]]; then
        echo "required runtime extension is missing: $extension" >&2
        exit 3
    fi
done

mkdir -p "$RUN_ROOT"
SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
SOURCE_BRANCH=$(git -C "$ROOT" branch --show-current)
printf '%s\n' "$SOURCE_REVISION" > "$RUN_ROOT/source_revision.txt"
printf '%s\n' "$SOURCE_BRANCH" > "$RUN_ROOT/source_branch.txt"
printf '%s\n' "$GPU_INDEX" > "$RUN_ROOT/physical_gpu.txt"

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
export PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH"
export LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
export PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/openmpi/bin
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

write_runner_status() {
    local rc=$1
    python3 - "$RUN_ROOT" "$rc" \
            "$CURRENT_STAGE" "$SOURCE_REVISION" "$SOURCE_BRANCH" \
            "$INSTANCE" "$GPU_INDEX" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def read_rc(name):
    path = root / name
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None

def sha(name):
    path = root / name
    return hashlib.sha256(path.read_bytes()).hexdigest() \
        if path.is_file() else None

gates = {
    "cleanup": read_rc("cleanup.rc"),
    "service_postflight": read_rc("service_postflight.rc"),
    "preflight_before": read_rc("preflight_before.rc"),
    "preflight_after": read_rc("preflight_after.rc"),
    "preflight_comparison": read_rc("preflight_comparison.rc"),
    "fatal_scan": read_rc("fatal_scan.rc"),
    "timeout_scan": read_rc("timeout_scan.rc"),
}
returncode = int(sys.argv[2])
report = {
    "schema": "qwen36-diagnostic-component-runner-v2",
    "version": 2,
    "qualified": returncode == 0 and all(
        value == 0 for value in gates.values()),
    "returncode": returncode,
    "last_stage": sys.argv[3],
    "source_revision": sys.argv[4],
    "source_branch": sys.argv[5],
    "instance": sys.argv[6],
    "physical_gpu": int(sys.argv[7]),
    "gates": gates,
    "artifact_sha256": {
        name: sha(name) for name in (
            "runtime_identity.json",
            "preflight_before.json",
            "service_postflight.json",
            "preflight_after.json",
            "preflight_comparison.json",
            "fatal_scan.txt",
            "timeout_scan.txt",
            "qualification.json",
        )
    },
    "production_promotion_authorized": False,
}
(root / "runner_status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

stop_active_probe() {
    local can_stop=0
    local child_pid=""
    local cleanup_rc=0
    local current_start=""
    local live_count=0
    local observed_pgid=""

    if [[ -z "$ACTIVE_PGID" && -s "$ACTIVE_READY_FILE" ]]; then
        read -r ACTIVE_PID ACTIVE_START_TIME < "$ACTIVE_READY_FILE"
        if bi100_validate_pid "$ACTIVE_PID" \
                && [[ "$ACTIVE_START_TIME" =~ ^[1-9][0-9]*$ ]]; then
            observed_pgid=$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null \
                | tr -d ' ')
            if [[ -z "$observed_pgid" || "$observed_pgid" == "$ACTIVE_PID" ]]; then
                ACTIVE_PGID=$ACTIVE_PID
            fi
        fi
    fi
    if [[ -z "$ACTIVE_PGID" && -n "$ACTIVE_LAUNCHER_PID" ]]; then
        child_pid=$(ps -o pid= --ppid "$ACTIVE_LAUNCHER_PID" 2>/dev/null \
            | awk 'NF { print $1; exit }')
        if bi100_validate_pid "$child_pid"; then
            observed_pgid=$(ps -o pgid= -p "$child_pid" 2>/dev/null \
                | tr -d ' ')
            if [[ "$observed_pgid" == "$child_pid" ]]; then
                ACTIVE_PID=$child_pid
                ACTIVE_PGID=$child_pid
                ACTIVE_START_TIME=$(awk '{ print $22 }' \
                    "/proc/$child_pid/stat" 2>/dev/null)
            fi
        fi
    fi

    if [[ -n "$ACTIVE_PGID" ]]; then
        if [[ -r "/proc/$ACTIVE_PID/stat" ]]; then
            current_start=$(awk '{ print $22 }' \
                "/proc/$ACTIVE_PID/stat" 2>/dev/null)
            observed_pgid=$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null \
                | tr -d ' ')
            if [[ "$current_start" != "$ACTIVE_START_TIME" \
                    || "$observed_pgid" != "$ACTIVE_PGID" ]]; then
                echo "probe process-group leader identity changed" >&2
                cleanup_rc=2
            else
                can_stop=1
            fi
        else
            live_count=$(bi100_process_group_count \
                "$ACTIVE_PGID" live) || cleanup_rc=2
            if [[ $cleanup_rc -eq 0 && $live_count -gt 0 ]]; then
                if [[ -r "/proc/$ACTIVE_PID/stat" ]]; then
                    echo "probe process-group id was reused" >&2
                    cleanup_rc=2
                else
                    can_stop=1
                fi
            fi
        fi
        if [[ $cleanup_rc -eq 0 && $can_stop -eq 1 ]]; then
            bi100_stop_process_group \
                "$ACTIVE_PGID" "" 60 20 || cleanup_rc=$?
        fi
    elif [[ -n "$ACTIVE_LAUNCHER_PID" ]] \
            && kill -0 "$ACTIVE_LAUNCHER_PID" 2>/dev/null; then
        echo "probe launcher lacks a verified process group" >&2
        kill -TERM "$ACTIVE_LAUNCHER_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$ACTIVE_LAUNCHER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$ACTIVE_LAUNCHER_PID" 2>/dev/null; then
            kill -KILL "$ACTIVE_LAUNCHER_PID" 2>/dev/null || true
        fi
        cleanup_rc=2
    fi
    if [[ -n "$ACTIVE_LAUNCHER_PID" ]]; then
        wait "$ACTIVE_LAUNCHER_PID" 2>/dev/null || true
    fi
    if [[ -n "$ACTIVE_READY_FILE" ]]; then
        rm -f "$ACTIVE_READY_FILE"
    fi
    ACTIVE_LAUNCHER_PID=""
    ACTIVE_PID=""
    ACTIVE_PGID=""
    ACTIVE_START_TIME=""
    ACTIVE_READY_FILE=""
    return "$cleanup_rc"
}

run_physical_preflight() {
    local name=$1
    local rc=0
    timeout --signal=TERM --kill-after=90s 180s \
        python3 "$ROOT/tests/bi100_preflight.py" \
        --gpus "$GPU_INDEX" --timeout-s 25 --matmul-size 1024 \
        --json-out "$RUN_ROOT/${name}.json" \
        > "$RUN_ROOT/${name}.stdout" \
        2> "$RUN_ROOT/${name}.stderr" || rc=$?
    printf '%s\n' "$rc" > "$RUN_ROOT/${name}.rc"
    return "$rc"
}

run_service_postflight() {
    local rc=0
    timeout --signal=TERM --kill-after=70s 240s \
        env -u CUDA_VISIBLE_DEVICES PYTHONPATH="$ROOT/tests" \
        python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus "$GPU_INDEX" \
        --settle-timeout-s 90 --clean-samples 3 \
        --sample-interval-s 2 \
        --out "$RUN_ROOT/service_postflight.json" \
        > "$RUN_ROOT/service_postflight.stdout" \
        2> "$RUN_ROOT/service_postflight.stderr" || rc=$?
    printf '%s\n' "$rc" > "$RUN_ROOT/service_postflight.rc"
    return "$rc"
}

run_preflight_comparison() {
    local rc=0
    timeout --signal=TERM --kill-after=70s 240s \
        env -u CUDA_VISIBLE_DEVICES PYTHONPATH="$ROOT/tests" \
        python3 "$ROOT/tests/compare_bi100_preflights.py" \
        --preflight "before=$RUN_ROOT/preflight_before.json" \
        --preflight "after=$RUN_ROOT/preflight_after.json" \
        --expected-gpus "$GPU_INDEX" \
        --max-free-memory-drop-bytes 1073741824 \
        --out "$RUN_ROOT/preflight_comparison.json" \
        > "$RUN_ROOT/preflight_comparison.stdout" \
        2> "$RUN_ROOT/preflight_comparison.stderr" || rc=$?
    printf '%s\n' "$rc" > "$RUN_ROOT/preflight_comparison.rc"
    return "$rc"
}

scan_fatal_logs() {
    local file
    local found=0
    : > "$RUN_ROOT/fatal_scan.txt"
    while IFS= read -r -d '' file; do
        if grep -Eiq "$FATAL_PATTERN" "$file"; then
            printf 'file=%s\n' "$(basename "$file")" \
                >> "$RUN_ROOT/fatal_scan.txt"
            found=1
        fi
    done < <(find "$RUN_ROOT" -maxdepth 1 -type f \
        \( -name '*.stdout' -o -name '*.stderr' \) -print0)
    printf '%s\n' "$found" > "$RUN_ROOT/fatal_scan.rc"
    return "$found"
}

scan_timeout_rcs() {
    local file
    local value
    local found=0
    : > "$RUN_ROOT/timeout_scan.txt"
    while IFS= read -r -d '' file; do
        value=$(tr -d '[:space:]' < "$file")
        case "$value" in
            124|137)
                printf '%s=%s\n' "$(basename "$file")" "$value" \
                    >> "$RUN_ROOT/timeout_scan.txt"
                found=1
                ;;
        esac
    done < <(find "$RUN_ROOT" -maxdepth 1 -type f -name '*.rc' -print0)
    printf '%s\n' "$found" > "$RUN_ROOT/timeout_scan.rc"
    return "$found"
}

write_postflight_status() {
    python3 - "$RUN_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def read_rc(name):
    path = root / name
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None

gates = {
    "cleanup": read_rc("cleanup.rc"),
    "service_postflight": read_rc("service_postflight.rc"),
    "preflight_before": read_rc("preflight_before.rc"),
    "preflight_after": read_rc("preflight_after.rc"),
    "preflight_comparison": read_rc("preflight_comparison.rc"),
    "fatal_scan": read_rc("fatal_scan.rc"),
    "timeout_scan": read_rc("timeout_scan.rc"),
}
report = {
    "schema": "qwen36-diagnostic-component-postflight-v1",
    "version": 1,
    "qualified": all(value == 0 for value in gates.values()),
    "gates": gates,
    "production_promotion_authorized": False,
}
(root / "postflight_status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
raise SystemExit(0 if report["qualified"] else 1)
PY
}

perform_postflight() {
    local cleanup_rc=0
    local service_rc=0
    local preflight_rc=0
    local comparison_rc=0
    local fatal_rc=0
    local timeout_rc=0
    local status_rc=0

    if [[ "$POSTFLIGHT_COMPLETE" == 1 ]]; then
        return 0
    fi
    stop_active_probe
    cleanup_rc=$?
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/cleanup.rc"
    unset CUDA_VISIBLE_DEVICES

    run_service_postflight
    service_rc=$?
    run_physical_preflight preflight_after
    preflight_rc=$?
    run_preflight_comparison
    comparison_rc=$?
    scan_fatal_logs
    fatal_rc=$?
    scan_timeout_rcs
    timeout_rc=$?
    write_postflight_status
    status_rc=$?
    POSTFLIGHT_COMPLETE=1

    if [[ $cleanup_rc -ne 0 || $service_rc -ne 0 \
            || $preflight_rc -ne 0 || $comparison_rc -ne 0 \
            || $fatal_rc -ne 0 || $timeout_rc -ne 0 \
            || $status_rc -ne 0 ]]; then
        return 1
    fi
    return 0
}

cleanup() {
    local rc=$?
    local fatal_rc=0
    local postflight_rc=0
    local status_rc=0
    local timeout_rc=0
    trap - EXIT TERM INT
    set +e
    if [[ "$PREFLIGHT_ATTEMPTED" == 1 ]]; then
        perform_postflight
        postflight_rc=$?
        scan_fatal_logs
        fatal_rc=$?
        scan_timeout_rcs
        timeout_rc=$?
        write_postflight_status
        status_rc=$?
    else
        stop_active_probe
        postflight_rc=$?
        printf '%s\n' "$postflight_rc" > "$RUN_ROOT/cleanup.rc"
    fi
    if [[ $postflight_rc -ne 0 || $fatal_rc -ne 0 \
            || $timeout_rc -ne 0 || $status_rc -ne 0 ]]; then
        rc=1
    fi
    write_runner_status "$rc"
    status_rc=$?
    if [[ $status_rc -ne 0 ]]; then
        rc=1
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

abort_after_probe_failure() {
    local message=$1
    echo "$message" >&2
    exit 5
}

run_probe() {
    local name=$1
    local report=$2
    local timeout_s=$3
    local cleanup_rc=0
    local rc=0
    shift 3
    CURRENT_STAGE=$name
    ACTIVE_READY_FILE="$RUN_ROOT/.${name}.session"
    rm -f "$ACTIVE_READY_FILE"
    setsid --fork --wait bash -c '
        ready=$1
        timeout_s=$2
        shift 2
        start_time=$(awk "{ print \$22 }" "/proc/$$/stat")
        printf "%s %s\n" "$$" "$start_time" > "$ready"
        exec timeout --signal=TERM --kill-after=60s "$timeout_s" "$@"
    ' bi100-scoped-probe "$ACTIVE_READY_FILE" "$timeout_s" "$@" \
        > "$RUN_ROOT/${name}.stdout" \
        2> "$RUN_ROOT/${name}.stderr" &
    ACTIVE_LAUNCHER_PID=$!

    for _ in $(seq 1 100); do
        if [[ -s "$ACTIVE_READY_FILE" ]]; then
            read -r ACTIVE_PID ACTIVE_START_TIME < "$ACTIVE_READY_FILE"
            if bi100_validate_pid "$ACTIVE_PID" \
                    && [[ "$ACTIVE_START_TIME" =~ ^[1-9][0-9]*$ ]]; then
                ACTIVE_PGID=$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null \
                    | tr -d ' ')
                if [[ "$ACTIVE_PGID" == "$ACTIVE_PID" ]]; then
                    break
                fi
            fi
            ACTIVE_PID=""
            ACTIVE_PGID=""
            ACTIVE_START_TIME=""
        fi
        if ! kill -0 "$ACTIVE_LAUNCHER_PID" 2>/dev/null; then
            break
        fi
        sleep 0.05
    done
    if [[ -z "$ACTIVE_PGID" ]]; then
        stop_active_probe || true
        rc=125
    else
        wait "$ACTIVE_LAUNCHER_PID"
        rc=$?
        stop_active_probe
        cleanup_rc=$?
        if [[ $cleanup_rc -ne 0 && $rc -eq 0 ]]; then
            rc=125
        fi
    fi
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/${name}.cleanup.rc"
    printf '%s\n' "$rc" > "$RUN_ROOT/${name}.rc"
    printf '%s\n' "$rc" > "$RUN_ROOT/${name}.returncode"
    if [[ $rc -ne 0 ]]; then
        abort_after_probe_failure "$name failed (rc=$rc)"
    fi
    if [[ ! -s "$report" ]]; then
        abort_after_probe_failure \
            "$name did not produce its structured report (rc=$rc)"
    fi
}

CURRENT_STAGE=preflight_before
PREFLIGHT_ATTEMPTED=1
if ! run_physical_preflight preflight_before; then
    echo "selected GPU preflight failed" >&2
    exit 4
fi

CURRENT_STAGE=runtime_identity
runtime_identity_rc=0
timeout --signal=TERM --kill-after=10s 180s \
        python3 "$ROOT/tests/verify_bare_host_runtime_identity.py" \
        --source-root "$ROOT" \
        --runtime-site-packages "$BI100_RUNTIME_SITE_PACKAGES" \
        --runtime-install "$RUNTIME_INSTALL" \
        --out "$RUN_ROOT/runtime_identity.json" \
        > "$RUN_ROOT/runtime_identity.stdout" \
        2> "$RUN_ROOT/runtime_identity.stderr" || runtime_identity_rc=$?
printf '%s\n' "$runtime_identity_rc" > "$RUN_ROOT/runtime_identity.rc"
if [[ $runtime_identity_rc -ne 0 ]]; then
    echo "immutable runtime identity failed" >&2
    exit 4
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
QGKV_REPORTS=()
for rank in 0 1 2 3; do
    report="$RUN_ROOT/qgkv_rank${rank}.json"
    QGKV_REPORTS+=("$report")
    run_probe "qgkv_rank${rank}" "$report" 240s \
        python3 "$ROOT/tests/bi100_full_attention_qgkv_runtime.py" \
        --device cuda:0 --tp-rank "$rank" --out "$report"
done

run_probe moe "$RUN_ROOT/moe.json" 1200s \
    python3 "$ROOT/tests/bench_moe_direct_routed.py" \
    --direct-extension "$VLLM_ROOT/corex_moe_direct_routed.so" \
    --gather-extension "$VLLM_ROOT/corex_moe_weight_gather.so" \
    --reduce-extension "$VLLM_ROOT/corex_moe_exact_reduce.so" \
    --device cuda:0 --out "$RUN_ROOT/moe.json"

run_probe gdn "$RUN_ROOT/gdn.json" 1200s \
    python3 "$ROOT/tests/bench_gdn_packed_production_boundary.py" \
    --packed-extension "$VLLM_ROOT/corex_gdn_packed_decode.so" \
    --beta-decay-extension "$VLLM_ROOT/corex_gdn_beta_decay.so" \
    --qk-map-extension "$VLLM_ROOT/corex_gdn_qk_map.so" \
    --device cuda:0 --out "$RUN_ROOT/gdn.json"

run_probe paged_kv "$RUN_ROOT/paged_kv.json" 1800s \
    python3 "$ROOT/tests/bench_paged_kv_gather.py" \
    --extension "$VLLM_ROOT/corex_paged_kv_gather.so" \
    --device cuda:0 --lengths 32768,65536,131072,235000 \
    --out "$RUN_ROOT/paged_kv.json"

run_probe cache_engine "$RUN_ROOT/cache_engine.json" 900s \
    python3 "$ROOT/tests/bench_m1_57_cache_engine_integration.py" \
    --device cuda:0 --source-revision "$SOURCE_REVISION" \
    --instance "$INSTANCE" --out "$RUN_ROOT/cache_engine.json"

unset CUDA_VISIBLE_DEVICES
CURRENT_STAGE=postflight
if ! perform_postflight; then
    echo "diagnostic component lifecycle postflight failed" >&2
    exit 5
fi

QUALIFY_ARGS=(
    --moe "$RUN_ROOT/moe.json"
    --gdn "$RUN_ROOT/gdn.json"
    --paged "$RUN_ROOT/paged_kv.json"
    --cache "$RUN_ROOT/cache_engine.json"
    --preflight-before "$RUN_ROOT/preflight_before.json"
    --preflight-after "$RUN_ROOT/preflight_after.json"
    --runtime-identity "$RUN_ROOT/runtime_identity.json"
    --source-revision "$SOURCE_REVISION"
    --source-branch "$SOURCE_BRANCH"
    --instance "$INSTANCE"
    --physical-gpu "$GPU_INDEX"
    --out "$RUN_ROOT/qualification.json"
)
for report in "${QGKV_REPORTS[@]}"; do
    QUALIFY_ARGS+=(--qgkv "$report")
done
for log in "$RUN_ROOT"/*.stdout "$RUN_ROOT"/*.stderr; do
    QUALIFY_ARGS+=(--log "$log")
done

CURRENT_STAGE=qualification
qualification_rc=0
timeout --signal=TERM --kill-after=10s 180s \
    python3 "$ROOT/tests/qualify_qwen36_diagnostic_components.py" \
        "${QUALIFY_ARGS[@]}" \
        > "$RUN_ROOT/qualification.stdout" \
        2> "$RUN_ROOT/qualification.stderr" || qualification_rc=$?
printf '%s\n' "$qualification_rc" > "$RUN_ROOT/qualification.rc"
if [[ $qualification_rc -ne 0 ]]; then
    echo "diagnostic component qualification rejected the result" >&2
    exit 6
fi

CURRENT_STAGE=final_audit
final_audit_rc=0
scan_fatal_logs || final_audit_rc=1
scan_timeout_rcs || final_audit_rc=1
write_postflight_status || final_audit_rc=1
if [[ $final_audit_rc -ne 0 ]]; then
    echo "diagnostic component final audit failed" >&2
    exit 6
fi

CURRENT_STAGE=completed
exit 0
