#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

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
case "$RUN_ROOT/" in
    "$ROOT/"*)
        echo "M1-99 output must stay outside the source repository" >&2
        exit 2
        ;;
esac
if [[ "$RUN_ROOT" != /tmp/* ]]; then
    echo "M1-99 output must use a private /tmp path" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
    echo "M1-99 output already exists: $RUN_ROOT" >&2
    exit 2
fi

MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
ACTIVE_PID=""
ACTIVE_PGID=""
ACTIVE_STARTTIME=""
ACTIVE_SESSION_TOKEN=""
ACTIVE_LABEL=""
BEFORE_PREFLIGHT_PASSED=0
CURRENT_STAGE=argument_validation

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
COREX_LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
COREX_PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
        . ':(exclude)bench_runs/**')" ]]; then
    echo "M1-99 runner refuses a dirty source tree" >&2
    exit 3
fi
if [[ -z "${BI100_RUNTIME_SITE_PACKAGES:-}" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/vllm" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/transformers" ]]; then
    echo "M1-99 requires an immutable bare-host runtime overlay" >&2
    exit 3
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model directory is missing: $MODEL_PATH" >&2
    exit 3
fi
if pgrep -f 'vllm\.entrypoints\.openai\.api_server' >/dev/null 2>&1; then
    echo "an API server process is already running" >&2
    exit 3
fi

BI100_RUNTIME_SITE_PACKAGES=$(python3 - \
        "$BI100_RUNTIME_SITE_PACKAGES" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=True))
PY
)
MODEL_PATH=$(python3 - "$MODEL_PATH" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=True))
PY
)
RUNTIME_ROOT=$(dirname "$BI100_RUNTIME_SITE_PACKAGES")
RUNTIME_INSTALL=${BI100_RUNTIME_INSTALL_REPORT:-${RUNTIME_INSTALL_REPORT:-$RUNTIME_ROOT/install.json}}
if [[ ! -f "$RUNTIME_INSTALL" ]]; then
    echo "M1-99 runtime install report is missing: $RUNTIME_INSTALL" >&2
    exit 3
fi

mkdir -p "$RUN_ROOT"
SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
SOURCE_BRANCH=$(git -C "$ROOT" branch --show-current)
printf '%s\n' "$SOURCE_REVISION" > "$RUN_ROOT/source_revision.txt"
printf '%s\n' "$SOURCE_BRANCH" > "$RUN_ROOT/source_branch.txt"
printf '%s\n' "$INSTANCE" > "$RUN_ROOT/instance.txt"
printf '%s\n' "$MODEL_PATH" > "$RUN_ROOT/model_path.txt"
printf '%s\n' "$BI100_RUNTIME_SITE_PACKAGES" \
    > "$RUN_ROOT/runtime_site_packages.txt"

export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
unset CUDA_VISIBLE_DEVICES

read_process_starttime() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys

value = (Path("/proc") / sys.argv[1] / "stat").read_text(encoding="ascii")
fields = value[value.rfind(")") + 2:].split()
print(fields[19])
PY
}

active_pid_is_live() {
    [[ -n "$ACTIVE_PID" && -n "$ACTIVE_STARTTIME" ]] || return 1
    python3 - "$ACTIVE_PID" "$ACTIVE_STARTTIME" <<'PY'
from pathlib import Path
import sys

path = Path("/proc") / sys.argv[1] / "stat"
try:
    value = path.read_text(encoding="ascii")
except (FileNotFoundError, ProcessLookupError):
    raise SystemExit(1)
fields = value[value.rfind(")") + 2:].split()
raise SystemExit(
    0 if fields[0] != "Z" and fields[19] == sys.argv[2] else 1)
PY
}

stop_active_group() {
    local rc=0
    if [[ -z "$ACTIVE_PID" ]]; then
        return 0
    fi
    if [[ -n "$ACTIVE_PGID" ]]; then
        bi100_stop_process_group \
            "$ACTIVE_PGID" "$ACTIVE_PID" 60 20 \
            "$ACTIVE_STARTTIME" "$ACTIVE_SESSION_TOKEN" || rc=$?
    else
        if active_pid_is_live; then
            kill -TERM "$ACTIVE_PID" 2>/dev/null || true
            for _ in $(seq 1 60); do
                active_pid_is_live || break
                sleep 1
            done
        fi
        if active_pid_is_live; then
            kill -KILL "$ACTIVE_PID" 2>/dev/null || true
            for _ in $(seq 1 20); do
                active_pid_is_live || break
                sleep 1
            done
        fi
        wait "$ACTIVE_PID" 2>/dev/null || true
        if active_pid_is_live; then
            echo "$ACTIVE_LABEL process survived scoped cleanup" >&2
            rc=1
        fi
    fi
    if [[ $rc -eq 0 ]]; then
        ACTIVE_PID=""
        ACTIVE_PGID=""
        ACTIVE_STARTTIME=""
        ACTIVE_SESSION_TOKEN=""
        ACTIVE_LABEL=""
    fi
    return "$rc"
}

health() {
    python3 - <<'PY' >/dev/null 2>&1
import urllib.request

urllib.request.urlopen(
    "http://127.0.0.1:8000/health", timeout=5).read()
PY
}

port_is_free() {
    python3 - <<'PY' >/dev/null 2>&1
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 8000))
PY
}

wait_for_port_free() {
    for _ in $(seq 1 180); do
        port_is_free && return 0
        sleep 1
    done
    echo "port 8000 remained busy after scoped service cleanup" >&2
    return 1
}

run_preflight() {
    local output=$1
    timeout --signal=TERM --kill-after=70s 480s \
        env PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
        LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
        python3 "$ROOT/tests/bi100_preflight.py" \
        --gpus 0,1,2,3 --timeout-s 25 --matmul-size 1024 \
        --json-out "$output.json" \
        > "$output.stdout" 2> "$output.stderr"
}

run_postflight() {
    local output=$1
    timeout --signal=TERM --kill-after=70s 240s \
        env PYTHONPATH="$ROOT/tests" \
        python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus 0,1,2,3 \
        --settle-timeout-s 90 --clean-samples 3 \
        --sample-interval-s 2 \
        --out "$output.json" \
        > "$output.stdout" 2> "$output.stderr"
}

scan_log() {
    local log=$1
    local output=$2
    local pattern
    pattern='CUDA error|illegal memory access|SIGSEGV|Fatal Python error|out of memory|device-side assert|AssertionError|CoreX.*(failed|fatal)|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|Timeout(Error|Expired)|engine iteration timed out|watchdog.*tim(e|ed) out|scheduler requested a missing GDN prefix state|non-finite GatedDeltaNet'
    if [[ ! -f "$log" ]]; then
        echo "missing service log: $log" > "$output"
        return 1
    fi
    if grep -Eiq "$pattern" "$log"; then
        grep -Ein "$pattern" "$log" > "$output" || true
        return 1
    fi
    : > "$output"
}

start_service() {
    local arm=$1
    local selector=$2
    local identity=$arm/service_identity.json
    local observed=""

    (
        exec env \
            BI100_RUNTIME_SITE_PACKAGES="$BI100_RUNTIME_SITE_PACKAGES" \
            BI100_RUNTIME_INSTALL_REPORT="$RUNTIME_INSTALL" \
            BI100_RUNTIME_WORKDIR="$arm/runtime-workdir" \
            MODEL_PATH="$MODEL_PATH" \
            HOST=0.0.0.0 PORT=8000 \
            ENABLE_CUSTOM_IPC=1 \
            VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
            BI100_MOE_COREX_DIRECT_ROUTED=1 \
            BI100_GDN_COREX_PACKED_DECODE=1 \
            BI100_GDN_COMBINED_QK_NORM=0 \
            BI100_GDN_CACHE_POLICY=admission64 \
            BI100_GDN_RESTORE_MODE=hybrid64 \
            BI100_HYBRID_KV_ACCOUNTING=full_attention \
            BI100_CPU_KV_OFFLOAD=0 \
            BI100_BLOCK_MAJOR_CPU_KV=0 \
            BI100_CACHE_TRACE=0 \
            BI100_ATTN_COREX_FUSED_PREFILL="$selector" \
            BI100_ATTN_COREX_FUSED_PREFILL_DIAGNOSTICS=0 \
            BI100_PROFILE=0 \
            BI100_PROFILE_INCLUDE_STARTUP=0 \
            BI100_PAGED_ATTN_DIAGNOSTICS=0 \
            BI100_GDN_ALLOW_NAN_ZERO=0 \
            BI100_GDN_FINITE_CHECK=0 \
            PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH" \
            LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" \
            PATH="$COREX_PATH" \
            python3 "$ROOT/scripts/exec_bi100_session.py" \
            "$identity" -- "$ROOT/launch_service"
    ) > "$arm/server.log" 2>&1 &
    ACTIVE_PID=$!
    ACTIVE_LABEL=$(basename "$arm")
    ACTIVE_STARTTIME=""
    for _ in $(seq 1 50); do
        ACTIVE_STARTTIME=$(
            read_process_starttime "$ACTIVE_PID" 2>/dev/null || true)
        [[ -n "$ACTIVE_STARTTIME" ]] && break
        kill -0 "$ACTIVE_PID" 2>/dev/null || break
        sleep 0.1
    done
    if [[ -z "$ACTIVE_STARTTIME" ]]; then
        echo "$ACTIVE_LABEL service leader disappeared before attestation" >&2
        return 1
    fi

    for _ in $(seq 1 100); do
        if [[ -s "$identity" ]]; then
            observed=$(python3 - "$identity" "$ACTIVE_PID" \
                    "$ACTIVE_STARTTIME" <<'PY' 2>/dev/null || true
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pid = int(sys.argv[2])
starttime = int(sys.argv[3])
token = value.get("session_token")
if (
    value.get("schema") != "bi100-process-session-v1"
    or value.get("version") != 1
    or value.get("pid") != pid
    or value.get("pgid") != pid
    or value.get("sid") != pid
    or value.get("starttime_ticks") != starttime
    or not isinstance(token, str)
    or len(token) != 32
    or any(character not in "0123456789abcdef" for character in token)
):
    raise SystemExit(1)
print(value["pgid"], token)
PY
            )
            [[ -n "$observed" ]] && break
        fi
        active_pid_is_live || break
        sleep 0.1
    done
    if [[ -z "$observed" ]]; then
        echo "$ACTIVE_LABEL service identity attestation failed" >&2
        return 1
    fi
    read -r ACTIVE_PGID ACTIVE_SESSION_TOKEN <<< "$observed"
    printf '%s\n' "$ACTIVE_PID" > "$arm/server.pid"
    printf '%s\n' "$ACTIVE_PGID" > "$arm/server.pgid"

    for _ in $(seq 1 360); do
        if health; then
            return 0
        fi
        active_pid_is_live || break
        sleep 10
    done
    echo "$ACTIVE_LABEL service did not become healthy within 3600 seconds" >&2
    return 1
}

write_arm_status() {
    local arm=$1
    local label=$2
    local pair=$3
    local selector=$4
    local final_rc=$5
    python3 - "$arm" "$label" "$pair" "$selector" "$final_rc" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def rc(name):
    path = root / name
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return int(value) if value.isdigit() else None

def sha(name):
    path = root / name
    return hashlib.sha256(path.read_bytes()).hexdigest() \
        if path.is_file() else None

report = {
    "schema": "bi100-m1-99-fused-prefill-arm-v1",
    "version": 1,
    "qualified": int(sys.argv[5]) == 0,
    "label": sys.argv[2],
    "pair": int(sys.argv[3]),
    "fused_prefill": int(sys.argv[4]),
    "gates": {
        "preflight_before": rc("preflight_before.rc"),
        "startup": rc("startup.rc"),
        "startup_contract": rc("startup_contract.rc"),
        "measurement": rc("measurement.rc"),
        "health_after": rc("health_after.rc"),
        "dispatch": rc("dispatch.rc"),
        "cleanup": rc("cleanup.rc"),
        "fatal_scan": rc("fatal_scan.rc"),
        "service_postflight": rc("service_postflight.rc"),
        "preflight_after": rc("preflight_after.rc"),
        "preflight_comparison": rc("preflight_comparison.rc"),
    },
    "artifact_sha256": {
        "measurement": sha("measurement.json"),
        "service_identity": sha("service_identity.json"),
        "service_postflight": sha("service_postflight.json"),
        "preflight_comparison": sha("preflight_comparison.json"),
    },
    "full_model_used": True,
    "model_quality_evaluated": False,
    "production_promotion_authorized": False,
}
(root / "status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8")
PY
}

run_arm() {
    local pair=$1
    local label=$2
    local selector=$3
    local arm=$RUN_ROOT/pair${pair}_${label}
    local arm_rc=0
    local rc=0
    local startup_rc=1
    local contract_rc=1
    local measurement_rc=1
    local health_rc=1
    local dispatch_rc=1
    local cleanup_rc=1
    local fatal_rc=1
    local postflight_rc=1
    local preflight_after_rc=1
    local comparison_rc=1
    local dispatch_count=0
    local run_id="m1-99-pair-${pair}-20260728"

    mkdir -p "$arm/runtime-workdir"
    set +e
    run_preflight "$arm/preflight_before"
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$arm/preflight_before.rc"
    if [[ $rc -ne 0 ]]; then
        write_arm_status "$arm" "$label" "$pair" "$selector" 1
        return 1
    fi

    set +e
    wait_for_port_free
    rc=$?
    if [[ $rc -eq 0 ]]; then
        start_service "$arm" "$selector"
        startup_rc=$?
    fi
    set -e
    printf '%s\n' "$startup_rc" > "$arm/startup.rc"
    [[ $startup_rc -eq 0 ]] || arm_rc=1

    if [[ $startup_rc -eq 0 ]]; then
        contract_rc=0
        grep -Fq '[BI100] fixed evaluator contract;' \
            "$arm/server.log" || contract_rc=1
        grep -Fq '[BI100] fixed kernels; moe_direct=1 gdn_packed=1' \
            "$arm/server.log" || contract_rc=1
        grep -Fq '[BI100] GDN cache; policy=admission64 restore=hybrid64' \
            "$arm/server.log" || contract_rc=1
        grep -Fq 'accounting=full_attention' \
            "$arm/server.log" || contract_rc=1
        grep -Fq "fused_prefill=$selector" \
            "$arm/server.log" || contract_rc=1
    fi
    printf '%s\n' "$contract_rc" > "$arm/startup_contract.rc"
    [[ $contract_rc -eq 0 ]] || arm_rc=1

    if [[ $contract_rc -eq 0 ]]; then
        set +e
        timeout --signal=TERM --kill-after=60s 10800s \
            env PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH" \
            LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
            python3 "$ROOT/tests/bench_fused_prefill_service.py" \
            --base http://127.0.0.1:8000 \
            --model-path "$MODEL_PATH" \
            --targets 65536,235000 --max-tokens 32 \
            --timeout-s 1800 --run-id "$run_id" \
            --mode "$label" --out "$arm/measurement.json" \
            > "$arm/measurement.stdout" \
            2> "$arm/measurement.stderr"
        measurement_rc=$?
        set -e
    fi
    printf '%s\n' "$measurement_rc" > "$arm/measurement.rc"
    [[ $measurement_rc -eq 0 ]] || arm_rc=1

    if [[ $measurement_rc -eq 0 ]]; then
        set +e
        health
        health_rc=$?
        set -e
    fi
    printf '%s\n' "$health_rc" > "$arm/health_after.rc"
    [[ $health_rc -eq 0 ]] || arm_rc=1

    if [[ -f "$arm/server.log" ]]; then
        dispatch_count=$(grep -Fc 'path=corex_split4' \
            "$arm/server.log" || true)
        printf '%s\n' "$dispatch_count" > "$arm/dispatch_count.txt"
        if [[ "$selector" == 1 && "$dispatch_count" -ge 4 ]]; then
            dispatch_rc=0
        elif [[ "$selector" == 0 && "$dispatch_count" -eq 0 ]]; then
            dispatch_rc=0
        fi
    fi
    printf '%s\n' "$dispatch_rc" > "$arm/dispatch.rc"
    [[ $dispatch_rc -eq 0 ]] || arm_rc=1

    set +e
    stop_active_group
    cleanup_rc=$?
    if [[ $cleanup_rc -eq 0 ]]; then
        wait_for_port_free
        cleanup_rc=$?
    fi
    set -e
    printf '%s\n' "$cleanup_rc" > "$arm/cleanup.rc"
    [[ $cleanup_rc -eq 0 ]] || arm_rc=1

    set +e
    scan_log "$arm/server.log" "$arm/fatal_scan.txt"
    fatal_rc=$?
    set -e
    printf '%s\n' "$fatal_rc" > "$arm/fatal_scan.rc"
    [[ $fatal_rc -eq 0 ]] || arm_rc=1

    set +e
    run_postflight "$arm/service_postflight"
    postflight_rc=$?
    set -e
    printf '%s\n' "$postflight_rc" > "$arm/service_postflight.rc"
    [[ $postflight_rc -eq 0 ]] || arm_rc=1

    if [[ $cleanup_rc -eq 0 && $postflight_rc -eq 0 ]]; then
        set +e
        run_preflight "$arm/preflight_after"
        preflight_after_rc=$?
        set -e
    fi
    printf '%s\n' "$preflight_after_rc" > "$arm/preflight_after.rc"
    if [[ $preflight_after_rc -eq 0 ]]; then
        set +e
        python3 "$ROOT/tests/compare_bi100_preflights.py" \
            --preflight "before=$arm/preflight_before.json" \
            --preflight "after=$arm/preflight_after.json" \
            --expected-gpus 0,1,2,3 \
            --max-free-memory-drop-bytes 1073741824 \
            --out "$arm/preflight_comparison.json" \
            > "$arm/preflight_comparison.stdout" \
            2> "$arm/preflight_comparison.stderr"
        comparison_rc=$?
        set -e
    fi
    printf '%s\n' "$comparison_rc" > "$arm/preflight_comparison.rc"
    [[ $preflight_after_rc -eq 0 && $comparison_rc -eq 0 ]] || arm_rc=1

    write_arm_status "$arm" "$label" "$pair" "$selector" "$arm_rc"
    return "$arm_rc"
}

write_runner_status() {
    local final_rc=$1
    python3 - "$RUN_ROOT" "$INSTANCE" "$SOURCE_REVISION" \
            "$SOURCE_BRANCH" "$CURRENT_STAGE" "$final_rc" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def rc(name):
    path = root / name
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return int(value) if value.isdigit() else None

def sha(name):
    path = root / name
    return hashlib.sha256(path.read_bytes()).hexdigest() \
        if path.is_file() else None

comparison = {}
path = root / "comparison.json"
if path.is_file():
    try:
        comparison = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        comparison = {}

arm_labels = [
    "pair1_control", "pair1_candidate",
    "pair2_candidate", "pair2_control",
    "pair3_control", "pair3_candidate",
]
arm_valid = all(rc(f"{label}.rc") == 0 for label in arm_labels)
candidate_screen_passed = comparison.get("qualified") is True
report = {
    "schema": "bi100-m1-99-fused-prefill-runner-v1",
    "version": 1,
    "qualified": int(sys.argv[6]) == 0,
    "measurement_valid": arm_valid,
    "candidate_screen_passed": candidate_screen_passed,
    "source_revision": sys.argv[3],
    "source_branch": sys.argv[4],
    "instance": sys.argv[2],
    "terminal_stage": sys.argv[5],
    "returncode": int(sys.argv[6]),
    "gates": {
        "postflight_before": rc("postflight_before.rc"),
        "preflight_before": rc("preflight_before.rc"),
        "runtime_identity": rc("runtime_identity.rc"),
        **{label: rc(f"{label}.rc") for label in arm_labels},
        "comparison": rc("comparison.rc"),
        "scoped_cleanup": rc("scoped_cleanup.rc"),
        "scoped_cleanup_clean": rc("scoped_cleanup_clean.rc"),
        "source_unchanged": rc("source_unchanged.rc"),
        "fatal_scan": rc("fatal_scan.rc"),
        "timeout_scan": rc("timeout_scan.rc"),
        "final_postflight": rc("final_postflight.rc"),
        "final_preflight": rc("final_preflight.rc"),
        "final_preflight_comparison": rc(
            "final_preflight_comparison.rc"),
    },
    "artifact_sha256": {
        "runtime_identity": sha("runtime_identity.json"),
        "comparison": sha("comparison.json"),
        "scoped_cleanup_clean": sha("scoped_cleanup_clean.json"),
        "final_postflight": sha("final_postflight.json"),
        "final_preflight_comparison": sha(
            "final_preflight_comparison.json"),
    },
    "decision": {
        "full_tp4_quality_gate_authorized": candidate_screen_passed,
        "official_style_replay_authorized": False,
        "production_promotion_authorized": False,
        "yaml_change_authorized": False,
        "main_merge_authorized": False,
    },
    "official_881_evaluated": False,
    "full_model_quality_suite_evaluated": False,
}
(root / "runner_status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8")
PY
}

finish() {
    local primary_rc=$?
    local final_rc=$primary_rc
    local cleanup_rc=0
    local cleanup_clean_rc=0
    local source_rc=0
    local fatal_rc=0
    local timeout_rc=0
    local postflight_rc=0
    local preflight_rc=0
    local comparison_rc=0
    local artifact
    local identity
    local value
    local pattern
    local identity_args=()
    local expected_args=()
    trap - EXIT
    trap '' INT TERM
    set +e

    stop_active_group || cleanup_rc=1
    unset CUDA_VISIBLE_DEVICES

    while IFS= read -r -d '' identity; do
        identity_args+=(--identity "$identity")
        expected_args+=(--expected-identity "$identity")
    done < <(find "$RUN_ROOT" -type f -name service_identity.json \
        -print0 | sort -z)
    if [[ ${#identity_args[@]} -gt 0 ]]; then
        timeout --signal=TERM --kill-after=70s 600s \
            python3 "$ROOT/scripts/cleanup_recorded_bi100_sessions.py" \
            "${identity_args[@]}" \
            --out "$RUN_ROOT/scoped_cleanup.json" \
            > "$RUN_ROOT/scoped_cleanup.stdout" \
            2> "$RUN_ROOT/scoped_cleanup.stderr"
        [[ $? -eq 0 ]] || cleanup_rc=1
        python3 "$ROOT/tests/qualify_recorded_session_cleanup.py" \
            "$RUN_ROOT/scoped_cleanup.json" \
            "${expected_args[@]}" \
            --out "$RUN_ROOT/scoped_cleanup_clean.json" \
            > "$RUN_ROOT/scoped_cleanup_clean.stdout" \
            2> "$RUN_ROOT/scoped_cleanup_clean.stderr"
        cleanup_clean_rc=$?
    else
        printf '%s\n' \
            '{"schema":"bi100-no-recorded-session","qualified":true}' \
            > "$RUN_ROOT/scoped_cleanup.json"
        cp "$RUN_ROOT/scoped_cleanup.json" \
            "$RUN_ROOT/scoped_cleanup_clean.json"
    fi
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/scoped_cleanup.rc"
    printf '%s\n' "$cleanup_clean_rc" \
        > "$RUN_ROOT/scoped_cleanup_clean.rc"

    current_revision=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)
    current_status=$(git -C "$ROOT" status --porcelain \
        --untracked-files=all -- . ':(exclude)bench_runs/**')
    if [[ "$current_revision" != "$SOURCE_REVISION" \
            || -n "$current_status" ]]; then
        source_rc=1
    fi
    printf '%s\n' "$source_rc" > "$RUN_ROOT/source_unchanged.rc"

    pattern='CUDA error|illegal memory access|SIGSEGV|Fatal Python error|out of memory|device-side assert|AssertionError|CoreX.*(failed|fatal)|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|Timeout(Error|Expired)|engine iteration timed out|watchdog.*tim(e|ed) out|scheduler requested a missing GDN prefix state|non-finite GatedDeltaNet'
    : > "$RUN_ROOT/fatal_scan.txt"
    while IFS= read -r -d '' artifact; do
        if grep -Eiq "$pattern" "$artifact"; then
            printf 'file=%s\n' "$artifact" >> "$RUN_ROOT/fatal_scan.txt"
            grep -Ein "$pattern" "$artifact" \
                >> "$RUN_ROOT/fatal_scan.txt" || true
            fatal_rc=1
        fi
    done < <(find "$RUN_ROOT" -type f \
        \( -name '*.log' -o -name '*.stdout' -o -name '*.stderr' \) \
        -print0)
    printf '%s\n' "$fatal_rc" > "$RUN_ROOT/fatal_scan.rc"

    : > "$RUN_ROOT/timeout_scan.txt"
    while IFS= read -r -d '' artifact; do
        value=$(tr -d '[:space:]' < "$artifact")
        case "$value" in
            124|137|143)
                printf '%s=%s\n' "$artifact" "$value" \
                    >> "$RUN_ROOT/timeout_scan.txt"
                timeout_rc=1
                ;;
        esac
    done < <(find "$RUN_ROOT" -type f -name '*.rc' -print0)
    printf '%s\n' "$timeout_rc" > "$RUN_ROOT/timeout_scan.rc"

    run_postflight "$RUN_ROOT/final_postflight"
    postflight_rc=$?
    printf '%s\n' "$postflight_rc" > "$RUN_ROOT/final_postflight.rc"

    if [[ "$BEFORE_PREFLIGHT_PASSED" == 1 && $cleanup_rc -eq 0 \
            && $cleanup_clean_rc -eq 0 && $postflight_rc -eq 0 ]]; then
        run_preflight "$RUN_ROOT/final_preflight"
        preflight_rc=$?
        printf '%s\n' "$preflight_rc" > "$RUN_ROOT/final_preflight.rc"
        if [[ $preflight_rc -eq 0 ]]; then
            python3 "$ROOT/tests/compare_bi100_preflights.py" \
                --preflight "before=$RUN_ROOT/preflight_before.json" \
                --preflight "after=$RUN_ROOT/final_preflight.json" \
                --expected-gpus 0,1,2,3 \
                --max-free-memory-drop-bytes 1073741824 \
                --out "$RUN_ROOT/final_preflight_comparison.json" \
                > "$RUN_ROOT/final_preflight_comparison.stdout" \
                2> "$RUN_ROOT/final_preflight_comparison.stderr"
            comparison_rc=$?
        else
            comparison_rc=1
        fi
        printf '%s\n' "$comparison_rc" \
            > "$RUN_ROOT/final_preflight_comparison.rc"
    fi

    if [[ $cleanup_rc -ne 0 || $cleanup_clean_rc -ne 0 \
            || $source_rc -ne 0 || $fatal_rc -ne 0 \
            || $timeout_rc -ne 0 || $postflight_rc -ne 0 \
            || $preflight_rc -ne 0 || $comparison_rc -ne 0 ]]; then
        final_rc=1
    fi
    printf '%s\n' "$final_rc" > "$RUN_ROOT/overall.rc"
    write_runner_status "$final_rc"
    exit "$final_rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CURRENT_STAGE=postflight_before
set +e
run_postflight "$RUN_ROOT/postflight_before"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/postflight_before.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=preflight_before
set +e
run_preflight "$RUN_ROOT/preflight_before"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/preflight_before.rc"
[[ $rc -eq 0 ]]
BEFORE_PREFLIGHT_PASSED=1

CURRENT_STAGE=runtime_identity
set +e
timeout --signal=TERM --kill-after=70s 240s \
    env PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH" \
    LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
    python3 "$ROOT/tests/verify_bare_host_runtime_identity.py" \
    --source-root "$ROOT" \
    --runtime-site-packages "$BI100_RUNTIME_SITE_PACKAGES" \
    --runtime-install "$RUNTIME_INSTALL" \
    --out "$RUN_ROOT/runtime_identity.json" \
    > "$RUN_ROOT/runtime_identity.stdout" \
    2> "$RUN_ROOT/runtime_identity.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/runtime_identity.rc"
[[ $rc -eq 0 ]]

for specification in \
        "1 control 0" \
        "1 candidate 1" \
        "2 candidate 1" \
        "2 control 0" \
        "3 control 0" \
        "3 candidate 1"; do
    read -r pair label selector <<< "$specification"
    CURRENT_STAGE="pair${pair}_${label}"
    set +e
    run_arm "$pair" "$label" "$selector"
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$RUN_ROOT/${CURRENT_STAGE}.rc"
    [[ $rc -eq 0 ]]
done

CURRENT_STAGE=comparison
set +e
python3 "$ROOT/tests/compare_m1_99_fused_prefill_paired_ab.py" \
    --control "$RUN_ROOT/pair1_control/measurement.json" \
    --control "$RUN_ROOT/pair2_control/measurement.json" \
    --control "$RUN_ROOT/pair3_control/measurement.json" \
    --candidate "$RUN_ROOT/pair1_candidate/measurement.json" \
    --candidate "$RUN_ROOT/pair2_candidate/measurement.json" \
    --candidate "$RUN_ROOT/pair3_candidate/measurement.json" \
    --out "$RUN_ROOT/comparison.json" \
    > "$RUN_ROOT/comparison.stdout" \
    2> "$RUN_ROOT/comparison.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/comparison.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=complete
echo "M1-99 three-pair TP4 performance screen passed; full quality is next"
