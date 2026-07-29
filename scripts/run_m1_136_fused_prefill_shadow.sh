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
SHADOW_VARIANT=${BI100_FUSED_PREFILL_SHADOW_VARIANT:-legacy}
case "$SHADOW_VARIANT" in
    legacy)
        EXPERIMENT_LABEL=M1-136
        RUN_ID_PREFIX=m1-136-shadow
        NUMERIC_MODE=legacy
        FAILURE_ACTION=raise
        QUALIFIER="$ROOT/tests/qualify_fused_prefill_shadow.py"
        CONTRACT="$ROOT/quality/layered_quality_gate.v2.json"
        RUNNER_SCHEMA=bi100-m1-136-fused-prefill-shadow-runner-v1
        ;;
    calibrated)
        EXPERIMENT_LABEL=M1-138
        RUN_ID_PREFIX=m1-138-calibrated-shadow
        NUMERIC_MODE=calibrated
        FAILURE_ACTION=record
        QUALIFIER="$ROOT/tests/qualify_fused_prefill_calibrated_shadow.py"
        CONTRACT="$ROOT/quality/fused_prefill_numeric_adjudication.v1.json"
        RUNNER_SCHEMA=bi100-m1-138-fused-prefill-calibrated-shadow-runner-v1
        ;;
    *)
        echo "BI100_FUSED_PREFILL_SHADOW_VARIANT is invalid" >&2
        exit 2
        ;;
esac
RUN_ROOT=$(python3 - "$2" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)
if [[ "$RUN_ROOT/" == "$ROOT/"* ]]; then
    echo "$EXPERIMENT_LABEL output must stay outside the source repository" \
        >&2
    exit 2
fi
if [[ "$RUN_ROOT" != /tmp/* ]]; then
    echo "$EXPERIMENT_LABEL output must use a private /tmp path" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
    echo "$EXPERIMENT_LABEL output already exists: $RUN_ROOT" >&2
    exit 2
fi

MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
COREX_LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
COREX_PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
        . ':(exclude)bench_runs/**')" ]]; then
    echo "$EXPERIMENT_LABEL runner refuses a dirty source tree" >&2
    exit 3
fi
if [[ -z "${BI100_RUNTIME_SITE_PACKAGES:-}" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/vllm" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/transformers" ]]; then
    echo "$EXPERIMENT_LABEL requires an immutable bare-host runtime overlay" \
        >&2
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
RUNTIME_INSTALL=${BI100_RUNTIME_INSTALL_REPORT:-$RUNTIME_ROOT/install.json}
if [[ ! -f "$RUNTIME_INSTALL" ]]; then
    echo "$EXPERIMENT_LABEL runtime install report is missing: " \
        "$RUNTIME_INSTALL" >&2
    exit 3
fi

mkdir -p "$RUN_ROOT/runtime-workdir" "$RUN_ROOT/shadow"
SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
SOURCE_BRANCH=$(git -C "$ROOT" branch --show-current)
RUN_ID="${RUN_ID_PREFIX}-${SOURCE_REVISION:0:12}"
printf '%s\n' "$SOURCE_REVISION" > "$RUN_ROOT/source_revision.txt"
printf '%s\n' "$SOURCE_BRANCH" > "$RUN_ROOT/source_branch.txt"
printf '%s\n' "$INSTANCE" > "$RUN_ROOT/instance.txt"
printf '%s\n' "$MODEL_PATH" > "$RUN_ROOT/model_path.txt"
printf '%s\n' "$BI100_RUNTIME_SITE_PACKAGES" \
    > "$RUN_ROOT/runtime_site_packages.txt"

ACTIVE_PID=""
ACTIVE_PGID=""
ACTIVE_STARTTIME=""
ACTIVE_SESSION_TOKEN=""
BEFORE_PREFLIGHT_PASSED=0
CURRENT_STAGE=initialization

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

wait_for_recorded_group_empty() {
    local pgid=$1
    local attempts=$2
    local live_count
    local zombie_count
    for _ in $(seq 1 "$attempts"); do
        live_count=$(bi100_process_group_count "$pgid" live) || return 2
        zombie_count=$(bi100_process_group_count "$pgid" zombie) || return 2
        if [[ "$live_count" == 0 && "$zombie_count" == 0 ]]; then
            return 0
        fi
        sleep 1
    done
    echo "recorded service process group did not fully reap: pgid=$pgid" >&2
    return 1
}

stop_active_group() {
    local rc=0
    local stopped_pgid
    if [[ -z "$ACTIVE_PID" ]]; then
        return 0
    fi
    if [[ -n "$ACTIVE_PGID" ]]; then
        stopped_pgid=$ACTIVE_PGID
        bi100_stop_process_group \
            "$ACTIVE_PGID" "$ACTIVE_PID" 60 20 \
            "$ACTIVE_STARTTIME" "$ACTIVE_SESSION_TOKEN" || rc=$?
        wait "$ACTIVE_PID" 2>/dev/null || true
        wait_for_recorded_group_empty "$stopped_pgid" 20 || rc=$?
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
        active_pid_is_live && rc=1
    fi
    if [[ $rc -eq 0 ]]; then
        ACTIVE_PID=""
        ACTIVE_PGID=""
        ACTIVE_STARTTIME=""
        ACTIVE_SESSION_TOKEN=""
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
    timeout --signal=TERM --kill-after=70s 300s \
        env PYTHONPATH="$ROOT/tests" \
        python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus 0,1,2,3 \
        --settle-timeout-s 90 --clean-samples 3 \
        --sample-interval-s 2 \
        --out "$output.json" \
        > "$output.stdout" 2> "$output.stderr"
}

start_service() {
    local identity=$RUN_ROOT/service_identity.json
    local observed=""
    (
        exec env \
            BI100_RUNTIME_SITE_PACKAGES="$BI100_RUNTIME_SITE_PACKAGES" \
            BI100_RUNTIME_INSTALL_REPORT="$RUNTIME_INSTALL" \
            BI100_RUNTIME_WORKDIR="$RUN_ROOT/runtime-workdir" \
            MODEL_PATH="$MODEL_PATH" HOST=0.0.0.0 PORT=8000 \
            ENABLE_CUSTOM_IPC=1 VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
            BI100_MOE_COREX_DIRECT_ROUTED=1 \
            BI100_GDN_COREX_PACKED_DECODE=1 \
            BI100_GDN_COMBINED_QK_NORM=0 \
            BI100_GDN_CACHE_POLICY=admission64 \
            BI100_GDN_RESTORE_MODE=hybrid64 \
            BI100_HYBRID_KV_ACCOUNTING=full_attention \
            BI100_CPU_KV_OFFLOAD=0 BI100_BLOCK_MAJOR_CPU_KV=0 \
            BI100_CACHE_TRACE=0 \
            BI100_ATTN_COREX_FUSED_PREFILL=1 \
            BI100_ATTN_COREX_FUSED_PREFILL_DIAGNOSTICS=0 \
            BI100_ATTN_COREX_FUSED_PREFILL_SHADOW=1 \
            BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_REPORT_DIR="$RUN_ROOT/shadow" \
            BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_RUN_ID="$RUN_ID" \
            BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_CONTEXTS=49152,114688 \
            BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_MAX_CALLS_PER_CONTEXT=2 \
            BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_NUMERIC_MODE="$NUMERIC_MODE" \
            BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_FAILURE_ACTION="$FAILURE_ACTION" \
            BI100_PROFILE=0 BI100_PROFILE_INCLUDE_STARTUP=0 \
            BI100_PAGED_ATTN_DIAGNOSTICS=0 \
            BI100_GDN_ALLOW_NAN_ZERO=0 BI100_GDN_FINITE_CHECK=0 \
            PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH" \
            LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
            python3 "$ROOT/scripts/exec_bi100_session.py" \
            "$identity" -- "$ROOT/launch_service"
    ) > "$RUN_ROOT/server.log" 2>&1 &
    ACTIVE_PID=$!
    for _ in $(seq 1 50); do
        ACTIVE_STARTTIME=$(
            read_process_starttime "$ACTIVE_PID" 2>/dev/null || true)
        [[ -n "$ACTIVE_STARTTIME" ]] && break
        kill -0 "$ACTIVE_PID" 2>/dev/null || break
        sleep 0.1
    done
    [[ -n "$ACTIVE_STARTTIME" ]] || return 1
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
    [[ -n "$observed" ]] || return 1
    read -r ACTIVE_PGID ACTIVE_SESSION_TOKEN <<< "$observed"
    printf '%s\n' "$ACTIVE_PID" > "$RUN_ROOT/server.pid"
    printf '%s\n' "$ACTIVE_PGID" > "$RUN_ROOT/server.pgid"
    for _ in $(seq 1 360); do
        health && return 0
        active_pid_is_live || break
        sleep 10
    done
    return 1
}

write_runner_status() {
    local final_rc=$1
    python3 - "$RUN_ROOT" "$INSTANCE" "$SOURCE_REVISION" \
            "$SOURCE_BRANCH" "$CURRENT_STAGE" "$final_rc" \
            "$RUNNER_SCHEMA" "$NUMERIC_MODE" <<'PY'
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
    "schema": sys.argv[7],
    "version": 1,
    "qualified": int(sys.argv[6]) == 0,
    "source_revision": sys.argv[3],
    "source_branch": sys.argv[4],
    "instance": sys.argv[2],
    "terminal_stage": sys.argv[5],
    "returncode": int(sys.argv[6]),
    "numeric_mode": sys.argv[8],
    "gpu_count": 4,
    "tensor_parallel_size": 4,
    "max_model_len": 262144,
    "gates": {
        name: rc(f"{name}.rc")
        for name in (
            "postflight_before", "preflight_before", "runtime_identity",
            "startup", "startup_contract", "measurement", "health_after",
            "dispatch", "shadow_qualification", "scoped_cleanup",
            "scoped_cleanup_clean", "source_unchanged", "fatal_scan",
            "timeout_scan", "final_postflight", "final_preflight",
            "final_preflight_comparison",
        )
    },
    "artifact_sha256": {
        name: sha(name)
        for name in (
            "runtime_identity.json", "measurement.json",
            "shadow_qualification.json", "scoped_cleanup_clean.json",
            "final_postflight.json", "final_preflight_comparison.json",
        )
    },
    "decision": {
        "operator_shadow_reference_qualified": (
            rc("shadow_qualification.rc") == 0),
        "capability_noninferiority_evaluated": False,
        "production_promotion_authorized": False,
        "yaml_change_authorized": False,
        "main_merge_authorized": False,
    },
    "privacy": {
        "contains_prompts": False,
        "contains_model_outputs": False,
        "contains_tensor_values": False,
        "contains_token_ids": False,
        "contains_credentials": False,
    },
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
    local preflight_rc=1
    local comparison_rc=1
    local current_revision current_status artifact value
    local identity_args=()
    local expected_args=()
    local pattern
    trap - EXIT
    trap '' INT TERM
    set +e

    stop_active_group || cleanup_rc=1
    unset CUDA_VISIBLE_DEVICES
    if [[ -f "$RUN_ROOT/service_identity.json" ]]; then
        identity_args+=(--identity "$RUN_ROOT/service_identity.json")
        expected_args+=(--expected-identity "$RUN_ROOT/service_identity.json")
    fi
    if [[ ${#identity_args[@]} -gt 0 ]]; then
        timeout --signal=TERM --kill-after=70s 600s \
            python3 "$ROOT/scripts/cleanup_recorded_bi100_sessions.py" \
            "${identity_args[@]}" --out "$RUN_ROOT/scoped_cleanup.json" \
            > "$RUN_ROOT/scoped_cleanup.stdout" \
            2> "$RUN_ROOT/scoped_cleanup.stderr"
        [[ $? -eq 0 ]] || cleanup_rc=1
        python3 "$ROOT/tests/qualify_recorded_session_cleanup.py" \
            "$RUN_ROOT/scoped_cleanup.json" "${expected_args[@]}" \
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
run_postflight "$RUN_ROOT/postflight_before"
printf '0\n' > "$RUN_ROOT/postflight_before.rc"
port_is_free

CURRENT_STAGE=preflight_before
run_preflight "$RUN_ROOT/preflight_before"
printf '0\n' > "$RUN_ROOT/preflight_before.rc"
BEFORE_PREFLIGHT_PASSED=1

CURRENT_STAGE=runtime_identity
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
printf '0\n' > "$RUN_ROOT/runtime_identity.rc"
RUNTIME_IDENTITY=$(python3 - "$RUN_ROOT/runtime_identity.json" <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tree_sha = value.get("runtime_tree_sha256")
if (
    value.get("qualified") is not True
    or not isinstance(tree_sha, str)
    or len(tree_sha) != 64
    or any(character not in "0123456789abcdef" for character in tree_sha)
):
    raise SystemExit("qualified runtime identity lacks a valid tree SHA-256")
print(f"bare-host-overlay-v1:{tree_sha[:20]}")
PY
)

CURRENT_STAGE=startup
start_service
printf '0\n' > "$RUN_ROOT/startup.rc"

CURRENT_STAGE=startup_contract
grep -Fq '[BI100] fixed evaluator contract;' "$RUN_ROOT/server.log"
grep -Fq 'fused_prefill=1' "$RUN_ROOT/server.log"
printf '0\n' > "$RUN_ROOT/startup_contract.rc"

CURRENT_STAGE=measurement
timeout --signal=TERM --kill-after=70s 7200s \
    env PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH" \
    LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
    python3 "$ROOT/tests/bench_fused_prefill_service.py" \
    --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
    --targets 65536,131072 --max-tokens 8 --timeout-s 1800 \
    --run-id "$RUN_ID" --mode candidate \
    --out "$RUN_ROOT/measurement.json" \
    > "$RUN_ROOT/measurement.stdout" \
    2> "$RUN_ROOT/measurement.stderr"
printf '0\n' > "$RUN_ROOT/measurement.rc"

CURRENT_STAGE=health_after
health
printf '0\n' > "$RUN_ROOT/health_after.rc"

CURRENT_STAGE=dispatch
DISPATCH_COUNT=$(awk '
    { count += gsub(/path=corex_split4/, "&") }
    END { print count + 0 }
' "$RUN_ROOT/server.log")
printf '%s\n' "$DISPATCH_COUNT" > "$RUN_ROOT/dispatch_count.txt"
[[ "$DISPATCH_COUNT" -ge 4 ]]
printf '0\n' > "$RUN_ROOT/dispatch.rc"

CURRENT_STAGE=shadow_qualification
python3 "$QUALIFIER" \
    --report-dir "$RUN_ROOT/shadow" --run-id "$RUN_ID" \
    --source-revision "$SOURCE_REVISION" \
    --runtime-identity "$RUNTIME_IDENTITY" \
    --contract "$CONTRACT" \
    --out "$RUN_ROOT/shadow_qualification.json"
printf '0\n' > "$RUN_ROOT/shadow_qualification.rc"

CURRENT_STAGE=complete
