#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

if [[ $# -ne 3 ]]; then
    echo "usage: $0 GPU_INDEX INSTANCE RUN_ROOT" >&2
    exit 2
fi

GPU_INDEX=$1
INSTANCE=$2
RUN_ROOT=$3
BI100_RUNTIME_SITE_PACKAGES=${BI100_RUNTIME_SITE_PACKAGES:-}
ACTIVE_PID=""
ACTIVE_PGID=""
ACTIVE_STARTTIME=""
ACTIVE_SESSION_TOKEN=""
ACTIVE_LABEL=""
BEFORE_PREFLIGHT_PASSED=0
CURRENT_STAGE=argument_validation

if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
    echo "GPU_INDEX must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "$INSTANCE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    echo "INSTANCE must be a short non-sensitive label" >&2
    exit 2
fi
if [[ -z "$BI100_RUNTIME_SITE_PACKAGES" ]]; then
    echo "BI100_RUNTIME_SITE_PACKAGES is required" >&2
    exit 3
fi

RUN_ROOT=$(python3 - "$RUN_ROOT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)
case "$RUN_ROOT/" in
    "$ROOT/"*)
        echo "RUN_ROOT must stay outside the source repository" >&2
        exit 3
        ;;
esac
if [[ "$RUN_ROOT" != /tmp/* ]]; then
    echo "RUN_ROOT must use a private /tmp path" >&2
    exit 3
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "RUN_ROOT already exists: $RUN_ROOT" >&2
    exit 3
fi
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
        . ':(exclude)bench_runs/**')" ]]; then
    echo "M1-88 runner refuses a dirty source tree" >&2
    exit 3
fi

BI100_RUNTIME_SITE_PACKAGES=$(python3 - \
        "$BI100_RUNTIME_SITE_PACKAGES" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=True))
PY
)
RUNTIME_ROOT=$(dirname "$BI100_RUNTIME_SITE_PACKAGES")
RUNTIME_INSTALL=${BI100_RUNTIME_INSTALL_REPORT:-$RUNTIME_ROOT/install.json}
DIRECT_EXTENSION=$BI100_RUNTIME_SITE_PACKAGES/vllm/corex_moe_direct_routed.so
if [[ ! -f "$RUNTIME_INSTALL" || ! -f "$DIRECT_EXTENSION" ]]; then
    echo "runtime identity or production direct W13 extension is missing" >&2
    exit 3
fi
if pgrep -f 'vllm\.entrypoints\.openai\.api_server' >/dev/null 2>&1; then
    echo "an API server is already running" >&2
    exit 3
fi

mkdir -p "$RUN_ROOT"
SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
SOURCE_BRANCH=$(git -C "$ROOT" branch --show-current)
printf '%s\n' "$SOURCE_REVISION" > "$RUN_ROOT/source_revision.txt"
printf '%s\n' "$SOURCE_BRANCH" > "$RUN_ROOT/source_branch.txt"
printf '%s\n' "$GPU_INDEX" > "$RUN_ROOT/physical_gpu.txt"

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
COREX_LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
COREX_PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
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
state = fields[0]
starttime = fields[19]
raise SystemExit(0 if state != "Z" and starttime == sys.argv[2] else 1)
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

run_scoped() {
    local label=$1
    local max_seconds=$2
    local identity="$RUN_ROOT/${label}_process_identity.json"
    local observed
    local elapsed=0
    local command_rc=0
    local timed_out=0
    shift 2

    (
        exec python3 "$ROOT/scripts/exec_bi100_session.py" \
            "$identity" -- "$@"
    ) > "$RUN_ROOT/${label}.stdout" 2> "$RUN_ROOT/${label}.stderr" &
    ACTIVE_PID=$!
    ACTIVE_LABEL=$label
    ACTIVE_STARTTIME=""
    for _ in $(seq 1 50); do
        ACTIVE_STARTTIME=$(
            read_process_starttime "$ACTIVE_PID" 2>/dev/null || true)
        [[ -n "$ACTIVE_STARTTIME" ]] && break
        kill -0 "$ACTIVE_PID" 2>/dev/null || break
        sleep 0.1
    done
    if [[ -z "$ACTIVE_STARTTIME" ]]; then
        wait "$ACTIVE_PID" 2>/dev/null || true
        ACTIVE_PID=""
        ACTIVE_LABEL=""
        return 125
    fi

    observed=""
    for _ in $(seq 1 50); do
        if [[ -s "$identity" ]]; then
            observed=$(python3 - "$identity" "$ACTIVE_PID" \
                    "$ACTIVE_STARTTIME" <<'PY'
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
            ) || observed=""
            [[ -n "$observed" ]] && break
        fi
        active_pid_is_live || break
        sleep 0.1
    done
    if [[ -z "$observed" ]]; then
        stop_active_group || true
        return 125
    fi
    read -r ACTIVE_PGID ACTIVE_SESSION_TOKEN <<< "$observed"

    while active_pid_is_live; do
        if ((elapsed >= max_seconds)); then
            timed_out=1
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    if [[ $timed_out -eq 1 ]]; then
        printf 'timeout after %s seconds\n' "$max_seconds" \
            > "$RUN_ROOT/${label}.timeout"
        stop_active_group || command_rc=1
        command_rc=124
    else
        set +e
        wait "$ACTIVE_PID"
        command_rc=$?
        set -e
        ACTIVE_PID=""
        ACTIVE_PGID=""
        ACTIVE_STARTTIME=""
        ACTIVE_SESSION_TOKEN=""
        ACTIVE_LABEL=""
        : > "$RUN_ROOT/${label}.timeout"
    fi
    printf '%s\n' "$command_rc" > "$RUN_ROOT/${label}.rc"
    return "$command_rc"
}

run_preflight() {
    local output=$1
    timeout --signal=TERM --kill-after=70s 240s \
        env PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
        LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
        python3 "$ROOT/tests/bi100_preflight.py" \
        --gpus "$GPU_INDEX" --timeout-s 25 --matmul-size 1024 \
        --json-out "$output.json" \
        > "$output.stdout" 2> "$output.stderr"
}

run_postflight() {
    local output=$1
    timeout --signal=TERM --kill-after=70s 240s \
        env PYTHONPATH="$ROOT/tests" \
        python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus "$GPU_INDEX" \
        --settle-timeout-s 90 --clean-samples 3 \
        --sample-interval-s 2 \
        --out "$output.json" \
        > "$output.stdout" 2> "$output.stderr"
}

write_status() {
    local final_rc=$1
    python3 - "$RUN_ROOT" "$final_rc" "$CURRENT_STAGE" \
            "$SOURCE_REVISION" "$SOURCE_BRANCH" "$INSTANCE" \
            "$GPU_INDEX" <<'PY'
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

qualification = None
path = root / "qualification.json"
if path.is_file():
    try:
        qualification = json.loads(path.read_text(
            encoding="utf-8")).get("qualified")
    except (json.JSONDecodeError, OSError):
        qualification = None

gates = {
    "postflight_before": read_rc("postflight_before.rc"),
    "preflight_before": read_rc("preflight_before.rc"),
    "runtime_identity": read_rc("runtime_identity.rc"),
    "build": read_rc("build.rc"),
    "benchmark": read_rc("benchmark.rc"),
    "qualification": read_rc("qualification.rc"),
    "scoped_cleanup": read_rc("scoped_cleanup.rc"),
    "fatal_scan": read_rc("fatal_scan.rc"),
    "timeout_scan": read_rc("timeout_scan.rc"),
    "postflight_after": read_rc("postflight_after.rc"),
    "preflight_after": read_rc("preflight_after.rc"),
    "preflight_comparison": read_rc("preflight_comparison.rc"),
}
report = {
    "schema": "bi100-m1-88-w13-rounding-guard-runner-v1",
    "version": 1,
    "returncode": int(sys.argv[2]),
    "last_stage": sys.argv[3],
    "source_revision": sys.argv[4],
    "source_branch": sys.argv[5],
    "instance": sys.argv[6],
    "physical_gpu": int(sys.argv[7]),
    "candidate_qualified": qualification,
    "gates": gates,
    "artifact_sha256": {
        "benchmark": sha("benchmark.json"),
        "qualification": sha("qualification.json"),
        "runtime_identity": sha("runtime_identity.json"),
        "scoped_cleanup": sha("scoped_cleanup.json"),
        "preflight_comparison": sha("preflight_comparison.json"),
    },
    "limits": {
        "relative_l2": 1.0e-5,
        "max_flagged_fraction": 0.05,
        "max_step_flagged_fraction": 0.10,
        "sequence_steps_per_seed": 500,
        "term_grace_s": 60,
    },
    "production_runtime_changed": False,
    "production_promotion_authorized": False,
    "yaml_change_authorized": False,
    "main_merge_authorized": False,
}
(root / "runner_status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

finish() {
    local primary_rc=$?
    local final_rc=$primary_rc
    local cleanup_rc=0
    local fatal_rc=0
    local timeout_rc=0
    local postflight_rc=0
    local after_rc=0
    local comparison_rc=0
    local identity_args=()
    trap - EXIT INT TERM
    set +e

    stop_active_group || cleanup_rc=1
    unset CUDA_VISIBLE_DEVICES
    for identity in \
            "$RUN_ROOT/build_process_identity.json" \
            "$RUN_ROOT/benchmark_process_identity.json"; do
        if [[ -f "$identity" ]]; then
            identity_args+=(--identity "$identity")
        fi
    done
    timeout --signal=TERM --kill-after=70s 300s \
        python3 "$ROOT/scripts/cleanup_recorded_bi100_sessions.py" \
        "${identity_args[@]}" \
        --out "$RUN_ROOT/scoped_cleanup.json" \
        > "$RUN_ROOT/scoped_cleanup.stdout" \
        2> "$RUN_ROOT/scoped_cleanup.stderr"
    [[ $? -eq 0 ]] || cleanup_rc=1
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/scoped_cleanup.rc"

    if grep -Eiq \
            'CUDA error|SIGSEGV|Fatal Python error|out of memory|device-side assert|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|worker.*(died|lost|exited unexpectedly)' \
            "$RUN_ROOT"/*.stdout "$RUN_ROOT"/*.stderr 2>/dev/null; then
        grep -Ein \
            'CUDA error|SIGSEGV|Fatal Python error|out of memory|device-side assert|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|worker.*(died|lost|exited unexpectedly)' \
            "$RUN_ROOT"/*.stdout "$RUN_ROOT"/*.stderr \
            > "$RUN_ROOT/fatal_scan.txt" 2>/dev/null || true
        fatal_rc=1
    else
        : > "$RUN_ROOT/fatal_scan.txt"
    fi
    printf '%s\n' "$fatal_rc" > "$RUN_ROOT/fatal_scan.rc"

    if find "$RUN_ROOT" -maxdepth 1 -name '*.timeout' \
            -type f -size +0c -print -quit | grep -q .; then
        find "$RUN_ROOT" -maxdepth 1 -name '*.timeout' \
            -type f -size +0c -print > "$RUN_ROOT/timeout_scan.txt"
        timeout_rc=1
    else
        : > "$RUN_ROOT/timeout_scan.txt"
    fi
    printf '%s\n' "$timeout_rc" > "$RUN_ROOT/timeout_scan.rc"

    run_postflight "$RUN_ROOT/postflight_after"
    postflight_rc=$?
    printf '%s\n' "$postflight_rc" > "$RUN_ROOT/postflight_after.rc"

    if [[ "$BEFORE_PREFLIGHT_PASSED" == 1 && $cleanup_rc -eq 0 \
            && $postflight_rc -eq 0 ]]; then
        run_preflight "$RUN_ROOT/preflight_after"
        after_rc=$?
        printf '%s\n' "$after_rc" > "$RUN_ROOT/preflight_after.rc"
        if [[ $after_rc -eq 0 ]]; then
            python3 "$ROOT/tests/compare_bi100_preflights.py" \
                --preflight "before=$RUN_ROOT/preflight_before.json" \
                --preflight "after=$RUN_ROOT/preflight_after.json" \
                --expected-gpus "$GPU_INDEX" \
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

    if [[ $cleanup_rc -ne 0 || $fatal_rc -ne 0 || $timeout_rc -ne 0 \
            || $postflight_rc -ne 0 || $after_rc -ne 0 \
            || $comparison_rc -ne 0 ]]; then
        final_rc=1
    fi
    printf '%s\n' "$final_rc" > "$RUN_ROOT/overall.rc"
    write_status "$final_rc"
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

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH"
export LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH"
export PATH="$COREX_PATH"

CURRENT_STAGE=build
set +e
run_scoped build 900 \
    "$ROOT/tests/build_corex_moe_w13_rounding_probe.sh" \
    "$RUN_ROOT/extensions"
rc=$?
set -e
[[ $rc -eq 0 ]]

CURRENT_STAGE=benchmark
set +e
run_scoped benchmark 3600 \
    python3 "$ROOT/tests/bench_moe_w13_rounding_guard.py" \
    --probe-extension \
        "$RUN_ROOT/extensions/corex_moe_w13_rounding_probe.so" \
    --direct-extension "$DIRECT_EXTENSION" \
    --device cuda:0 \
    --seeds 20260716,20260727 \
    --sequence-steps 500 \
    --cpu-threads 8 \
    --out "$RUN_ROOT/benchmark.json"
rc=$?
set -e
[[ $rc -eq 0 ]]

CURRENT_STAGE=qualification
set +e
python3 "$ROOT/tests/qualify_moe_w13_rounding_guard.py" \
    --report "$RUN_ROOT/benchmark.json" \
    --out "$RUN_ROOT/qualification.json" \
    > "$RUN_ROOT/qualification.stdout" \
    2> "$RUN_ROOT/qualification.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/qualification.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=completed
exit 0
