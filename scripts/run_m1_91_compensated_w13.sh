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
    echo "M1-91 runner refuses a dirty source tree" >&2
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
sha256sum "$DIRECT_EXTENSION" > "$RUN_ROOT/direct_extension.identity"

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
import os
from pathlib import Path
import tempfile
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

def identity_sha(name):
    path = root / name
    if not path.is_file():
        return None
    fields = path.read_text(encoding="utf-8").split()
    if (
        not fields
        or len(fields[0]) != 64
        or any(character not in "0123456789abcdef"
               for character in fields[0])
    ):
        return None
    return fields[0]

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
    "artifact_binding": read_rc("artifact_binding.rc"),
    "scoped_cleanup": read_rc("scoped_cleanup.rc"),
    "scoped_cleanup_clean": read_rc("scoped_cleanup_clean.rc"),
    "source_unchanged": read_rc("source_unchanged.rc"),
    "fatal_scan": read_rc("fatal_scan.rc"),
    "timeout_scan": read_rc("timeout_scan.rc"),
    "postflight_after": read_rc("postflight_after.rc"),
    "preflight_after": read_rc("preflight_after.rc"),
    "preflight_comparison": read_rc("preflight_comparison.rc"),
}
all_gates_passed = all(value == 0 for value in gates.values())
report = {
    "schema": "bi100-m1-91-compensated-w13-runner-v1",
    "version": 1,
    "returncode": int(sys.argv[2]),
    "last_stage": sys.argv[3],
    "source_revision": sys.argv[4],
    "source_branch": sys.argv[5],
    "instance": sys.argv[6],
    "physical_gpu": int(sys.argv[7]),
    "candidate_qualified": qualification,
    "experiment_valid": (
        int(sys.argv[2]) == 0
        and qualification is True
        and all_gates_passed
    ),
    "gates": gates,
    "artifact_sha256": {
        "benchmark": sha("benchmark.json"),
        "qualification": sha("qualification.json"),
        "artifact_binding": sha("artifact_binding.json"),
        "candidate_extension":
            sha("extensions/corex_moe_compensated_w13.so"),
        "direct_extension": identity_sha("direct_extension.identity"),
        "runtime_identity": sha("runtime_identity.json"),
        "scoped_cleanup": sha("scoped_cleanup.json"),
        "scoped_cleanup_clean": sha("scoped_cleanup_clean.json"),
        "preflight_comparison": sha("preflight_comparison.json"),
    },
    "limits": {
        "relative_l2": 1.0e-5,
        "fixed_speedup": 1.5,
        "routed_speedup": 1.25,
        "seeds": [20260716, 20260727],
        "sequence_steps_per_seed": 500,
        "warmup": 30,
        "iterations": 300,
        "repeats": 9,
        "term_grace_s": 60,
        "kill_grace_s": 20,
        "complete_token_scan_required": True,
    },
    "production_runtime_changed": False,
    "production_promotion_authorized": False,
    "yaml_change_authorized": False,
    "main_merge_authorized": False,
}
payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
descriptor, temporary_name = tempfile.mkstemp(
    prefix=".runner_status.",
    suffix=".tmp",
    dir=root,
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, root / "runner_status.json")
finally:
    temporary.unlink(missing_ok=True)
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
    local after_rc=0
    local comparison_rc=0
    local artifact
    local identity_args=()
    local pattern
    local value
    trap - EXIT
    trap '' INT TERM
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
    python3 "$ROOT/tests/qualify_recorded_session_cleanup.py" \
        "$RUN_ROOT/scoped_cleanup.json" \
        --expected-identity "$RUN_ROOT/build_process_identity.json" \
        --expected-identity "$RUN_ROOT/benchmark_process_identity.json" \
        --out "$RUN_ROOT/scoped_cleanup_clean.json" \
        > "$RUN_ROOT/scoped_cleanup_clean.stdout" \
        2> "$RUN_ROOT/scoped_cleanup_clean.stderr"
    cleanup_clean_rc=$?
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
    while IFS= read -r -d '' artifact; do
        printf '%s\n' "$artifact" >> "$RUN_ROOT/timeout_scan.txt"
        timeout_rc=1
    done < <(find "$RUN_ROOT" -type f -name '*.timeout' -size +0c -print0)
    printf '%s\n' "$timeout_rc" > "$RUN_ROOT/timeout_scan.rc"

    run_postflight "$RUN_ROOT/postflight_after"
    postflight_rc=$?
    printf '%s\n' "$postflight_rc" > "$RUN_ROOT/postflight_after.rc"

    if [[ "$BEFORE_PREFLIGHT_PASSED" == 1 && $cleanup_rc -eq 0 \
            && $cleanup_clean_rc -eq 0 && $postflight_rc -eq 0 ]]; then
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

    if [[ $cleanup_rc -ne 0 || $cleanup_clean_rc -ne 0 \
            || $source_rc -ne 0 || $fatal_rc -ne 0 \
            || $timeout_rc -ne 0 || $postflight_rc -ne 0 \
            || $after_rc -ne 0 || $comparison_rc -ne 0 ]]; then
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
    "$ROOT/tests/build_corex_moe_compensated_w13.sh" \
    "$RUN_ROOT/extensions"
rc=$?
set -e
[[ $rc -eq 0 ]]

CURRENT_STAGE=benchmark
set +e
run_scoped benchmark 3600 \
    python3 "$ROOT/tests/bench_moe_compensated_w13.py" \
    --candidate-extension \
        "$RUN_ROOT/extensions/corex_moe_compensated_w13.so" \
    --direct-extension "$DIRECT_EXTENSION" \
    --device cuda:0 \
    --seeds 20260716,20260727 \
    --sequence-steps 500 \
    --warmup 30 \
    --iterations 300 \
    --repeats 9 \
    --cpu-threads 8 \
    --out "$RUN_ROOT/benchmark.json"
rc=$?
set -e
[[ $rc -eq 0 ]]

CURRENT_STAGE=qualification
set +e
python3 "$ROOT/tests/qualify_moe_compensated_w13.py" \
    --report "$RUN_ROOT/benchmark.json" \
    --candidate-extension \
        "$RUN_ROOT/extensions/corex_moe_compensated_w13.so" \
    --direct-extension "$DIRECT_EXTENSION" \
    --out "$RUN_ROOT/qualification.json" \
    > "$RUN_ROOT/qualification.stdout" \
    2> "$RUN_ROOT/qualification.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/qualification.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=artifact_binding
set +e
python3 - "$RUN_ROOT/benchmark.json" "$RUN_ROOT/qualification.json" \
        "$RUN_ROOT/extensions/corex_moe_compensated_w13.so" \
        "$DIRECT_EXTENSION" "$RUN_ROOT/artifact_binding.json" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys

benchmark_path = Path(sys.argv[1])
qualification_path = Path(sys.argv[2])
candidate_path = Path(sys.argv[3])
direct_path = Path(sys.argv[4])
output_path = Path(sys.argv[5])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
qualification = json.loads(
    qualification_path.read_text(encoding="utf-8"))
observed = {
    "benchmark_sha256": sha(benchmark_path),
    "candidate_extension_sha256": sha(candidate_path),
    "direct_extension_sha256": sha(direct_path),
}
expected = {
    "benchmark_sha256":
        qualification.get("evidence", {}).get("report_sha256"),
    "candidate_extension_sha256":
        qualification.get("evidence", {}).get(
            "candidate_extension_sha256"),
    "direct_extension_sha256":
        qualification.get("evidence", {}).get(
            "direct_extension_sha256"),
}
benchmark_extensions = benchmark.get("extensions", {})
qualified = (
    qualification.get("qualified") is True
    and observed == expected
    and observed["candidate_extension_sha256"]
        == benchmark_extensions.get("candidate_sha256")
    and observed["direct_extension_sha256"]
        == benchmark_extensions.get("direct_sha256")
)
report = {
    "schema": "bi100-m1-91-artifact-binding-v1",
    "version": 1,
    "qualified": qualified,
    "observed": observed,
    "expected": expected,
}
descriptor, temporary_name = tempfile.mkstemp(
    prefix=".artifact_binding.",
    suffix=".tmp",
    dir=output_path.parent,
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, output_path)
finally:
    temporary.unlink(missing_ok=True)
raise SystemExit(0 if qualified else 1)
PY
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/artifact_binding.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=completed
exit 0
