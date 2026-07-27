#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

if [[ $# -ne 3 ]]; then
    echo "usage: $0 GPU_INDEX INSTANCE RUN_ROOT" >&2
    exit 2
fi

GPU_INDEX=$1
INSTANCE=$2
RUN_ROOT=$3
ACTIVE_PID=""
ACTIVE_PGID=""
BEFORE_PREFLIGHT_PASSED=0
CURRENT_STAGE=argument_validation

if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
    echo "GPU_INDEX must be a non-negative integer" >&2
    exit 2
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
    echo "combined q/k gate refuses a dirty source tree" >&2
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
)
RUNTIME_ROOT=$(dirname "$BI100_RUNTIME_SITE_PACKAGES")
RUNTIME_INSTALL=${BI100_RUNTIME_INSTALL_REPORT:-$RUNTIME_ROOT/install.json}
QK_MAP_EXTENSION=$BI100_RUNTIME_SITE_PACKAGES/vllm/corex_gdn_qk_map.so
if [[ ! -f "$RUNTIME_INSTALL" || ! -f "$QK_MAP_EXTENSION" ]]; then
    echo "runtime identity or q/k map extension is missing" >&2
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
export PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH"
export LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64
export PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
unset CUDA_VISIBLE_DEVICES

run_preflight() {
    local name=$1
    timeout --signal=TERM --kill-after=60s 180s \
        python3 "$ROOT/tests/bi100_preflight.py" \
        --gpus "$GPU_INDEX" --timeout-s 25 --matmul-size 1024 \
        --json-out "$RUN_ROOT/preflight_${name}.json" \
        > "$RUN_ROOT/preflight_${name}.stdout" \
        2> "$RUN_ROOT/preflight_${name}.stderr"
}

stop_active_group() {
    if [[ -n "$ACTIVE_PGID" ]]; then
        bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID" 60 20
    elif [[ -n "$ACTIVE_PID" ]]; then
        echo "benchmark PID lacks a verified process group" >&2
        return 2
    fi
    ACTIVE_PID=""
    ACTIVE_PGID=""
}

scan_fatal() {
    local pattern
    pattern='CUDA error|SIGSEGV|Fatal Python error|out of memory|device-side assert|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|TimeoutError'
    if grep -Eiq "$pattern" \
            "$RUN_ROOT/benchmark.stdout" "$RUN_ROOT/benchmark.stderr" \
            2>/dev/null; then
        grep -Ein "$pattern" \
            "$RUN_ROOT/benchmark.stdout" "$RUN_ROOT/benchmark.stderr" \
            > "$RUN_ROOT/fatal_scan.txt" 2>/dev/null || true
        return 1
    fi
    : > "$RUN_ROOT/fatal_scan.txt"
}

write_status() {
    local final_rc=$1
    python3 - "$RUN_ROOT" "$final_rc" "$CURRENT_STAGE" \
            "$SOURCE_REVISION" "$SOURCE_BRANCH" "$INSTANCE" "$GPU_INDEX" <<'PY'
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
    return int(value) if value.isdigit() else None

def sha256(name):
    path = root / name
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

report = {
    "schema": "bi100-gdn-combined-qk-norm-runner-v1",
    "version": 1,
    "returncode": int(sys.argv[2]),
    "last_stage": sys.argv[3],
    "source_revision": sys.argv[4],
    "source_branch": sys.argv[5],
    "instance": sys.argv[6],
    "physical_gpu": int(sys.argv[7]),
    "gates": {
        "preflight_before": read_rc("preflight_before.rc"),
        "runtime_identity": read_rc("runtime_identity.rc"),
        "benchmark": read_rc("benchmark.rc"),
        "qualification": read_rc("qualification.rc"),
        "cleanup": read_rc("cleanup.rc"),
        "service_postflight": read_rc("service_postflight.rc"),
        "fatal_scan": read_rc("fatal_scan.rc"),
        "timeout_scan": read_rc("timeout_scan.rc"),
        "preflight_after": read_rc("preflight_after.rc"),
        "preflight_comparison": read_rc("preflight_comparison.rc"),
    },
    "artifacts": {
        "benchmark_sha256": sha256("benchmark.json"),
        "qualification_sha256": sha256("qualification.json"),
        "runtime_identity_sha256": sha256("runtime_identity.json"),
        "service_postflight_sha256": sha256("service_postflight.json"),
    },
    "limits": {
        "relative_l2": 1.0e-5,
        "speedup": 1.25,
        "saving_ms": 0.02,
        "sequence_steps": 500,
    },
    "production_promotion_authorized": False,
}
(root / "runner_status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

finish() {
    local primary_rc=$?
    local final_rc=$primary_rc
    local cleanup_rc=0
    local postflight_rc=0
    local fatal_rc=0
    local timeout_rc=0
    local after_rc=0
    local comparison_rc=0
    trap - EXIT
    set +e

    stop_active_group
    cleanup_rc=$?
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/cleanup.rc"
    unset CUDA_VISIBLE_DEVICES

    python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus "$GPU_INDEX" \
        --out "$RUN_ROOT/service_postflight.json" \
        > "$RUN_ROOT/service_postflight.stdout" \
        2> "$RUN_ROOT/service_postflight.stderr"
    postflight_rc=$?
    printf '%s\n' "$postflight_rc" > "$RUN_ROOT/service_postflight.rc"

    scan_fatal
    fatal_rc=$?
    printf '%s\n' "$fatal_rc" > "$RUN_ROOT/fatal_scan.rc"

    if [[ -f "$RUN_ROOT/benchmark.rc" ]] \
            && grep -Eq '^(124|137)$' "$RUN_ROOT/benchmark.rc"; then
        timeout_rc=1
        cp "$RUN_ROOT/benchmark.rc" "$RUN_ROOT/timeout_scan.txt"
    else
        : > "$RUN_ROOT/timeout_scan.txt"
    fi
    printf '%s\n' "$timeout_rc" > "$RUN_ROOT/timeout_scan.rc"

    if [[ "$BEFORE_PREFLIGHT_PASSED" == 1 ]]; then
        run_preflight after
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

    if [[ $cleanup_rc -ne 0 || $postflight_rc -ne 0 \
            || $fatal_rc -ne 0 || $timeout_rc -ne 0 \
            || $after_rc -ne 0 || $comparison_rc -ne 0 ]]; then
        final_rc=1
    fi
    printf '%s\n' "$final_rc" > "$RUN_ROOT/overall.rc"
    write_status "$final_rc"
    exit "$final_rc"
}
trap finish EXIT

CURRENT_STAGE=preflight_before
set +e
run_preflight before
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/preflight_before.rc"
[[ $rc -eq 0 ]]
BEFORE_PREFLIGHT_PASSED=1

CURRENT_STAGE=runtime_identity
set +e
timeout --signal=TERM --kill-after=60s 180s \
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

CURRENT_STAGE=benchmark
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
setsid timeout --signal=TERM --kill-after=60s 3600s \
    python3 "$ROOT/tests/bench_gdn_combined_qk_norm.py" \
    --qk-map-extension "$QK_MAP_EXTENSION" \
    --device cuda:0 --sequence-steps 500 \
    --out "$RUN_ROOT/benchmark.json" \
    > "$RUN_ROOT/benchmark.stdout" 2> "$RUN_ROOT/benchmark.stderr" &
ACTIVE_PID=$!
ACTIVE_PGID=$ACTIVE_PID
observed_pgid=""
for _ in $(seq 1 20); do
    observed_pgid=$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null | tr -d ' ')
    [[ -n "$observed_pgid" ]] && break
    sleep 1
done
if [[ -z "$observed_pgid" || "$observed_pgid" != "$ACTIVE_PGID" ]]; then
    echo "benchmark did not enter its isolated process group" >&2
    exit 1
fi
set +e
wait "$ACTIVE_PID"
rc=$?
stop_active_group
cleanup_rc=$?
set -e
if [[ $cleanup_rc -ne 0 ]]; then
    rc=1
fi
printf '%s\n' "$rc" > "$RUN_ROOT/benchmark.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=qualification
set +e
python3 "$ROOT/tests/qualify_gdn_combined_qk_norm.py" \
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
