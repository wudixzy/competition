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
BI100_RUNTIME_SITE_PACKAGES=${BI100_RUNTIME_SITE_PACKAGES:-}
CONTROL_RUNTIME_SITE_PACKAGES=${CONTROL_RUNTIME_SITE_PACKAGES:-}
CONTROL_REVISION=${CONTROL_REVISION:-cdb1bc41f728a5610a3632ad7923d73a90748919}
CANDIDATE_REVISION=${CANDIDATE_REVISION:-37001edff643d98bf41bf4a52e0a145329003315}
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
    echo "M1-71 gate refuses a dirty source tree" >&2
    exit 3
fi
if [[ -z "$BI100_RUNTIME_SITE_PACKAGES" \
        || -z "$CONTROL_RUNTIME_SITE_PACKAGES" ]]; then
    echo "candidate and control immutable overlays are required" >&2
    exit 3
fi
BI100_RUNTIME_SITE_PACKAGES=$(python3 - \
        "$BI100_RUNTIME_SITE_PACKAGES" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=True))
PY
)
CONTROL_RUNTIME_SITE_PACKAGES=$(python3 - \
        "$CONTROL_RUNTIME_SITE_PACKAGES" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=True))
PY
)
CANDIDATE_INSTALL=$(dirname "$BI100_RUNTIME_SITE_PACKAGES")/install.json
CONTROL_INSTALL=$(dirname "$CONTROL_RUNTIME_SITE_PACKAGES")/install.json
DIRECT_EXTENSION=$BI100_RUNTIME_SITE_PACKAGES/vllm/corex_moe_direct_routed.so
GATHER_EXTENSION=$BI100_RUNTIME_SITE_PACKAGES/vllm/corex_moe_weight_gather.so
REDUCE_EXTENSION=$BI100_RUNTIME_SITE_PACKAGES/vllm/corex_moe_exact_reduce.so
for path in \
        "$CANDIDATE_INSTALL" "$CONTROL_INSTALL" \
        "$DIRECT_EXTENSION" "$GATHER_EXTENSION" "$REDUCE_EXTENSION"; do
    if [[ ! -f "$path" ]]; then
        echo "required runtime artifact is missing: $path" >&2
        exit 3
    fi
done
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
export LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
export PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
unset CUDA_VISIBLE_DEVICES

run_preflight() {
    local name=$1
    timeout --signal=TERM --kill-after=70s 240s \
        python3 "$ROOT/tests/bi100_preflight.py" \
        --gpus "$GPU_INDEX" --timeout-s 25 --matmul-size 1024 \
        --json-out "$RUN_ROOT/preflight_${name}.json" \
        > "$RUN_ROOT/preflight_${name}.stdout" \
        2> "$RUN_ROOT/preflight_${name}.stderr"
}

stop_active_group() {
    local rc=0
    if [[ -n "$ACTIVE_PGID" ]]; then
        bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID" 60 20 || rc=$?
    elif [[ -n "$ACTIVE_PID" ]]; then
        echo "benchmark PID lacks a verified process group" >&2
        rc=2
    fi
    ACTIVE_PID=""
    ACTIVE_PGID=""
    return "$rc"
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
    "schema": "bi100-m1-71-moe-hybrid-exact-tail-runner-v1",
    "version": 1,
    "qualified": int(sys.argv[2]) == 0,
    "returncode": int(sys.argv[2]),
    "last_stage": sys.argv[3],
    "source_revision": sys.argv[4],
    "source_branch": sys.argv[5],
    "instance": sys.argv[6],
    "physical_gpu": int(sys.argv[7]),
    "gates": {
        "preflight_before": rc("preflight_before.rc"),
        "runtime_pair": rc("runtime_pair.rc"),
        "benchmark": rc("benchmark.rc"),
        "qualification": rc("qualification.rc"),
        "cleanup": rc("cleanup.rc"),
        "service_postflight": rc("service_postflight.rc"),
        "fatal_scan": rc("fatal_scan.rc"),
        "timeout_scan": rc("timeout_scan.rc"),
        "preflight_after": rc("preflight_after.rc"),
        "preflight_comparison": rc("preflight_comparison.rc"),
    },
    "artifact_sha256": {
        "runtime_pair": sha("runtime_pair.json"),
        "benchmark": sha("benchmark.json"),
        "qualification": sha("qualification.json"),
        "service_postflight": sha("service_postflight.json"),
    },
    "limits": {
        "relative_l2": 1.0e-5,
        "speedup": 1.25,
        "saving_ms": 0.02,
        "sequence_steps": 500,
    },
    "semantic_quality_evaluated": False,
    "full_model_evaluated": False,
    "production_promotion_authorized": False,
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
    local postflight_rc=0
    local fatal_rc=0
    local timeout_rc=0
    local after_rc=0
    local comparison_rc=0
    trap - EXIT TERM INT
    set +e

    stop_active_group
    cleanup_rc=$?
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/cleanup.rc"
    unset CUDA_VISIBLE_DEVICES

    python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus "$GPU_INDEX" \
        --settle-timeout-s 90 --clean-samples 3 \
        --sample-interval-s 2 \
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
trap 'exit 143' TERM
trap 'exit 130' INT
trap finish EXIT

CURRENT_STAGE=preflight_before
set +e
run_preflight before
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/preflight_before.rc"
[[ $rc -eq 0 ]]
BEFORE_PREFLIGHT_PASSED=1

CURRENT_STAGE=runtime_pair
set +e
PYTHONPATH="$ROOT/tests" python3 "$ROOT/tests/verify_m1_70_runtime_pair.py" \
    --source-root "$ROOT" \
    --control-site "$CONTROL_RUNTIME_SITE_PACKAGES" \
    --control-install "$CONTROL_INSTALL" \
    --control-revision "$CONTROL_REVISION" \
    --candidate-site "$BI100_RUNTIME_SITE_PACKAGES" \
    --candidate-install "$CANDIDATE_INSTALL" \
    --candidate-revision "$CANDIDATE_REVISION" \
    --out "$RUN_ROOT/runtime_pair.json" \
    > "$RUN_ROOT/runtime_pair.stdout" \
    2> "$RUN_ROOT/runtime_pair.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/runtime_pair.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=benchmark
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
setsid timeout --signal=TERM --kill-after=60s 3600s \
    python3 "$ROOT/tests/bench_moe_direct_routed.py" \
    --direct-extension "$DIRECT_EXTENSION" \
    --gather-extension "$GATHER_EXTENSION" \
    --reduce-extension "$REDUCE_EXTENSION" \
    --device cuda:0 \
    --warmup 30 --iterations 300 --repeats 9 \
    --sequence-steps 500 --seed 20260716 \
    --out "$RUN_ROOT/benchmark.json" \
    > "$RUN_ROOT/benchmark.stdout" \
    2> "$RUN_ROOT/benchmark.stderr" &
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
python3 "$ROOT/tests/qualify_moe_hybrid_exact_tail.py" \
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
