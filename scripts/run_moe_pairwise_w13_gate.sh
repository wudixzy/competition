#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)

if [[ $# -ne 3 ]]; then
    echo "usage: $0 GPU_INDEX INSTANCE RUN_ROOT" >&2
    exit 2
fi

GPU_INDEX=$1
INSTANCE=$2
RUN_ROOT=$3
CURRENT_STAGE=argument_validation
BEFORE_PREFLIGHT_PASSED=0

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
    echo "pairwise W13 gate refuses a dirty source tree" >&2
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
DIRECT_EXTENSION=$BI100_RUNTIME_SITE_PACKAGES/vllm/corex_moe_direct_routed.so
if [[ ! -f "$RUNTIME_INSTALL" || ! -f "$DIRECT_EXTENSION" ]]; then
    echo "runtime identity or direct W13 reference extension is missing" >&2
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

run_preflight() {
    local name=$1
    timeout --signal=TERM --kill-after=10s 180s \
        python3 "$ROOT/tests/bi100_preflight.py" \
        --gpus "$GPU_INDEX" --timeout-s 25 --matmul-size 1024 \
        --json-out "$RUN_ROOT/preflight_${name}.json" \
        > "$RUN_ROOT/preflight_${name}.stdout" \
        2> "$RUN_ROOT/preflight_${name}.stderr"
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
    "schema": "bi100-moe-pairwise-w13-runner-v1",
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
        "build": read_rc("build.rc"),
        "benchmark": read_rc("benchmark.rc"),
        "qualification": read_rc("qualification.rc"),
        "fatal_scan": read_rc("fatal_scan.rc"),
        "preflight_after": read_rc("preflight_after.rc"),
        "preflight_comparison": read_rc("preflight_comparison.rc"),
    },
    "artifacts": {
        "benchmark_sha256": sha256("benchmark.json"),
        "qualification_sha256": sha256("qualification.json"),
        "runtime_identity_sha256": sha256("runtime_identity.json"),
    },
    "limits": {
        "relative_l2": 1.0e-5,
        "fixed_speedup": 1.5,
        "routed_speedup": 1.25,
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
    local fatal_rc=0
    local after_rc=0
    local comparison_rc=0
    trap - EXIT
    set +e
    unset CUDA_VISIBLE_DEVICES

    if grep -Eiq \
            'CUDA error|SIGSEGV|Fatal Python error|out of memory|device-side assert' \
            "$RUN_ROOT"/*.stdout "$RUN_ROOT"/*.stderr 2>/dev/null; then
        grep -Ein \
            'CUDA error|SIGSEGV|Fatal Python error|out of memory|device-side assert' \
            "$RUN_ROOT"/*.stdout "$RUN_ROOT"/*.stderr \
            > "$RUN_ROOT/fatal_scan.txt" 2>/dev/null || true
        fatal_rc=1
    else
        : > "$RUN_ROOT/fatal_scan.txt"
    fi
    printf '%s\n' "$fatal_rc" > "$RUN_ROOT/fatal_scan.rc"

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

    if [[ $fatal_rc -ne 0 || $after_rc -ne 0 || $comparison_rc -ne 0 ]]; then
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
timeout --signal=TERM --kill-after=10s 180s \
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
CURRENT_STAGE=build
set +e
timeout --signal=TERM --kill-after=15s 600s \
    "$ROOT/tests/build_corex_moe_pairwise_w13.sh" \
    "$RUN_ROOT/extensions" \
    > "$RUN_ROOT/build.stdout" 2> "$RUN_ROOT/build.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/build.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=benchmark
set +e
timeout --signal=TERM --kill-after=30s 3600s \
    python3 "$ROOT/tests/bench_moe_pairwise_w13.py" \
    --candidate-extension \
        "$RUN_ROOT/extensions/corex_moe_pairwise_w13.so" \
    --direct-extension "$DIRECT_EXTENSION" \
    --device cuda:0 --sequence-steps 500 \
    --out "$RUN_ROOT/benchmark.json" \
    > "$RUN_ROOT/benchmark.stdout" 2> "$RUN_ROOT/benchmark.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/benchmark.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=qualification
set +e
python3 "$ROOT/tests/qualify_moe_pairwise_w13.py" \
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
