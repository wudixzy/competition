#!/bin/bash
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)

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

if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
    echo "GPU_INDEX must be a non-negative integer" >&2
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
    python3 - "$RUN_ROOT/runner_status.json" "$rc" \
            "$CURRENT_STAGE" "$SOURCE_REVISION" "$SOURCE_BRANCH" \
            "$INSTANCE" "$GPU_INDEX" <<'PY'
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "qwen36-diagnostic-component-runner-v1",
    "version": 1,
    "returncode": int(sys.argv[2]),
    "last_stage": sys.argv[3],
    "source_revision": sys.argv[4],
    "source_branch": sys.argv[5],
    "instance": sys.argv[6],
    "physical_gpu": int(sys.argv[7]),
    "production_promotion_authorized": False,
}, indent=2, sort_keys=True) + "\n")
PY
}

cleanup() {
    local rc=$?
    trap - EXIT
    unset CUDA_VISIBLE_DEVICES
    write_runner_status "$rc"
    exit "$rc"
}
trap cleanup EXIT

run_physical_preflight() {
    local name=$1
    timeout --signal=TERM --kill-after=10s 180s \
        python3 "$ROOT/tests/bi100_preflight.py" \
        --gpus "$GPU_INDEX" --timeout-s 25 --matmul-size 1024 \
        --json-out "$RUN_ROOT/${name}.json" \
        > "$RUN_ROOT/${name}.stdout" \
        2> "$RUN_ROOT/${name}.stderr"
}

abort_after_probe_failure() {
    local message=$1
    echo "$message" >&2
    unset CUDA_VISIBLE_DEVICES
    CURRENT_STAGE=preflight_after_probe_failure
    run_physical_preflight preflight_after || true
    exit 5
}

run_probe() {
    local name=$1
    local report=$2
    local timeout_s=$3
    shift 3
    CURRENT_STAGE=$name
    timeout --signal=TERM --kill-after=15s "$timeout_s" \
        "$@" > "$RUN_ROOT/${name}.stdout" \
        2> "$RUN_ROOT/${name}.stderr"
    local rc=$?
    printf '%s\n' "$rc" > "$RUN_ROOT/${name}.returncode"
    if [[ ! -s "$report" ]]; then
        abort_after_probe_failure \
            "$name did not produce its structured report (rc=$rc)"
    fi
}

CURRENT_STAGE=preflight_before
if ! run_physical_preflight preflight_before; then
    echo "selected GPU preflight failed" >&2
    exit 4
fi

CURRENT_STAGE=runtime_identity
if ! timeout --signal=TERM --kill-after=10s 180s \
        python3 "$ROOT/tests/verify_bare_host_runtime_identity.py" \
        --source-root "$ROOT" \
        --runtime-site-packages "$BI100_RUNTIME_SITE_PACKAGES" \
        --runtime-install "$RUNTIME_INSTALL" \
        --out "$RUN_ROOT/runtime_identity.json" \
        > "$RUN_ROOT/runtime_identity.stdout" \
        2> "$RUN_ROOT/runtime_identity.stderr"; then
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
CURRENT_STAGE=preflight_after
if ! run_physical_preflight preflight_after; then
    echo "selected GPU postflight failed" >&2
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
if ! python3 "$ROOT/tests/qualify_qwen36_diagnostic_components.py" \
        "${QUALIFY_ARGS[@]}" \
        > "$RUN_ROOT/qualification.stdout" \
        2> "$RUN_ROOT/qualification.stderr"; then
    echo "diagnostic component qualification rejected the result" >&2
    exit 6
fi

CURRENT_STAGE=completed
exit 0
