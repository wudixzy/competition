#!/bin/bash
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

usage() {
    cat >&2 <<'EOF'
Usage:
  run_qwen36_diagnostic_gate.sh MODEL_PATH TP_SIZE GPU_LIST INSTANCE RUN_ROOT

TP_SIZE must be 1 or 2. GPU_LIST contains physical BI100 indices, for example
"3" or "1,2". The immutable runtime overlay is supplied through
BI100_RUNTIME_SITE_PACKAGES. Results must stay outside the source repository.
EOF
}

if [[ $# -ne 5 ]]; then
    usage
    exit 2
fi

MODEL_PATH=$1
TP_SIZE=$2
GPU_LIST=$3
INSTANCE=$4
RUN_ROOT=$5
SOURCE_MODEL_PATH=${SOURCE_MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
PORT=${PORT:-8000}
STARTUP_TIMEOUT_S=${STARTUP_TIMEOUT_S:-900}
ACTIVE_PID=""
ACTIVE_PGID=""

case "$TP_SIZE" in
    1|2) ;;
    *) echo "TP_SIZE must be 1 or 2" >&2; exit 2 ;;
esac
if [[ ! "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPU_LIST must be a comma-separated list of indices" >&2
    exit 2
fi
IFS=, read -r -a GPU_ARRAY <<< "$GPU_LIST"
if [[ ${#GPU_ARRAY[@]} -ne $TP_SIZE ]]; then
    echo "GPU_LIST count must equal TP_SIZE" >&2
    exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ || "$PORT" -lt 1 || "$PORT" -gt 65535 ]]; then
    echo "PORT must be between 1 and 65535" >&2
    exit 2
fi
if [[ ! "$STARTUP_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]]; then
    echo "STARTUP_TIMEOUT_S must be a positive integer" >&2
    exit 2
fi

MODEL_PATH=$(python3 - "$MODEL_PATH" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=True))
PY
) || exit 3
SOURCE_MODEL_PATH=$(python3 - "$SOURCE_MODEL_PATH" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=True))
PY
) || exit 3
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
    echo "diagnostic gate refuses a dirty source tree" >&2
    exit 3
fi
if [[ -z "${BI100_RUNTIME_SITE_PACKAGES:-}" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/vllm" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/transformers" ]]; then
    echo "BI100_RUNTIME_SITE_PACKAGES must identify an immutable overlay" >&2
    exit 3
fi
BI100_RUNTIME_SITE_PACKAGES=$(python3 - "$BI100_RUNTIME_SITE_PACKAGES" <<'PY'
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

mkdir -p "$RUN_ROOT"
git -C "$ROOT" rev-parse HEAD > "$RUN_ROOT/source_revision.txt"
git -C "$ROOT" branch --show-current > "$RUN_ROOT/source_branch.txt"
printf '%s\n' "$GPU_LIST" > "$RUN_ROOT/gpu_list.txt"
printf '%s\n' "$TP_SIZE" > "$RUN_ROOT/tp_size.txt"

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
export PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH"
export LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
export PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/openmpi/bin

stop_service() {
    local rc=0
    if [[ -n "$ACTIVE_PGID" ]]; then
        bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID" || rc=$?
    fi
    ACTIVE_PID=""
    ACTIVE_PGID=""
    return "$rc"
}

cleanup() {
    local rc=$?
    trap - EXIT
    set +e
    stop_service
    exit "$rc"
}
trap cleanup EXIT

if ! python3 "$ROOT/scripts/verify_qwen36_diagnostic_checkpoint.py" \
        --source "$SOURCE_MODEL_PATH" \
        --checkpoint "$MODEL_PATH" \
        --full-hash \
        --json-out "$RUN_ROOT/checkpoint_verify.json" \
        > "$RUN_ROOT/checkpoint_verify.stdout" \
        2> "$RUN_ROOT/checkpoint_verify.stderr"; then
    echo "diagnostic checkpoint verification failed" >&2
    exit 4
fi

if ! timeout --signal=TERM --kill-after=10s 180s \
        python3 "$ROOT/tests/bi100_preflight.py" \
        --gpus "$GPU_LIST" --timeout-s 25 --matmul-size 1024 \
        --json-out "$RUN_ROOT/preflight_before.json" \
        > "$RUN_ROOT/preflight_before.stdout" \
        2> "$RUN_ROOT/preflight_before.stderr"; then
    echo "selected GPU preflight failed" >&2
    exit 4
fi

if [[ "$TP_SIZE" == 2 ]]; then
    if ! timeout --signal=TERM --kill-after=10s 180s \
            python3 "$ROOT/tests/bi100_nccl_preflight.py" \
            --gpus "$GPU_LIST" --timeout-s 60 \
            --json-out "$RUN_ROOT/nccl_before.json" \
            > "$RUN_ROOT/nccl_before.stdout" \
            2> "$RUN_ROOT/nccl_before.stderr"; then
        echo "selected TP2 NCCL preflight failed" >&2
        exit 4
    fi
fi

if ! python3 "$ROOT/tests/gdn_action_broadcast_gate.py" \
        --out "$RUN_ROOT/gdn_action_broadcast.json" \
        > "$RUN_ROOT/gdn_action_broadcast.stdout" \
        2> "$RUN_ROOT/gdn_action_broadcast.stderr"; then
    echo "GDN action broadcast gate failed" >&2
    exit 4
fi

python3 - "$ROOT" "$MODEL_PATH" "$SOURCE_MODEL_PATH" \
        "$BI100_RUNTIME_SITE_PACKAGES" "$RUNTIME_INSTALL" "$GPU_LIST" \
        "$TP_SIZE" "$INSTANCE" "$RUN_ROOT/runtime_identity.json" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
model = Path(sys.argv[2])
source_model = Path(sys.argv[3])
site = Path(sys.argv[4])
install = Path(sys.argv[5])
out = Path(sys.argv[9])

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

report = {
    "schema": "qwen36-diagnostic-runtime-identity-v1",
    "version": 1,
    "source_revision": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "source_branch": subprocess.check_output(
        ["git", "-C", str(root), "branch", "--show-current"],
        text=True).strip(),
    "diagnostic_model": str(model),
    "source_model": str(source_model),
    "diagnostic_manifest_sha256": sha(
        model / "diagnostic-checkpoint-manifest.json"),
    "diagnostic_config_sha256": sha(model / "config.json"),
    "diagnostic_index_sha256": sha(model / "model.safetensors.index.json"),
    "runtime_site_packages": str(site),
    "runtime_install_report": str(install),
    "runtime_install_sha256": sha(install),
    "physical_gpus": sys.argv[6].split(","),
    "tensor_parallel_size": int(sys.argv[7]),
    "instance": sys.argv[8],
    "max_model_len": 262144,
    "semantic_quality_evaluated": False,
    "production_promotion_authorized": False,
}
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export ENABLE_CUSTOM_IPC=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export BI100_EXECUTOR_STARTUP_DEBUG=1
export BI100_DIAGNOSTIC_LAYER_TRACE=1
export BI100_HYBRID_KV_ACCOUNTING=full_attention
export BI100_GDN_CACHE_POLICY=fine32
export BI100_GDN_RESTORE_MODE=direct
export BI100_GDN_ALLOW_NAN_ZERO=0
export BI100_GDN_FINITE_CHECK=0
export BI100_CACHE_TRACE=1
export BI100_PREFIX_MODEL_FINGERPRINT=Qwen3.6-35B-A3B-diagnostic-4L-real
export BI100_PREFIX_DTYPE=float16
export BI100_PREFIX_TP_SIZE="$TP_SIZE"
export BI100_BLOCK_MAJOR_CPU_KV=0
export BI100_CPU_KV_OFFLOAD=0
export BI100_MOE_COREX_DIRECT_ROUTED=1
export BI100_GDN_COREX_PACKED_DECODE=1

RUNTIME_WORKDIR="$RUN_ROOT/runtime-workdir"
mkdir -p "$RUNTIME_WORKDIR"
COMMAND=(
    python3
    -m vllm.entrypoints.openai.api_server
    --host 0.0.0.0
    --port "$PORT"
    --model "$MODEL_PATH"
    --served-model-name llm
    --max-model-len 262144
    --gpu-memory-utilization 0.9
    --trust-remote-code
    --tensor-parallel-size "$TP_SIZE"
    --max-num-seqs 1
    --disable-log-requests
    --disable-frontend-multiprocessing
    --max-num-batched-tokens 8192
    --enable-chunked-prefill
    --max-seq-len-to-capture 32768
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --enable-prefix-caching
)
printf '%q ' "${COMMAND[@]}" > "$RUN_ROOT/service_command.txt"
printf '\n' >> "$RUN_ROOT/service_command.txt"

(
    cd "$RUNTIME_WORKDIR" || exit 1
    exec setsid "${COMMAND[@]}"
) > "$RUN_ROOT/server.log" 2>&1 &
ACTIVE_PID=$!
sleep 1
ACTIVE_PGID=$(ps -o pgid= -p "$ACTIVE_PID" | tr -d ' ')
if [[ -z "$ACTIVE_PGID" || "$ACTIVE_PGID" != "$ACTIVE_PID" ]]; then
    echo "service did not start as a dedicated process-group leader" >&2
    exit 5
fi
printf '%s\n' "$ACTIVE_PID" > "$RUN_ROOT/service.pid"
printf '%s\n' "$ACTIVE_PGID" > "$RUN_ROOT/service.pgid"

health() {
    python3 - "$PORT" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
urllib.request.urlopen(
    f"http://127.0.0.1:{sys.argv[1]}/health", timeout=5).read()
PY
}

startup_ok=0
for _ in $(seq 1 "$STARTUP_TIMEOUT_S"); do
    if health; then
        startup_ok=1
        break
    fi
    if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
        break
    fi
    sleep 1
done
if [[ "$startup_ok" != 1 ]]; then
    echo "diagnostic service failed to become healthy" >&2
    exit 5
fi
printf '%s\n' 0 > "$RUN_ROOT/startup.rc"

overall_rc=0
if python3 "$ROOT/tests/qwen36_diagnostic_api.py" \
        --base "http://127.0.0.1:$PORT" \
        --model-path "$MODEL_PATH" \
        --timeout-s 300 \
        --json-out "$RUN_ROOT/api_gate.json" \
        > "$RUN_ROOT/api_gate.stdout" \
        2> "$RUN_ROOT/api_gate.stderr"; then
    printf '%s\n' 0 > "$RUN_ROOT/api_gate.rc"
else
    rc=$?
    printf '%s\n' "$rc" > "$RUN_ROOT/api_gate.rc"
    overall_rc=1
fi

if python3 "$ROOT/tests/prefix_boundary_api.py" \
        --base "http://127.0.0.1:$PORT" \
        --model-path "$MODEL_PATH" \
        --block-context-len 11296 \
        --prefix-query-len 320 \
        --max-tokens 1 \
        --timeout-s 600 \
        --run-id "m1-60-${TP_SIZE}-${INSTANCE}" \
        --json-out "$RUN_ROOT/prefix_boundary.json" \
        > "$RUN_ROOT/prefix_boundary.stdout" \
        2> "$RUN_ROOT/prefix_boundary.stderr"; then
    printf '%s\n' 0 > "$RUN_ROOT/prefix_boundary.rc"
else
    rc=$?
    printf '%s\n' "$rc" > "$RUN_ROOT/prefix_boundary.rc"
    overall_rc=1
fi

if ! stop_service; then
    overall_rc=1
fi
unset CUDA_VISIBLE_DEVICES

FATAL_PATTERN='CUDA error|SIGSEGV|Fatal Python error|out of memory|worker process.*died|Gloo.*failed|Connection reset by peer|scheduler requested a missing GDN prefix state|non-finite GatedDeltaNet'
if grep -Eiq "$FATAL_PATTERN" "$RUN_ROOT/server.log"; then
    grep -Ein "$FATAL_PATTERN" "$RUN_ROOT/server.log" \
        > "$RUN_ROOT/fatal_scan.txt" || true
    printf '%s\n' 1 > "$RUN_ROOT/fatal_scan.rc"
    overall_rc=1
else
    : > "$RUN_ROOT/fatal_scan.txt"
    printf '%s\n' 0 > "$RUN_ROOT/fatal_scan.rc"
fi

expected_layer_lines=$((TP_SIZE * 4))
actual_layer_lines=$(grep -Ec \
    '\[BI100 DIAGNOSTIC\].*layer=[0-3].*mlp=Qwen3_5MoeSparseBlock stage=completed' \
    "$RUN_ROOT/server.log" || true)
printf '%s\n' "$actual_layer_lines" > "$RUN_ROOT/layer_trace_count.txt"
if [[ "$actual_layer_lines" -lt "$expected_layer_lines" ]]; then
    echo "expected at least $expected_layer_lines completed layer traces, got $actual_layer_lines" \
        > "$RUN_ROOT/layer_trace_error.txt"
    printf '%s\n' 1 > "$RUN_ROOT/layer_trace.rc"
    overall_rc=1
else
    : > "$RUN_ROOT/layer_trace_error.txt"
    printf '%s\n' 0 > "$RUN_ROOT/layer_trace.rc"
fi

if timeout --signal=TERM --kill-after=10s 180s \
        python3 "$ROOT/tests/bi100_preflight.py" \
        --gpus "$GPU_LIST" --timeout-s 25 --matmul-size 1024 \
        --json-out "$RUN_ROOT/preflight_after.json" \
        > "$RUN_ROOT/preflight_after.stdout" \
        2> "$RUN_ROOT/preflight_after.stderr"; then
    printf '%s\n' 0 > "$RUN_ROOT/preflight_after.rc"
else
    rc=$?
    printf '%s\n' "$rc" > "$RUN_ROOT/preflight_after.rc"
    overall_rc=1
fi

python3 - "$RUN_ROOT" "$overall_rc" <<'PY'
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
        return int(path.read_text().strip())
    except ValueError:
        return None

def sha(name):
    path = root / name
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

api = json.loads((root / "api_gate.json").read_text()) \
    if (root / "api_gate.json").is_file() else None
prefix = json.loads((root / "prefix_boundary.json").read_text()) \
    if (root / "prefix_boundary.json").is_file() else None
report = {
    "schema": "qwen36-diagnostic-service-gate-v1",
    "version": 1,
    "qualified": int(sys.argv[2]) == 0,
    "runtime_identity": json.loads(
        (root / "runtime_identity.json").read_text()),
    "gates": {
        "api": read_rc("api_gate.rc"),
        "prefix_boundary": read_rc("prefix_boundary.rc"),
        "fatal_scan": read_rc("fatal_scan.rc"),
        "layer_trace": read_rc("layer_trace.rc"),
        "preflight_after": read_rc("preflight_after.rc"),
    },
    "layer_trace_count": int(
        (root / "layer_trace_count.txt").read_text().strip()),
    "api_summary": {
        "qualified": api.get("qualified"),
        "case_count": api.get("case_count"),
    } if api else None,
    "prefix_summary": {
        "partial_cached_tokens": (
            prefix.get("partial_cache", {}).get("cached_tokens")),
        "warm_cached_tokens": (
            prefix.get("warm_cache", {}).get("cached_tokens")),
    } if prefix else None,
    "artifact_sha256": {
        name: sha(name) for name in (
            "checkpoint_verify.json",
            "preflight_before.json",
            "nccl_before.json",
            "gdn_action_broadcast.json",
            "api_gate.json",
            "prefix_boundary.json",
            "server.log",
            "preflight_after.json",
        )
    },
    "semantic_quality_evaluated": False,
    "production_promotion_authorized": False,
}
(root / "status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

trap - EXIT
exit "$overall_rc"
