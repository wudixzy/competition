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
RUN_ROOT=$2
GPU_INDEX=${GPU_INDEX:-1}
PORT=${PORT:-8030}
STARTUP_TIMEOUT_S=${STARTUP_TIMEOUT_S:-900}
MODEL_PATH=${MODEL_PATH:-/root/shared-storage/models/Qwen/Qwen3.6-35B-A3B-diagnostic-4L-real}
SOURCE_MODEL_PATH=${SOURCE_MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
BI100_RUNTIME_SITE_PACKAGES=${BI100_RUNTIME_SITE_PACKAGES:-}
ACTIVE_PID=""
ACTIVE_PGID=""
INITIAL_PREFLIGHT_PASSED=0
CURRENT_STAGE=argument_validation

if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
    echo "GPU_INDEX must be a non-negative integer" >&2
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
if [[ ! "$INSTANCE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    echo "INSTANCE must be a short non-sensitive label" >&2
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
    echo "M1-86 runner refuses a dirty source tree" >&2
    exit 3
fi
if [[ -z "$BI100_RUNTIME_SITE_PACKAGES" ]]; then
    echo "BI100_RUNTIME_SITE_PACKAGES is required" >&2
    exit 3
fi

MODEL_PATH=$(python3 - "$MODEL_PATH" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=True))
PY
)
SOURCE_MODEL_PATH=$(python3 - "$SOURCE_MODEL_PATH" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=True))
PY
)
BI100_RUNTIME_SITE_PACKAGES=$(python3 - \
        "$BI100_RUNTIME_SITE_PACKAGES" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=True))
PY
)
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

check_port_available() {
    python3 - "$PORT" <<'PY'
import socket
import sys
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", int(sys.argv[1])))
PY
}

stop_active_group() {
    local rc=0
    if [[ -n "$ACTIVE_PGID" ]]; then
        bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID" 60 20 || rc=$?
    elif [[ -n "$ACTIVE_PID" ]]; then
        if kill -0 "$ACTIVE_PID" 2>/dev/null; then
            kill -TERM "$ACTIVE_PID" 2>/dev/null || true
            for _ in $(seq 1 60); do
                kill -0 "$ACTIVE_PID" 2>/dev/null || break
                sleep 1
            done
        fi
        if kill -0 "$ACTIVE_PID" 2>/dev/null; then
            kill -KILL "$ACTIVE_PID" 2>/dev/null || true
            for _ in $(seq 1 20); do
                kill -0 "$ACTIVE_PID" 2>/dev/null || break
                sleep 1
            done
        fi
        wait "$ACTIVE_PID" 2>/dev/null || true
        if kill -0 "$ACTIVE_PID" 2>/dev/null; then
            echo "unisolated service leader survived cleanup" >&2
            rc=1
        fi
    fi
    if [[ $rc -eq 0 ]]; then
        ACTIVE_PID=""
        ACTIVE_PGID=""
    fi
    return "$rc"
}

scan_log() {
    local log=$1
    local output=$2
    local pattern
    if [[ ! -f "$log" ]]; then
        printf 'missing log: %s\n' "$log" > "$output"
        return 1
    fi
    pattern='CUDA error|SIGSEGV|Fatal Python error|out of memory|device-side assert|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|engine iteration timed out|watchdog.*tim(e|ed) out|scheduler requested a missing GDN prefix state|non-finite GatedDeltaNet'
    if grep -Eiq "$pattern" "$log" 2>/dev/null; then
        grep -Ein "$pattern" "$log" > "$output" 2>/dev/null || true
        return 1
    fi
    : > "$output"
}

write_service_contract() {
    local arm=$1
    local image_limit=$2
    shift 2
    python3 - "$arm/service_contract.json" "$ROOT" "$MODEL_PATH" \
            "$RUNTIME_INSTALL" "$RUN_ROOT/runtime_overlay_identity.json" \
            "$SOURCE_REVISION" "$image_limit" "$@" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

out = Path(sys.argv[1])
root = Path(sys.argv[2])
model = Path(sys.argv[3])
install_path = Path(sys.argv[4])
overlay_path = Path(sys.argv[5])
source_revision = sys.argv[6]
image_limit = int(sys.argv[7])
command = sys.argv[8:]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
environment_names = (
    "BI100_ATTN_COREX_PAGED_GATHER",
    "BI100_BLOCK_MAJOR_CPU_KV",
    "BI100_CACHE_TRACE",
    "BI100_CPU_KV_OFFLOAD",
    "BI100_DIAGNOSTIC_LAYER_TRACE",
    "BI100_EXECUTOR_STARTUP_DEBUG",
    "BI100_GDN_ALLOW_NAN_ZERO",
    "BI100_GDN_CACHE_POLICY",
    "BI100_GDN_COMBINED_QK_NORM",
    "BI100_GDN_COREX_PACKED_DECODE",
    "BI100_GDN_FINITE_CHECK",
    "BI100_GDN_RESTORE_MODE",
    "BI100_HYBRID_KV_ACCOUNTING",
    "BI100_MOE_COREX_DIRECT_ROUTED",
    "BI100_PREFIX_DTYPE",
    "BI100_PREFIX_MODEL_FINGERPRINT",
    "BI100_PREFIX_TP_SIZE",
    "CUDA_VISIBLE_DEVICES",
    "ENABLE_CUSTOM_IPC",
    "VLLM_ENGINE_ITERATION_TIMEOUT_S",
)
report = {
    "schema": "bi100-m1-86-service-contract-v1",
    "version": 1,
    "source_revision": source_revision,
    "source_branch": os.environ["SOURCE_BRANCH"],
    "runtime_tree_sha256": overlay["runtime_tree_sha256"],
    "runtime_install_sha256": sha(install_path),
    "model_path": str(model),
    "model_manifest_sha256": sha(
        model / "diagnostic-checkpoint-manifest.json"),
    "command": command,
    "environment": {
        name: os.environ.get(name) for name in environment_names
    },
    "tensor_parallel_size": 1,
    "max_model_len": 262144,
    "image_limit": image_limit,
    "runtime_source_files_match": overlay.get("qualified"),
    "semantic_quality_evaluated": False,
    "production_promotion_authorized": False,
}
out.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

write_arm_status() {
    local arm=$1
    local label=$2
    local image_limit=$3
    local final_rc=$4
    python3 - "$arm" "$label" "$image_limit" "$final_rc" "$PORT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def rc(name):
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
    "preflight_before": rc("preflight_before.rc"),
    "port_preflight": rc("port_preflight.rc"),
    "service_contract": rc("service_contract.rc"),
    "process_group": rc("process_group.rc"),
    "startup": rc("startup.rc"),
    "probe": rc("probe.rc"),
    "capacity": rc("capacity.rc"),
    "cleanup": rc("cleanup.rc"),
    "cache_trace": rc("cache_trace.rc"),
    "attribution": rc("attribution.rc"),
    "fatal_scan": rc("fatal_scan.rc"),
    "service_postflight": rc("service_postflight.rc"),
    "preflight_after": rc("preflight_after.rc"),
    "preflight_comparison": rc("preflight_comparison.rc"),
}
report = {
    "schema": "bi100-m1-86-multi-image-arm-v1",
    "version": 1,
    "qualified": int(sys.argv[4]) == 0 and all(
        value == 0 for value in gates.values()),
    "label": sys.argv[2],
    "image_limit": int(sys.argv[3]),
    "returncode": int(sys.argv[4]),
    "port": int(sys.argv[5]),
    "gates": gates,
    "artifact_sha256": {
        "probe": sha("probe.json"),
        "attribution": sha("attribution.json"),
        "capacity": sha("capacity.json"),
        "cache_trace": sha("cache_trace.json"),
        "service_contract": sha("service_contract.json"),
        "process_group_identity": sha("process_group_identity.json"),
        "service_postflight": sha("service_postflight.json"),
        "preflight_comparison": sha("preflight_comparison.json"),
    },
    "full_model_evaluated": False,
    "semantic_quality_evaluated": False,
    "production_promotion_authorized": False,
}
(root / "status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

run_arm() {
    local label=$1
    local image_limit=$2
    local expected_status=$3
    local arm="$RUN_ROOT/$label"
    local arm_rc=0
    local rc=0
    local observed_pgid=""
    local process_group_ok=0
    local startup_ok=0
    local startup_deadline=0
    local cleanup_rc=0
    local trace_rc=0
    local attribution_rc=0
    local fatal_rc=0
    local postflight_rc=0
    local preflight_after_rc=0
    local comparison_rc=0

    mkdir -p "$arm/runtime-workdir"
    set +e
    run_preflight "$arm/preflight_before"
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$arm/preflight_before.rc"
    [[ $rc -eq 0 ]] || arm_rc=1

    set +e
    check_port_available \
        > "$arm/port_preflight.stdout" \
        2> "$arm/port_preflight.stderr"
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$arm/port_preflight.rc"
    [[ $rc -eq 0 ]] || arm_rc=1

    COMMAND=(
        python3
        -m vllm.entrypoints.openai.api_server
        --host 127.0.0.1
        --port "$PORT"
        --model "$MODEL_PATH"
        --served-model-name llm
        --max-model-len 262144
        --gpu-memory-utilization 0.9
        --trust-remote-code
        --tensor-parallel-size 1
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
    if [[ "$image_limit" == 2 ]]; then
        COMMAND+=(--limit-mm-per-prompt image=2)
    fi
    printf '%q ' "${COMMAND[@]}" > "$arm/service_command.txt"
    printf '\n' >> "$arm/service_command.txt"

    export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
    export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
    export ENABLE_CUSTOM_IPC=1
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
    export BI100_PREFIX_TP_SIZE=1
    export BI100_BLOCK_MAJOR_CPU_KV=0
    export BI100_CPU_KV_OFFLOAD=0
    export BI100_MOE_COREX_DIRECT_ROUTED=0
    export BI100_GDN_COREX_PACKED_DECODE=0
    export BI100_GDN_COMBINED_QK_NORM=0
    export BI100_ATTN_COREX_PAGED_GATHER=1
    export SOURCE_BRANCH

    set +e
    write_service_contract "$arm" "$image_limit" "${COMMAND[@]}" \
        > "$arm/service_contract.stdout" \
        2> "$arm/service_contract.stderr"
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$arm/service_contract.rc"
    [[ $rc -eq 0 ]] || arm_rc=1

    if [[ $arm_rc -eq 0 ]]; then
        (
            cd "$arm/runtime-workdir" || exit 1
            export PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH"
            export LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH"
            export PATH="$COREX_PATH"
            exec python3 "$ROOT/scripts/exec_bi100_session.py" \
                "$arm/process_group_identity.json" -- "${COMMAND[@]}"
        ) > "$arm/server.log" 2>&1 &
        ACTIVE_PID=$!
        ACTIVE_PGID=$ACTIVE_PID
        for _ in $(seq 1 20); do
            if [[ -s "$arm/process_group_identity.json" ]]; then
                if observed_pgid=$(python3 - \
                        "$arm/process_group_identity.json" \
                        "$ACTIVE_PID" <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = int(sys.argv[2])
if (
    value.get("schema") != "bi100-process-session-v1"
    or value.get("version") != 1
    or value.get("pid") != expected
    or value.get("pgid") != expected
    or value.get("sid") != expected
):
    raise SystemExit(1)
print(value["pgid"])
PY
                ); then
                    process_group_ok=1
                    break
                fi
            fi
            kill -0 "$ACTIVE_PID" 2>/dev/null || break
            sleep 1
        done
        if [[ "$process_group_ok" != 1 \
                || "$observed_pgid" != "$ACTIVE_PGID" ]]; then
            echo "$label service did not enter an isolated process group" >&2
            arm_rc=1
        else
            startup_deadline=$((SECONDS + STARTUP_TIMEOUT_S))
            while ((SECONDS < startup_deadline)); do
                if python3 - "$PORT" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
urllib.request.urlopen(
    f"http://127.0.0.1:{sys.argv[1]}/health", timeout=5).read()
PY
                then
                    startup_ok=1
                    break
                fi
                if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
        fi
    fi
    printf '%s\n' "$((1 - process_group_ok))" > "$arm/process_group.rc"

    if [[ "$startup_ok" == 1 ]]; then
        printf '%s\n' 0 > "$arm/startup.rc"
        set +e
        python3 "$ROOT/scripts/check_startup_capacity.py" \
            "$arm/server.log" \
            --max-model-len 262144 --block-size 16 \
            --out "$arm/capacity.json" \
            > "$arm/capacity.stdout" 2> "$arm/capacity.stderr"
        rc=$?
        set -e
        printf '%s\n' "$rc" > "$arm/capacity.rc"
        [[ $rc -eq 0 ]] || arm_rc=1

        set +e
        timeout --signal=TERM --kill-after=70s 1800s \
            env PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH" \
            LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
            python3 "$ROOT/tests/qwen36_multi_image_http_gate.py" \
            --base "http://127.0.0.1:$PORT" \
            --model-path "$MODEL_PATH" \
            --timeout-s 600 \
            --expected-two-image-status "$expected_status" \
            --json-out "$arm/probe.json" \
            > "$arm/probe.stdout" 2> "$arm/probe.stderr"
        rc=$?
        set -e
        printf '%s\n' "$rc" > "$arm/probe.rc"
        [[ $rc -eq 0 ]] || arm_rc=1
    else
        printf '%s\n' 1 > "$arm/startup.rc"
        printf '%s\n' 1 > "$arm/capacity.rc"
        printf '%s\n' 1 > "$arm/probe.rc"
        arm_rc=1
    fi

    set +e
    stop_active_group
    cleanup_rc=$?
    set -e
    printf '%s\n' "$cleanup_rc" > "$arm/cleanup.rc"
    [[ $cleanup_rc -eq 0 ]] || arm_rc=1
    unset CUDA_VISIBLE_DEVICES

    if [[ -f "$arm/server.log" && -f "$arm/probe.json" ]]; then
        set +e
        python3 "$ROOT/tests/qualify_m1_86_multi_image_trace.py" \
            "$arm/server.log" "$arm/probe.json" \
            --mode "$label" --out "$arm/cache_trace.json" \
            > "$arm/cache_trace.stdout" \
            2> "$arm/cache_trace.stderr"
        trace_rc=$?
        set -e
    else
        trace_rc=1
    fi
    printf '%s\n' "$trace_rc" > "$arm/cache_trace.rc"
    [[ $trace_rc -eq 0 ]] || arm_rc=1

    if [[ -f "$arm/server.log" ]]; then
        set +e
        python3 "$ROOT/tests/summarize_api_4xx_log.py" \
            "$arm/server.log" --out "$arm/attribution.json" \
            > "$arm/attribution.stdout" 2> "$arm/attribution.stderr"
        attribution_rc=$?
        set -e
    else
        attribution_rc=1
    fi
    printf '%s\n' "$attribution_rc" > "$arm/attribution.rc"
    [[ $attribution_rc -eq 0 ]] || arm_rc=1

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

    set +e
    run_preflight "$arm/preflight_after"
    preflight_after_rc=$?
    set -e
    printf '%s\n' "$preflight_after_rc" > "$arm/preflight_after.rc"
    if [[ $preflight_after_rc -eq 0 ]]; then
        set +e
        python3 "$ROOT/tests/compare_bi100_preflights.py" \
            --preflight "before=$arm/preflight_before.json" \
            --preflight "after=$arm/preflight_after.json" \
            --expected-gpus "$GPU_INDEX" \
            --max-free-memory-drop-bytes 1073741824 \
            --out "$arm/preflight_comparison.json" \
            > "$arm/preflight_comparison.stdout" \
            2> "$arm/preflight_comparison.stderr"
        comparison_rc=$?
        set -e
    else
        comparison_rc=1
    fi
    printf '%s\n' "$comparison_rc" > "$arm/preflight_comparison.rc"
    [[ $preflight_after_rc -eq 0 && $comparison_rc -eq 0 ]] || arm_rc=1

    write_arm_status "$arm" "$label" "$image_limit" "$arm_rc"
    return "$arm_rc"
}

write_runner_status() {
    local final_rc=$1
    python3 - "$RUN_ROOT" "$INSTANCE" "$SOURCE_REVISION" \
            "$SOURCE_BRANCH" "$GPU_INDEX" "$final_rc" "$PORT" \
            "$CURRENT_STAGE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def rc(name):
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
    "checkpoint_verify": rc("checkpoint_verify.rc"),
    "runtime_overlay_identity": rc("runtime_overlay_identity.rc"),
    "initial_preflight": rc("initial_preflight.rc"),
    "control": rc("control.rc"),
    "candidate": rc("candidate.rc"),
    "comparison": rc("comparison.rc"),
    "cleanup": rc("cleanup.rc"),
    "final_postflight": rc("final_postflight.rc"),
    "final_preflight": rc("final_preflight.rc"),
    "final_preflight_comparison": rc("final_preflight_comparison.rc"),
    "fatal_scan": rc("fatal_scan.rc"),
    "timeout_scan": rc("timeout_scan.rc"),
}
report = {
    "schema": "bi100-m1-86-multi-image-ab-runner-v1",
    "version": 1,
    "qualified": int(sys.argv[6]) == 0 and all(
        value == 0 for value in gates.values()),
    "source_revision": sys.argv[3],
    "source_branch": sys.argv[4],
    "instance": sys.argv[2],
    "physical_gpu": int(sys.argv[5]),
    "returncode": int(sys.argv[6]),
    "port": int(sys.argv[7]),
    "terminal_stage": sys.argv[8],
    "gates": gates,
    "artifact_sha256": {
        "runtime_overlay_identity": sha("runtime_overlay_identity.json"),
        "comparison": sha("comparison.json"),
        "final_postflight": sha("final_postflight.json"),
        "final_preflight_comparison": sha(
            "final_preflight_comparison.json"),
    },
    "full_model_evaluated": False,
    "semantic_quality_evaluated": False,
    "performance_evaluated": False,
    "production_promotion_authorized": False,
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
    local postflight_rc=0
    local preflight_rc=0
    local comparison_rc=0
    local fatal_rc=0
    local timeout_rc=0
    local file
    local value
    local pattern
    trap - EXIT TERM INT
    set +e

    stop_active_group
    cleanup_rc=$?
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/cleanup.rc"
    unset CUDA_VISIBLE_DEVICES

    run_postflight "$RUN_ROOT/final_postflight"
    postflight_rc=$?
    printf '%s\n' "$postflight_rc" > "$RUN_ROOT/final_postflight.rc"

    if [[ "$INITIAL_PREFLIGHT_PASSED" == 1 ]]; then
        run_preflight "$RUN_ROOT/final_preflight"
        preflight_rc=$?
        printf '%s\n' "$preflight_rc" > "$RUN_ROOT/final_preflight.rc"
        if [[ $preflight_rc -eq 0 ]]; then
            python3 "$ROOT/tests/compare_bi100_preflights.py" \
                --preflight "initial=$RUN_ROOT/initial_preflight.json" \
                --preflight "final=$RUN_ROOT/final_preflight.json" \
                --expected-gpus "$GPU_INDEX" \
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
    else
        preflight_rc=1
        comparison_rc=1
        printf '%s\n' 1 > "$RUN_ROOT/final_preflight.rc"
        printf '%s\n' 1 > "$RUN_ROOT/final_preflight_comparison.rc"
    fi

    pattern='CUDA error|SIGSEGV|Fatal Python error|out of memory|device-side assert|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|engine iteration timed out|watchdog.*tim(e|ed) out|scheduler requested a missing GDN prefix state|non-finite GatedDeltaNet'
    : > "$RUN_ROOT/fatal_scan.txt"
    while IFS= read -r -d '' file; do
        if grep -Eiq "$pattern" "$file"; then
            printf 'file=%s\n' "$file" >> "$RUN_ROOT/fatal_scan.txt"
            grep -Ein "$pattern" "$file" \
                >> "$RUN_ROOT/fatal_scan.txt" || true
            fatal_rc=1
        fi
    done < <(find "$RUN_ROOT" -type f \
        \( -name server.log -o -name '*.stdout' -o -name '*.stderr' \) \
        -print0)
    printf '%s\n' "$fatal_rc" > "$RUN_ROOT/fatal_scan.rc"

    : > "$RUN_ROOT/timeout_scan.txt"
    while IFS= read -r -d '' file; do
        value=$(tr -d '[:space:]' < "$file")
        case "$value" in
            124|137)
                printf '%s=%s\n' "$file" "$value" \
                    >> "$RUN_ROOT/timeout_scan.txt"
                timeout_rc=1
                ;;
        esac
    done < <(find "$RUN_ROOT" -type f -name '*.rc' -print0)
    printf '%s\n' "$timeout_rc" > "$RUN_ROOT/timeout_scan.rc"

    if [[ $cleanup_rc -ne 0 || $postflight_rc -ne 0 \
            || $preflight_rc -ne 0 || $comparison_rc -ne 0 \
            || $fatal_rc -ne 0 || $timeout_rc -ne 0 ]]; then
        final_rc=1
    fi
    write_runner_status "$final_rc"
    exit "$final_rc"
}
trap 'exit 143' TERM
trap 'exit 130' INT
trap finish EXIT

CURRENT_STAGE=checkpoint_verify
set +e
python3 "$ROOT/scripts/verify_qwen36_diagnostic_checkpoint.py" \
    --source "$SOURCE_MODEL_PATH" \
    --checkpoint "$MODEL_PATH" \
    --full-hash \
    --json-out "$RUN_ROOT/checkpoint_verify.json" \
    > "$RUN_ROOT/checkpoint_verify.stdout" \
    2> "$RUN_ROOT/checkpoint_verify.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/checkpoint_verify.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=runtime_overlay_identity
set +e
timeout --signal=TERM --kill-after=70s 240s \
    env PYTHONPATH="$ROOT/tests" \
    python3 "$ROOT/tests/verify_bare_host_runtime_identity.py" \
    --source-root "$ROOT" \
    --runtime-site-packages "$BI100_RUNTIME_SITE_PACKAGES" \
    --runtime-install "$RUNTIME_INSTALL" \
    --out "$RUN_ROOT/runtime_overlay_identity.json" \
    > "$RUN_ROOT/runtime_overlay_identity.stdout" \
    2> "$RUN_ROOT/runtime_overlay_identity.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/runtime_overlay_identity.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=initial_preflight
set +e
run_preflight "$RUN_ROOT/initial_preflight"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/initial_preflight.rc"
[[ $rc -eq 0 ]]
INITIAL_PREFLIGHT_PASSED=1

CURRENT_STAGE=control
set +e
run_arm control 1 400
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/control.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=candidate
set +e
run_arm candidate 2 200
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/candidate.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=comparison
set +e
python3 "$ROOT/tests/compare_m1_86_multi_image_ab.py" \
    --control-report "$RUN_ROOT/control/probe.json" \
    --candidate-report "$RUN_ROOT/candidate/probe.json" \
    --control-attribution "$RUN_ROOT/control/attribution.json" \
    --candidate-attribution "$RUN_ROOT/candidate/attribution.json" \
    --control-status "$RUN_ROOT/control/status.json" \
    --candidate-status "$RUN_ROOT/candidate/status.json" \
    --control-contract "$RUN_ROOT/control/service_contract.json" \
    --candidate-contract "$RUN_ROOT/candidate/service_contract.json" \
    --control-capacity "$RUN_ROOT/control/capacity.json" \
    --candidate-capacity "$RUN_ROOT/candidate/capacity.json" \
    --control-trace "$RUN_ROOT/control/cache_trace.json" \
    --candidate-trace "$RUN_ROOT/candidate/cache_trace.json" \
    --control-process-group \
        "$RUN_ROOT/control/process_group_identity.json" \
    --candidate-process-group \
        "$RUN_ROOT/candidate/process_group_identity.json" \
    --out "$RUN_ROOT/comparison.json" \
    > "$RUN_ROOT/comparison.stdout" \
    2> "$RUN_ROOT/comparison.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/comparison.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=completed
exit 0
