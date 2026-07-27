#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

if [[ $# -ne 2 ]]; then
    echo "usage: $0 INSTANCE RUN_ROOT" >&2
    exit 2
fi

INSTANCE=$1
RUN_ROOT=$2
GPU_INDEX=${GPU_INDEX:-1}
PORT=${PORT:-8018}
STARTUP_TIMEOUT_S=${STARTUP_TIMEOUT_S:-900}
MODEL_PATH=${MODEL_PATH:-/root/shared-storage/models/Qwen/Qwen3.6-35B-A3B-diagnostic-4L-real}
SOURCE_MODEL_PATH=${SOURCE_MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
CONTROL_RUNTIME_SITE_PACKAGES=${CONTROL_RUNTIME_SITE_PACKAGES:-}
CANDIDATE_RUNTIME_SITE_PACKAGES=${CANDIDATE_RUNTIME_SITE_PACKAGES:-}
CONTROL_REVISION=${CONTROL_REVISION:-cdb1bc41f728a5610a3632ad7923d73a90748919}
CANDIDATE_REVISION=${CANDIDATE_REVISION:-37001edff643d98bf41bf4a52e0a145329003315}
ACTIVE_PID=""
ACTIVE_PGID=""
INITIAL_PREFLIGHT_PASSED=0
CURRENT_STAGE=argument_validation

if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
    echo "GPU_INDEX must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ || "$PORT" -lt 1 || "$PORT" -gt 65533 ]]; then
    echo "PORT must be between 1 and 65533 for three isolated arms" >&2
    exit 2
fi
if [[ ! "$STARTUP_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]]; then
    echo "STARTUP_TIMEOUT_S must be a positive integer" >&2
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
    echo "M1-70 runner refuses a dirty source tree" >&2
    exit 3
fi
if [[ -z "$CONTROL_RUNTIME_SITE_PACKAGES" \
        || -z "$CANDIDATE_RUNTIME_SITE_PACKAGES" ]]; then
    echo "control and candidate immutable overlays are required" >&2
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
CONTROL_RUNTIME_SITE_PACKAGES=$(python3 - \
        "$CONTROL_RUNTIME_SITE_PACKAGES" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=True))
PY
)
CANDIDATE_RUNTIME_SITE_PACKAGES=$(python3 - \
        "$CANDIDATE_RUNTIME_SITE_PACKAGES" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=True))
PY
)
CONTROL_INSTALL=$(dirname "$CONTROL_RUNTIME_SITE_PACKAGES")/install.json
CANDIDATE_INSTALL=$(dirname "$CANDIDATE_RUNTIME_SITE_PACKAGES")/install.json
if [[ ! -f "$CONTROL_INSTALL" || ! -f "$CANDIDATE_INSTALL" ]]; then
    echo "runtime install report is missing" >&2
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

read_rc() {
    local path=$1
    if [[ ! -f "$path" ]]; then
        printf '%s\n' null
        return
    fi
    local value
    value=$(tr -d '[:space:]' < "$path")
    if [[ "$value" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$value"
    else
        printf '%s\n' null
    fi
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

check_port_available() {
    local port=$1
    python3 - "$port" <<'PY'
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
        echo "service PID lacks a verified process group" >&2
        rc=2
    fi
    ACTIVE_PID=""
    ACTIVE_PGID=""
    return "$rc"
}

run_postflight() {
    local output=$1
    env PYTHONPATH="$ROOT/tests" \
        python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus "$GPU_INDEX" \
        --settle-timeout-s 90 --clean-samples 3 \
        --sample-interval-s 2 \
        --out "$output.json" \
        > "$output.stdout" 2> "$output.stderr"
}

scan_log() {
    local log=$1
    local output=$2
    local pattern
    pattern='CUDA error|SIGSEGV|Fatal Python error|out of memory|device-side assert|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|engine iteration timed out|watchdog.*tim(e|ed) out|scheduler requested a missing GDN prefix state|non-finite GatedDeltaNet'
    if grep -Eiq "$pattern" "$log" 2>/dev/null; then
        grep -Ein "$pattern" "$log" > "$output" 2>/dev/null || true
        return 1
    fi
    : > "$output"
}

write_arm_status() {
    local arm=$1
    local final_rc=$2
    local expected_system_status=$3
    local image_limit=$4
    local runtime_revision=$5
    local arm_port=$6
    python3 - "$arm" "$final_rc" "$expected_system_status" \
            "$image_limit" "$runtime_revision" "$arm_port" <<'PY'
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
    "schema": "bi100-m1-70-diagnostic-http-arm-v3",
    "version": 3,
    "qualified": int(sys.argv[2]) == 0,
    "runtime_revision": sys.argv[5],
    "port": int(sys.argv[6]),
    "multiple_system_parts_expected_status": int(sys.argv[3]),
    "image_limit": int(sys.argv[4]),
    "gates": {
        "preflight_before": rc("preflight_before.rc"),
        "port_preflight": rc("port_preflight.rc"),
        "startup": rc("startup.rc"),
        "probe": rc("probe.rc"),
        "cleanup": rc("cleanup.rc"),
        "attribution": rc("attribution.rc"),
        "fatal_scan": rc("fatal_scan.rc"),
        "service_postflight": rc("service_postflight.rc"),
        "preflight_after": rc("preflight_after.rc"),
        "preflight_comparison": rc("preflight_comparison.rc"),
    },
    "artifact_sha256": {
        "probe": sha("probe.json"),
        "attribution": sha("attribution.json"),
        "service_postflight": sha("service_postflight.json"),
    },
    "semantic_quality_evaluated": False,
    "full_model_evaluated": False,
    "production_promotion_authorized": False,
}
(root / "status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8")
PY
}

run_arm() {
    local label=$1
    local site=$2
    local runtime_revision=$3
    local expected_system_status=$4
    local image_limit=$5
    local require_attribution=$6
    local arm_port=$7
    local arm="$RUN_ROOT/$label"
    local arm_rc=0
    local rc=0
    local observed_pgid=""
    local startup_ok=0
    local cleanup_rc=0
    local postflight_rc=0
    local preflight_after_rc=0
    local comparison_rc=0
    local attribution_rc=0
    local fatal_rc=0

    mkdir -p "$arm/runtime-workdir"
    set +e
    run_preflight "$arm/preflight_before"
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$arm/preflight_before.rc"
    if [[ $rc -ne 0 ]]; then
        write_arm_status \
            "$arm" 1 "$expected_system_status" "$image_limit" \
            "$runtime_revision" "$arm_port"
        return 1
    fi

    set +e
    check_port_available "$arm_port" \
        > "$arm/port_preflight.stdout" \
        2> "$arm/port_preflight.stderr"
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$arm/port_preflight.rc"
    if [[ $rc -ne 0 ]]; then
        write_arm_status \
            "$arm" 1 "$expected_system_status" "$image_limit" \
            "$runtime_revision" "$arm_port"
        return 1
    fi

    COMMAND=(
        python3
        -m vllm.entrypoints.openai.api_server
        --host 127.0.0.1
        --port "$arm_port"
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

    (
        cd "$arm/runtime-workdir" || exit 1
        export PYTHONPATH="$ROOT/tests:$site:$SYSTEM_PYTHONPATH"
        export LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH"
        export PATH="$COREX_PATH"
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
        exec setsid "${COMMAND[@]}"
    ) > "$arm/server.log" 2>&1 &
    ACTIVE_PID=$!
    ACTIVE_PGID=$ACTIVE_PID
    for _ in $(seq 1 20); do
        observed_pgid=$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null \
            | tr -d ' ')
        [[ -n "$observed_pgid" ]] && break
        sleep 1
    done
    if [[ -z "$observed_pgid" || "$observed_pgid" != "$ACTIVE_PGID" ]]; then
        echo "$label service did not enter an isolated process group" >&2
        arm_rc=1
    else
        for _ in $(seq 1 "$STARTUP_TIMEOUT_S"); do
            if python3 - "$arm_port" <<'PY' >/dev/null 2>&1
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
    if [[ "$startup_ok" == 1 ]]; then
        printf '%s\n' 0 > "$arm/startup.rc"
        set +e
        python3 "$ROOT/tests/qwen36_compat_http_gate.py" \
            --base "http://127.0.0.1:$arm_port" \
            --model-path "$MODEL_PATH" \
            --timeout-s 600 \
            --multiple-system-parts-expected-status \
            "$expected_system_status" \
            --image-limit "$image_limit" \
            --json-out "$arm/probe.json" \
            > "$arm/probe.stdout" 2> "$arm/probe.stderr"
        rc=$?
        set -e
        printf '%s\n' "$rc" > "$arm/probe.rc"
        [[ $rc -eq 0 ]] || arm_rc=1
    else
        printf '%s\n' 1 > "$arm/startup.rc"
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

    if [[ "$require_attribution" == 1 && -f "$arm/server.log" ]]; then
        set +e
        python3 "$ROOT/tests/summarize_api_4xx_log.py" \
            "$arm/server.log" --out "$arm/attribution.json" \
            > "$arm/attribution.stdout" 2> "$arm/attribution.stderr"
        attribution_rc=$?
        set -e
    else
        attribution_rc=0
        printf '%s\n' \
            '{"schema":"baseline-v2-attribution-not-compared","qualified":true}' \
            > "$arm/attribution.json"
        : > "$arm/attribution.stdout"
        : > "$arm/attribution.stderr"
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

    write_arm_status \
        "$arm" "$arm_rc" "$expected_system_status" "$image_limit" \
        "$runtime_revision" "$arm_port"
    return "$arm_rc"
}

write_runner_status() {
    local final_rc=$1
    python3 - "$RUN_ROOT" "$INSTANCE" "$SOURCE_REVISION" \
            "$SOURCE_BRANCH" "$GPU_INDEX" "$final_rc" "$PORT" <<'PY'
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
    "schema": "bi100-m1-70-diagnostic-http-ab-runner-v3",
    "version": 3,
    "qualified": int(sys.argv[6]) == 0,
    "source_revision": sys.argv[3],
    "source_branch": sys.argv[4],
    "instance": sys.argv[2],
    "physical_gpu": int(sys.argv[5]),
    "returncode": int(sys.argv[6]),
    "arm_ports": {
        "baseline_default": int(sys.argv[7]),
        "candidate_default": int(sys.argv[7]) + 1,
        "candidate_image2": int(sys.argv[7]) + 2,
    },
    "gates": {
        "checkpoint_verify": rc("checkpoint_verify.rc"),
        "runtime_pair": rc("runtime_pair.rc"),
        "initial_preflight": rc("initial_preflight.rc"),
        "baseline_default": rc("baseline_default.rc"),
        "candidate_default": rc("candidate_default.rc"),
        "candidate_image2": rc("candidate_image2.rc"),
        "comparison": rc("comparison.rc"),
        "cleanup": rc("cleanup.rc"),
        "final_postflight": rc("final_postflight.rc"),
        "final_preflight": rc("final_preflight.rc"),
        "final_preflight_comparison": rc(
            "final_preflight_comparison.rc"),
        "fatal_scan": rc("fatal_scan.rc"),
        "timeout_scan": rc("timeout_scan.rc"),
    },
    "artifact_sha256": {
        "runtime_pair": sha("runtime_pair.json"),
        "comparison": sha("comparison.json"),
        "final_postflight": sha("final_postflight.json"),
    },
    "semantic_quality_evaluated": False,
    "full_model_evaluated": False,
    "default_image_limit_change_authorized": False,
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
    local preflight_rc=0
    local preflight_comparison_rc=0
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
            preflight_comparison_rc=$?
        else
            preflight_comparison_rc=1
        fi
        printf '%s\n' "$preflight_comparison_rc" \
            > "$RUN_ROOT/final_preflight_comparison.rc"
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
    done < <(find "$RUN_ROOT" -type f -name server.log -print0)
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
    done < <(find "$RUN_ROOT" -type f \
        \( -name startup.rc -o -name probe.rc \) -print0)
    printf '%s\n' "$timeout_rc" > "$RUN_ROOT/timeout_scan.rc"

    if [[ $cleanup_rc -ne 0 || $postflight_rc -ne 0 \
            || $preflight_rc -ne 0 || $preflight_comparison_rc -ne 0 \
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

CURRENT_STAGE=runtime_pair
set +e
PYTHONPATH="$ROOT/tests" python3 "$ROOT/tests/verify_m1_70_runtime_pair.py" \
    --source-root "$ROOT" \
    --control-site "$CONTROL_RUNTIME_SITE_PACKAGES" \
    --control-install "$CONTROL_INSTALL" \
    --control-revision "$CONTROL_REVISION" \
    --candidate-site "$CANDIDATE_RUNTIME_SITE_PACKAGES" \
    --candidate-install "$CANDIDATE_INSTALL" \
    --candidate-revision "$CANDIDATE_REVISION" \
    --out "$RUN_ROOT/runtime_pair.json" \
    > "$RUN_ROOT/runtime_pair.stdout" \
    2> "$RUN_ROOT/runtime_pair.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/runtime_pair.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=initial_preflight
set +e
run_preflight "$RUN_ROOT/initial_preflight"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/initial_preflight.rc"
[[ $rc -eq 0 ]]
INITIAL_PREFLIGHT_PASSED=1

CURRENT_STAGE=baseline_default
set +e
run_arm baseline_default "$CONTROL_RUNTIME_SITE_PACKAGES" \
    "$CONTROL_REVISION" 400 1 0 "$PORT"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/baseline_default.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=candidate_default
set +e
run_arm candidate_default "$CANDIDATE_RUNTIME_SITE_PACKAGES" \
    "$CANDIDATE_REVISION" 200 1 1 "$((PORT + 1))"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/candidate_default.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=candidate_image2
set +e
run_arm candidate_image2 "$CANDIDATE_RUNTIME_SITE_PACKAGES" \
    "$CANDIDATE_REVISION" 200 2 1 "$((PORT + 2))"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/candidate_image2.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=comparison
set +e
python3 "$ROOT/tests/compare_qwen36_compat_http_ab.py" \
    "$RUN_ROOT/baseline_default/probe.json" \
    "$RUN_ROOT/candidate_default/probe.json" \
    "$RUN_ROOT/candidate_image2/probe.json" \
    --candidate-default-4xx \
    "$RUN_ROOT/candidate_default/attribution.json" \
    --candidate-image2-4xx \
    "$RUN_ROOT/candidate_image2/attribution.json" \
    --out "$RUN_ROOT/comparison.json" \
    > "$RUN_ROOT/comparison.stdout" \
    2> "$RUN_ROOT/comparison.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/comparison.rc"
[[ $rc -eq 0 ]]

CURRENT_STAGE=completed
exit 0
