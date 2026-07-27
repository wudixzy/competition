#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

if [[ $# -ne 2 ]]; then
    echo "usage: $0 INSTANCE RUN_ROOT" >&2
    exit 2
fi

INSTANCE=$1
RUN_ROOT=$(python3 - "$2" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)
case "$RUN_ROOT/" in
    "$ROOT/"*)
        echo "M1-58 output must stay outside the source repository" >&2
        exit 2
        ;;
esac
if [[ "$RUN_ROOT" != /tmp/* ]]; then
    echo "M1-58 output must use a private /tmp path" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
    echo "M1-58 output already exists; refusing to overwrite: $RUN_ROOT" >&2
    exit 2
fi
MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
RUN_ID=m158-block-major-fixed-20260726
ACTIVE_PID=""
ACTIVE_PGID=""
INITIAL_PREFLIGHT_PASSED=0
CURRENT_STAGE=argument_validation

export LD_LIBRARY_PATH="/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}"
export HOST=0.0.0.0
export PORT=8000
export MODEL_PATH
export ENABLE_CUSTOM_IPC=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export BI100_MOE_COREX_DIRECT_ROUTED=1
export BI100_GDN_COREX_PACKED_DECODE=1
export BI100_CPU_KV_OFFLOAD=1
export BI100_GDN_CACHE_POLICY=admission64
export BI100_GDN_RESTORE_MODE=direct
export BI100_CACHE_TRACE=0
export BI100_BLOCK_MAJOR_CPU_KV_TRACE=0
export BI100_ATTN_COREX_FUSED_PREFILL=0
export BI100_KV_EVICTION_POLICY=lru
export BI100_HYBRID_KV_ACCOUNTING=full_attention
export BI100_PROFILE=0
export BI100_PROFILE_INCLUDE_STARTUP=0
export BI100_PAGED_ATTN_DIAGNOSTICS=0
export BI100_GDN_ALLOW_NAN_ZERO=0
export BI100_GDN_FINITE_CHECK=0
unset CUDA_VISIBLE_DEVICES NUM_GPU_BLOCKS_OVERRIDE BI100_MOE_COREX_THREE_BUCKET

if [[ -z "${BI100_RUNTIME_SITE_PACKAGES:-}" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/vllm" \
        || ! -d "$BI100_RUNTIME_SITE_PACKAGES/transformers" ]]; then
    echo "M1-58 requires an atomic bare-host runtime overlay" >&2
    exit 3
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model directory is missing: $MODEL_PATH" >&2
    exit 3
fi
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
        . ':(exclude)bench_runs/**')" ]]; then
    echo "M1-58 refuses a dirty source tree" >&2
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
RUNTIME_INSTALL=${BI100_RUNTIME_INSTALL_REPORT:-${RUNTIME_INSTALL_REPORT:-$RUNTIME_ROOT/install.json}}
if [[ ! -f "$RUNTIME_INSTALL" ]]; then
    echo "M1-58 runtime install report is missing: $RUNTIME_INSTALL" >&2
    exit 3
fi

mkdir -p "$RUN_ROOT"
SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
SOURCE_BRANCH=$(git -C "$ROOT" branch --show-current)
printf '%s\n' "$SOURCE_REVISION" > "$RUN_ROOT/source_revision.txt"
printf '%s\n' "$SOURCE_BRANCH" > "$RUN_ROOT/source_branch.txt"
printf '%s\n' "$INSTANCE" > "$RUN_ROOT/instance.txt"
export BI100_RUNTIME_SITE_PACKAGES
export PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages:${PYTHONPATH:-}"

health() {
    python3 -c 'import urllib.request; urllib.request.urlopen(
        "http://127.0.0.1:8000/health", timeout=5).read()' >/dev/null 2>&1
}

port_is_free() {
    python3 - <<'PY'
import socket

sock = socket.socket()
try:
    sock.bind(("127.0.0.1", 8000))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

wait_for_port_free() {
    for _ in $(seq 1 180); do
        port_is_free && return 0
        sleep 1
    done
    echo "port 8000 remained busy; refusing to mix service lifetimes" >&2
    return 1
}

stop_service() {
    local rc=0
    if [[ -n "$ACTIVE_PGID" ]]; then
        # TP4 workers and collective runtimes get a full minute to unwind.
        bi100_stop_process_group \
            "$ACTIVE_PGID" "$ACTIVE_PID" 60 20 || rc=$?
    elif [[ -n "$ACTIVE_PID" ]]; then
        echo "service PID $ACTIVE_PID has no verified process group" >&2
        rc=2
    fi
    if [[ $rc -eq 0 ]]; then
        ACTIVE_PID=""
        ACTIVE_PGID=""
    fi
    return "$rc"
}

run_offline_gate() {
    local name=$1
    local timeout_s=$2
    local rc=0
    shift 2
    if timeout --signal=TERM --kill-after=30s "${timeout_s}s" "$@" \
            > "$RUN_ROOT/${name}.stdout" \
            2> "$RUN_ROOT/${name}.stderr"; then
        rc=0
    else
        rc=$?
    fi
    printf '%s\n' "$rc" > "$RUN_ROOT/${name}.rc"
    if [[ $rc -ne 0 ]]; then
        echo "M1-58 offline gate failed: $name rc=$rc" >&2
        return "$rc"
    fi
}

run_preflight() {
    local label=$1
    local rc=0
    if timeout --signal=TERM --kill-after=70s 480s \
            python3 "$ROOT/tests/bi100_preflight.py" \
            --gpus 0,1,2,3 --timeout-s 25 --matmul-size 1024 \
            --json-out "$RUN_ROOT/preflight_${label}.json" \
            > "$RUN_ROOT/preflight_${label}.stdout" \
            2> "$RUN_ROOT/preflight_${label}.stderr"; then
        rc=0
    else
        rc=$?
    fi
    printf '%s\n' "$rc" > "$RUN_ROOT/preflight_${label}.rc"
    if [[ $rc -ne 0 ]]; then
        echo "M1-58 four-GPU preflight failed at $label" >&2
        return "$rc"
    fi
}

fatal_scan() {
    local log=$1
    local output=$2
    local pattern
    pattern='CUDA error|SIGSEGV|Fatal Python error|out of memory|device-side assert|AssertionError|Traceback \(most recent call last\)|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|TimeoutError|engine iteration timed out|watchdog.*tim(e|ed) out|scheduler requested a missing GDN prefix state|non-finite GatedDeltaNet'
    if [[ ! -f "$log" ]]; then
        echo "service log is missing: $log" > "$output"
        return 1
    fi
    if grep -Eiq "$pattern" "$log"; then
        grep -Ein "$pattern" "$log" > "$output" || true
        return 1
    fi
    : > "$output"
}

run_service_postflight() {
    local output=$1
    timeout --signal=TERM --kill-after=30s 180s \
        python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus 0,1,2,3 \
        --settle-timeout-s 90 --clean-samples 3 \
        --sample-interval-s 2 \
        --out "$output.json" \
        > "$output.stdout" 2> "$output.stderr"
}

scan_timeout_rcs() {
    local root=$1
    local output=$2
    local file
    local found=0
    local value
    : > "$output"
    while IFS= read -r -d '' file; do
        value=$(tr -d '[:space:]' < "$file")
        case "$value" in
            124|137)
                printf '%s=%s\n' "$file" "$value" >> "$output"
                found=1
                ;;
        esac
    done < <(find "$root" -type f -name '*.rc' -print0)
    return "$found"
}

write_arm_status() {
    local output=$1
    local label=$2
    local selector=$3
    local final_rc=$4
    python3 - "$output" "$label" "$selector" "$final_rc" <<'PY'
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
    return hashlib.sha256(path.read_bytes()).hexdigest() \
        if path.is_file() else None

report = {
    "schema": "bi100-m1-58-block-major-arm-v2",
    "version": 2,
    "qualified": int(sys.argv[4]) == 0,
    "label": sys.argv[2],
    "block_major_cpu_kv": int(sys.argv[3]),
    "gates": {
        "startup": read_rc("startup.rc"),
        "startup_markers": read_rc("startup_markers.rc"),
        "startup_contract": read_rc("startup_gate.rc"),
        "pressure": read_rc("pressure.rc"),
        "health_after_pressure": read_rc("health_after_pressure.rc"),
        "cleanup": read_rc("cleanup.rc"),
        "fatal_scan": read_rc("fatal_scan.rc"),
        "timeout_scan": read_rc("timeout_scan.rc"),
        "service_postflight": read_rc("service_postflight.rc"),
    },
    "artifact_sha256": {
        "startup_contract": sha256("startup_gate.json"),
        "pressure": sha256("pressure.json"),
        "service_postflight": sha256("service_postflight.json"),
    },
    "model_quality_evaluated": False,
    "production_promotion_authorized": False,
}
(root / "status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8")
PY
}

run_arm() {
    local label=$1
    local selector=$2
    local output=$RUN_ROOT/$label
    local arm_rc=0
    local cleanup_rc=0
    local fatal_rc=0
    local health_rc=1
    local markers_rc=1
    local observed_pgid=""
    local postflight_rc=0
    local pressure_rc=1
    local startup_gate_rc=1
    local startup_ok=0
    local timeout_rc=0

    mkdir -p "$output/runtime-workdir"
    printf '%s\n' "$selector" > "$output/block_major_cpu_kv.txt"
    if ! wait_for_port_free; then
        write_arm_status "$output" "$label" "$selector" 1
        return 1
    fi

    BI100_BLOCK_MAJOR_CPU_KV="$selector" \
        BI100_RUNTIME_WORKDIR="$output/runtime-workdir" \
        setsid "$ROOT/launch_service" \
        > "$output/server.log" 2>&1 < /dev/null &
    ACTIVE_PID=$!
    ACTIVE_PGID=$ACTIVE_PID
    printf '%s\n' "$ACTIVE_PID" > "$output/server.pid"

    for _ in $(seq 1 20); do
        observed_pgid=$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null | tr -d ' ')
        [[ -n "$observed_pgid" ]] && break
        sleep 1
    done
    if [[ -z "$observed_pgid" || "$observed_pgid" != "$ACTIVE_PGID" ]]; then
        echo "$label service did not enter an isolated process group" >&2
        arm_rc=1
    else
        printf '%s\n' "$ACTIVE_PGID" > "$output/server.pgid"
        for _ in $(seq 1 360); do
            if health; then
                startup_ok=1
                break
            fi
            if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
                break
            fi
            sleep 10
        done
    fi

    if [[ "$startup_ok" == 1 ]]; then
        printf '%s\n' 0 > "$output/startup.rc"
    else
        printf '%s\n' 1 > "$output/startup.rc"
        echo "M1-58 $label service did not become healthy" >&2
        tail -120 "$output/server.log" >&2 || true
        arm_rc=1
    fi

    if [[ "$startup_ok" == 1 ]]; then
        markers_rc=0
        grep -Fq '[BI100] fixed evaluator contract;' \
            "$output/server.log" || markers_rc=1
        grep -Fq '[BI100] fixed kernels; moe_direct=1 gdn_packed=1' \
            "$output/server.log" || markers_rc=1
        grep -Fq '[BI100] GDN cache; policy=admission64 restore=direct' \
            "$output/server.log" || markers_rc=1
        grep -Fq \
            '[BI100] M1-49 runtime contract; accounting=full_attention' \
            "$output/server.log" || markers_rc=1
        grep -Fq "cpu_kv_offload=1 block_major_cpu_kv=$selector" \
            "$output/server.log" || markers_rc=1
        if [[ "$selector" == "1" ]]; then
            grep -Fq '[BI100 BLOCK KV] capacity reserve' \
                "$output/server.log" || markers_rc=1
            grep -Fq '[BI100 BLOCK KV] enabled' \
                "$output/server.log" || markers_rc=1
        elif grep -Fq '[BI100 BLOCK KV]' "$output/server.log"; then
            echo "control emitted block-major runtime markers" >&2
            markers_rc=1
        fi
    fi
    printf '%s\n' "$markers_rc" > "$output/startup_markers.rc"
    [[ $markers_rc -eq 0 ]] || arm_rc=1

    if [[ "$startup_ok" == 1 && $markers_rc -eq 0 ]]; then
        set +e
        BI100_BLOCK_MAJOR_CPU_KV="$selector" \
            timeout --signal=TERM --kill-after=20s 180s \
            python3 "$ROOT/tests/hybrid_kv_startup_gate.py" \
            "$output/server.log" --mode full_attention \
            --model-path "$MODEL_PATH" --max-model-len 262144 \
            --block-size 16 --tensor-parallel-size 4 \
            --expected-cpu-kv-offload 1 \
            --expected-block-major-cpu-kv "$selector" \
            --expected-block-major-cpu-kv-trace 0 \
            --out "$output/startup_gate.json" \
            > "$output/startup_gate.stdout" \
            2> "$output/startup_gate.stderr"
        startup_gate_rc=$?
        set -e
    fi
    printf '%s\n' "$startup_gate_rc" > "$output/startup_gate.rc"
    [[ $startup_gate_rc -eq 0 ]] || arm_rc=1

    if [[ $startup_gate_rc -eq 0 ]]; then
        set +e
        timeout --signal=TERM --kill-after=60s 18000s \
            python3 "$ROOT/tests/cpu_kv_offload_pressure_api.py" \
            --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" \
            --target-prompt-tokens 65536 \
            --pressure-prompt-tokens 135040 --pressure-count 9 \
            --max-tokens 8 --timeout-s 900 --run-id "$RUN_ID" \
            --mode candidate --block-size 16 \
            --min-candidate-cached 65504 --max-control-cached 16 \
            --json-out "$output/pressure.json" \
            > "$output/pressure.stdout" 2> "$output/pressure.stderr"
        pressure_rc=$?
        set -e
    fi
    printf '%s\n' "$pressure_rc" > "$output/pressure.rc"
    [[ $pressure_rc -eq 0 ]] || arm_rc=1

    if [[ $pressure_rc -eq 0 ]]; then
        set +e
        health
        health_rc=$?
        set -e
    fi
    printf '%s\n' "$health_rc" > "$output/health_after_pressure.rc"
    [[ $health_rc -eq 0 ]] || arm_rc=1

    set +e
    stop_service
    cleanup_rc=$?
    if [[ $cleanup_rc -eq 0 ]]; then
        wait_for_port_free
        cleanup_rc=$?
    fi
    set -e
    printf '%s\n' "$cleanup_rc" > "$output/cleanup.rc"
    [[ $cleanup_rc -eq 0 ]] || arm_rc=1

    set +e
    fatal_scan "$output/server.log" "$output/fatal_scan.txt"
    fatal_rc=$?
    set -e
    printf '%s\n' "$fatal_rc" > "$output/fatal_scan.rc"
    [[ $fatal_rc -eq 0 ]] || arm_rc=1

    set +e
    scan_timeout_rcs "$output" "$output/timeout_scan.txt"
    timeout_rc=$?
    set -e
    printf '%s\n' "$timeout_rc" > "$output/timeout_scan.rc"
    [[ $timeout_rc -eq 0 ]] || arm_rc=1

    set +e
    run_service_postflight "$output/service_postflight"
    postflight_rc=$?
    set -e
    printf '%s\n' "$postflight_rc" > "$output/service_postflight.rc"
    [[ $postflight_rc -eq 0 ]] || arm_rc=1

    write_arm_status "$output" "$label" "$selector" "$arm_rc"
    return "$arm_rc"
}

scan_all_fatal_logs() {
    local file
    local found=0
    local pattern
    pattern='CUDA error|SIGSEGV|Fatal Python error|out of memory|device-side assert|AssertionError|Traceback \(most recent call last\)|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|TimeoutError|engine iteration timed out|watchdog.*tim(e|ed) out|scheduler requested a missing GDN prefix state|non-finite GatedDeltaNet'
    : > "$RUN_ROOT/fatal_scan.txt"
    while IFS= read -r -d '' file; do
        if grep -Eiq "$pattern" "$file"; then
            printf 'file=%s\n' "$file" >> "$RUN_ROOT/fatal_scan.txt"
            grep -Ein "$pattern" "$file" \
                >> "$RUN_ROOT/fatal_scan.txt" || true
            found=1
        fi
    done < <(find "$RUN_ROOT" -type f -name server.log -print0)
    return "$found"
}

write_runner_status() {
    local final_rc=$1
    python3 - "$RUN_ROOT" "$INSTANCE" "$SOURCE_REVISION" \
            "$SOURCE_BRANCH" "$CURRENT_STAGE" "$final_rc" <<'PY'
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
    return hashlib.sha256(path.read_bytes()).hexdigest() \
        if path.is_file() else None

report = {
    "schema": "bi100-m1-58-block-major-ab-runner-v2",
    "version": 2,
    "qualified": int(sys.argv[6]) == 0,
    "source_revision": sys.argv[3],
    "source_branch": sys.argv[4],
    "instance": sys.argv[2],
    "terminal_stage": sys.argv[5],
    "returncode": int(sys.argv[6]),
    "gates": {
        "runtime_identity": read_rc("runtime_identity.rc"),
        "preflight_before_control": read_rc("preflight_before_control.rc"),
        "control": read_rc("control.rc"),
        "preflight_after_control": read_rc("preflight_after_control.rc"),
        "candidate": read_rc("candidate.rc"),
        "preflight_after_candidate": read_rc(
            "preflight_after_candidate.rc"),
        "preflight_comparison": read_rc("preflight_comparison.rc"),
        "comparison": read_rc("comparison.rc"),
        "cleanup": read_rc("cleanup.rc"),
        "final_postflight": read_rc("final_postflight.rc"),
        "final_preflight": read_rc("preflight_final.rc"),
        "final_preflight_comparison": read_rc(
            "final_preflight_comparison.rc"),
        "fatal_scan": read_rc("fatal_scan.rc"),
        "timeout_scan": read_rc("timeout_scan.rc"),
    },
    "arm_gates": {
        label: {
            "cleanup": read_rc(f"{label}/cleanup.rc"),
            "fatal_scan": read_rc(f"{label}/fatal_scan.rc"),
            "timeout_scan": read_rc(f"{label}/timeout_scan.rc"),
            "service_postflight": read_rc(
                f"{label}/service_postflight.rc"),
        }
        for label in ("control", "candidate")
    },
    "artifact_sha256": {
        "runtime_identity": sha256("runtime_identity.json"),
        "comparison": sha256("comparison.json"),
        "final_postflight": sha256("final_postflight.json"),
        "final_preflight_comparison": sha256(
            "final_preflight_comparison.json"),
    },
    "model_quality_evaluated": False,
    "official_881_evaluated": False,
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
    local fatal_rc=0
    local postflight_rc=0
    local preflight_rc=0
    local preflight_comparison_rc=0
    local timeout_rc=0
    trap - EXIT TERM INT
    set +e

    stop_service
    cleanup_rc=$?
    if [[ $cleanup_rc -eq 0 ]]; then
        wait_for_port_free
        cleanup_rc=$?
    fi
    printf '%s\n' "$cleanup_rc" > "$RUN_ROOT/cleanup.rc"

    run_service_postflight "$RUN_ROOT/final_postflight"
    postflight_rc=$?
    printf '%s\n' "$postflight_rc" > "$RUN_ROOT/final_postflight.rc"

    if [[ "$INITIAL_PREFLIGHT_PASSED" == 1 ]]; then
        run_preflight final
        preflight_rc=$?
        if [[ $preflight_rc -eq 0 ]]; then
            python3 "$ROOT/tests/compare_bi100_preflights.py" \
                --preflight \
                "before_control=$RUN_ROOT/preflight_before_control.json" \
                --preflight "final=$RUN_ROOT/preflight_final.json" \
                --expected-gpus 0,1,2,3 \
                --max-free-memory-drop-bytes 268435456 \
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

    scan_all_fatal_logs
    fatal_rc=$?
    printf '%s\n' "$fatal_rc" > "$RUN_ROOT/fatal_scan.rc"

    scan_timeout_rcs "$RUN_ROOT" "$RUN_ROOT/timeout_scan.txt"
    timeout_rc=$?
    printf '%s\n' "$timeout_rc" > "$RUN_ROOT/timeout_scan.rc"

    if [[ $cleanup_rc -ne 0 || $postflight_rc -ne 0 \
            || $preflight_rc -ne 0 || $preflight_comparison_rc -ne 0 \
            || $fatal_rc -ne 0 || $timeout_rc -ne 0 ]]; then
        final_rc=1
    fi
    printf '%s\n' "$final_rc" > "$RUN_ROOT/overall.rc"
    write_runner_status "$final_rc"
    exit "$final_rc"
}
trap 'exit 143' TERM
trap 'exit 130' INT
trap finish EXIT

CURRENT_STAGE=runtime_identity
run_offline_gate runtime_identity 90 \
    python3 "$ROOT/tests/verify_bare_host_runtime_identity.py" \
    --source-root "$ROOT" \
    --runtime-site-packages "$BI100_RUNTIME_SITE_PACKAGES" \
    --runtime-install "$RUNTIME_INSTALL" \
    --out "$RUN_ROOT/runtime_identity.json"

CURRENT_STAGE=preflight_before_control
run_preflight before_control
INITIAL_PREFLIGHT_PASSED=1

CURRENT_STAGE=control
if run_arm control 0; then
    control_rc=0
else
    control_rc=$?
fi
printf '%s\n' "$control_rc" > "$RUN_ROOT/control.rc"
[[ $control_rc -eq 0 ]]

CURRENT_STAGE=preflight_after_control
run_preflight after_control

CURRENT_STAGE=candidate
if run_arm candidate 1; then
    candidate_rc=0
else
    candidate_rc=$?
fi
printf '%s\n' "$candidate_rc" > "$RUN_ROOT/candidate.rc"
[[ $candidate_rc -eq 0 ]]

CURRENT_STAGE=preflight_after_candidate
run_preflight after_candidate

CURRENT_STAGE=preflight_comparison
run_offline_gate preflight_comparison 120 \
    python3 "$ROOT/tests/compare_bi100_preflights.py" \
    --preflight \
    "before_control=$RUN_ROOT/preflight_before_control.json" \
    --preflight \
    "after_control=$RUN_ROOT/preflight_after_control.json" \
    --preflight \
    "after_candidate=$RUN_ROOT/preflight_after_candidate.json" \
    --max-free-memory-drop-bytes 268435456 \
    --out "$RUN_ROOT/preflight_comparison.json"

CURRENT_STAGE=comparison
run_offline_gate comparison 120 \
    python3 "$ROOT/tests/compare_m1_58_block_major_ab.py" \
    --control-startup "$RUN_ROOT/control/startup_gate.json" \
    --candidate-startup "$RUN_ROOT/candidate/startup_gate.json" \
    --control-pressure "$RUN_ROOT/control/pressure.json" \
    --candidate-pressure "$RUN_ROOT/candidate/pressure.json" \
    --runtime-identity "$RUN_ROOT/runtime_identity.json" \
    --preflight-comparison "$RUN_ROOT/preflight_comparison.json" \
    --out "$RUN_ROOT/comparison.json"

CURRENT_STAGE=complete
echo "M1-58 fixed TP4 block-major CacheEngine A/B passed"
