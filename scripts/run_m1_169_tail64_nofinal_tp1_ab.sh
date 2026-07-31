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
PORT=${PORT:-8061}
CANDIDATE_POLICY=${CANDIDATE_POLICY:-tail64_nofinal}
ARM_ORDER=${ARM_ORDER:-admission64,$CANDIDATE_POLICY}
BENCH_SALT_ORDER=${BENCH_SALT_ORDER:-namespace-first}
BENCH_SALT_NAMESPACE=${BENCH_SALT_NAMESPACE:-m1-169-tail64-nofinal-tp1-v1}
BENCH_TOOL_COUNT=${BENCH_TOOL_COUNT:-29}
MODEL_PATH=${MODEL_PATH:-/root/shared-storage/models/Qwen/Qwen3.6-35B-A3B-diagnostic-4L-real}
BI100_RUNTIME_SITE_PACKAGES=${BI100_RUNTIME_SITE_PACKAGES:-}
STARTUP_TIMEOUT_S=${STARTUP_TIMEOUT_S:-900}
NUM_GPU_BLOCKS_OVERRIDE=${NUM_GPU_BLOCKS_OVERRIDE:-}
BLOCK_OVERRIDE_ARGS=()
ACTIVE_PID=""
ACTIVE_PGID=""
ACTIVE_STARTTIME=""
ACTIVE_SESSION_TOKEN=""
CURRENT_STAGE=argument_validation
FINALIZED=0
SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
COREX_LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
COREX_PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/iluvatar/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
FATAL_PATTERN='CUDA error|illegal memory access|SIGSEGV|Fatal Python error|out of memory|device-side assert|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|engine iteration timed out|scheduler requested a missing GDN prefix state|non-finite GatedDeltaNet'

[[ "$INSTANCE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
    echo "INSTANCE must be a short non-sensitive label" >&2
    exit 2
}
[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || { echo "GPU_INDEX must be non-negative" >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ && "$PORT" -ge 1 && "$PORT" -le 65535 ]] || {
    echo "PORT must be between 1 and 65535" >&2
    exit 2
}
case "$CANDIDATE_POLICY" in
    tail64_nofinal|off) ;;
    *) echo "CANDIDATE_POLICY must be tail64_nofinal or off" >&2; exit 2 ;;
esac
[[ "$ARM_ORDER" == "admission64,$CANDIDATE_POLICY" \
    || "$ARM_ORDER" == "$CANDIDATE_POLICY,admission64" ]] || {
    echo "ARM_ORDER must contain admission64 and CANDIDATE_POLICY once" >&2
    exit 2
}
case "$BENCH_SALT_ORDER" in
    namespace-first|identity-first) ;;
    *) echo "BENCH_SALT_ORDER is invalid" >&2; exit 2 ;;
esac
[[ "$BENCH_SALT_NAMESPACE" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$ ]] || {
    echo "BENCH_SALT_NAMESPACE must be a short non-sensitive label" >&2
    exit 2
}
[[ "$BENCH_TOOL_COUNT" == 0 || "$BENCH_TOOL_COUNT" == 29 ]] || {
    echo "BENCH_TOOL_COUNT must be 0 or 29" >&2
    exit 2
}
[[ "$STARTUP_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || {
    echo "STARTUP_TIMEOUT_S must be positive" >&2
    exit 2
}
if [[ -n "$NUM_GPU_BLOCKS_OVERRIDE" ]]; then
    [[ "$NUM_GPU_BLOCKS_OVERRIDE" =~ ^[1-9][0-9]*$ ]] || {
        echo "NUM_GPU_BLOCKS_OVERRIDE must be a positive integer" >&2
        exit 2
    }
    BLOCK_OVERRIDE_ARGS=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE")
fi

RUN_ROOT=$(python3 - "$RUN_ROOT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)
[[ "$RUN_ROOT" == /tmp/* && ! -e "$RUN_ROOT" ]] || {
    echo "RUN_ROOT must be a new private /tmp path" >&2
    exit 3
}
case "$RUN_ROOT/" in "$ROOT/"*) echo "RUN_ROOT must stay outside source" >&2; exit 3;; esac
MODEL_PATH=$(python3 - "$MODEL_PATH" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=True))
PY
)
[[ -n "$BI100_RUNTIME_SITE_PACKAGES" ]] || {
    echo "BI100_RUNTIME_SITE_PACKAGES is required" >&2
    exit 3
}
BI100_RUNTIME_SITE_PACKAGES=$(python3 - "$BI100_RUNTIME_SITE_PACKAGES" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=True))
PY
)
[[ -d "$BI100_RUNTIME_SITE_PACKAGES/vllm" && -d "$BI100_RUNTIME_SITE_PACKAGES/transformers" ]] || {
    echo "runtime overlay must contain vllm and transformers" >&2
    exit 3
}
RUNTIME_INSTALL=$(dirname "$BI100_RUNTIME_SITE_PACKAGES")/install.json
[[ -f "$RUNTIME_INSTALL" ]] || { echo "runtime install report is missing" >&2; exit 3; }
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all -- . ':(exclude)bench_runs/**')" ]] || {
    echo "M1-169 refuses a dirty source tree" >&2
    exit 3
}

mkdir -p "$RUN_ROOT"
SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
SOURCE_BRANCH=$(git -C "$ROOT" branch --show-current)
printf '%s\n' "$SOURCE_REVISION" > "$RUN_ROOT/source_revision.txt"
printf '%s\n' "$SOURCE_BRANCH" > "$RUN_ROOT/source_branch.txt"
printf '%s\n' "$GPU_INDEX" > "$RUN_ROOT/gpu_index.txt"
printf '%s\n' "$MODEL_PATH" > "$RUN_ROOT/model_path.txt"
printf '%s\n' "$BI100_RUNTIME_SITE_PACKAGES" > "$RUN_ROOT/runtime_site_packages.txt"
printf '%s\n' "$ARM_ORDER" > "$RUN_ROOT/arm_order.txt"
printf '%s\n' "$CANDIDATE_POLICY" > "$RUN_ROOT/candidate_policy.txt"
printf '%s\n' "$BENCH_SALT_ORDER" > "$RUN_ROOT/bench_salt_order.txt"
printf '%s\n' "$BENCH_SALT_NAMESPACE" > "$RUN_ROOT/bench_salt_namespace.txt"
printf '%s\n' "$BENCH_TOOL_COUNT" > "$RUN_ROOT/bench_tool_count.txt"
printf '%s\n' "$NUM_GPU_BLOCKS_OVERRIDE" > "$RUN_ROOT/num_gpu_blocks_override.txt"

write_stage() {
    printf '%s\n' "$CURRENT_STAGE" > "$RUN_ROOT/stage.txt"
}

read_starttime() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys
value = (Path("/proc") / sys.argv[1] / "stat").read_text(encoding="ascii")
print(value[value.rfind(")") + 2:].split()[19])
PY
}

active_live() {
    [[ -n "$ACTIVE_PID" && -n "$ACTIVE_STARTTIME" ]] || return 1
    [[ "$(read_starttime "$ACTIVE_PID" 2>/dev/null || true)" == "$ACTIVE_STARTTIME" ]]
}

stop_active() {
    local rc=0
    [[ -n "$ACTIVE_PID" ]] || return 0
    if active_live && [[ -n "$ACTIVE_PGID" && -n "$ACTIVE_SESSION_TOKEN" ]]; then
        bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID" 60 20 \
            "$ACTIVE_STARTTIME" "$ACTIVE_SESSION_TOKEN" || rc=$?
    elif active_live; then
        kill -TERM "$ACTIVE_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do active_live || break; sleep 1; done
        if active_live; then kill -KILL "$ACTIVE_PID" 2>/dev/null || true; fi
    fi
    wait "$ACTIVE_PID" 2>/dev/null || true
    active_live && rc=1
    if [[ $rc -eq 0 ]]; then
        ACTIVE_PID=""; ACTIVE_PGID=""; ACTIVE_STARTTIME=""; ACTIVE_SESSION_TOKEN=""
    fi
    return "$rc"
}

port_free() {
    python3 - "$PORT" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
}

wait_port_free() {
    for _ in $(seq 1 120); do
        port_free && return 0
        sleep 1
    done
    return 1
}

health() {
    python3 - "$PORT" <<'PY'
import sys
import urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=5).read()
PY
}

run_preflight() {
    local output=$1
    timeout --signal=TERM --kill-after=90s 240s env -u CUDA_VISIBLE_DEVICES \
        PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
        LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
        python3 "$ROOT/tests/bi100_preflight.py" --gpus "$GPU_INDEX" \
        --timeout-s 25 --matmul-size 1024 --json-out "$output.json" \
        > "$output.stdout" 2> "$output.stderr"
}

run_postflight() {
    local output=$1
    timeout --signal=TERM --kill-after=70s 240s env -u CUDA_VISIBLE_DEVICES \
        PYTHONPATH="$ROOT/tests" python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus "$GPU_INDEX" --settle-timeout-s 90 --clean-samples 3 \
        --sample-interval-s 2 --out "$output.json" \
        > "$output.stdout" 2> "$output.stderr"
}

start_service() {
    local arm=$1 policy=$2 identity="$arm/service_identity.json" observed
    port_free || return 1
    (
        cd /tmp
        exec env CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
            BI100_GDN_CACHE_POLICY="$policy" BI100_GDN_RESTORE_MODE=hybrid64 \
            BI100_HYBRID_KV_ACCOUNTING=full_attention BI100_CACHE_TRACE=1 \
            BI100_ATTN_COREX_FUSED_PREFILL=0 BI100_MOE_COREX_DIRECT_ROUTED=1 \
            BI100_GDN_COREX_PACKED_DECODE=1 BI100_GDN_COMBINED_QK_NORM=0 \
            BI100_GDN_ALLOW_NAN_ZERO=0 BI100_GDN_FINITE_CHECK=0 \
            VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
            PYTHONPATH="$BI100_RUNTIME_SITE_PACKAGES:$ROOT/tests:$SYSTEM_PYTHONPATH" \
            LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
            python3 "$ROOT/scripts/exec_bi100_session.py" "$identity" -- \
            python3 -m vllm.entrypoints.openai.api_server \
                --host 127.0.0.1 --port "$PORT" --model "$MODEL_PATH" \
                --served-model-name llm --max-model-len 262144 \
                --gpu-memory-utilization 0.9 --trust-remote-code \
                --tensor-parallel-size 1 --max-num-seqs 1 \
                "${BLOCK_OVERRIDE_ARGS[@]}" \
                --disable-log-requests --disable-frontend-multiprocessing \
                --max-num-batched-tokens 8192 --enable-chunked-prefill \
                --max-seq-len-to-capture 32768 --enable-auto-tool-choice \
                --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
                --enable-prefix-caching
    ) > "$arm/server.log" 2>&1 &
    ACTIVE_PID=$!
    for _ in $(seq 1 50); do
        ACTIVE_STARTTIME=$(read_starttime "$ACTIVE_PID" 2>/dev/null || true)
        [[ -n "$ACTIVE_STARTTIME" ]] && break
        kill -0 "$ACTIVE_PID" 2>/dev/null || break
        sleep 0.1
    done
    [[ -n "$ACTIVE_STARTTIME" ]] || return 1
    for _ in $(seq 1 100); do
        if [[ -s "$identity" ]]; then
            observed=$(python3 - "$identity" "$ACTIVE_PID" "$ACTIVE_STARTTIME" <<'PY'
import json
from pathlib import Path
import sys
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
token = value.get("session_token")
if (value.get("pid") != int(sys.argv[2]) or value.get("pgid") != int(sys.argv[2])
        or value.get("sid") != int(sys.argv[2])
        or value.get("starttime_ticks") != int(sys.argv[3])
        or not isinstance(token, str) or len(token) != 32):
    raise SystemExit(1)
print(value["pgid"], token)
PY
            ) && read -r ACTIVE_PGID ACTIVE_SESSION_TOKEN <<< "$observed" && break
        fi
        active_live || break
        sleep 0.1
    done
    [[ -n "$ACTIVE_PGID" && "$ACTIVE_SESSION_TOKEN" =~ ^[0-9a-f]{32}$ ]] || return 1
    for _ in $(seq 1 "$((STARTUP_TIMEOUT_S / 5))"); do
        health >/dev/null 2>&1 && return 0
        active_live || break
        sleep 5
    done
    return 1
}

verify_service_import_contract() {
    local output=$1
    (
    cd /tmp
    env PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="$BI100_RUNTIME_SITE_PACKAGES:$ROOT/tests:$SYSTEM_PYTHONPATH" \
        LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
        python3 - "$BI100_RUNTIME_SITE_PACKAGES" "$output" <<'PY'
import inspect
import json
from pathlib import Path
import sys

from vllm.entrypoints.openai import api_server, cli_args
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.utils import FlexibleArgumentParser

site = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
api_path = Path(inspect.getfile(api_server)).resolve()
cli_path = Path(inspect.getfile(cli_args)).resolve()
parser = cli_args.make_arg_parser(FlexibleArgumentParser())
options = {
    option
    for action in parser._actions
    for option in action.option_strings
}
qualified = all((
    api_path.is_relative_to(site),
    cli_path.is_relative_to(site),
    "--reasoning-parser" in options,
    "--tool-call-parser" in options,
    "qwen3_coder" in ToolParserManager.tool_parsers,
    "qwen3" in ReasoningParserManager.list_registered(),
))
report = {
    "schema": "bi100-m1-170-service-import-contract-v1",
    "version": 1,
    "qualified": qualified,
    "api_server_from_overlay": api_path.is_relative_to(site),
    "cli_args_from_overlay": cli_path.is_relative_to(site),
    "reasoning_parser_option": "--reasoning-parser" in options,
    "tool_call_parser_option": "--tool-call-parser" in options,
    "qwen3_coder_registered": (
        "qwen3_coder" in ToolParserManager.tool_parsers),
    "qwen3_reasoning_registered": (
        "qwen3" in ReasoningParserManager.list_registered()),
}
output.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
raise SystemExit(0 if qualified else 1)
PY
    )
}

scan_fatal() {
    local output=$1 file found=0
    : > "$output"
    while IFS= read -r -d '' file; do
        if grep -Eiq "$FATAL_PATTERN" "$file" 2>/dev/null; then
            printf 'file=%s\n' "$file" >> "$output"
            grep -Ein "$FATAL_PATTERN" "$file" >> "$output" 2>/dev/null || true
            found=1
        fi
    done < <(find "$RUN_ROOT" -type f \( -name '*.log' -o -name '*.stderr' \) -print0)
    return "$found"
}

verify_policy_contract() {
    local arm=$1 policy=$2
    (
    cd /tmp
    env BI100_GDN_CACHE_POLICY="$policy" BI100_GDN_RESTORE_MODE=hybrid64 \
        PYTHONPATH="$BI100_RUNTIME_SITE_PACKAGES:$ROOT/tests:$SYSTEM_PYTHONPATH" \
        LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
        python3 - "$policy" "$arm/policy_contract.json" <<'PY'
import json
from pathlib import Path
import sys

from vllm.gdn_prefix import (
    gdn_cache_policy_from_env,
    gdn_restore_mode_from_env,
)

expected = sys.argv[1]
observed = gdn_cache_policy_from_env()
restore = gdn_restore_mode_from_env()
report = {
    "schema": "bi100-m1-169-policy-contract-v1",
    "version": 1,
    "expected_policy": expected,
    "observed_policy": observed,
    "observed_restore_mode": restore,
    "qualified": observed == expected and restore == "hybrid64",
}
Path(sys.argv[2]).write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(0 if report["qualified"] else 1)
PY
    )
}

run_arm() {
    local policy=$1 arm="$RUN_ROOT/$policy" rc=0
    local startup_rc=0 policy_rc=125 measurement_rc=125 health_rc=0
    local cleanup_rc=0 port_rc=0
    mkdir -p "$arm"
    start_service "$arm" "$policy" || { startup_rc=$?; rc=1; }
    printf '%s\n' "$startup_rc" > "$arm/startup.rc"
    if [[ $rc -eq 0 ]]; then
        policy_rc=0
        verify_policy_contract "$arm" "$policy" \
            > "$arm/policy_contract.stdout" \
            2> "$arm/policy_contract.stderr" \
            || { policy_rc=$?; rc=1; }
    fi
    printf '%s\n' "$policy_rc" > "$arm/policy_contract.rc"
    if [[ $rc -eq 0 ]]; then
        measurement_rc=0
        timeout --signal=TERM --kill-after=60s 7200s env \
            PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH" \
            LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
            python3 "$ROOT/tests/bench_m1_104_admission64_policy_matrix.py" \
            --base "http://127.0.0.1:$PORT" --model-path "$MODEL_PATH" \
            --policy "$policy" --ab-pair 1 \
            --salt-namespace "$BENCH_SALT_NAMESPACE" \
            --salt-order "$BENCH_SALT_ORDER" \
            --tool-count "$BENCH_TOOL_COUNT" \
            --out "$arm/measurement.json" \
            > "$arm/measurement.stdout" 2> "$arm/measurement.stderr" \
            || { measurement_rc=$?; rc=1; }
    fi
    printf '%s\n' "$measurement_rc" > "$arm/measurement.rc"
    health > "$arm/health_after.stdout" 2> "$arm/health_after.stderr" \
        || { health_rc=$?; rc=1; }
    printf '%s\n' "$health_rc" > "$arm/health_after.rc"
    stop_active || { cleanup_rc=$?; rc=1; }
    printf '%s\n' "$cleanup_rc" > "$arm/cleanup.rc"
    wait_port_free || { port_rc=$?; rc=1; }
    printf '%s\n' "$port_rc" > "$arm/port_free.rc"
    printf '%s\n' "$rc" > "$arm/arm.rc"
    return "$rc"
}

finish() {
    local primary=$? final cleanup=0 post=0 after=0 fatal=0
    final=$primary
    trap - EXIT INT TERM
    set +e
    stop_active || cleanup=1
    run_postflight "$RUN_ROOT/postflight_after"; post=$?
    if [[ $post -eq 0 ]]; then run_preflight "$RUN_ROOT/preflight_after"; after=$?; else after=1; fi
    scan_fatal "$RUN_ROOT/fatal_scan.txt"; fatal=$?
    printf '%s\n' "$cleanup" > "$RUN_ROOT/cleanup.rc"
    printf '%s\n' "$post" > "$RUN_ROOT/postflight_after.rc"
    printf '%s\n' "$after" > "$RUN_ROOT/preflight_after.rc"
    printf '%s\n' "$fatal" > "$RUN_ROOT/fatal_scan.rc"
    if [[ $cleanup -ne 0 || $post -ne 0 || $after -ne 0 || $fatal -ne 0 ]]; then final=1; fi
    python3 - "$RUN_ROOT" "$SOURCE_REVISION" "$SOURCE_BRANCH" "$INSTANCE" "$GPU_INDEX" "$ARM_ORDER" "$NUM_GPU_BLOCKS_OVERRIDE" "$CANDIDATE_POLICY" "$BENCH_SALT_ORDER" "$BENCH_TOOL_COUNT" "$final" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
def rc(path):
    target = root / path
    try:
        return int(target.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
report = {
    "schema": (
        "bi100-m1-169-tail64-nofinal-tp1-runner-v1"
        if sys.argv[8] == "tail64_nofinal"
        else "bi100-m1-170-cold-capture-overhead-runner-v2"
    ),
    "version": 1 if sys.argv[8] == "tail64_nofinal" else 2,
    "source_revision": sys.argv[2],
    "source_branch": sys.argv[3],
    "instance": sys.argv[4],
    "gpu_index": int(sys.argv[5]),
    "arm_order": sys.argv[6].split(","),
    "num_gpu_blocks_override": (
        int(sys.argv[7]) if sys.argv[7] else None),
    "candidate_policy": sys.argv[8],
    "bench_salt_order": sys.argv[9],
    "bench_tool_count": int(sys.argv[10]),
    "returncode": int(sys.argv[11]),
    "qualified_development_screen": int(sys.argv[11]) == 0,
    "gates": {
        "preflight_before": rc("preflight_before.rc"),
        "service_import_contract": rc("service_import_contract.rc"),
        "admission64": rc("admission64/arm.rc"),
        "candidate": rc(f"{sys.argv[8]}/arm.rc"),
        "comparison": rc("comparison.rc"),
        "cleanup": rc("cleanup.rc"),
        "postflight_after": rc("postflight_after.rc"),
        "preflight_after": rc("preflight_after.rc"),
        "fatal_scan": rc("fatal_scan.rc"),
    },
    "model_quality_evaluated": False,
    "tp4_evaluated": False,
    "production_promotion_authorized": False,
}
(root / "runner_status.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    FINALIZED=1
    exit "$final"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CURRENT_STAGE=preflight_before
write_stage
run_preflight "$RUN_ROOT/preflight_before"
printf '0\n' > "$RUN_ROOT/preflight_before.rc"
CURRENT_STAGE=runtime_identity
write_stage
env PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH" \
    LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
    python3 "$ROOT/tests/verify_bare_host_runtime_identity.py" \
    --source-root "$ROOT" --runtime-site-packages "$BI100_RUNTIME_SITE_PACKAGES" \
    --runtime-install "$RUNTIME_INSTALL" --out "$RUN_ROOT/runtime_identity.json"
CURRENT_STAGE=service_import_contract
write_stage
service_import_rc=0
verify_service_import_contract "$RUN_ROOT/service_import_contract.json" \
    > "$RUN_ROOT/service_import_contract.stdout" \
    2> "$RUN_ROOT/service_import_contract.stderr" \
    || service_import_rc=$?
printf '%s\n' "$service_import_rc" > "$RUN_ROOT/service_import_contract.rc"
[[ $service_import_rc -eq 0 ]]

IFS=, read -r first second <<< "$ARM_ORDER"
for policy in "$first" "$second"; do
    CURRENT_STAGE=$policy
    write_stage
    run_arm "$policy"
done
CURRENT_STAGE=comparison
write_stage
comparison_rc=0
if [[ "$CANDIDATE_POLICY" == tail64_nofinal ]]; then
    python3 "$ROOT/tests/compare_m1_169_tail64_nofinal_tp1.py" \
        --control "$RUN_ROOT/admission64/measurement.json" \
        --candidate "$RUN_ROOT/tail64_nofinal/measurement.json" \
        --out "$RUN_ROOT/comparison.json" \
        > "$RUN_ROOT/comparison.stdout" 2> "$RUN_ROOT/comparison.stderr" \
        || comparison_rc=$?
else
    python3 "$ROOT/tests/compare_m1_170_cold_capture_overhead.py" \
        --control "$RUN_ROOT/admission64/measurement.json" \
        --candidate "$RUN_ROOT/off/measurement.json" \
        --out "$RUN_ROOT/comparison.json" \
        > "$RUN_ROOT/comparison.stdout" 2> "$RUN_ROOT/comparison.stderr" \
        || comparison_rc=$?
fi
printf '%s\n' "$comparison_rc" > "$RUN_ROOT/comparison.rc"
[[ $comparison_rc -eq 0 ]]
CURRENT_STAGE=complete
write_stage
exit 0
