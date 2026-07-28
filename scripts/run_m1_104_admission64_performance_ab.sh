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
RUN_ROOT=$(python3 - "$2" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)
if [[ "$RUN_ROOT" != /tmp/* || -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
    echo "RUN_ROOT must be a new private /tmp path" >&2
    exit 2
fi
case "$RUN_ROOT/" in
    "$ROOT/"*) echo "RUN_ROOT must stay outside the source tree" >&2; exit 2;;
esac
if [[ ! "$INSTANCE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    echo "INSTANCE must be a short non-sensitive label" >&2
    exit 2
fi

MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
BI100_RUNTIME_SITE_PACKAGES=${BI100_RUNTIME_SITE_PACKAGES:-}
RUNTIME_INSTALL=${BI100_RUNTIME_INSTALL_REPORT:-${RUNTIME_INSTALL_REPORT:-}}
SALT_NAMESPACE=m1-104-admission64-policy-ab-v1
TOTAL_TIMEOUT_S=43200
ARM_TIMEOUT_S=10800
RUN_DEADLINE=0
WATCHDOG_PID=""
CURRENT_STAGE=argument_validation
BEFORE_PREFLIGHT_PASSED=0
FINALIZED=0
ACTIVE_PID=""
ACTIVE_PGID=""
ACTIVE_STARTTIME=""
ACTIVE_SESSION_TOKEN=""
ACTIVE_LABEL=""

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
COREX_LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
COREX_PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
FATAL_PATTERN='CUDA error|illegal memory access|SIGSEGV|Fatal Python error|out of memory|device-side assert|CoreX.*(failed|fatal)|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|engine iteration timed out|watchdog.*tim(e|ed) out|Timeout(Error|Expired)|scheduler requested a missing GDN prefix state|non-finite GatedDeltaNet'

[[ -d "$MODEL_PATH" ]] || { echo "model directory is missing: $MODEL_PATH" >&2; exit 3; }
[[ -d "$BI100_RUNTIME_SITE_PACKAGES/vllm" && -d "$BI100_RUNTIME_SITE_PACKAGES/transformers" ]] || {
    echo "BI100_RUNTIME_SITE_PACKAGES must contain vllm and transformers" >&2; exit 3;
}
BI100_RUNTIME_SITE_PACKAGES=$(python3 - "$BI100_RUNTIME_SITE_PACKAGES" <<'PY'
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
[[ -n "$RUNTIME_INSTALL" ]] || RUNTIME_INSTALL="$RUNTIME_ROOT/install.json"
[[ -f "$RUNTIME_INSTALL" ]] || { echo "runtime install report is missing" >&2; exit 3; }
pgrep -f 'vllm\.entrypoints\.openai\.api_server' >/dev/null 2>&1 && {
    echo "an API server process is already running" >&2; exit 3;
}
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
        . ':(exclude)bench_runs/**')" ]]; then
    echo "M1-104 refuses a dirty source tree" >&2
    exit 3
fi

mkdir -p "$RUN_ROOT"
SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
SOURCE_BRANCH=$(git -C "$ROOT" branch --show-current)
printf '%s\n' "$SOURCE_REVISION" > "$RUN_ROOT/source_revision.txt"
printf '%s\n' "$SOURCE_BRANCH" > "$RUN_ROOT/source_branch.txt"
printf '%s\n' "$INSTANCE" > "$RUN_ROOT/instance.txt"
printf '%s\n' "$MODEL_PATH" > "$RUN_ROOT/model_path.txt"
printf '%s\n' "$BI100_RUNTIME_SITE_PACKAGES" > "$RUN_ROOT/runtime_site_packages.txt"
printf '%s\n' "$RUNTIME_INSTALL" > "$RUN_ROOT/runtime_install.txt"
printf '%s\n' "$SALT_NAMESPACE" > "$RUN_ROOT/salt_namespace.txt"
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
unset CUDA_VISIBLE_DEVICES
start_watchdog() {
    (sleep "$TOTAL_TIMEOUT_S"; kill -TERM "$$" 2>/dev/null || true) &
    WATCHDOG_PID=$!
}
stop_watchdog() {
    if [[ -n "$WATCHDOG_PID" ]]; then
        kill -TERM "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
        WATCHDOG_PID=""
    fi
}

write_stage() { printf '%s\n' "$CURRENT_STAGE" > "$RUN_ROOT/stage.txt"; }

run_preflight() {
    local out=$1
    timeout --signal=TERM --kill-after=90s 480s env -u CUDA_VISIBLE_DEVICES \
        PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
        python3 "$ROOT/tests/bi100_preflight.py" --gpus 0,1,2,3 --timeout-s 25 --matmul-size 1024 \
        --json-out "$out.json" > "$out.stdout" 2> "$out.stderr"
}

run_postflight() {
    local out=$1
    timeout --signal=TERM --kill-after=90s 300s env -u CUDA_VISIBLE_DEVICES \
        PYTHONPATH="$ROOT/tests" python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus 0,1,2,3 --settle-timeout-s 90 --clean-samples 3 --sample-interval-s 2 \
        --out "$out.json" > "$out.stdout" 2> "$out.stderr"
}

read_starttime() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys
p = Path('/proc') / sys.argv[1] / 'stat'
v = p.read_text(encoding='ascii'); print(v[v.rfind(')') + 2:].split()[19])
PY
}

active_live() {
    [[ -n "$ACTIVE_PID" && -n "$ACTIVE_STARTTIME" ]] || return 1
    python3 - "$ACTIVE_PID" "$ACTIVE_STARTTIME" <<'PY'
from pathlib import Path
import sys
try:
    v = (Path('/proc') / sys.argv[1] / 'stat').read_text(encoding='ascii')
except (FileNotFoundError, ProcessLookupError):
    raise SystemExit(1)
f = v[v.rfind(')') + 2:].split()
raise SystemExit(0 if f[0] != 'Z' and f[19] == sys.argv[2] else 1)
PY
}

stop_active() {
    local rc=0
    [[ -n "$ACTIVE_PID" ]] || return 0
    if active_live; then
        if [[ -n "$ACTIVE_PGID" && -n "$ACTIVE_SESSION_TOKEN" ]]; then
            bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID" 60 20 \
                "$ACTIVE_STARTTIME" "$ACTIVE_SESSION_TOKEN" || rc=$?
        else
            kill -TERM "$ACTIVE_PID" 2>/dev/null || true
            for _ in $(seq 1 60); do active_live || break; sleep 1; done
            if active_live; then kill -KILL "$ACTIVE_PID" 2>/dev/null || true; fi
            for _ in $(seq 1 20); do active_live || break; sleep 1; done
            active_live && rc=1
        fi
    fi
    wait "$ACTIVE_PID" 2>/dev/null || true
    [[ $rc -eq 0 ]] && ACTIVE_PID="" ACTIVE_PGID="" ACTIVE_STARTTIME="" ACTIVE_SESSION_TOKEN="" ACTIVE_LABEL=""
    return "$rc"
}

port_free() { python3 - <<'PY'
import socket
with socket.socket() as s: s.bind(('127.0.0.1', 8000))
PY
}

wait_port_free() {
    for _ in $(seq 1 180); do port_free && return 0; sleep 1; done
    return 1
}

health() { python3 - <<'PY'
import urllib.request
urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read()
PY
}

attest_service() {
    local identity=$1 pid=$2 start=$3 observed
    observed=$(python3 - "$identity" "$pid" "$start" <<'PY'
import json
from pathlib import Path
import sys
v = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
token = v.get('session_token')
if (v.get('schema') != 'bi100-process-session-v1' or v.get('version') != 1
        or v.get('pid') != int(sys.argv[2]) or v.get('pgid') != int(sys.argv[2])
        or v.get('sid') != int(sys.argv[2]) or v.get('starttime_ticks') != int(sys.argv[3])
        or not isinstance(token, str) or len(token) != 32
        or any(c not in '0123456789abcdef' for c in token)):
    raise SystemExit(1)
print(v['pgid'], token)
PY
    ) || return 1
    read -r ACTIVE_PGID ACTIVE_SESSION_TOKEN <<< "$observed"
}

start_service() {
    local arm=$1 policy=$2 identity=$arm/service_identity.json observed=""
    (
        exec env BI100_RUNTIME_SITE_PACKAGES="$BI100_RUNTIME_SITE_PACKAGES" \
            BI100_RUNTIME_INSTALL_REPORT="$RUNTIME_INSTALL" BI100_RUNTIME_WORKDIR="$arm/runtime-workdir" \
            MODEL_PATH="$MODEL_PATH" HOST=0.0.0.0 PORT=8000 ENABLE_CUSTOM_IPC=1 \
            VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
            BI100_MOE_COREX_DIRECT_ROUTED=1 BI100_GDN_COREX_PACKED_DECODE=1 BI100_GDN_COMBINED_QK_NORM=0 \
            BI100_GDN_CACHE_POLICY="$policy" BI100_GDN_RESTORE_MODE=direct \
            BI100_HYBRID_KV_ACCOUNTING=full_attention BI100_CPU_KV_OFFLOAD=0 BI100_BLOCK_MAJOR_CPU_KV=0 \
            BI100_CACHE_TRACE=1 BI100_ATTN_COREX_FUSED_PREFILL=0 \
            BI100_KV_EVICTION_POLICY=lru \
            BI100_ATTN_COREX_FUSED_PREFILL_DIAGNOSTICS=0 BI100_PROFILE=0 BI100_PROFILE_INCLUDE_STARTUP=0 \
            BI100_PAGED_ATTN_DIAGNOSTICS=0 BI100_GDN_ALLOW_NAN_ZERO=0 BI100_GDN_FINITE_CHECK=0 \
            PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH" \
            LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
            python3 "$ROOT/scripts/exec_bi100_session.py" "$identity" -- "$ROOT/launch_service"
    ) > "$arm/server.log" 2>&1 &
    ACTIVE_PID=$!; ACTIVE_LABEL=$(basename "$arm"); ACTIVE_STARTTIME=""
    for _ in $(seq 1 50); do
        ACTIVE_STARTTIME=$(read_starttime "$ACTIVE_PID" 2>/dev/null || true)
        [[ -n "$ACTIVE_STARTTIME" ]] && break
        kill -0 "$ACTIVE_PID" 2>/dev/null || break; sleep 0.1
    done
    [[ -n "$ACTIVE_STARTTIME" ]] || return 1
    for _ in $(seq 1 100); do
        if [[ -s "$identity" ]] && attest_service "$identity" "$ACTIVE_PID" "$ACTIVE_STARTTIME"; then
            printf '%s\n' "$ACTIVE_PID" > "$arm/server.pid"
            printf '%s\n' "$ACTIVE_PGID" > "$arm/server.pgid"
            break
        fi
        active_live || break; sleep 0.1
    done
    [[ -s "$identity" && -n "$ACTIVE_PGID" ]] || return 1
    for _ in $(seq 1 360); do health && return 0; active_live || break; sleep 10; done
    return 1
}

scan_fatal() {
    local out=$1 file found=0
    : > "$out"
    while IFS= read -r -d '' file; do
        if grep -Eiq "$FATAL_PATTERN" "$file" 2>/dev/null; then
            printf 'file=%s\n' "$file" >> "$out"
            grep -Ein "$FATAL_PATTERN" "$file" >> "$out" 2>/dev/null || true
            found=1
        fi
    done < <(find "$RUN_ROOT" -type f \( -name '*.log' -o -name '*.stdout' -o -name '*.stderr' \) -print0)
    return "$found"
}

scan_timeouts() {
    local out=$1 file value found=0
    : > "$out"
    while IFS= read -r -d '' file; do
        value=$(tr -d '[:space:]' < "$file")
        if [[ ! "$value" =~ ^[0-9]+$ ]]; then
            printf '%s=malformed:%s\n' "$file" "$value" >> "$out"
            found=1
            continue
        fi
        case "$value" in 124|137|143) printf '%s=%s\n' "$file" "$value" >> "$out"; found=1;; esac
    done < <(find "$RUN_ROOT" -type f -name '*.rc' -print0)
    return "$found"
}

write_arm_status() {
    local arm=$1 pair=$2 label=$3 policy=$4 rc=$5
    python3 - "$arm" "$pair" "$label" "$policy" "$rc" <<'PY'
import hashlib, json
from pathlib import Path
import sys
r = Path(sys.argv[1])
def code(name):
    p = r / name
    if not p.is_file(): return None
    v = p.read_text(encoding='utf-8').strip()
    return int(v) if v.isdigit() else None
def sha(name):
    p = r / name
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
report = {
    'schema': 'bi100-m1-104-admission64-performance-arm-v1', 'version': 1,
    'pair': int(sys.argv[2]), 'label': sys.argv[3], 'policy': sys.argv[4],
    'qualified': int(sys.argv[5]) == 0, 'valid_measurement': code('measurement.rc') == 0,
    'gates': {n: code(n + '.rc') for n in (
        'preflight_before', 'startup', 'startup_contract', 'measurement', 'health_after',
        'cleanup', 'fatal_scan', 'service_postflight', 'preflight_after', 'preflight_comparison')},
    'artifact_sha256': {n: sha(n) for n in ('measurement.json', 'service_identity.json',
                                             'service_postflight.json', 'preflight_comparison.json')},
    'model_quality_evaluated': False, 'production_promotion_authorized': False,
}
(r / 'status.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
}

run_arm() {
    local pair=$1 label=$2 policy=$3 arm="$RUN_ROOT/pair${pair}_${label}" rc=0 arm_rc=0
    local startup_rc=1 contract_rc=1 measurement_rc=1 health_rc=1 cleanup_rc=1 port_rc=1 fatal_rc=1 postflight_rc=1 after_rc=1 compare_rc=1
    mkdir -p "$arm/runtime-workdir"
    set +e; run_preflight "$arm/preflight_before"; rc=$?; set -e
    printf '%s\n' "$rc" > "$arm/preflight_before.rc"
    if [[ $rc -ne 0 ]]; then write_arm_status "$arm" "$pair" "$label" "$policy" 1; return 1; fi
    set +e; wait_port_free && start_service "$arm" "$policy"; startup_rc=$?; set -e
    printf '%s\n' "$startup_rc" > "$arm/startup.rc"
    if [[ $startup_rc -eq 0 ]]; then
        contract_rc=0
        grep -Fq '[BI100] fixed evaluator contract;' "$arm/server.log" || contract_rc=1
        grep -Fq '[BI100] M1-49 runtime contract;' "$arm/server.log" || contract_rc=1
        grep -Fq '[BI100] fixed kernels; moe_direct=1 gdn_packed=1 gdn_combined_qk=0' "$arm/server.log" || contract_rc=1
        grep -Fq "[BI100] GDN cache; policy=$policy restore=direct" "$arm/server.log" || contract_rc=1
        grep -Fq 'accounting=full_attention' "$arm/server.log" || contract_rc=1
        grep -Fq 'cpu_kv_offload=0' "$arm/server.log" || contract_rc=1
        grep -Fq 'cache_trace=1' "$arm/server.log" || contract_rc=1
        grep -Fq 'fused_prefill=0' "$arm/server.log" || contract_rc=1
        grep -Fq 'kv_eviction_policy=lru' "$arm/server.log" || contract_rc=1
    else
        contract_rc=1
    fi
    printf '%s\n' "$contract_rc" > "$arm/startup_contract.rc"; [[ $contract_rc -eq 0 ]] || arm_rc=1
    if [[ $contract_rc -eq 0 ]]; then
        set +e
        timeout --signal=TERM --kill-after=60s "$ARM_TIMEOUT_S" \
            env PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH" \
            LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
            python3 "$ROOT/tests/bench_m1_104_admission64_policy_matrix.py" \
            --base http://127.0.0.1:8000 --model-path "$MODEL_PATH" --policy "$policy" \
            --salt-namespace "$SALT_NAMESPACE" --out "$arm/measurement.json" \
            > "$arm/measurement.stdout" 2> "$arm/measurement.stderr"
        measurement_rc=$?; set -e
    fi
    printf '%s\n' "$measurement_rc" > "$arm/measurement.rc"
    [[ $measurement_rc -eq 0 ]] || arm_rc=1
    if [[ $startup_rc -eq 0 ]]; then set +e; health; health_rc=$?; set -e; fi
    printf '%s\n' "$health_rc" > "$arm/health_after.rc"; [[ $health_rc -eq 0 ]] || arm_rc=1
    set +e
    stop_active
    cleanup_rc=$?
    wait_port_free
    port_rc=$?
    [[ $port_rc -eq 0 ]] || cleanup_rc=1
    set -e
    printf '%s\n' "$cleanup_rc" > "$arm/cleanup.rc"; [[ $cleanup_rc -eq 0 ]] || arm_rc=1
    set +e; scan_fatal "$arm/fatal_scan.txt"; fatal_rc=$?; set -e
    printf '%s\n' "$fatal_rc" > "$arm/fatal_scan.rc"; [[ $fatal_rc -eq 0 ]] || arm_rc=1
    set +e; run_postflight "$arm/service_postflight"; postflight_rc=$?; set -e
    printf '%s\n' "$postflight_rc" > "$arm/service_postflight.rc"; [[ $postflight_rc -eq 0 ]] || arm_rc=1
    if [[ $cleanup_rc -eq 0 && $postflight_rc -eq 0 ]]; then
        set +e; run_preflight "$arm/preflight_after"; after_rc=$?; set -e
        if [[ $after_rc -eq 0 ]]; then
            set +e
            python3 "$ROOT/tests/compare_bi100_preflights.py" \
                --preflight "before=$arm/preflight_before.json" --preflight "after=$arm/preflight_after.json" \
                --expected-gpus 0,1,2,3 --max-free-memory-drop-bytes 1073741824 \
                --out "$arm/preflight_comparison.json" > "$arm/preflight_comparison.stdout" 2> "$arm/preflight_comparison.stderr"
            compare_rc=$?; set -e
        fi
    fi
    printf '%s\n' "$after_rc" > "$arm/preflight_after.rc"; printf '%s\n' "$compare_rc" > "$arm/preflight_comparison.rc"
    [[ $after_rc -eq 0 && $compare_rc -eq 0 ]] || arm_rc=1
    write_arm_status "$arm" "$pair" "$label" "$policy" "$arm_rc"
    return "$arm_rc"
}

write_status() {
    local final_rc=$1
    python3 - "$RUN_ROOT" "$INSTANCE" "$SOURCE_REVISION" "$SOURCE_BRANCH" "$CURRENT_STAGE" "$final_rc" <<'PY'
import hashlib, json
from pathlib import Path
import sys
r = Path(sys.argv[1])
def code(n):
    p = r / n
    if not p.is_file(): return None
    v = p.read_text(encoding='utf-8').strip()
    return int(v) if v.isdigit() else None
def sha(n):
    p = r / n
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
arms = ('pair1_control','pair1_candidate','pair2_candidate','pair2_control','pair3_control','pair3_candidate')
cmp = {}
try: cmp = json.loads((r / 'comparison.json').read_text(encoding='utf-8'))
except (OSError, ValueError): pass
report = {
  'schema': 'bi100-m1-104-admission64-performance-runner-v1', 'version': 1,
  'qualified': int(sys.argv[6]) == 0, 'returncode': int(sys.argv[6]),
  'source_revision': sys.argv[3], 'source_branch': sys.argv[4], 'instance': sys.argv[2],
  'terminal_stage': sys.argv[5], 'comparison_qualified': cmp.get('qualified') is True,
  'arms': list(arms), 'gates': {n: code(n) for n in (
      'postflight_before.rc','preflight_before.rc','runtime_identity.rc','comparison.rc',
      'recovery.rc','recovery_clean.rc','scoped_cleanup.rc','scoped_cleanup_clean.rc',
      'fatal_scan.rc','timeout_scan.rc','source_unchanged.rc','final_postflight.rc',
      'final_preflight.rc','final_preflight_comparison.rc')},
  'arm_status': {a: code(a + '.rc') for a in arms},
  'artifact_sha256': {n: sha(n) for n in (
      'runtime_identity.json','comparison.json','recovery_clean.json','scoped_cleanup_clean.json',
      'final_postflight.json','final_preflight_comparison.json')},
  'decision': {'full_quality_m1_85_authorized': cmp.get('qualified') is True,
      'official_style_replay_authorized': False, 'production_promotion_authorized': False,
      'yaml_change_authorized': False, 'main_merge_authorized': False},
  'official_881_evaluated': False, 'full_model_quality_suite_evaluated': False,
}
(r / 'runner_status.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
}

finish() {
    local primary=$? final=$primary cleanup=0 recovery=0 recovery_clean=0 fatal=0 timeouts=0 source=0 post=0 after=0 comparison=0 cleanup_clean=0
    local identity identity_args=() expected_args=() current_revision current_status file value
    trap - EXIT; trap '' INT TERM; set +e
    stop_watchdog
    stop_active || cleanup=1
    while IFS= read -r -d '' identity; do
        identity_args+=(--identity "$identity"); expected_args+=(--expected-identity "$identity")
    done < <(find "$RUN_ROOT" -type f -name service_identity.json -print0 | sort -z)
    if [[ ${#identity_args[@]} -gt 0 ]]; then
        timeout --signal=TERM --kill-after=70s 600s python3 "$ROOT/scripts/cleanup_recorded_bi100_sessions.py" \
            "${identity_args[@]}" --out "$RUN_ROOT/scoped_cleanup.json" > "$RUN_ROOT/scoped_cleanup.stdout" 2> "$RUN_ROOT/scoped_cleanup.stderr"
        [[ $? -eq 0 ]] || cleanup=1
        python3 "$ROOT/tests/qualify_recorded_session_cleanup.py" "$RUN_ROOT/scoped_cleanup.json" \
            "${expected_args[@]}" --out "$RUN_ROOT/scoped_cleanup_clean.json" > "$RUN_ROOT/scoped_cleanup_clean.stdout" 2> "$RUN_ROOT/scoped_cleanup_clean.stderr"
        cleanup_clean=$?
    else
        printf '%s\n' '{"schema":"bi100-no-recorded-session","qualified":true}' > "$RUN_ROOT/scoped_cleanup.json"
        cp "$RUN_ROOT/scoped_cleanup.json" "$RUN_ROOT/scoped_cleanup_clean.json"
    fi
    printf '%s\n' "$cleanup" > "$RUN_ROOT/scoped_cleanup.rc"; printf '%s\n' "$cleanup_clean" > "$RUN_ROOT/scoped_cleanup_clean.rc"
    if [[ ${#identity_args[@]} -gt 0 ]]; then
        timeout --signal=TERM --kill-after=70s 600s python3 "$ROOT/scripts/cleanup_recorded_bi100_sessions.py" \
            "${identity_args[@]}" --out "$RUN_ROOT/recovery.json" \
            > "$RUN_ROOT/recovery.stdout" 2> "$RUN_ROOT/recovery.stderr"; recovery=$?
        printf '%s\n' "$recovery" > "$RUN_ROOT/recovery.rc"
        python3 "$ROOT/tests/qualify_recorded_session_cleanup.py" "$RUN_ROOT/recovery.json" \
            "${expected_args[@]}" --out "$RUN_ROOT/recovery_clean.json" \
            > "$RUN_ROOT/recovery_clean.stdout" 2> "$RUN_ROOT/recovery_clean.stderr"; recovery_clean=$?
        printf '%s\n' "$recovery_clean" > "$RUN_ROOT/recovery_clean.rc"
    else
        printf '%s\n' '{"schema":"bi100-no-recorded-session","qualified":true}' > "$RUN_ROOT/recovery.json"
        cp "$RUN_ROOT/recovery.json" "$RUN_ROOT/recovery_clean.json"
        printf '%s\n' 0 > "$RUN_ROOT/recovery.rc"
        printf '%s\n' 0 > "$RUN_ROOT/recovery_clean.rc"
    fi
    current_revision=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null); current_status=$(git -C "$ROOT" status --porcelain --untracked-files=all -- . ':(exclude)bench_runs/**')
    [[ "$current_revision" == "$SOURCE_REVISION" && -z "$current_status" ]] || source=1
    printf '%s\n' "$source" > "$RUN_ROOT/source_unchanged.rc"
    scan_fatal "$RUN_ROOT/fatal_scan.txt"; fatal=$?; printf '%s\n' "$fatal" > "$RUN_ROOT/fatal_scan.rc"
    scan_timeouts "$RUN_ROOT/timeout_scan.txt"; timeouts=$?; printf '%s\n' "$timeouts" > "$RUN_ROOT/timeout_scan.rc"
    run_postflight "$RUN_ROOT/final_postflight"; post=$?; printf '%s\n' "$post" > "$RUN_ROOT/final_postflight.rc"
    if [[ $BEFORE_PREFLIGHT_PASSED -eq 1 && $cleanup -eq 0 && $cleanup_clean -eq 0 && $post -eq 0 ]]; then
        run_preflight "$RUN_ROOT/final_preflight"; after=$?
        if [[ $after -eq 0 ]]; then
            python3 "$ROOT/tests/compare_bi100_preflights.py" --preflight "before=$RUN_ROOT/preflight_before.json" \
                --preflight "after=$RUN_ROOT/final_preflight.json" --expected-gpus 0,1,2,3 \
                --max-free-memory-drop-bytes 1073741824 --out "$RUN_ROOT/final_preflight_comparison.json" \
                > "$RUN_ROOT/final_preflight_comparison.stdout" 2> "$RUN_ROOT/final_preflight_comparison.stderr"; comparison=$?
        else comparison=1; fi
    else after=1; comparison=1; fi
    printf '%s\n' "$after" > "$RUN_ROOT/final_preflight.rc"; printf '%s\n' "$comparison" > "$RUN_ROOT/final_preflight_comparison.rc"
    if [[ $cleanup -ne 0 || $cleanup_clean -ne 0 || $recovery -ne 0 || $recovery_clean -ne 0 || $fatal -ne 0 || $timeouts -ne 0 || $source -ne 0 || $post -ne 0 || $after -ne 0 || $comparison -ne 0 ]]; then final=1; fi
    printf '%s\n' "$final" > "$RUN_ROOT/overall.rc"; write_status "$final"; FINALIZED=1; exit "$final"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_watchdog
CURRENT_STAGE=postflight_before; write_stage; set +e; run_postflight "$RUN_ROOT/postflight_before"; rc=$?; set -e; printf '%s\n' "$rc" > "$RUN_ROOT/postflight_before.rc"; [[ $rc -eq 0 ]]
CURRENT_STAGE=preflight_before; write_stage; set +e; run_preflight "$RUN_ROOT/preflight_before"; rc=$?; set -e; printf '%s\n' "$rc" > "$RUN_ROOT/preflight_before.rc"; [[ $rc -eq 0 ]]; BEFORE_PREFLIGHT_PASSED=1
CURRENT_STAGE=runtime_identity; write_stage; set +e
timeout --signal=TERM --kill-after=70s 240s env PYTHONPATH="$ROOT/tests:$BI100_RUNTIME_SITE_PACKAGES:$SYSTEM_PYTHONPATH" LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
    python3 "$ROOT/tests/verify_bare_host_runtime_identity.py" --source-root "$ROOT" --runtime-site-packages "$BI100_RUNTIME_SITE_PACKAGES" \
    --runtime-install "$RUNTIME_INSTALL" --out "$RUN_ROOT/runtime_identity.json" > "$RUN_ROOT/runtime_identity.stdout" 2> "$RUN_ROOT/runtime_identity.stderr"; rc=$?; set -e
printf '%s\n' "$rc" > "$RUN_ROOT/runtime_identity.rc"; [[ $rc -eq 0 ]]
RUN_DEADLINE=$((SECONDS + TOTAL_TIMEOUT_S))

for specification in '1 control fine32' '1 candidate admission64' '2 candidate admission64' '2 control fine32' '3 control fine32' '3 candidate admission64'; do
    read -r pair label policy <<< "$specification"; CURRENT_STAGE="pair${pair}_${label}"; write_stage
    if (( SECONDS >= RUN_DEADLINE )); then
        echo "M1-104 fixed total timeout reached before $CURRENT_STAGE" >&2
        exit 124
    fi
    set +e; run_arm "$pair" "$label" "$policy"; rc=$?; set -e
    printf '%s\n' "$rc" > "$RUN_ROOT/pair${pair}_${label}.rc"
    [[ $rc -eq 0 ]] || exit 1
done

CURRENT_STAGE=comparison; write_stage; set +e
python3 "$ROOT/tests/compare_m1_104_admission64_paired_ab.py" \
    --control "$RUN_ROOT/pair1_control/measurement.json" \
    --control "$RUN_ROOT/pair2_control/measurement.json" \
    --control "$RUN_ROOT/pair3_control/measurement.json" \
    --candidate "$RUN_ROOT/pair1_candidate/measurement.json" \
    --candidate "$RUN_ROOT/pair2_candidate/measurement.json" \
    --candidate "$RUN_ROOT/pair3_candidate/measurement.json" \
    --out "$RUN_ROOT/comparison.json" \
    > "$RUN_ROOT/comparison.stdout" 2> "$RUN_ROOT/comparison.stderr"; rc=$?; set -e
printf '%s\n' "$rc" > "$RUN_ROOT/comparison.rc"; [[ $rc -eq 0 ]]
CURRENT_STAGE=complete; write_stage; exit 0
