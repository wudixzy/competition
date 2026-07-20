#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_ROOT=${M1_33_RUN_ROOT:-$ROOT/bench_runs/m1_33}
BASELINE_DIR=${M1_33_BASELINE_DIR:-$ROOT/bench_runs/m1_32/fine32_direct_fixed}
MODEL_PATH=${MODEL_PATH:-/root/public-storage/models/Qwen/Qwen3.6-35B-A3B}
OPERATOR_DIR="$RUN_ROOT/operator_exactness"
RUNTIME_DIR="$RUN_ROOT/runtime_preflight"

export PYTHONPATH="$ROOT/tests:/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$RUNTIME_DIR"
set +e
timeout --signal=TERM --kill-after=5s 60s \
    python3 - "$ROOT/qwen3_6_scripts/gdn_prefix.py" \
    > "$RUNTIME_DIR/stdout.log" 2> "$RUNTIME_DIR/stderr.log" <<'PY'
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

source = Path(sys.argv[1]).resolve()
candidates = [
    (Path(entry) / "vllm" / "gdn_prefix.py").resolve()
    for entry in sys.path
    if entry and (Path(entry) / "vllm" / "gdn_prefix.py").is_file()
]
if not candidates:
    raise SystemExit("installed vllm/gdn_prefix.py was not found on sys.path")
runtime = candidates[0]
source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
runtime_digest = hashlib.sha256(runtime.read_bytes()).hexdigest()
if runtime_digest != source_digest:
    raise SystemExit(
        "stale GDN runtime override: "
        f"source={source_digest} runtime={runtime_digest} path={runtime}")

spec = importlib.util.spec_from_file_location("bi100_runtime_gdn_prefix", runtime)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load runtime module: {runtime}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
os.environ["BI100_GDN_RESTORE_MODE"] = "chunk64"
if module.gdn_restore_mode_from_env() != "chunk64":
    raise SystemExit("deployed runtime rejected chunk64 restore mode")
if module.gdn_restore_alignment("chunk64", 16, 8192) != 64:
    raise SystemExit("deployed runtime returned the wrong chunk64 alignment")
print(f"runtime={runtime}")
print(f"sha256={runtime_digest}")
print("chunk64_alignment=64")
PY
runtime_rc=$?
set -e
printf '%s\n' "$runtime_rc" > "$RUNTIME_DIR/runtime.rc"
if [[ $runtime_rc -ne 0 ]]; then
    echo "M1-33 runtime deployment preflight failed: rc=$runtime_rc" >&2
    cat "$RUNTIME_DIR/stderr.log" >&2 || true
    exit "$runtime_rc"
fi

mkdir -p "$OPERATOR_DIR"
set +e
timeout --signal=TERM --kill-after=20s 900s \
    python3 "$ROOT/tests/gdn_split_exactness.py" \
    --device cuda:0 --out "$OPERATOR_DIR/result.json" \
    > "$OPERATOR_DIR/stdout.log" 2> "$OPERATOR_DIR/stderr.log"
operator_rc=$?
set -e
printf '%s\n' "$operator_rc" > "$OPERATOR_DIR/operator.rc"
if [[ $operator_rc -ne 0 ]]; then
    echo "M1-33 native-chunk exactness failed: rc=$operator_rc" >&2
    exit "$operator_rc"
fi

exec env \
    M1_32_RUN_ROOT="$RUN_ROOT" \
    M1_32_FIXED_DIR="$BASELINE_DIR" \
    M1_32_START_AT=aligned \
    M1_32_FALLBACK_LABEL=admission64_chunk64_gate \
    M1_32_FALLBACK_POLICY=admission64 \
    M1_32_FALLBACK_MODE=chunk64 \
    M1_32_FALLBACK_MIN_CACHED=234944 \
    M1_32_FALLBACK_PRESSURE_RUN_ID=m1_33_chunk64_pressure \
    M1_32_FALLBACK_RUN_ID=m1_33_chunk64_235k \
    MODEL_PATH="$MODEL_PATH" \
    bash "$ROOT/scripts/run_m1_32_remaining_gates.sh"
