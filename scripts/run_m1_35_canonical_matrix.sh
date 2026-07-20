#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_ROOT=${M1_35_RUN_ROOT:-$ROOT/bench_runs/m1_35}
GUARD_DIR=${M1_35_GUARD_DIR:-$ROOT/bench_runs/m1_34/legacy_m1_32_admission64_guard2}
BASELINE_DIR=${M1_35_BASELINE_DIR:-$ROOT/bench_runs/m1_32/fine32_direct_fixed}
CANDIDATE_DIR=${M1_35_MATRIX_DIR:-$RUN_ROOT/admission64_direct_canonical_fixed}
RUNTIME_DIR="$RUN_ROOT/runtime_preflight"

export PYTHONPATH="$ROOT/tests:/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$RUNTIME_DIR"
set +e
timeout --signal=TERM --kill-after=5s 60s \
    python3 - "$ROOT/qwen3_6_scripts/gdn_prefix.py" \
    "$ROOT/qwen3_6_scripts/scheduler.py" \
    > "$RUNTIME_DIR/stdout.log" 2> "$RUNTIME_DIR/stderr.log" <<'PY'
import hashlib
import importlib.util
from pathlib import Path
import sys


def installed(relative: str) -> Path:
    candidates = [
        (Path(entry) / "vllm" / relative).resolve()
        for entry in sys.path
        if entry and (Path(entry) / "vllm" / relative).is_file()
    ]
    if not candidates:
        raise SystemExit(f"installed vllm/{relative} was not found")
    return candidates[0]


source_gdn = Path(sys.argv[1]).resolve()
source_scheduler = Path(sys.argv[2]).resolve()
runtime_gdn = installed("gdn_prefix.py")
runtime_scheduler = installed("core/scheduler.py")
for source, runtime in (
        (source_gdn, runtime_gdn),
        (source_scheduler, runtime_scheduler)):
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    runtime_digest = hashlib.sha256(runtime.read_bytes()).hexdigest()
    if runtime_digest != source_digest:
        raise SystemExit(
            f"stale runtime override: source={source_digest} "
            f"runtime={runtime_digest} path={runtime}")
    print(f"sha256={runtime_digest} path={runtime}")

spec = importlib.util.spec_from_file_location(
    "bi100_runtime_gdn_prefix", runtime_gdn)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load runtime module: {runtime_gdn}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
key = module.make_prefix_key(2, b"x" * 32)
admission = module.GdnPrefixStatePolicy("admission64")
if not admission.should_capture_final(key):
    raise SystemExit("admission64 rejected a new canonical final state")
admission.admit([key])
if admission.should_capture_final(key):
    raise SystemExit("admission64 would recapture a resident final state")
fine = module.GdnPrefixStatePolicy("fine32")
fine.admit([key])
if not fine.should_capture_final(key):
    raise SystemExit("canonical retention changed fine32 behavior")
print("canonical_final_retention=ok")
PY
runtime_rc=$?
set -e
printf '%s\n' "$runtime_rc" > "$RUNTIME_DIR/runtime.rc"
if [[ $runtime_rc -ne 0 ]]; then
    echo "M1-35 runtime deployment preflight failed: rc=$runtime_rc" >&2
    cat "$RUNTIME_DIR/stderr.log" >&2 || true
    exit "$runtime_rc"
fi

exec env \
    M1_34_GUARD_DIR="$GUARD_DIR" \
    M1_34_BASELINE_DIR="$BASELINE_DIR" \
    M1_34_MATRIX_DIR="$CANDIDATE_DIR" \
    M1_34_CANDIDATE_LABEL=admission64_direct_canonical_fixed \
    bash "$ROOT/scripts/run_m1_34_fixed_matrix.sh"
