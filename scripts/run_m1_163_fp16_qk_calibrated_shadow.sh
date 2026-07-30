#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ $# -ne 3 ]]; then
    echo "usage: $0 INSTANCE PRIVATE_EXTENSION RUN_ROOT" >&2
    exit 2
fi

export BI100_FUSED_PREFILL_SHADOW_VARIANT=calibrated_v2
export BI100_FUSED_PREFILL_SHADOW_EXTENSION=$2
exec "$ROOT/scripts/run_m1_136_fused_prefill_shadow.sh" "$1" "$3"
