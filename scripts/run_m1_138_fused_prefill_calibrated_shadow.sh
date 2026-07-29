#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)

if [[ $# -ne 2 ]]; then
    echo "usage: $0 INSTANCE RUN_ROOT" >&2
    exit 2
fi

export BI100_FUSED_PREFILL_SHADOW_VARIANT=calibrated
exec "$ROOT/scripts/run_m1_136_fused_prefill_shadow.sh" "$1" "$2"
