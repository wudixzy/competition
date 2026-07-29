#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)

if [[ $# -ne 3 ]]; then
    echo "usage: $0 INSTANCE IFEVAL_ENV RUN_ROOT" >&2
    exit 2
fi

export BI100_QUALITY_AB_VARIANT=m1-122-fused-prefill-ifeval
export BI100_IFEVAL_ENV=$2
exec "$ROOT/scripts/run_m1_85_admission64_quality_ab.sh" "$1" "$3"
