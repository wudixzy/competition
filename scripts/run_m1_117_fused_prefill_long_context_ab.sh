#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
export BI100_QUALITY_AB_VARIANT=m1-117-fused-prefill-long-context
exec "$ROOT/scripts/run_m1_85_admission64_quality_ab.sh" "$@"
