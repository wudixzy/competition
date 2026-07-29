#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
export BI100_COMPONENT_AB_VARIANT=m1-126-grouped-qk-split512
exec "$ROOT/scripts/run_m1_109_fused_softmax_component_ab.sh" "$@"
