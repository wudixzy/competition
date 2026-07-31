#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)

export CANDIDATE_POLICY=off
export ARM_ORDER=${ARM_ORDER:-admission64,off}
export BENCH_SALT_ORDER=identity-first
export BENCH_SALT_NAMESPACE=m1-170-cold-capture-overhead-v1
export PORT=${PORT:-8062}

exec "$ROOT/scripts/run_m1_169_tail64_nofinal_tp1_ab.sh" "$@"
