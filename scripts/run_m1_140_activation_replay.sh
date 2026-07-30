#!/bin/bash
set -Eeuo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/lib/process_group.sh"

if [[ $# -ne 5 ]]; then
    echo "usage: $0 INSTANCE BANK_DIR EXTENSION RUN_ROOT PROFILE" >&2
    exit 2
fi

INSTANCE=$1
BANK_DIR=$(realpath "$2")
EXTENSION=$(realpath "$3")
RUN_ROOT=$(python3 - "$4" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)
PROFILE=$5
case "$PROFILE" in
    smoke|qualification) ;;
    *) echo "profile must be smoke or qualification" >&2; exit 2 ;;
esac
if [[ "$RUN_ROOT" != /tmp/* || "$RUN_ROOT" == /tmp ]]; then
    echo "replay output must be a new private path under /tmp" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
    echo "replay output already exists: $RUN_ROOT" >&2
    exit 2
fi
if [[ ! -s "$EXTENSION" ]]; then
    echo "candidate extension is missing" >&2
    exit 2
fi
for rank in 0 1 2 3; do
    test -s "$BANK_DIR/rank-$rank.manifest.json" || {
        echo "activation manifest is missing for rank $rank" >&2
        exit 2
    }
done
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
        . ':(exclude)bench_runs/**')" ]]; then
    echo "replay runner requires a clean candidate source tree" >&2
    exit 3
fi

SYSTEM_PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
COREX_LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
COREX_PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD)
SOURCE_BRANCH=$(git -C "$ROOT" branch --show-current)
EXTENSION_SHA=$(sha256sum "$EXTENSION" | cut -d' ' -f1)
read -r CAPTURE_SOURCE_REVISION RUNTIME_IDENTITY < <(
    python3 - "$BANK_DIR" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
values = [
    json.loads((root / f"rank-{rank}.manifest.json").read_text())
    for rank in range(4)
]
sources = {value.get("source_revision") for value in values}
runtimes = {value.get("runtime_identity") for value in values}
if len(sources) != 1 or len(runtimes) != 1:
    raise SystemExit("activation bank identities differ")
source = next(iter(sources))
runtime = next(iter(runtimes))
if not isinstance(source, str) or not isinstance(runtime, str):
    raise SystemExit("activation bank identity is missing")
print(source, runtime)
PY
)

mkdir -p "$RUN_ROOT"
RUN_ID="m1-140-replay-${SOURCE_REVISION:0:12}-${EXTENSION_SHA:0:12}"
TIMELINE="$RUN_ROOT/timeline.jsonl"
PIDS=()
PGIDS=()
CURRENT_STAGE=initialization
BEFORE_PREFLIGHT_PASSED=0

printf '%s\n' "$SOURCE_REVISION" > "$RUN_ROOT/candidate_source_revision.txt"
printf '%s\n' "$CAPTURE_SOURCE_REVISION" \
    > "$RUN_ROOT/capture_source_revision.txt"
printf '%s\n' "$SOURCE_BRANCH" > "$RUN_ROOT/source_branch.txt"
printf '%s\n' "$EXTENSION_SHA" > "$RUN_ROOT/extension_sha256.txt"
printf '%s\n' "$INSTANCE" > "$RUN_ROOT/instance.txt"

timeline_mark() {
    local stage=$1
    local event=$2
    local status=${3:-}
    local args=(
        python3 "$ROOT/scripts/record_experiment_timeline.py" mark
        --timeline "$TIMELINE" --run-id "$RUN_ID"
        --stage "$stage" --event "$event"
    )
    [[ -z "$status" ]] || args+=(--status "$status")
    "${args[@]}" >/dev/null
}

run_preflight() {
    local label=$1
    timeout --foreground --signal=TERM --kill-after=90s 480s \
        env PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
        LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
        python3 "$ROOT/tests/bi100_parallel_preflight.py" \
        --gpus 0,1,2,3 --timeout-s 25 --matmul-size 1024 \
        --json-out "$RUN_ROOT/$label.json" \
        --work-dir "$RUN_ROOT/$label-parallel" \
        > "$RUN_ROOT/$label.stdout" 2> "$RUN_ROOT/$label.stderr"
}

run_postflight() {
    local label=$1
    timeout --foreground --signal=TERM --kill-after=90s 300s \
        env PYTHONPATH="$ROOT/tests" \
        python3 "$ROOT/tests/service_postflight_gate.py" \
        --gpus 0,1,2,3 --settle-timeout-s 90 \
        --clean-samples 3 --sample-interval-s 2 \
        --out "$RUN_ROOT/$label.json" \
        > "$RUN_ROOT/$label.stdout" 2> "$RUN_ROOT/$label.stderr"
}

write_status() {
    local final_rc=$1
    python3 - "$RUN_ROOT" "$RUN_ID" "$SOURCE_REVISION" "$SOURCE_BRANCH" \
            "$CAPTURE_SOURCE_REVISION" "$RUNTIME_IDENTITY" "$INSTANCE" \
            "$PROFILE" "$EXTENSION_SHA" "$CURRENT_STAGE" "$final_rc" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def sha(name):
    path = root / name
    return hashlib.sha256(path.read_bytes()).hexdigest() \
        if path.is_file() else None

timeline = json.loads((root / "timeline_report.json").read_text()) \
    if (root / "timeline_report.json").is_file() else {}
value = {
    "schema": "bi100-m1-140-activation-replay-runner-v1",
    "version": 1,
    "qualified": int(sys.argv[11]) == 0,
    "returncode": int(sys.argv[11]),
    "terminal_stage": sys.argv[10],
    "run_id": sys.argv[2],
    "candidate_source_revision": sys.argv[3],
    "candidate_source_branch": sys.argv[4],
    "capture_source_revision": sys.argv[5],
    "runtime_identity": sys.argv[6],
    "instance": sys.argv[7],
    "profile": sys.argv[8],
    "candidate_extension_sha256": sys.argv[9],
    "gpu_count": 4,
    "parallel_rank_replays": 4,
    "timing": {
        "wall_span_s": timeline.get("wall_span_s"),
        "summed_stage_s": timeline.get("summed_stage_s"),
        "effective_parallelism": timeline.get("effective_parallelism"),
    },
    "artifact_sha256": {
        name: sha(name)
        for name in (
            "qualification.json", "timeline_report.json",
            "preflight_comparison.json", "final_postflight.json",
        )
    },
    "authorization": {
        "short_tp4_authorized": (
            int(sys.argv[11]) == 0 and sys.argv[8] == "qualification"),
        "long_context_authorized": False,
        "main_or_yaml_change_authorized": False,
    },
    "privacy": {
        "raw_activation_tensors_in_report": False,
        "credentials_recorded": False,
    },
}
(root / "runner_status.json").write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
}

finish() {
    local primary_rc=$?
    local cleanup_rc=0
    local fatal_rc=0
    local postflight_rc=0
    local preflight_rc=1
    local comparison_rc=1
    local pid pgid
    trap - EXIT
    trap '' INT TERM
    set +e

    CURRENT_STAGE=scoped_cleanup
    timeline_mark scoped_cleanup start
    for index in "${!PIDS[@]}"; do
        pid=${PIDS[$index]}
        pgid=${PGIDS[$index]}
        bi100_stop_process_group "$pgid" "$pid" 60 20 || cleanup_rc=1
    done
    PIDS=()
    PGIDS=()
    if [[ $cleanup_rc -eq 0 ]]; then
        timeline_mark scoped_cleanup end pass
    else
        timeline_mark scoped_cleanup end fail
        primary_rc=1
    fi

    CURRENT_STAGE=fatal_scan
    timeline_mark fatal_scan start
    : > "$RUN_ROOT/fatal_scan.txt"
    while IFS= read -r -d '' artifact; do
        if grep -Eiq \
                'CUDA error|illegal memory access|SIGSEGV|Fatal Python error|out of memory|device-side assert|CoreX.*(failed|fatal)|Gloo.*(failed|reset|error)|NCCL.*(failed|abort|error)|Connection reset by peer|worker.*(died|lost|exited unexpectedly)|Timeout(Error|Expired)' \
                "$artifact"; then
            printf 'file=%s\n' "$artifact" >> "$RUN_ROOT/fatal_scan.txt"
            fatal_rc=1
        fi
    done < <(find "$RUN_ROOT" -type f \
        \( -name '*.stdout' -o -name '*.stderr' \) -print0)
    if [[ $fatal_rc -eq 0 ]]; then
        timeline_mark fatal_scan end pass
    else
        timeline_mark fatal_scan end fail
        primary_rc=1
    fi

    CURRENT_STAGE=final_postflight
    timeline_mark final_postflight start
    run_postflight final_postflight
    postflight_rc=$?
    if [[ $postflight_rc -eq 0 ]]; then
        timeline_mark final_postflight end pass
    else
        timeline_mark final_postflight end fail
        primary_rc=1
    fi

    if [[ "$BEFORE_PREFLIGHT_PASSED" == 1 \
            && $cleanup_rc -eq 0 && $postflight_rc -eq 0 ]]; then
        CURRENT_STAGE=final_preflight
        timeline_mark final_preflight start
        run_preflight final_preflight
        preflight_rc=$?
        if [[ $preflight_rc -eq 0 ]]; then
            timeline_mark final_preflight end pass
            python3 "$ROOT/tests/compare_bi100_preflights.py" \
                --preflight "before=$RUN_ROOT/preflight_before.json" \
                --preflight "after=$RUN_ROOT/final_preflight.json" \
                --expected-gpus 0,1,2,3 \
                --max-free-memory-drop-bytes 1073741824 \
                --out "$RUN_ROOT/preflight_comparison.json" \
                > "$RUN_ROOT/preflight_comparison.stdout" \
                2> "$RUN_ROOT/preflight_comparison.stderr"
            comparison_rc=$?
        else
            timeline_mark final_preflight end fail
        fi
    fi
    if [[ $preflight_rc -ne 0 || $comparison_rc -ne 0 ]]; then
        primary_rc=1
    fi

    python3 "$ROOT/scripts/record_experiment_timeline.py" summarize \
        --timeline "$TIMELINE" --run-id "$RUN_ID" \
        --out "$RUN_ROOT/timeline_report.json" >/dev/null || primary_rc=1
    write_status "$primary_rc"
    exit "$primary_rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CURRENT_STAGE=postflight_before
timeline_mark postflight_before start
run_postflight postflight_before
timeline_mark postflight_before end pass

CURRENT_STAGE=preflight_before
timeline_mark preflight_before start
run_preflight preflight_before
timeline_mark preflight_before end pass
BEFORE_PREFLIGHT_PASSED=1

CURRENT_STAGE=parallel_replay
timeline_mark parallel_replay start
for rank in 0 1 2 3; do
    setsid timeout --foreground --signal=TERM --kill-after=90s 7200s \
        env CUDA_VISIBLE_DEVICES="$rank" \
        PYTHONPATH="$ROOT/tests:$SYSTEM_PYTHONPATH" \
        LD_LIBRARY_PATH="$COREX_LD_LIBRARY_PATH" PATH="$COREX_PATH" \
        python3 "$ROOT/tests/replay_fused_prefill_activation.py" \
        --bank-manifest "$BANK_DIR/rank-$rank.manifest.json" \
        --candidate-extension "$EXTENSION" \
        --expected-candidate-sha256 "$EXTENSION_SHA" \
        --capture-source-revision "$CAPTURE_SOURCE_REVISION" \
        --candidate-source-revision "$SOURCE_REVISION" \
        --runtime-identity "$RUNTIME_IDENTITY" \
        --instance "$INSTANCE" --visible-physical-gpu "$rank" \
        --out "$RUN_ROOT/rank-$rank.replay.json" \
        > "$RUN_ROOT/rank-$rank.stdout" \
        2> "$RUN_ROOT/rank-$rank.stderr" &
    PIDS+=("$!")
    PGIDS+=("$!")
done
rc=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || rc=1
done
PIDS=()
PGIDS=()
if [[ $rc -ne 0 ]]; then
    timeline_mark parallel_replay end fail
    exit 1
fi
timeline_mark parallel_replay end pass

CURRENT_STAGE=qualification
timeline_mark qualification start
args=()
for rank in 0 1 2 3; do
    args+=(--report "$RUN_ROOT/rank-$rank.replay.json")
done
python3 "$ROOT/tests/qualify_fused_prefill_activation_replay.py" \
    "${args[@]}" \
    --contract "$ROOT/quality/experiment_funnel.v1.json" \
    --numeric-contract \
    "$ROOT/quality/fused_prefill_numeric_adjudication.v1.json" \
    --profile "$PROFILE" --out "$RUN_ROOT/qualification.json" \
    > "$RUN_ROOT/qualification.stdout" \
    2> "$RUN_ROOT/qualification.stderr"
timeline_mark qualification end pass

CURRENT_STAGE=source_unchanged
timeline_mark source_unchanged start
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$SOURCE_REVISION" ]]
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all -- \
    . ':(exclude)bench_runs/**')" ]]
timeline_mark source_unchanged end pass

CURRENT_STAGE=complete
