# M1-87 single-GPU diagnostic queue

Date: 2026-07-28

## Objective

M1-89, M1-84, and M1-86 cover different risks. The installed-runtime gate
checks the real block manager, `Sequence` empty-multimodal behavior, Pillow
palette/transparency hashing, request-local fallback, and namespace release.
The current diagnostic service gate checks API, quality-contract,
compatibility, streaming tool history, prefix reuse, capacity, and lifecycle
behavior. M1-86 isolates the sole
`--limit-mm-per-prompt image=2` command delta and checks deterministic
multi-image output and cache isolation. Running any result alone does not prove
that all three used the same source and runtime overlay, or that the service
stages used the same diagnostic checkpoint and physical GPU.

M1-87 v2 runs those three gates sequentially and produces one fail-closed
identity and lifecycle decision. The v2 integration is commit `9ecbf30` on
the private `fix/M1-89-multimodal-cache-namespace-20260728` branch. It changes
test infrastructure only. It does not
change model code, weights, dtype, tokenizer, chat template, request semantics,
cache policy, `computility-run.yaml`, Dockerfile, or a production default.

## Fixed queue

`scripts/run_m1_87_single_gpu_queue.sh` uses one declared physical BI100 and one
immutable overlay installed from the exact current HEAD:

1. verify that the immutable overlay matches the exact current source;
2. from `/tmp`, run the M1-89 installed-runtime v2 gate without model or GPU
   execution;
3. run the current M1-84 diagnostic service gate at TP1;
4. require an independent service postflight and GPU preflight;
5. run the fixed M1-86 control/candidate multi-image A/B at TP1;
6. recover only process groups recorded by this run, then require final service
   postflight, GPU preflight, recursive fatal scan, and timeout scan;
7. bind all three stages into the v2 `queue_status.json`.

The diagnostic and multi-image services use different fixed loopback ports.
Those two service stages use the same four-layer structural real-weight
checkpoint, source model, source revision, overlay tree, physical GPU,
262,144-token capacity, reference compute switches, and privacy-safe output
summaries. M1-89 performs no model or GPU execution and is bound to that same
source revision, runtime path, and overlay tree.

## Lifecycle contract

Every service and queue child is launched through
`scripts/exec_bi100_session.py`. Before `exec`, it creates a new session and
atomically records PID, PGID, SID, `/proc` starttime, and a random private
session token. The token is inherited by that process tree and is not written
to service contracts, logs, aggregate status, or repository artifacts.

Normal cleanup sends SIGTERM only to the recorded process group and waits at
least 60 seconds. SIGKILL is permitted only for verified survivors, followed
by wait/reap. Cleanup ignores repeated TERM/INT so a second signal cannot
interrupt the cleanup sequence.

The outer queue previously allowed 900 seconds before killing an interrupted
child group. Commit `9ecbf30` reduces that bound to 60 seconds while preserving
the exact PID/PGID/starttime/session-token check and mandatory wait/reap.

If a child stage exits abnormally, the outer queue examines only the two queue
child identities and three service identities created under its private run
root. Recovery requires exact PID/PGID/SID/starttime structure and the inherited
session token on every live member before it can signal a group. It also scans
for descendants that retained the private token after escaping the original
process group and signals only those exact token-bearing PIDs. A token or
identity mismatch is never signalled. Emergency recovery can make the machine
clean, but it cannot qualify an experiment: a valid M1-87 result requires all
five sessions to have already been quiescent, with no recovery TERM or KILL.
The root-run recovery scan is complete-or-fail: an unreadable `/proc/*/environ`
entry invalidates recovery instead of being silently skipped.

Startup uses one monotonic deadline. Each HTTP health attempt is bounded by the
remaining time, and the service starttime must remain unchanged. Recursive
scans cover all `*.log`, `*.stdout`, `*.stderr`, and `*.rc` artifacts. Timeout,
forced-kill, termination, malformed rc, fatal CoreX/CUDA, Gloo/NCCL reset,
worker loss, missing GDN state, and non-finite GDN evidence invalidate the run.

## Evidence contract

`tests/qualify_m1_87_single_gpu_queue.py` requires exact gate and artifact
sets. It rejects missing or extra gates, missing or extra artifact entries,
path traversal, symlinked evidence, digest mismatch, source or overlay drift,
checkpoint drift, GPU drift, nonzero lifecycle rc, and incomplete cleanup. The
M1-86 aggregate additionally binds both arm-level service postflights and GPU
preflight comparisons to the declared `CUDA_VISIBLE_DEVICES`.

The aggregate binds:

- the M1-89 overlay identity and nine-check installed-runtime report;
- the full M1-84 status artifact manifest;
- the M1-86 runner manifest and every input consumed by its comparison;
- both queue-child session identities;
- interstage and final process/GPU postflights;
- the recorded-service recovery report.

Only digests, model paths, non-sensitive process identity, lifecycle summaries,
and qualification decisions enter aggregate evidence. Raw prompts, images,
tokens, generated output, credentials, and session tokens are not copied into
the aggregate.

## Current status

Implementation and CPU-only validation are complete on the current private
M1-89 branch. Focused queue/runtime tests passed 25 of 25; complete tests-root
discovery passed 1030 tests with 25 dependency skips. Submission preflight
passed 9 of 9, and the fixed quality-data and 53-case metric manifests passed.
No BI100 result has been claimed. The latest bounded SSH probe still failed in
the TLS ProxyCommand layer before authentication, and the local host has no
usable CoreX GPU.

The four-layer checkpoint is suitable for parser, compatibility, cache
isolation, capacity, and lifecycle diagnostics. It does not establish
full-model semantic quality, TP4 correctness, the complete official functional
matrix, the 881-request performance result, or any competition threshold.
The M1-89 installed-runtime gate proves real Pillow namespace behavior but does
not replace a palette/transparency service-level cold/warm test.

## Invocation after GPU recovery

Install an immutable runtime overlay from the exact current committed
revision, then run:

```bash
export BI100_RUNTIME_SITE_PACKAGES=/absolute/path/to/immutable/site-packages
export MODEL_PATH=/root/shared-storage/models/Qwen/Qwen3.6-35B-A3B-diagnostic-4L-real
export SOURCE_MODEL_PATH=/root/public-storage/models/Qwen/Qwen3.6-35B-A3B

GPU_INDEX=<healthy-index> \
DIAGNOSTIC_PORT=8040 \
MULTI_IMAGE_PORT=8050 \
scripts/run_m1_87_single_gpu_queue.sh \
  private-bi100-instance \
  /tmp/m1-87-single-gpu-YYYYMMDDTHHMMSSZ
```

The run root must be a new private path under `/tmp`. Long execution must be
monitored by a lightweight subagent. Do not commit raw run output. Retain only
privacy-safe structured evidence after manual review.

## Interpretation

A qualified M1-87 result authorizes only the single-GPU structural diagnostic
phase. Full-model TP4 functional, cold/warm correctness, long-context,
multimodal, tool/reasoning, semantic-quality, and performance gates remain
mandatory. M1-87 never authorizes changing `main`, formal YAML, repository
visibility, or a production default.
