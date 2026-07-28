# M1-87 single-GPU diagnostic queue

Date: 2026-07-28

## Objective

M1-84 and M1-86 cover different risks. The current diagnostic service gate
checks API, quality-contract, compatibility, streaming tool history, prefix
reuse, capacity, and lifecycle behavior. M1-86 isolates the sole
`--limit-mm-per-prompt image=2` command delta and checks deterministic
multi-image output and cache isolation. Running either result alone does not
prove that both used the same source, runtime overlay, diagnostic checkpoint,
or physical GPU.

M1-87 runs those two gates sequentially and produces one fail-closed identity
and lifecycle decision. It changes test infrastructure only. It does not
change model code, weights, dtype, tokenizer, chat template, request semantics,
cache policy, `computility-run.yaml`, Dockerfile, or a production default.

## Fixed queue

`scripts/run_m1_87_single_gpu_queue.sh` uses one declared physical BI100 and one
immutable overlay installed from the exact current HEAD:

1. run the current M1-84 diagnostic service gate at TP1;
2. require an independent service postflight and GPU preflight;
3. run the fixed M1-86 control/candidate multi-image A/B at TP1;
4. recover only process groups recorded by this run, then require final service
   postflight, GPU preflight, recursive fatal scan, and timeout scan;
5. bind both stages into `queue_status.json`.

The diagnostic and multi-image services use different fixed loopback ports.
Each stage uses the same four-layer structural real-weight checkpoint, source
model, source revision, overlay tree, physical GPU, 262,144-token capacity,
reference compute switches, and privacy-safe output summaries.

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

Implementation and CPU-only validation are complete on the private M1-87
experiment branch. No BI100 result has been claimed. The latest bounded SSH
probe failed before authentication and the local host has no usable CoreX GPU.

The four-layer checkpoint is suitable for parser, compatibility, cache
isolation, capacity, and lifecycle diagnostics. It does not establish
full-model semantic quality, TP4 correctness, the complete official functional
matrix, the 881-request performance result, or any competition threshold.

## Invocation after GPU recovery

Install an immutable runtime overlay from the exact committed M1-87 revision,
then run:

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
