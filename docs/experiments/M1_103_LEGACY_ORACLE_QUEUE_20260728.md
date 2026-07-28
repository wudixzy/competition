# M1-103 legacy-candidate oracle queue

Date: 2026-07-28

Status: private two-GPU execution queue prepared but not yet run. It changes no
production runtime, default, YAML, model, tokenizer, request semantics, or
repository visibility.

## Purpose

M1-100 and M1-101 are bounded policy-v2 retests of historical candidates that
showed useful performance but were rejected against a vendor reduction order:

- M1-100 rechecks E-PREFIX-08 at the actual `Q=8176` production segment against
  sampled CPU FP64 full-sequence attention;
- M1-101 rechecks the frozen M1-28 WMMA QK tile against CPU FP64 QK, softmax,
  and PV across all fixed output rows.

The numerical questions are independent and each uses one GPU. M1-103 runs
them concurrently on distinct declared physical GPUs after the active TP4
experiment has fully released the machine.

## Fixed queue

`scripts/run_m1_103_legacy_oracle_queue.sh` accepts only a non-sensitive
instance label and a new private `/tmp` run root. GPU placement defaults to:

```text
M1-100 prefix oracle: physical GPU 0
M1-101 WMMA oracle:   physical GPU 1
```

`PREFIX_GPU` and `WMMA_GPU` may select another pair of distinct healthy
physical devices. They are placement controls, not algorithm parameters.
Shapes, seeds, magnitudes, samples, timing trials, thresholds, and CPU thread
counts remain frozen inside the two oracle programs.

Before launch, the queue requires:

- a clean committed source tree;
- no residual API server, worker, or process holding either selected GPU;
- independent CoreX allocation and deterministic matmul preflight on both
  selected GPUs;
- successful compilation of the hash-pinned M1-28 WMMA extension.

Each oracle runs through `exec_bi100_session.py`, which records its PID, PGID,
SID, Linux starttime, and private session token before `exec`. The queue
attests that identity and then runs both children concurrently with
`CUDA_VISIBLE_DEVICES` mapping each physical GPU to logical `cuda:0`.

## Lifecycle

The complete parallel stage has a fixed two-hour timeout. Normal exits are
waited and reaped. On interruption or timeout, cleanup targets only an
identity-verified process group created by this queue:

1. send SIGTERM;
2. wait 60 seconds;
3. send SIGKILL only to verified survivors;
4. wait and reap the child.

If an identity cannot be attested, fallback cleanup may signal only the exact
child PID whose Linux starttime still matches; it never performs a broad
process search or kill.

Finalization always performs:

- recorded-session recovery and a requirement that no emergency TERM/KILL was
  needed after normal child completion;
- stable service/GPU postflight;
- repeated CoreX preflight and before/after topology, checksum, and memory
  comparison;
- recursive fatal, OOM, collective, worker-loss, and timeout scans.

Any lifecycle or postflight failure invalidates the queue result.

## Negative evidence

The oracle programs return zero when their candidate passes and one when it is
a valid numerical/performance rejection. M1-103 deliberately accepts either
return code only when it matches a complete schema-qualified report. This
allows one candidate to be retained as valid negative evidence while the other
passes.

Timeout, forced termination, malformed or missing reports, artifact drift,
candidate/report return-code mismatch, or any production authorization field
invalidates the queue.

`tests/qualify_m1_103_legacy_oracle_queue.py` writes one privacy-safe status
containing scalar candidate decisions and SHA-256 bindings only. It does not
copy raw tensors, model output, prompts, session tokens, credentials, or
environment data.

## Decision boundary

A passing queue means only that both experiments executed with valid identity
and lifecycle evidence. Each candidate retains its own result:

- a passing M1-100 may proceed only to a fixed greedy next-token gate;
- a passing M1-101 may proceed only to a separately designed integration
  benefit gate;
- a rejected candidate closes without parameter or tolerance scans.

No outcome directly authorizes service integration, production promotion,
YAML/default changes, `main` merge, or repository visibility changes.
