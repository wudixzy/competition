# M1-83 single-GPU component lifecycle

Date: 2026-07-28

## Objective

Make the existing single-BI100 component suite safe to run unattended and
fail closed before using it for optimization decisions. The suite exercises
TP4-rank-local QGKV mapping, MoE, GDN, long paged-KV gather, and CacheEngine
integration on one physical GPU.

## Findings

The previous component runner had three lifecycle gaps:

- a probe received only 15 seconds between timeout SIGTERM and SIGKILL;
- an interrupted run did not own or clean a dedicated probe process group;
- it accepted a structured report without requiring the producing command to
  return zero.

It also ran independent before/after GPU preflights without the shared
topology and persistent-memory comparison used by the HTTP runner. A failed
probe path repeated a preflight, but did not require process-residue,
fatal-log, timeout, and comparison gates to pass.

These gaps do not invalidate the component implementations by themselves,
but they can leave GPU work behind or qualify partial output after a timeout.

## Change

`scripts/run_qwen36_diagnostic_component_gates.sh` now:

- rejects unsafe instance labels before model or GPU work;
- launches each probe in its own `setsid` session and process group;
- binds the probe leader PID to its `/proc` start time before any group
  signal, rejecting a reused or changed identity;
- sends SIGTERM first and allows 60 seconds before timeout SIGKILL;
- uses the shared process-group helper to terminate remaining group members,
  wait/reap the launcher, and reject cleanup failures;
- makes the physical-GPU preflight catch parent SIGTERM/SIGINT, then terminate
  and reap its separately-sessioned GPU child with the same 60-second TERM
  and 20-second KILL fallback;
- gives the outer diagnostic preflight wrappers 90 seconds after TERM, so the
  child cleanup can reach its KILL fallback and reap before the wrapper dies;
- requires both a zero probe return code and a nonempty structured report;
- records probe, cleanup, preflight, runtime-identity, qualification, fatal,
  and timeout return codes;
- scans for residual API, worker, and selected-GPU processes after cleanup;
- repeats the physical-GPU preflight and compares topology, deterministic
  matmul results, and free memory with a fixed 1 GiB maximum drop;
- reruns fatal and timeout scans from the EXIT trap so qualification failures
  and interruptions cannot bypass the final audit;
- emits a fail-closed v2 runner status with lifecycle gates and artifact
  SHA-256 values.

No model implementation, custom kernel, weight, dtype, tokenizer, chat
template, request semantics, cache policy, formal YAML, Dockerfile, or default
runtime switch changed.

## Local validation

- 913 repository `unittest` cases pass; 25 optional-dependency cases skip;
- 55 focused component, HTTP diagnostic, preflight-signal, process-group,
  preflight-comparison, and process-residue tests pass;
- the frozen 53-case quality manifest and all seven quality-data provenance
  sources qualify;
- submission preflight passes 9 of 9 checks;
- shell syntax and `git diff --check` pass;
- an unsafe instance label exits with status 2 before creating a run or
  touching a GPU.

`shellcheck` is not installed in the local environment.

## GPU status

One bounded probe of `ssh-73ca29ba` failed during the Proxy/TLS/SSH handshake
with `Connection closed by UNKNOWN port 65535`; no remote command ran. Local
NVML initialization also fails, so no real single-GPU probe was executed in
this change.

## Next run

Install a new immutable runtime overlay from the exact M1-83 source revision
on one healthy BI100, then run the component suite. A valid result requires
all component numerical and speed gates, zero probe return codes, no residual
GPU process, matching before/after topology, at most 1 GiB persistent free
memory loss, and empty fatal and timeout scans.

The component result remains structural and TP4-rank-local. It cannot qualify
M1-73 HTTP behavior, full-model capability, TP4 correctness, or production
performance, and it does not authorize `main` or formal YAML changes.
