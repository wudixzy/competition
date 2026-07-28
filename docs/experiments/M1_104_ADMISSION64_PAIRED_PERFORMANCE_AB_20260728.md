# M1-104 admission64 paired performance A/B

Date: 2026-07-28

Status: runner and CPU contract tests prepared. No GPU run is performed by this
change. This experiment reopens the historical admission64 candidate under the
revised cache-benefit policy; it does not change runtime code, defaults, YAML,
Dockerfile, model, tokenizer, or repository visibility.

## Purpose

M1-35 had a large effective-cache improvement and lower TTFT, but was rejected
because the former screen required both a five-point hit-rate gain and a five
percent weighted-proxy gain. The revised policy permits either a repeated
effective-hit gain of at least two percentage points or a repeated weighted
gain of at least three percent without reducing hit rate, subject to the
unchanged correctness and performance floors. M1-104 measures that candidate
again on the current source and overlay.

## Fixed experiment

The runner is `scripts/run_m1_104_admission64_performance_ab.sh INSTANCE
RUN_ROOT`. It requires a new private `/tmp` output directory, the full Qwen3.6
model, and an immutable `BI100_RUNTIME_SITE_PACKAGES` overlay containing vLLM
and transformers. Every arm runs a fresh TP4 service and a four-card preflight.

The only arm variable is `BI100_GDN_CACHE_POLICY`:

| Arm | Policy |
|---|---|
| control | `fine32` |
| candidate | `admission64` |

The order is fixed and alternating:

1. control, candidate
2. candidate, control
3. control, candidate

All arms use direct GDN restore, `full_attention` accounting, fused prefill
disabled, CPU KV offload disabled, submission kernels, TP4, max model length
262144, and the fixed service launch contract. The benchmark is delegated to
`tests/bench_m1_104_admission64_policy_matrix.py` with a unique salt namespace
per arm. After all six arms, the runner calls
`tests/compare_m1_104_admission64_paired_ab.py --root RUN_ROOT --out
RUN_ROOT/comparison.json`.

No parameter scan, YAML tuning, request-semantic change, quantization, model
change, or context truncation is allowed.

## Lifecycle and failure semantics

Each arm records preflight, startup contract, benchmark, health, cleanup,
fatal/timeout scans, service postflight, after-preflight, and GPU preflight
comparison. Services are started through `exec_bi100_session.py`; cleanup is
restricted to the attested PID/PGID/SID/starttime/token session. It sends
SIGTERM and waits 60 seconds, sends SIGKILL only to verified survivors, then
waits/reaps. The finalizer also performs recorded-session recovery and a
machine-wide postflight/preflight check. No broad `pkill` or `killall` is used.

A candidate benchmark return code of 1 is retained as a valid negative
measurement when its report is present. The runner still emits arm status,
cleanup evidence, and final runner status. The final shell result follows the
comparison qualification and infrastructure gates; a control infrastructure
failure or incomplete evidence fails the run.

## Promotion boundary

Even a qualified M1-104 result authorizes only the next M1-85 full functional
and Agent quality A/B gate. It does not authorize an official-style replay,
default switch, `computility-run.yaml` modification, `main` merge, Docker or
ModelHub submission, or any visibility change. Performance and model quality
remain separate gates. The eventual candidate must still pass the complete
quality suite, cache cold/warm correctness, 256K capacity, long-context,
multimodal, tool-calling, reasoning, fatal/worker-loss, and official metric
requirements before any promotion proposal.

## Local validation

The companion `tests/test_m1_104_admission64_runner_unit.py` checks shell syntax,
fixed arm order, immutable launch settings, attested scoped cleanup, negative
candidate-result handling, and the promotion boundary. These are static and
pure-CPU checks. They do not establish GPU performance, cache benefit, TTFT,
throughput, output quality, or an official score.
