# M1-117 Fused Prefill Long-Context A/B

## Purpose

M1-117 supplies the complete long-context quality gate required after the
M1-114 performance screen and M1-116 focused functional/output diagnostic.
It does not relax or replace either earlier gate.

The runner starts two fresh TP4 services from one immutable source/runtime
overlay in fixed order:

1. admission64/hybrid64 with fused prefill disabled;
2. admission64/hybrid64 with fused prefill enabled.

Each arm runs the complete 12-case extended matrix at `max_model_len=262144`.
The existing `compare_long_context_quality_reports.py` strict comparator then
checks runtime identity, tokenizer/template identity, request contracts,
independent capability facts, exact-output cases, next-token cases, cache
correctness, and the capacity boundary. Only the fused-prefill environment
selector may differ.

## Lifecycle

M1-117 inherits the attested service and outer orchestrator lifecycle:

- preflight before each service;
- process-group identity and scoped cleanup;
- 60-second SIGTERM grace before any SIGKILL;
- service recovery qualification;
- fatal, timeout, Gloo/NCCL, worker-loss, and residual-process scans;
- postflight and repeated four-GPU preflight.

The strict comparator return code is part of the runner return code. Failed
exactness remains failed evidence; the runner does not reinterpret a mismatch
as a capability pass.

## Command

After M1-114 and M1-116 release all four GPUs:

```bash
scripts/run_m1_117_fused_prefill_long_context_ab.sh \
  INSTANCE \
  /tmp/m1-117-fused-long-context-SOURCE
```

`BI100_RUNTIME_SITE_PACKAGES` and `BI100_RUNTIME_INSTALL_REPORT` must identify
one exact overlay installed from the M1-117 commit. Output must stay under a
new private `/tmp` path.

## Authorization

Even a passing M1-117 comparison does not independently authorize performance
claims, a default selector, YAML changes, a `main` merge, or production
promotion. Those decisions also require the three-pair TP4 performance result,
M1-116 functional/agent/output evidence, component numerical evidence, and
clean lifecycle results.
