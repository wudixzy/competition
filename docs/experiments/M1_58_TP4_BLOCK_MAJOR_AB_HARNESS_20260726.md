# M1-58 TP4 Block-Major A/B Harness

## Purpose

`scripts/run_m1_58_block_major_kv_ab.sh` is the fixed model-level gate for the
M1-57/M1-58 block-major CacheEngine candidate. It requires one immutable
bare-host runtime and four healthy BI100 cards. Control and candidate use the
same model, tokenizer, command, full-attention accounting, CPU content cache,
GDN policy, restore mode, sampling semantics, and request order. The only A/B
selector is:

```text
control:   BI100_BLOCK_MAJOR_CPU_KV=0
candidate: BI100_BLOCK_MAJOR_CPU_KV=1
```

Both arms keep `BI100_CPU_KV_OFFLOAD=1`,
`BI100_HYBRID_KV_ACCOUNTING=full_attention`,
`BI100_GDN_CACHE_POLICY=admission64`,
`BI100_GDN_RESTORE_MODE=direct`, and all trace modes disabled. The formal
29-argument command remains unchanged.

The hardened runner requires the instance identity and a new private output
directory outside the repository:

```bash
BI100_RUNTIME_SITE_PACKAGES=/absolute/immutable/site-packages \
BI100_RUNTIME_INSTALL_REPORT=/absolute/immutable/install.json \
scripts/run_m1_58_block_major_kv_ab.sh \
  ssh-INSTANCE /tmp/m1-58-block-major-YYYYMMDD-HHMMSS
```

It refuses a dirty source tree, an existing API server, an existing output
path, a path outside `/tmp`, or a path inside the source repository. It records
the exact branch, source revision, instance, runtime identity, and gate return
codes. Raw service logs remain outside the repository.

## Fixed Workload

Each arm starts a fresh TP4 service and executes the same greedy sequence:

- one exact 65,536-token target cold request;
- one immediate target repeat;
- nine distinct exact 135,040-token cold pressure requests;
- the target again after pressure;
- one final target repeat;
- `max_tokens=8`, `temperature=0`, seed `20260721`;
- run identity `m158-block-major-fixed-20260726`.

The target plus pressure requests represent about 80,000 logical blocks, above
the observed full-attention GPU capacity even before the candidate's fixed
1,024-block staging reserve. The 4 GiB CPU tier is still large enough to retain
the evicted target prefix.

## Qualification

The harness fails closed unless all of the following pass:

1. hash-bound runtime identity, source revision, generated worker and
   CacheEngine, block-major module, and CoreX extension;
2. four-card deterministic preflight before control, after control, after
   candidate, and in the finalizer, with at most 256 MiB post-cleanup
   free-memory drift;
3. 262144 startup capacity and exactly four rank-local runtime reports;
4. no block-major marker in control, and exact 1,024-block reservation plus
   block-major cache allocation on every candidate rank;
5. both arms restore at least 65,504 target tokens after pressure;
6. every request matches across arms for prompt, cached and completion token
   counts, `finish_reason`, and privacy-safe full-message SHA-256;
7. candidate restored-request elapsed time is at least `1.20x` faster;
8. aggregate cold and pure GPU-warm elapsed time each regress by no more than
   2%;
9. no fatal, OOM, SIGSEGV, traceback, Gloo/NCCL failure, worker loss, timeout,
   process leak, or GPU health drift;
10. each arm and the finalizer pass `service_postflight_gate.py`, including API
    server/worker residue and GPU-process checks.

Every service is launched in its own process group. Cleanup sends `SIGTERM` to
that group and waits 60 seconds before considering `SIGKILL`, then waits/reaps
the leader. A per-arm postflight runs after cleanup, and the `EXIT`/`TERM`/`INT`
finalizer repeats cleanup, postflight, four-card preflight, fatal-log scanning,
and timeout-RC scanning. Any failure leaves `runner_status.json` unqualified,
even if the model-level comparison itself passed.

The fixed comparison is implemented by
`tests/compare_m1_58_block_major_ab.py`. Parameters and thresholds are not
exposed as tuning arguments. Existing output directories are never overwritten.

## Scope

The hardened lifecycle has not yet been run on TP4. Local tests validate its
static contract, but only a real four-card execution can qualify cleanup,
postflight, capacity, correctness, and latency evidence.

Local validation for the lifecycle hardening:

- `bash -n` passed;
- 13 focused M1-58 runner and comparison tests passed;
- complete discovery passed 901 tests with 25 optional-dependency skips;
- the 53-case official metric manifest and all seven quality-data sources
  qualified;
- submission preflight passed 9 of 9 checks;
- `git diff --check` and the scoped credential scan passed.

Even a qualified M1-58 A/B establishes only model-level offload correctness,
capacity, and the candidate's incremental latency effect. It does not replace:

- the complete functional API gate;
- cold/warm long-context output checks at 4K through 262144;
- tool calling, reasoning, multimodal, structured output, and quality suites;
- admissible Output/Input/Cache TPS, TTFT, success-rate, hit-rate, and weighted
  score measurement;
- official 881-request evidence.

No `main`, formal YAML, default selector, model precision, request semantics,
or repository visibility change is authorized by this harness alone.
