# M1-112 hybrid64 quality gate support

Date: 2026-07-29

Status: harness qualification complete; no model service result yet.

## Purpose

The current M1-107 cache candidate and M1-109 fused-prefill TP4 screen both
use `BI100_GDN_CACHE_POLICY=admission64` with
`BI100_GDN_RESTORE_MODE=hybrid64`. The existing quality-service harness only
accepted `direct` and `aligned`, so it could not validate M1-109 under the
same cache and scheduler configuration used by the performance experiment.

M1-112 adds `hybrid64` to:

- the quality-service argument contract;
- the serialized runtime-contract validator and builder;
- the startup-log contract gate.
- a top-level fused-prefill functional and Agent A/B runner.

The runtime contract additionally rejects `hybrid64` unless the cache policy
is `admission64`. No runtime implementation, request semantics, model code,
Dockerfile, `computility-run.yaml`, or default environment value changes.

`scripts/run_m1_112_fused_prefill_quality_ab.sh` starts two fresh TP4
services in fixed order. Both arms use `admission64/hybrid64`, LRU eviction,
the submission kernel profile, the same immutable runtime overlay, source,
model, command, request order, and four GPUs. The control sets fused prefill
to `0`; the candidate sets it to `1`.

The aggregate comparator fails closed unless:

- all child quality, Agent, runtime, startup, 4xx-attribution, cleanup,
  postflight, and repeated preflight gates pass;
- every functional and Agent case qualifies;
- the two runtime contracts differ only in
  `BI100_ATTN_COREX_FUSED_PREFILL` and the experiment label;
- intentional 4xx attribution and privacy-safe request shapes are identical;
- both isolated process sessions have valid, distinct identities.

## Validation

- focused quality-contract and harness tests: 46 passed;
- complete unit suite: 1,176 passed, 25 skipped;
- submission preflight: all checks passed;
- Python compilation, shell syntax, line-ending, and sensitive-artifact gates
  passed.

## Next gate

If the three-pair M1-109 TP4 performance screen qualifies, run fresh control
and candidate quality services with exactly:

```text
cache policy: admission64
restore mode: hybrid64
KV eviction: lru
fused prefill: 0 versus 1
```

Invocation:

```bash
BI100_RUNTIME_SITE_PACKAGES=/path/to/immutable/site-packages \
scripts/run_m1_112_fused_prefill_quality_ab.sh \
  ssh-73ca29ba /tmp/m1-112-fused-prefill-quality-ab-YYYYMMDDTHHMMSSZ
```

The functional, agent, long-context, cold/warm cache-correctness, lifecycle,
and output-comparison reports remain independent from the performance result.
This harness change does not authorize a default-on selector, official score
claim, YAML change, or merge to `main`.
