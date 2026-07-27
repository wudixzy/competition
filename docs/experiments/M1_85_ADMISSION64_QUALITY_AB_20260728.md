# M1-85 admission64 full-quality A/B

Date: 2026-07-28

## Objective

Historical TP4 evidence showed that `admission64/direct` can raise effective
cache reuse, and later long-context qualification showed exact outputs at the
131K, 235K, and 262K boundaries. It has not been authorized as the default
because there is no same-source, same-overlay comparison covering the complete
functional and Agent quality suites.

M1-85 adds that missing evidence path. It does not change cache
implementation, model weights, request semantics, sampling, tokenizer, chat
template, `computility-run.yaml`, or any default switch.

## Fixed experiment

`scripts/run_m1_85_admission64_quality_ab.sh` runs two fresh TP4 services in
fixed order:

1. control: `fine32/direct`;
2. candidate: `admission64/direct`.

Both arms use:

- the same source revision, private instance, immutable runtime overlay, model,
  tokenizer, command, and base image;
- the `submission` kernel profile;
- TP4, `max_model_len=262144`, chunk size 8192, prefix caching, and full
  attention KV accounting;
- fused prefill disabled and LRU KV eviction;
- the complete 53-case functional gate;
- the fixed 11-case Agent workload matrix;
- cache trace and privacy-safe 4xx attribution.

The aggregate comparator rejects any runtime environment change other than
`BI100_GDN_CACHE_POLICY=fine32->admission64`. The per-arm labels are also
allowed to differ and are bound to their status files.

## Qualification contract

Each service arm must pass every startup, runtime identity, allocator,
scheduler/worker broadcast, functional, Agent, 4xx attribution, cleanup,
postflight, fatal-scan, timeout-scan, and before/after GPU preflight gate.
Status artifact SHA-256 values must bind the exact runtime contract, functional
report, Agent report, and 4xx report used by the aggregate decision.

The functional comparator requires all 53 cases to preserve HTTP status,
finish reasons, tokenization, deterministic normalized outputs, completion
usage, or independently checked semantic facts as appropriate. The Agent
comparator requires all 11 fixed cases to preserve finish reason, content and
reasoning shape, tool-call count, prompt/completion usage, semantic output, and
validation facts. Cached-token counts are not required to be equal because the
policy is the variable under test.

Expected invalid requests must produce at least one attributed 4xx. Both arms
must have complete, classified, privacy-safe attribution with no malformed
marker or attribution delta, and their status-code, reason, endpoint, and
request-shape summaries must match exactly. This prevents a cache candidate
from hiding a request-compatibility regression behind an otherwise passing
quality report.

## Lifecycle

Every arm starts in its own process group. The child quality runner sends
SIGTERM, waits 60 seconds for the API server, TP4 workers, and collective
runtimes to exit, sends SIGKILL only to verified survivors, waits/reaps, checks
for process and GPU residue, repeats four-GPU preflight, and scans for fatal,
Gloo, NCCL, worker-loss, and timeout evidence.

The outer A/B runner has an independent trap. It allows the child up to 900
seconds to finish its own cleanup before escalating, then performs another
process postflight, four-GPU preflight, and aggregate fatal/timeout scan.

## Interpretation

A qualified M1-85 result authorizes only the functional and Agent
non-regression dimension for `admission64/direct`. It does not authorize a
default-policy change, performance claim, YAML change, `main` merge, or
production promotion. Admission still requires a same-source TP4 performance
A/B, cache correctness evidence, long-context stability, and all final hard
metrics.

The runner intentionally requires four healthy GPUs. Single-GPU component and
HTTP gates can be used while TP4 is unavailable, but they cannot substitute for
this result.

## Invocation after TP4 recovery

```bash
export BI100_RUNTIME_SITE_PACKAGES=/absolute/path/to/immutable/site-packages
scripts/run_m1_85_admission64_quality_ab.sh \
  private-tp4-instance \
  /tmp/m1-85-admission64-quality-ab-YYYYMMDDTHHMMSSZ
```

Raw service logs and model outputs remain under the private `/tmp` run root and
must not be committed. Only privacy-safe structured evidence may be reviewed
for a later repository commit.
