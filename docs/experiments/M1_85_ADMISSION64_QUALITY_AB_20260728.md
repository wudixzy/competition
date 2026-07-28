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
scheduler/worker broadcast, process-identity, functional, Agent, 4xx
attribution, cleanup, recorded-session recovery, recovery qualification,
postflight, fatal-scan, timeout-scan, and before/after GPU preflight gate.
Status artifact SHA-256 values must bind the exact runtime contract, process
identity, recovery reports, functional report, Agent report, and 4xx report
used by the aggregate decision. New runs use quality-service status v2 and
aggregate v2; v1 evidence cannot be mixed into this decision.

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

Every A/B arm and every API service starts through
`exec_bi100_session.py`. The atomic identity binds PID, PGID, SID, Linux
starttime, and a random per-session token before the target is executed. The
child quality runner sends SIGTERM only to the verified API-service group,
waits 60 seconds for the server, TP4 workers, and collective runtimes to exit,
sends SIGKILL only to verified survivors, and waits/reaps the leader.

The child then performs a recorded-session recovery scan. A recovery signal
may clean an abnormal run, but it invalidates that run: qualification requires
the identity to be already quiescent, a complete token scan, no TERM or KILL,
and zero live or escaped processes. Postflight, four-GPU preflight, fatal
scanning, and timeout scanning run after cleanup.

The outer A/B runner has an independent 60-second TERM and 20-second KILL
window. Its finalizer validates both A/B runner identities and both nested
service identities, so a service that escaped because its child runner was
forcibly terminated can still be cleaned without a broad process search.
Normal qualification requires all four identities to be already quiescent.
Repeated TERM or INT is ignored while either finalizer is completing its
bounded cleanup and evidence writes.

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
