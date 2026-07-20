# M1-32 Content-Keyed GDN Admission Qualification

## Objective

M1-32 qualifies the M1-31 content-keyed recurrent-state design on a real TP4
BI100 service. The submission command remains fixed at TP4, 262144 context,
8192-token scheduler chunks, and concurrency one. Runtime policy changes are
made only through `BI100_GDN_CACHE_POLICY` and
`BI100_GDN_RESTORE_MODE`.

The stage gate is deliberately stricter than a cache-hit improvement. A
candidate must preserve deterministic output under state eviction, improve the
effective KV/GDN intersection by at least five percentage points, improve the
weighted proxy by at least five percent, retain Output TPS P10 >= 20 without a
greater than two-percent regression, and preserve 256K capacity.

## Qualification Harness Fixes

The dataset-shaped harness now runs an exact 18-request contract: three prompt
lengths, three pairs per length, and cold/warm phases for every pair. Output
labels are independent from prompt salts, so different policies receive the
same rendered requests. The summary validates client/server token counts,
target error within one 16-token block, cold/warm salt identity, and complete
request coverage. Policy comparison refuses to qualify non-identical request
contracts.

Additional runtime fixes made before qualification:

- CoreX Python and shared-library paths are propagated to clients and NCCL
  spawn workers.
- The cache trace patch targets the authoritative block manager used by this
  vendor vLLM rather than an inactive compatibility copy.
- Prefix-pressure evidence is atomically persisted before an assertion, so a
  correctness failure retains the before/after token counts and hashes.
- Bare-host `launch_service` now mirrors the evaluator's fixed exact kernels:
  `BI100_MOE_COREX_DIRECT_ROUTED=1` and
  `BI100_GDN_COREX_PACKED_DECODE=1`. Earlier absolute TPS runs without these
  variables are not submission-equivalent.

## Strict Same-Request A/B

The following measurements used the same prompt salt namespace and passed the
18-request contract validation. They predate the `launch_service` fixed-kernel
environment correction, so only the relative cache-policy comparison is
usable; their absolute Output TPS and score are not production baselines.

| Metric | `fine32/direct` | `admission64/direct` | Delta |
| --- | ---: | ---: | ---: |
| Success | 100% | 100% | 0 pp |
| Effective cache hit | 49.9301% | 61.0671% | +11.1370 pp |
| Cache TPS | 7109.645 | 7630.875 | +7.33% |
| Input TPS | 732.366 | 824.430 | +12.57% |
| Output TPS P10 | 14.8843 | 15.7869 | +6.06% |
| TTFT P90 | 20.8028 s | 18.4650 s | -11.24% |
| Weighted proxy | 6281.291 | 6846.025 | +8.99% |

The candidate clears the relative hit, score, output-regression, success, and
request-identity gates. It does not clear the absolute Output TPS, TTFT, or
8000-score gates in this non-production-equivalent run.

Cold requests under `admission64` can legitimately report a small or shared
prefix hit. Tool schemas occur before the per-request system run id, producing
a repeated 3088/3104-token branch. This is workload structure, not cross-run
contamination.

## Correctness Result

`fine32/direct` passed two persisted 17-session pressure reruns: a 10593-token
test prompt reported zero cached tokens after pressure, then a refreshed warm
request restored 10592 tokens, with the same deterministic output hash across
all observations. One earlier non-persisted run failed output equivalence, so
the mode still has residual risk and requires the fixed long-context matrix.

The zero hit after pressure is important: 17 competing sessions create enough
fine-grained checkpoints to evict the original state from the 32-slot policy.
That request passed by recomputing its entire prompt, not by proving that the
10592-token direct state was replay-safe. `admission64` retains that state and
therefore exposes the arbitrary-boundary numerical path difference.

`admission64/direct` failed deterministically useful evidence. The eviction
prompt first ran cold with hash prefix `bc4f55`, then after 17 competing
sessions reported a 10592-token effective hit but produced hash prefix
`eba366`. The service remained healthy and no OOM or worker loss occurred.
Therefore the scheduler and workers agreed that a reusable state existed, but
restoring it did not preserve the generated result.

This is a hard rejection of `admission64/direct`, regardless of its measured
cache and proxy-score improvement. It must not become the submission default
or be merged to `main`.

The aligned fallback is correctness-oriented rather than score-capable for the
dataset-shaped matrix. With 8192-token alignment, the 4096- and 7800-token
requests have no restorable boundary and each 16000-token warm request can
restore at most 8192 tokens. Its resulting matrix-wide theoretical hit ceiling
is `3 * 8192 / (6 * (4096 + 7800 + 16000)) = 14.68%`, before any eviction.
Consequently a full aligned performance matrix cannot reach the 50% cache
gate; only its pressure and 235K exact-replay correctness tests remain useful.

The aligned correctness fallback passed both remaining TP4 gates. Under
17-session pressure, the aligned and unaligned test prompts restored 8,192
tokens and retained identical outputs; the eviction target also stayed
identical before pressure, after pressure, and after refresh. The corrected
235K gate then forced both responses to generate the full 1,000 tokens:

| Observation | Cold | Warm |
| --- | ---: | ---: |
| Cached tokens | 0 | 229,376 |
| Completion tokens | 1,000 | 1,000 |
| Elapsed | 762.489 s | 241.484 s |
| Finish reason | `length` | `length` |
| Message SHA-256 | `a7d5a63c...81799` | `a7d5a63c...81799` |

`long_235k_exact.rc` and `remaining_gates.rc` are both zero. An earlier
8-token run was explicitly archived as invalid and is not part of this
qualification.

## Decision And Remaining Work

- Keep `fine32/direct` as the submission default.
- Treat `admission64/aligned` as correctness-qualified but performance-
  disqualified: its fixed matrix hit ceiling is only 14.68%.
- Test M1-33 `admission64/chunk64`, which aligns to the native DeltaNet
  recurrence chunk while retaining much finer reuse. It must pass operator,
  pressure, matrix, and long-context gates before any default change.
- Do not start fused paged-attention implementation. Stage three is gated on a
  correct cache candidate meeting the stage-two thresholds; aligned does not
  meet that gate and M1-33 is still unqualified.
- Do not scan YAML thresholds, state capacities, or kernel tiles to work around
  the failure.

## Production-Equivalent Fixed Baseline

The interrupted `fine32/direct` matrix completed all 18 requests and was read
back on 2026-07-21. Its server log confirms both evaluator-fixed kernels were
enabled (`moe_direct=1`, `gdn_packed=1`). All request-contract checks passed,
including exact client/server token counts and cold/warm salt identity.

| Metric | Fixed baseline | Required |
| --- | ---: | ---: |
| Success rate | 100% | >= 99% |
| Output TPS P10 | 21.6563 | >= 20 |
| Effective cache hit | 49.9301% | >= 50% |
| Input TPS | 741.4479 | - |
| Cache TPS | 7607.9233 | - |
| TTFT P90, all | 20.8748 s | <= 5 s |
| TTFT P90, warm | 1.4438 s | - |
| Weighted proxy | 6699.4888 | >= 8000 |

The baseline misses cache hit by only 0.0699 percentage points, but misses the
weighted target by 1300.5112 and the all-request TTFT gate by 15.8748 seconds.
Its cold TTFT P90 is 20.9465 seconds while warm TTFT P90 is 1.4438 seconds, so
the remaining latency failure is the uncached prefill path rather than decode
or warm restore.

The production-equivalent result is stored under
`bench_runs/m1_32/fine32_direct_fixed`. The instance gateway outage ended on
2026-07-21. The direct 131K/256 exact and 235K/256 warm-repeat gates, aligned
pressure gate, and corrected aligned 235K/1,000 exact gate all have successful
persisted evidence. M1-32 is closed without a promotable cache policy:
`admission64/direct` is incorrect and `admission64/aligned` cannot meet the
cache-rate threshold.
