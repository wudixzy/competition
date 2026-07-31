# M1-174 query-tiled reassessment result

## Status

The fixed three-GPU screen closes the historical single-group query-tiled
kernel. Its numerical and lifecycle checks passed, but it was 47x to 75x
slower than the qualified M1-162 FP16-QK baseline at the production query
length. It does not authorize real-activation replay, TP4 service testing,
production overlay changes, YAML changes, or `main` promotion.

## Identity

- source revision:
  `1bdf7b826466ec7f0c98a20e8aa5d8d9391723d1`;
- instance: `ssh-73ca29ba`;
- physical GPUs: 1, 2, and 3; GPU 0 was not used;
- baseline extension SHA-256:
  `36e043f138aa87c635178e4aa6a30af710b87c3f3d7c2a3f1838fc0e365bd368`;
- candidate extension SHA-256:
  `ce75409a30b51e684f5384197b750952fdc63e9d19365f378791cf8ea3d3b67c`;
- candidate source SHA-256:
  `0217061a8803d2a181a01dd7316531d8cfed1fb84619d5f4e204acafe53b89c5`.

## Result

| Cell | M1-162 baseline ms | Query-tiled ms | Speedup |
| --- | ---: | ---: | ---: |
| 16K | 62.901 | 2966.640 | 0.02120x |
| 32K | 117.230 | 7315.856 | 0.01602x |
| 64K | 225.896 | 16852.207 | 0.01340x |

All candidate outputs and LSE values were finite. Candidate repeats were
bit-exact. Relative L2 error against the FP32 reference was between
`7.660e-6` and `7.847e-6`; the calibrated error was within `1.000001x` of
ordinary FP16 rounding, and LSE relative L2 was below `2.90e-8`. The cell
failures are therefore performance failures, not numerical failures.

The median speedup was `0.01602x`, far below the frozen `1.08x` aggregate
gate and the `0.98x` per-cell floor. The result also becomes worse as context
grows. The kernel assigns one 64-thread warp to each 16-query/head tile and
serially traverses 512-token groups, repeatedly synchronizing QK, scalar
online-softmax, and PV work. Eliminating the global logits workspace does not
compensate for that lost matrix-level parallelism on BI100.

## Decision

Do not tune this implementation's query tile, reduction group, or launch
threshold. M1-173 already showed that reducing split-PV launch count alone
does not materially improve M1-162, while M1-174 shows that the opposite
single-warp fusion extreme is structurally noncompetitive. Retain M1-162 as
the mature operator candidate and wait for healthy TP4 capacity to perform
real-activation replay and service-level validation.

All three fixed cells completed with expected fail-closed status. Scoped
cleanup reaped the run-owned children; postflight, repeated preflight, memory
comparison, and fatal scans passed. Runner wall time was 134.36 seconds.
Privacy-safe structured evidence is under
`docs/experiments/evidence/M1_174_QUERY_TILED_REASSESSMENT_20260801`.
