# M1-115 Stage-Profile Closure Fix

## Status

Harness fix implemented; repeat profile pending.

This change affects diagnostic qualification only. It does not alter the
production extension, model runtime, request semantics, YAML, or defaults.

## M1-110 Evidence

All four M1-110 cells completed with finite, numerically qualified outputs and
clean GPU lifecycle gates. The dense case qualified, while 65K, 128K, and 235K
were rejected with:

```text
stage timings do not sum to the event total
```

The raw evidence showed that every trial closed exactly:

| Case | Per-trial stage sum minus total (ms) | Sum of medians minus median total (ms) |
|---|---|---:|
| Dense | `0, 0, 0` | -0.002915 |
| 65K | `0, 0, 0` | +0.040130 |
| 128K | `0, 0, 0` | -0.040727 |
| 235K | `0, 0, 0` | -0.031435 |

The old harness summed independently selected per-stage medians and compared
that value with the independently selected median event total. Median is not a
linear operation, so this can fail even when every CUDA-event row closes
exactly.

## Fix

- Validate stage/event closure for each raw trial.
- Keep a fail-closed absolute residual limit of `1e-6 ms`.
- Report every row residual.
- Report the sum of stage medians and its diagnostic difference from the
  median event total.
- Normalize stage shares by the sum of stage medians so the reported shares
  sum to one.
- Continue using the median event total for instrumentation perturbation.

No performance or numerical threshold is relaxed.

## Repeat Gate

The unchanged M1-110 fixed four-GPU matrix must be rerun with the same
production and profile binaries. All four cells, lifecycle gates, numerical
checks, event perturbation checks, and runtime identity checks must qualify
before the stage ranking is used to authorize another implementation.
