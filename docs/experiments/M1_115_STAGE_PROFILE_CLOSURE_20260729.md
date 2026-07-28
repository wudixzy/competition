# M1-115 Stage-Profile Closure Fix

## Status

Qualified on the four-GPU BI100 instance. The repeat profile completed with
runner return code zero and all four cells qualified.

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

## Repeat Result

The fixed matrix ran on `ssh-73ca29ba` from source
`0d0a55918e2c39fc4de0cb7c7e609823d54679e1`. It used four concurrent
single-GPU component cells. This is not a full-model TP4 service result.

The production and instrumented extensions were byte-bound to:

```text
production ad4ea7707bb2f2bfe04e07a7ad5fd58a647232be70a3056937a0d738c8254bff
profile    0f22a98cde885829416a26ae58067a5684cb9d90067dd9e46a4bc1abbf4d9b48
```

| Case | Total (ms) | Gather | QK | Softmax | PV | Merge | Dominant |
|---|---:|---:|---:|---:|---:|---:|---|
| Dense q8176 | 40.174 | 0.23% | 28.99% | 16.97% | 28.99% | 4.30% | PV |
| 65K q8176 | 295.907 | 0.56% | 35.55% | 20.58% | 35.27% | 5.21% | QK |
| 128K q8176 | 519.789 | 0.58% | 36.01% | 20.79% | 35.69% | 5.30% | QK |
| 235K q5616 | 664.616 | 0.83% | 35.32% | 20.14% | 37.82% | 5.18% | PV |

Instrumentation event perturbation stayed between 0.23% and 0.43%. The
before/after four-GPU preflight, final postflight, fatal scan, and profile
comparison all passed.

For 235K, QK and PV account for 73.14% of measured time and gather accounts
for only 0.83%. The next implementation may therefore target QK/PV call
count, intermediate materialization, or GQA reuse. Further gather-only tuning
is not justified by this profile.

This result authorizes deeper-fusion design selection only. It does not
authorize a full-model TP4 experiment, production default, `main` merge,
`computility-run.yaml` change, or official score claim.

Structured evidence:
`docs/experiments/evidence/M1_115_STAGE_PROFILE_CLOSURE_20260729/qualification.json`.
