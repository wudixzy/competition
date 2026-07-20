# M1-38 Small-Batch Direct Routed MoE - 2026-07-21

## Context

The fixed cache matrix leaves only 8 or 16 physical prompt tokens after a
warm restore. The production routed-MoE direct kernel is used only when
`T == 1`; every `T > 1` call sorts all assignments, copies 256 expert counts
to the CPU, and executes a Python loop over active experts. This is an
uncovered warm-cache path and does not require changing the qualified decode
kernel.

M1-38 tests whether the existing direct W13 and W2 kernels can be reused one
token at a time for `T in {2, 8, 16}`. It is an isolated capability gate. It
does not change the production model, prebuilt bundle, Dockerfile, YAML, or a
running service.

## Fixed Gates

The operator gate was fixed before production integration:

- all 40 random-sequence steps must be finite;
- maximum absolute error must not exceed `0.0001220703125`;
- mean absolute error must not exceed `6.8e-6`, matching the accepted
  E-MOE-20 staged endpoint bound;
- relative L2 is recorded as a diagnostic, not reused from the unrelated
  attention-fusion gate;
- speedup must be at least `1.25x` for T=2 and `1.5x` for T=8/16;
- the static 40-layer warm-TTFT projection must improve by at least 5% for
  the scored T=8/16 suffixes.

The first probe revision accidentally applied the Phase-3 attention
`relative_l2 <= 1e-5` gate and used a looser `mean_abs <= 1e-5` limit. This was
corrected before the backup result was observed. It does not alter the first
decision: that run exceeded both the final maximum and mean error limits.

## Direct-Loop Result

Evidence:

```text
/root/competition-m1-32-latest/bench_runs/m1_38/small_batch_loop
```

| T | Baseline | Direct loop | Speedup | Max abs | Mean abs | Projected warm gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | `4.6857 ms` | `0.1794 ms` | `26.12x` | `2.4414e-4` | `1.3637e-5` | `12.48%` |
| 8 | `18.2700 ms` | `0.6017 ms` | `30.36x` | `3.6621e-4` | `1.3567e-5` | `48.95%` |
| 16 | `30.8677 ms` | `1.1192 ms` | `27.58x` | `3.6621e-4` | `1.3593e-5` | `82.41%` |

All outputs were finite and every speed/projection gate passed. Numerical
qualification failed for every token count.

## Reference-Order Backup

One bounded backup sorted each token's eight expert IDs into the same
ascending expert order used by the general path and rounded the accumulated
routed output to FP16 after every expert contribution. This tests whether the
error came only from top-k accumulation order. It does not change launch
geometry or scan parameters.

Evidence:

```text
/root/competition-m1-32-latest/bench_runs/m1_38/sorted_half
```

The CoreX extension built successfully (`build.rc=0`). The probe exited
fail-closed (`run.rc=1`):

| T | Baseline | Sorted-half | Speedup | Max abs | Mean abs | Projected warm gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | `4.6584 ms` | `0.4320 ms` | `10.78x` | `1.8311e-4` | `1.2249e-5` | `11.71%` |
| 8 | `18.4435 ms` | `0.9862 ms` | `18.70x` | `2.4414e-4` | `1.2264e-5` | `48.36%` |
| 16 | `31.4242 ms` | `1.6292 ms` | `19.29x` | `2.4414e-4` | `1.2293e-5` | `82.54%` |

Changing the routed reduction order reduced the maximum error but left the
mean error almost unchanged. The remaining difference is therefore produced
by the direct GEMV arithmetic versus the vendor GEMM boundary, not just by
the final expert accumulation.

## Decision

Status: `NUMERICAL_REJECTED`.

Do not integrate either small-batch direct loop, do not add a runtime knob,
and do not replace the production prebuilt extension. Do not scan token
thresholds, launch geometry, compiler flags, seeds, or tolerance. A future
small-batch MoE design would need a reference-compatible vendor/grouped GEMM
architecture and a separate memory-capacity proof; it is not an authorized
continuation of M1-38.
