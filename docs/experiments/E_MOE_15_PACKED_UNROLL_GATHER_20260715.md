# E-MOE-15: Packed allocation and unrolled weight gather

## Hypothesis

E-MOE-13 allocates selected W13 and W2 separately and copies 16 bytes per loop
iteration. E-MOE-15 tested whether one shared storage allocation plus copy-loop
unrolling could reduce allocator and loop-control overhead without changing
the output layout or arithmetic.

The candidate returned two contiguous views over one packed storage tensor and
scanned compile-time copy unroll factors `{1,2,4,8}`. The E-MOE-13 production
extension remained the baseline.

## Method

`tests/bench_moe_weight_gather_unroll.py` used the checkpoint's TP4 rank-local
shape, 30 warmups, 9 repeats of 300 iterations, and 100 random routed
exactness steps on GPU1.

## Results

| Unroll | Gather ms | Gather vs E-MOE-13 | Fixed full ms | Fixed full speedup |
| ---: | ---: | ---: | ---: | ---: |
| E-MOE-13 | 0.063764 | 1.0000x | 0.265972 | 1.0000x |
| 1 | 0.064096 | 0.9948x | 0.262055 | 1.0149x |
| 2 | 0.065120 | 0.9792x | 0.266979 | 0.9962x |
| 4 | 0.065405 | 0.9749x | 0.262725 | 1.0124x |
| 8 | 0.065693 | 0.9706x | 0.265648 | 1.0012x |

The best fixed-boundary candidate was unroll 1, but the complete routed result
was effectively unchanged and slightly slower:

```text
E-MOE-13 routed: 0.331078 ms
candidate routed: 0.331145 ms
speedup:          0.9998x
```

All fixed outputs and 100/100 random routed outputs were bit-exact with
`max_abs=0`.

Raw artifact:

```text
/root/competition/bench_runs/20260715_E_MOE_15/gpu1-scan.json
```

## Decision

`REJECT FOR PRODUCTION`. Packed allocation did not reduce measured gather
latency, and loop unrolling degraded it. The complete route fails the 5%
performance gate, so no cross-device run is warranted. E-MOE-13 remains the
production candidate.
