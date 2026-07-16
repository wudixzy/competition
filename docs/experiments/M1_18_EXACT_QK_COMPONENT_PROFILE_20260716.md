# M1-18 Exact-QK Component Profile

## Objective

Determine whether an exact-QK fused score-to-paged-V implementation has enough
theoretical headroom to beat the current E-ATTN-05 path by at least 5%. This is
a profiling gate, not an implementation experiment.

## Environment

- Device: Iluvatar BI-V100, healthy physical GPU1 only
- Torch: 2.1.0 CoreX
- Context lengths: 65,536 and 100,000
- Warmup: 5; timed iterations per trial: 5; trials: 7
- Fixed seed: 20260716
- M1-16 hybrid extension for K-only gather and direct paged PV
- E-ATTN-05 scalar extension for the production K+V gather reference

Each component was synchronized independently. The table reports P10, median,
and P90 trial latency in milliseconds.

## Results

| Component | 65,536 P10/P50/P90 | 100,000 P10/P50/P90 |
|---|---:|---:|
| E-ATTN-05 scalar K+V gather | 2.534/2.551/2.558 | 2.918/2.923/3.120 |
| M1-16 K-only gather | 2.967/2.975/2.987 | 2.694/2.698/2.702 |
| Authoritative PyTorch FP32 QK | 2.138/2.141/2.149 | 2.631/2.633/2.636 |
| Softmax on fixed logits | 0.072/0.073/0.075 | 0.102/0.103/0.103 |
| Reference contiguous PV | 2.303/2.305/2.309 | 3.517/3.519/3.526 |
| M1-16 direct paged PV | 3.710/3.719/3.733 | 5.587/5.647/5.679 |

Correctness checks passed for exact K and the fixed contiguous PV reference.
Direct paged PV had maximum absolute error `7.629e-06`.

The measured exact-QK/direct-PV component sums were `8.9086 ms` at 65K and
`11.0811 ms` at 100K. The known E-ATTN-05 end-to-end times are `7.0830 ms` and
`9.2203 ms`, so the component sums are slower by `25.77%` and `20.18%`.
They agree with the known M1-16 end-to-end measurements within `0.59%/0.28%`.

## Decision

`REJECT`. Removing only the materialized global softmax weights cannot recover
the required 5% while retaining authoritative FP32 QK. The estimate is already
optimistic because it adds isolated component medians and excludes integration
overhead. Do not implement M1-18, tune its launch geometry, or revive alternate
QK reduction orders. Close the current long-context direct-decode line and
return to the next full-model hotspot.

Raw evidence is preserved outside Git at
`result/20260716/M1-18-component-profile/` and on the profiling host at
`/root/m1_18_component_profile/`.
