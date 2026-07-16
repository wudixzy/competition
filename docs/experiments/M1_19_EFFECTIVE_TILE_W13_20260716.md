# M1-19 Effective-Tile W13 Gate

## Scope

M1-19 tests the first native grouped-MoE primitive after the padded cuBLAS
and heavy-tail hybrid paths were rejected. It covers only expert-major W13 and
SiluAndMul for the real 7,800-token route-count trace. It does not implement
W2, modify the model, or change the evaluator configuration.

The fixed kernel uses the vendor BI100 WMMA shape `M=16, N=32, K=32`, a
64-lane warp, FP16 matrix inputs, and FP32 accumulators. Each expert count is
rounded independently to 16 rows; the actual 40-layer padding ratio is only
`1.02897x-1.03282x`. Four output-column tiles produce the 128-wide activated
intermediate without copying a `[padded_assignments, 2048]` input.

## Correctness repair

The first attempt incorrectly treated BI100 WMMA shared memory as ordinary
row-major storage. The vendor example requires `wmma::CoordToOffset`, packed
half2 loads, a 64-lane warp, and zero stride in fragment loads/stores. That
attempt produced `11.48-13.56` maximum error and is invalid evidence.

Attempt 2 used the vendor address mapping. A deterministic tiny smoke reached
`1.53e-5` maximum error and padded rows became exactly zero. Review then found
that the fused activation consumed FP32 accumulators directly instead of first
rounding the gate/up GEMM outputs to FP16 as the reference does. Attempt 3
added only that required round trip; tile and launch geometry were unchanged.

## Final result

Physical GPU1, seed 20260716, five warmups and seven synchronized trials:

| Real-route layer | Padding | Reference P50 | Candidate P50 | Speedup | Max abs | Mean abs |
|---|---:|---:|---:|---:|---:|---:|
| minimum tiles | 1.02897x | 47.7740 ms | 28.0636 ms | 1.702x | 0.0078125 | 4.616e-5 |
| median tiles | 1.03077x | 47.7113 ms | 27.9875 ms | 1.705x | 0.0078125 | 4.611e-5 |
| maximum tiles | 1.03282x | 48.6355 ms | 28.0633 ms | 1.733x | 0.0078125 | 4.610e-5 |

All outputs were finite and all padded rows were exactly zero. The workspace
for output and metadata was `17,030,384` bytes. The kernel passed the 1.5x
performance gate but failed both fixed numerical gates (`max_abs <= 1e-3`,
`mean_abs <= 1e-5`). The remaining difference is consistent with the WMMA
reduction order versus the authoritative vendor GEMM boundary; it is not a
padding or address-mapping fault.

## Decision

`REJECT`. Do not implement grouped W2, integrate the model, scan WMMA tiles,
or relax the numerical threshold. The prototype remains evidence that
expert-tail scheduling removes the heavy-tail padding problem, but this CoreX
WMMA path does not preserve the accepted model-math boundary.

Remote artifacts:

```text
/root/M1_19_effective_w13/attempt1/
/root/M1_19_effective_w13/attempt2/
/root/M1_19_effective_w13/attempt3/
```

Local raw evidence is outside Git at
`result/20260716/M1-19-effective-w13/`.
