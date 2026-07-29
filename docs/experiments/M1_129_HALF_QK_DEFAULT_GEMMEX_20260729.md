# M1-129 half-input QK with default GemmEx

Date: 2026-07-29

Status: CoreX compilation and lifecycle gates passed, but all four fixed
shapes failed the output relative-L2 limit. The half-input full-pipeline route
is closed without a TP4 service run.

## Rationale

M1-128 used FP16 Q/K inputs, FP32 accumulation/output, and
`CUBLAS_GEMM_DEFAULT_TENSOR_OP`. It improved every fixed component shape by
`1.135x` to `1.170x`, but output relative-L2 was `1.47e-5` to `2.17e-5`,
above the fixed `1e-5` gate.

M1-129 is the one permitted accuracy-path alternative. It keeps the same
Q/K storage, gather, scale, FP32 accumulation/output, softmax, PV, merge, LSE,
shape guards, and fallback contract. The only computational change from
M1-128 is selecting `CUBLAS_GEMM_DEFAULT` instead of
`CUBLAS_GEMM_DEFAULT_TENSOR_OP`.

The M1-108/M1-109 and M1-128 sources remain unchanged. M1-129 has a separate
source, build script, and output artifact. No runtime overlay, selector,
prebuilt binary, YAML, model, tokenizer, or request behavior changes in this
branch.

## Gate

The candidate must pass the same fixed M1-109 four-shape component A/B:

- output and LSE relative-L2 at most `1e-5`;
- output maximum absolute error at most `1e-3`;
- median component speedup at least `1.10x`;
- at least three of four shapes improve;
- no shape regresses by more than 2%;
- all pre/postflight, cleanup, fatal, and timeout checks pass.

If this implementation also fails either numerical or performance gates, the
half-input full-pipeline route closes. No additional GemmEx algorithm, tile,
threshold, or YAML scan is authorized.

## Result

Source commit `de9d91e32eb1feff6ec176d7084a80095fb98ef2` compiled on
CoreX 3.2.3 for `ivcore10`. The 247,248-byte candidate extension has SHA-256
`62fc3af12eb863801abfb6e08336e9a04f5c377ba7d03511ad0a2412b0f37f82`.
The control remained the qualified M1-109 extension with SHA-256
`ad4ea7707bb2f2bfe04e07a7ad5fd58a647232be70a3056937a0d738c8254bff`.

| Case | M1-109 ms | M1-129 ms | Speedup | Output rel-L2 | LSE rel-L2 | Max abs |
|---|---:|---:|---:|---:|---:|---:|
| Dense q8176 | 39.797 | 35.088 | 1.134x | 1.470e-5 | 3.933e-8 | 2.441e-4 |
| 65K q8176 | 293.827 | 251.835 | 1.167x | 1.965e-5 | 3.196e-8 | 1.526e-5 |
| 128K q8176 | 516.138 | 441.246 | 1.170x | 1.988e-5 | 3.653e-8 | 1.526e-5 |
| 235K q5616 | 659.496 | 568.226 | 1.161x | 2.168e-5 | 4.286e-8 | 7.629e-6 |

All four shapes improved and median speedup was `1.164x`. Every LSE and
maximum-absolute-error check passed, but every output relative-L2 exceeded
`1e-5`. The numerical values are exactly equal to the M1-128 Tensor-Op run
for all four shapes, showing that the default algorithm selector did not
provide a distinct, more accurate CoreX path.

Preflight, postflight, fatal scan, cleanup, and repeated GPU-state comparison
all passed. This candidate is not authorized for TP4 service, runtime overlay,
YAML, or `main`. No further half-input GemmEx algorithm or tile variants will
be tested.
