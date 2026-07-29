# M1-129 half-input QK with default GemmEx

Date: 2026-07-29

Status: source implementation and local verification in progress. CoreX
compilation and the fixed four-shape component A/B are pending.

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
