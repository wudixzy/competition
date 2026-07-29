# M1-127 half-input QK GemmEx capability gate

Date: 2026-07-29

Status: source and local static validation in progress. CoreX compilation and
fixed production-shape GPU qualification are pending.

## Motivation

The qualified M1-115 profile shows that QK consumes approximately 35% of the
M1-109 component time at 65K, 128K, and 235K. M1-119 proved that merely
coalescing cuBLAS submissions improves the component by only 0.6% to 2.7%.
M1-124 and M1-126 proved that flattening the four GQA heads into a wider GEMM
is neutral or substantially slower on CoreX.

M1-127 tests a different boundary. Q and K already enter attention as FP16
model activations. The M1-109 extension converts both to FP32 before QK and
uses `cublasSgemmStridedBatched`. The candidate passes the unchanged FP16
values directly to `cublasGemmStridedBatchedEx`, retains FP32 multiplication
accumulation and FP32 scores, and applies the exact `1/16` attention scale as
the GEMM alpha.

This is not quantization or a model-dtype reduction. The operands remain at
their existing model dtype and the accumulation/output remain FP32. The gate
exists because a different GEMM implementation can still change reduction
order and therefore requires independent numerical qualification.

## Fixed capability gate

The extension contains both the exact M1-109 FP32 SGEMM submission shape and
the half-input/FP32-accumulate GemmEx candidate. The benchmark freezes:

- four query heads, one shared K tile, head dimension 256, and 512 key tokens;
- production query lengths 8176 and 5616;
- FP16 Q/K generated at magnitudes 0.5, 1.0, and 2.0;
- seed 20260729;
- an independent sampled CPU FP64 QK oracle;
- relative L2 at most `1e-5`, maximum absolute error at most `1e-3`;
- at least `1.25x` QK speedup for each production query shape;
- `CUBLAS_GEMM_DEFAULT_TENSOR_OP` as the sole candidate algorithm.

There is no algorithm, tile, threshold, split, dtype, or YAML scan. Failure
of either shape closes this candidate. A pass authorizes only integration
into an experimental copy of the M1-109 full pipeline, where the complete
output and TP4 gates must be rerun.

This capability source does not alter the runtime selector, prebuilt
extension, model, tokenizer, request semantics, `computility-run.yaml`,
defaults, `main`, or repository visibility.
