# M1-127 half-input QK GemmEx capability gate

Date: 2026-07-29

Status: corrected CoreX capability gate qualified. Full-pipeline component
integration is authorized; TP4 service, `main`, and YAML changes are not.

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

## Fixed runner

`scripts/run_m1_127_half_input_qk.sh` assigns `q8176` to physical GPU 0 and
`q5616` to physical GPU 1, then runs both fixed cells concurrently. Each cell
uses a private verified process session. The runner sends SIGTERM to only its
recorded process groups, waits 60 seconds before any SIGKILL fallback, reaps
children, scans fatal/timeout classes, and repeats four-card process and
compute preflight after completion. Its comparison can authorize only
experimental full-pipeline integration; it cannot authorize TP4 service,
`main`, YAML, or submission changes.

Local verification before remote execution:

- focused M1-127 tests: 5 passed;
- complete unit suite after evidence capture: 1109 passed, 13 skipped;
- submission preflight: 9/9 passed;
- shell/Python syntax and `git diff --check`: passed.

The first BI100 execution exposed a measurement defect: the GPU FP64 norm
reported zero relative L2 despite a nonzero maximum absolute difference.
CoreX GPU FP64 reduction is therefore not used for qualification. The fixed
gate copies complete candidate and control outputs to CPU and computes both
relative L2 and maximum absolute error in CPU FP64. The initial execution is
diagnostic only and must be repeated with the corrected gate.

## Corrected BI100 result

The corrected gate ran from source `da640c7` with complete candidate/control
errors reduced on CPU in FP64:

- q8176: `0.722681 ms` control, `0.428312 ms` candidate, `1.6873x`;
- q5616: `0.507642 ms` control, `0.310008 ms` candidate, `1.6375x`;
- candidate/control relative L2 stayed between `3.088e-7` and `3.094e-7`;
- worst candidate/control maximum absolute error was `1.908e-5`;
- all outputs were finite and the independent sampled FP64 checks passed;
- both cells, cleanup, fatal/timeout scans, process postflight, and four-card
  before/after preflight returned zero;
- the foreground parent session reaped the outer helper.

The result authorizes one experimental integration into the complete M1-109
split4 attention pipeline. It does not establish end-to-end or TP4 model
benefit. Privacy-safe evidence is stored under
`docs/experiments/evidence/M1_127_HALF_INPUT_QK_QUALIFIED_20260729`.
