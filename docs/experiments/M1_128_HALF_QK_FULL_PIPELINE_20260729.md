# M1-128 half-input QK full-pipeline candidate

Date: 2026-07-29

Status: the first CoreX component candidate improved all four fixed shapes but
failed the output relative-L2 gate. It is not authorized for TP4 service,
runtime overlay, YAML, or `main`.

## Scope

M1-127 qualified FP16 Q/K input with FP32 accumulation for the fixed
production QK operation at `1.64x` to `1.69x`. M1-128 integrates exactly that
data type boundary into an experimental copy of the M1-109 split4 pipeline:

- Q is transposed in FP16 without applying a separate FP32 scale conversion;
- K is gathered from FP16 cache/new-token storage into FP16 tiles;
- QK uses GemmEx with FP16 inputs, FP32 accumulation and output, and the exact
  attention scale as GEMM alpha;
- V gather, online FP32 softmax, PV SGEMM, split merge, LSE, output dtype and
  all shape/capacity checks remain unchanged.

The frozen M1-109 source remains byte-identical at SHA-256
`11c387e6012834fe634ffa8d038f7a4bf4ec19fa13ec23779ee1f414037e564b`.
M1-128 lives in a separate candidate source file, and only its separate build
defines `BI100_HALF_INPUT_QK=1` and writes a different extension artifact. No
runtime selector, prebuilt submission extension, overlay, YAML, model,
tokenizer, or request behavior changes in this branch.

## Gate

The candidate must first pass the existing M1-109 four-shape component A/B
against the original extension:

- output and LSE relative L2 at most `1e-5`;
- output maximum absolute error at most `1e-3`;
- median component speedup at least `1.10x`;
- at least three of four shapes improve;
- no shape regresses by more than 2%;
- all pre/postflight, cleanup, fatal and timeout checks pass.

Only that result may authorize an experimental TP4 service overlay. It cannot
authorize `main`, YAML, or submission changes.

## First component result

Source commit `2135cb276ffa678c6474aacbcc669e2806b2391b` compiled on
CoreX 3.2.3 for `ivcore10`. The 247,240-byte candidate extension has SHA-256
`acc89f2cbadb99dbe73dbb0af397ebfe9885e55e6505fa361a798bab92b345cd`.
The control was the qualified M1-109 extension with SHA-256
`ad4ea7707bb2f2bfe04e07a7ad5fd58a647232be70a3056937a0d738c8254bff`.

| Case | M1-109 ms | M1-128 ms | Speedup | Output rel-L2 | LSE rel-L2 | Max abs |
|---|---:|---:|---:|---:|---:|---:|
| Dense q8176 | 39.805 | 35.070 | 1.135x | 1.470e-5 | 3.933e-8 | 2.441e-4 |
| 65K q8176 | 293.955 | 251.879 | 1.167x | 1.965e-5 | 3.196e-8 | 1.526e-5 |
| 128K q8176 | 516.080 | 441.167 | 1.170x | 1.988e-5 | 3.653e-8 | 1.526e-5 |
| 235K q5616 | 658.571 | 568.041 | 1.159x | 2.168e-5 | 4.286e-8 | 7.629e-6 |

All four shapes improved and median M1-109/M1-128 speedup was `1.163x`.
Finite, LSE, and maximum-absolute-error checks passed, but every output
relative-L2 exceeded the fixed `1e-5` limit. The gate therefore failed
closed. Preflight, postflight, cleanup, fatal scan, and GPU state comparison
all passed.

One second implementation is allowed before closing this route: retain FP16
Q/K inputs and FP32 output/accumulation, but use the default GemmEx algorithm
instead of explicitly selecting the Tensor-Op algorithm. This is a bounded
accuracy-path comparison, not an algorithm or tile scan. It must still meet
the same numerical limits and at least `1.10x` median component speedup.
