# M1-128 half-input QK full-pipeline candidate

Date: 2026-07-29

Status: source implementation and local static verification in progress.
CoreX compilation and fixed four-shape component A/B are pending.

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
