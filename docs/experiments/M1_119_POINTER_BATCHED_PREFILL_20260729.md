# M1-119 pointer-batched split4 prefill

Date: 2026-07-29

Status: source and local static gates in progress. CoreX compilation,
component numerical/performance qualification, and TP4 service validation are
pending.

## Motivation

The qualified M1-115 profile attributes 73.14% of the 235K M1-109 component
time to QK and PV. M1-109 issues four
`cublasSgemmStridedBatched(batch=4)` calls for QK and four for PV in every
2048-token group. Each call covers one 512-token split and four GQA query
heads.

M1-119 keeps the M1-108 numerical path and only changes the cuBLAS submission
shape. A device kernel builds 16 pointers for the four split by four head
matrices, then one `cublasSgemmBatched` call performs QK and one performs PV.
Each individual matrix retains the original dimensions, operands, leading
dimensions, 512-token softmax boundary, ATen max/exp/sum operations, and
ordered online merge.

This is deliberately more conservative than M1-113 group2048. It does not
flatten heads into a wider GEMM or change the softmax partition. The
candidate lives in the separate
`tests/corex_fused_paged_prefill_batched16_ext.cu` source and is enabled only
by the compile-time `BI100_PREFILL_BATCHED16_EXPERIMENT` definition. The
frozen M1-109 production source and prebuilt binary remain byte-identical. The
experiment does not change the runtime selector, defaults, request behavior,
`computility-run.yaml`, or model weights.

## Fixed gates

The component A/B compares the frozen M1-108 binary with the M1-119 binary on
dense, 65K, 128K, and 235K production shapes. It requires:

- finite output and LSE;
- output and LSE relative L2 no greater than `1e-5`;
- maximum absolute output error no greater than `1e-3`;
- median M1-108/M1-119 speedup at least `1.10x`;
- at least three positive cases;
- no individual regression beyond 2%;
- clean scoped cleanup, fatal scan, postflight, and repeated four-GPU
  preflight.

A component pass only authorizes a TP4 service A/B. Full-model output,
next-token, functional, tool, reasoning, multimodal, long-context, and
lifecycle gates remain mandatory before any runtime promotion.

## Planned execution

```bash
tests/build_corex_fused_prefill_batched16.sh \
  /tmp/m1-119-batched16-build-SOURCE

scripts/run_m1_119_pointer_batched_component_ab.sh \
  INSTANCE \
  /path/to/frozen-m1-108/corex_fused_paged_prefill.so \
  /tmp/m1-119-batched16-build-SOURCE/corex_fused_paged_prefill_batched16.so \
  /tmp/m1-119-batched16-component-SOURCE
```

If the pointer-batched call fails compilation, numerical gates, or the fixed
component speed gate, record the evidence and stop this call-coalescing
variant. Do not scan batch sizes or modify YAML thresholds.
