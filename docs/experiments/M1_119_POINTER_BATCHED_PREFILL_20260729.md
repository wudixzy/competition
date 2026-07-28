# M1-119 pointer-batched split4 prefill

Date: 2026-07-29

Status: CoreX compilation and numerical gates passed. The fixed component
performance gate failed, so this route is closed without a TP4 service run.

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

## Compile qualification

The private branch source commit
`0cd4aa2d9145bf5b8cbd2a9553a97307071236e8` was cloned exactly on the
BI100 host and remained clean after validation. CoreX 3.2.3 compiled the
candidate for `ivcore10` without accessing a GPU.

```text
frozen M1-109 source:
  11c387e6012834fe634ffa8d038f7a4bf4ec19fa13ec23779ee1f414037e564b
frozen M1-108 control binary:
  f654eee2c0677812394ff419d316e7e8c98ed1bcc84853a7f8d2ed5755503009
M1-119 candidate source:
  5c2c79a120d6982db96f436b149d1afe93a1faa02e97253c370d3e61088c5107
M1-119 candidate binary:
  b82dc08d2033da967dbcebd812163e9b0363e3e4eaba2a2cb16443f316a1c83c
```

The binary exports `PyInit_corex_fused_paged_prefill`, imports
`cublasSgemmBatched`, and has complete linkage under the production CoreX
`LD_LIBRARY_PATH`. Ten focused tests passed both locally and on the remote
exact clone. The complete local suite passed 1,215 tests with 26 skipped, and
submission preflight passed all nine checks.

Structured evidence:
`docs/experiments/evidence/M1_119_POINTER_BATCHED_PREFILL_20260729/compile_qualification.json`.

## Component result

The fixed four-GPU component A/B ran on `ssh-73ca29ba` from exact source
`0cd4aa2d9145bf5b8cbd2a9553a97307071236e8`. Each GPU ran one production
shape and alternated old/new order by physical GPU parity.

All candidate outputs were finite. Output and LSE relative L2 errors remained
below `1e-5`, and maximum absolute output error remained below `1e-3`.
Lifecycle gates also passed: fatal scan, postflight, repeated four-GPU
preflight, and preflight comparison were all clean.

The performance result did not reach the frozen `1.10x` median gate:

| Shape | M1-108 ms | M1-119 ms | Speedup |
| --- | ---: | ---: | ---: |
| dense 8176 | 71.059 | 70.616 | 1.0063x |
| 65K 8176 | 573.257 | 569.439 | 1.0067x |
| 128K 8176 | 1011.773 | 1004.033 | 1.0077x |
| 235K 5616 | 1274.972 | 1241.004 | 1.0274x |

Median speedup was `1.0072x`. All four cases were positive, but the change
only removes a small amount of cuBLAS submission overhead and does not address
the dominant QK/PV data path.

Decision:

- component qualification: failed;
- TP4 service experiment: not authorized;
- main, YAML, or default change: not authorized;
- further pointer-batch or tile scanning: stopped.

Structured evidence:
`docs/experiments/evidence/M1_119_POINTER_BATCHED_PREFILL_20260729/component_qualification.json`.
