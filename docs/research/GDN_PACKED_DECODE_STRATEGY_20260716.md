# Packed GDN decode strategy for BI100

## Why this is the next algorithmic candidate

The current rank-local TP4 decode profile attributes the following median
latencies to adjacent GDN stages:

| Stage | Median (ms/layer) |
| --- | ---: |
| beta/decay preparation | 0.007248 |
| normalized q/k mapping | 0.097710 |
| recurrent update | 0.060277 |
| **candidate packed boundary** | **0.165235** |

This is a large enough boundary to justify a new kernel. Scanning another
cuBLAS mode or replacing one small pointwise operator is not.

SGLang independently reached the same architectural conclusion for Qwen3.5:

- its [packed GDN decode PR](https://github.com/sgl-project/sglang/pull/20627)
  combines q/k/v extraction, q/k normalization, beta/decay, state decay,
  delta update, and output reduction in one kernel;
- the PR reduces the decode path from six stages to three and reports a
  `2.59x` kernel speedup at batch one on its tested shape;
- its serving benchmark reports `+17.9%` output throughput, although that
  high-concurrency NVIDIA result is not a projection for BI100;
- the broader
  [Qwen3.5 optimization tracker](https://github.com/sgl-project/sglang/issues/18590)
  also prioritizes packed decode and larger GDN fusion boundaries.

The transferable design is the boundary, dataflow, and launch reduction. The
Triton implementation and its NVIDIA tuning parameters are not portable to
CoreX.

## Shape audit

The production TP4 rank-local shape is:

```text
batch                 1
local key heads       4
local value heads     8
key/value head dim    128/128
temporal state        [1, 8, 128, 128] FP32
mixed qkv             [1, 2048] FP16
```

SGLang also transposes recurrent state from `[K,V]` to `[V,K]` to improve
coalescing ([PR 20283](https://github.com/sgl-project/sglang/pull/20283)). Its
motivating decode tile is highly asymmetric (`256x8`). Our `128x128` tile is
symmetric and the existing CoreX prototype already assigns adjacent value
columns to adjacent threads. State-layout migration is therefore not the
first experiment; it must win an isolated BI100 memory-access probe before it
can justify changes to prefill, prefix-state caching, and state allocation.

The upstream projection fusion
([PR 21019](https://github.com/sgl-project/sglang/pull/21019)) is also not a
direct transplant. This repository already performs one merged projection,
and its following `torch.split` calls return views. A production trace must
show real copy/reshape cost before developing another projection kernel.

## E-GDN-14 design

Implement one guarded CoreX extension for the exact production decode shape.
Each `(batch, value_head, value-column tile)` program should:

1. load the corresponding raw q/k head and v values from packed `mixed_qkv`;
2. reproduce the current FP16 q/k normalization contract and map four key
   heads to eight value heads;
3. compute sigmoid beta and exponential decay from `a`, `b`, `A_log`, and
   `dt_bias`;
4. decay the FP32 state, compute `k @ state`, apply the rank-one update, and
   compute `q @ updated_state`;
5. write the FP32 state in place and one FP32 output vector.

The first version deliberately excludes causal convolution, gated RMSNorm,
and output projection. They have separate qualified kernels or GEMMs, and
including them would make numerical attribution and fallback recovery harder.

The implementation remains behind `BI100_GDN_COREX_PACKED_DECODE=0` and must
fail closed on any unsupported dtype, shape, stride, batch size, or device.

## Stop conditions

Development stops before service integration unless the complete packed
boundary meets all of these gates on each healthy experimental GPU:

- median latency at most `0.110 ms/layer`, equivalent to at least `1.50x`
  against the current `0.165235 ms/layer` boundary;
- at least 500 fixed-input and 1,000 random-sequence steps remain finite;
- output and state drift remain bounded and do not grow monotonically without
  limit across the random sequence;
- the candidate latency is stable across serial repeats; concurrent-host
  timing is diagnostic only.

If those gates fail, do not tune block sizes indefinitely. Capture the failed
roofline or numerical reason, close E-GDN-14, and re-profile the full service.

## Service qualification

A microbenchmark winner still needs a same-binary TP4 A/B:

- three clean serial pairs, with at least `5%` Output TPS improvement in all
  three pairs;
- full API smoke and the dataset-shaped Agent workload matrix;
- deterministic repeated candidate output, a sustained 1,000-token response,
  and no non-finite/native runtime errors;
- 99.5K and 235K cold/warm requests with correct cache accounting;
- multimodal and tool-call coverage.

Baseline token-for-token equality remains a useful diagnostic, but it is not
the competition's quality contract. A changed reduction order may proceed only
when the real workload gates pass and the drift is documented.

## Ordering

Finish the current E-MOE-20 same-binary TP4 qualification first. E-GDN-14 is
the next implementation candidate only if E-MOE-20 is stable, or if E-MOE-20
fails and a fresh profile still ranks this packed GDN boundary above other
unqualified work.
