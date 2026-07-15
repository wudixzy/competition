# E-ATTN-04: Exact CoreX paged K/V gather

## Scope

The long-context decode fallback previously transformed each layer's paged
FP16 K/V cache through advanced indexing, permutations, contiguous copies, and
separate FP32 conversions before running the existing FP32 matmul/softmax path.
At 100K, the final K/V matrices alone contain about 195 MiB per layer.

E-ATTN-04 replaces only those layout copies with one CoreX kernel. It reads the
physical block IDs directly and writes the same FP32 matrices:

```text
K: [kv_heads, head_size, sequence]
V: [kv_heads, sequence, head_size]
```

The query, scale, FP32 matmuls, softmax, output cast, short-context native V1
path, prefix attention, cache writes, and evaluator command are unchanged.
`BI100_ATTN_COREX_PAGED_GATHER=0` restores the tensor-indexing fallback.

## Method

The benchmark uses the checkpoint's TP4 rank-local full-attention shape:

```text
query_heads=6, kv_heads=1, head_size=256, block_size=16, cache=float16
```

Physical block IDs are randomly permuted so the test does not assume contiguous
cache allocation. Each device ran five warmups and seven repeats of ten full
operations at 32K, 64K, and 100K.

## Results

| GPU | Context | Gather speedup | Full attention speedup | K/V/output exact |
| ---: | ---: | ---: | ---: | --- |
| 1 | 32,768 | 1.789x | 1.326x | yes |
| 1 | 65,536 | 1.921x | 1.367x | yes |
| 1 | 100,000 | 4.218x | 2.024x | yes |
| 2 | 32,768 | 1.792x | 1.325x | yes |
| 2 | 65,536 | 1.922x | 1.365x | yes |
| 2 | 100,000 | 4.227x | 2.018x | yes |
| 3 | 32,768 | 1.782x | 1.325x | yes |
| 3 | 65,536 | 1.930x | 1.370x | yes |
| 3 | 100,000 | 4.230x | 2.021x | yes |

Median complete-attention savings are approximately 2.89 ms/layer at 64K and
9.38 ms/layer at 100K. Across the model's ten full-attention layers, that is a
conditional projection of about 28.9 ms/token at 64K and 93.8 ms/token at
100K. The 32K result is capability evidence only because sequence lengths at
or below 32,768 remain on native paged-attention V1.

Every compared K element, V element, and final FP16 output was bit-exact on all
devices. A separate production-dispatch probe loaded the candidate
`paged_attn.py` and production-built extension, then toggled the runtime flag
on the same GPU1 input:

```text
100K fallback:  18.6890 ms
100K candidate:  9.2588 ms
speedup:          2.0185x
exact/max_abs:    yes / 0.0
```

Raw artifacts are not committed:

```text
gpu1.json  9ca0e3e7...1a53117c
gpu2.json  29180aa6...d7a1130
gpu3.json  a1611776...e2a5492
runtime-prod-gpu1.json
```

## Safety and decision

The extension validates CUDA placement, FP16 cache dtype, contiguous layout,
int32 block tables, K/V shape agreement, and block-table capacity. Unsupported
runtime layouts fail closed to the existing PyTorch code before calling the
extension. Diagnostic physical-block range checks remain available through
`BI100_PAGED_ATTN_DIAGNOSTICS=1`.

`QUALIFY FOR TP4 SERVICE A/B`. The 100K full-boundary result meets the 2x plan
gate and the 64K result remains materially positive. This candidate does not
by itself close the prior decode SIGSEGV incident: that crash did not identify
the exact request length or prove the K/V gather was the corrupting operation.
The incident reproduction matrix and long-running TP4 safety gates still
apply. GPU0 remains unavailable, so no four-card service run has been made.
