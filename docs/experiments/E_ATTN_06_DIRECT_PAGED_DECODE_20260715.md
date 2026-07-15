# E-ATTN-06: Direct split-K paged decode prototype

## Scope

E-ATTN-06 tests the second-stage attention plan: read paged FP16 K/V directly,
compute 512-token online-softmax partitions, then reduce partition max/sum/
weighted-value statistics. It avoids the E-ATTN-05 FP32 K/V gather matrices.

The prototype is restricted to the checkpoint's TP4 rank-local shape:

```text
query_heads=6, kv_heads=1, head_size=256, block_size=16
partition_tokens=512, accumulation=float32
```

This is an isolated test extension. No production source, build script,
environment default, or evaluator parameter was changed.

## Performance

GPU1 used the same randomly permuted physical block table and tensor values for
the E-ATTN-05 reference and direct kernel.

| Context | E-ATTN-05 ms | Direct ms | Speedup |
| ---: | ---: | ---: | ---: |
| 65,536 | 7.0410 | 4.3066 | 1.6349x |
| 100,000 | 9.2189 | 6.7698 | 1.3618x |

The performance hypothesis is valid. Across ten full-attention layers, the
additional microbenchmark saving would be about 27.3 ms/token at 64K and 24.5
ms/token at 100K.

## Numerical gate

The direct kernel changes both QK dot-product reduction order and split-K
softmax/value reduction order. It therefore failed the project's exact-output
contract even though typical errors were small.

Initial 20-query results:

| Context | Exact steps | `1e-3` close steps | Worst max abs |
| ---: | ---: | ---: | ---: |
| 65,536 | 0/20 | 20/20 | 0.0000153 |
| 100,000 | 0/20 | 16/20 | 0.0232048 |

A separate 100-query 100K stress run used a different seed:

```text
exact:       0/100
1e-3 close: 83/100
worst abs:  0.05937195
```

The worst error is too large to classify as an innocuous final-FP16 rounding
difference. Running more devices cannot repair a deterministic arithmetic
contract failure, so cross-device performance repetition was intentionally
stopped.

Artifacts are not committed:

```text
gpu1.json          b57a2cf7...e2c3d24
numerics-100.json  557d4de5...95821ec
```

## Decision

`REJECT FOR PRODUCTION`. Keep E-ATTN-05 as the exact long-context winner. Do
not add a quality-equivalence exception for a custom model-math kernel whose
100K error reaches 0.059. A future direct kernel must either reproduce the
authoritative FP32 matmul/softmax arithmetic or pass model-level hash and
quality gates under an explicitly approved numerical policy before service
integration.
