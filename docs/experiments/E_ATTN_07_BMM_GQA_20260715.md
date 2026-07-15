# E-ATTN-07: Stride-zero GQA BMM scheduling

## Scope

E-ATTN-07 preserves E-ATTN-05's gathered FP32 K/V matrices and all arithmetic,
but compares the existing broadcast four-dimensional `torch.matmul` with:

- `torch.bmm` after physically repeating K/V across six query heads;
- `torch.bmm` on stride-zero expanded K/V views.

No production source or evaluator setting was changed.

## Results

GPU1 used TP4 rank-local dimensions and 100 random queries per context. Both
BMM variants were bit-exact for every query.

| Context | Variant | Attention-only speedup | Full-path speedup | Exact |
| ---: | --- | ---: | ---: | --- |
| 65,536 | physical repeat | 0.6476x | 0.7436x | 100/100 |
| 65,536 | stride-zero expand | 1.0022x | 1.0025x | 100/100 |
| 100,000 | physical repeat | 0.6349x | 0.7195x | 100/100 |
| 100,000 | stride-zero expand | 1.0010x | 1.0000x | 100/100 |

The stride-zero path reaches the same CoreX GEMM behavior as broadcast
`matmul`; its small 64K difference is measurement noise. Physical repetition
adds large K/V copies and is materially slower.

Artifact, intentionally untracked:

```text
gpu1.json  3da6a5b4...cc7c23d
```

## Decision

`REJECT AS PERFORMANCE WINNER`. Exactness passes, but the complete-path gain
is far below the 5% integration gate. Keep E-ATTN-05 unchanged.
