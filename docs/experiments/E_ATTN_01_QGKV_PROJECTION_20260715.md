# E-ATTN-01: Retracted full-attention QGKV projection probe

## Scope

This probe was initially built from stale source comments rather than the
authoritative checkpoint config. It used hidden size 5120, 24 query heads, and
four KV heads. The real Qwen3.6-35B-A3B config is hidden size 2048, 16 query
heads, and two KV heads.

## Primitive gate

The following numbers are retained only to make the invalid experiment
reproducible. They do not represent the competition model or fixed TP4 path.

| Tokens | Separate median | Merged median | Speedup | Exact |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.220872 ms | 0.514159 ms | 2.3745x | yes |
| 64 | 0.492510 ms | 0.204608 ms | 2.4071x | yes |

P10/P90 were 1.219950/1.221658 ms versus 0.512438/0.515130 ms for T=1,
and 0.492310/0.492917 ms versus 0.204331/0.204808 ms for T=64.

Artifact:

```text
/root/competition/bench_runs/20260715_E_ATTN_01/result.json
```

## Retraction

At fixed TP4, `tp_size=4 > num_kv_heads=2`. The candidate therefore selected
the existing replicated-K/V fallback and did not use its merged QGKV layer at
all. Its loader hook was also inserted only into the base model loader, while
the competition MoE class uses an override. The TP2 diagnostic launch was
stopped before model loading completed once these issues were identified.

## Status

`INVALID AND RETRACTED`. The corrective commit restores the production model
source. Do not merge commit `752677e` and do not cite its speedup. The valid
successor must benchmark the actual TP4 shapes: q/g output 2048 and replicated
K/V outputs 512 each for a 2048-wide input. Because Q/G is sharded while K/V
is replicated, the next candidate may merge only K and V unless it implements
a custom mixed-sharding linear.
