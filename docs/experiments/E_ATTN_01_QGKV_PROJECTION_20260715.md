# E-ATTN-01: Merge full-attention QGKV projections

## Scope

Each full-attention layer executes separate q/g, key, and value projections.
Under the fixed TP=4 configuration, their local output sizes are 3072, 256,
and 256 for a 5120-wide input. This experiment tests one
`MergedColumnParallelLinear` with three independently sharded checkpoint
segments.

## Primitive gate

The real-shape probe ran on physical GPU1. All q/g, key, and value output
segments were bit-exact with the three-linear reference.

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

## Integration candidate

The experiment branch introduces `qgkv_proj` when `tp_size <= num_kv_heads`.
The existing replicated K/V path remains the fallback when KV heads cannot be
sharded. The checkpoint loader maps q/g, key, and value tensors to shards 0,
1, and 2 without modifying checkpoint files.

Local Python compilation and the P0 static suite pass. The CoreX loader unit
test was staged but not executed in this turn because SCP failed and the
verified chunk uploader was disconnected five times at offset zero.

## Status

`INTEGRATION CANDIDATE, NOT QUALIFIED`. Do not merge into
`integration/perf-winners` until remote loader tests, model import, TP4 load,
full smoke, deterministic token checks, sustained decode, and paired service
performance all pass. Resume from the experiment branch after SSH transfer is
stable.
