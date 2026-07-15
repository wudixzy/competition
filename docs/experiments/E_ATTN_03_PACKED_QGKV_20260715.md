# E-ATTN-03: Rank-local packed QGKV projection

## Scope

At fixed TP4, full-attention Q/G is column-sharded while K/V is replicated.
Standard merged linears cannot express both layouts, but each rank can hold one
local packed weight:

```text
rank-local QG: 2048 x 2048
replicated K:   512 x 2048
replicated V:   512 x 2048
packed QGKV:   3072 x 2048
```

The custom loader applies the same contiguous Q/G row slice as
`ColumnParallelLinear.weight_loader` for the current TP rank, then appends the
complete K and V tensors. Parameter count and memory are unchanged. The packed
path is enabled only when `tp_size > num_key_value_heads`; other TP layouts
retain the original Q/G, K, and V layers.

## vLLM layer benchmark

The benchmark uses actual vLLM `ReplicatedLinear` instances rather than bare
PyTorch calls. It ran serially on physical GPU1 with the authoritative TP4
rank shapes:

| Tokens | Three vLLM layers | Packed layer | Speedup | Exact |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.049838 ms | 0.026160 ms | 1.9051x | yes |
| 64 | 0.092275 ms | 0.064861 ms | 1.4227x | yes |

The T=1 p10/p90 ranges are 0.049811/0.049954 ms and
0.026125/0.026234 ms. The T=64 ranges are 0.092201/0.092363 ms and
0.064823/0.064910 ms. Q/G, K, and V output segments are bit-exact with max
absolute difference zero.

Artifact:

```text
/root/competition/bench_runs/20260715_E_ATTN_03/result.json
```

## Loader and runtime gates

The loader tests cover all logical shards, rank 0/1 Q/G slicing, absent-target
fallback, invalid shapes, non-divisible Q/G dimensions, and unrelated weights.
Both the base and MoE override loaders invoke the packed helper.

Remote CoreX results:

```text
rank-slice loader unit tests        6/6 pass
P0 static suite                    40/40 pass
Python compile                     pass
patch_ops registry verification    both base and MoE classes found
installed-module rank=2 runtime    all weights and outputs bit-exact
```

The installed-module probe emulates TP rank 2. It loads Q/G rows 4096-6143
from the global 8192-row checkpoint tensor, appends full 512-row K/V tensors,
and compares one packed vLLM layer against three baseline vLLM layers on GPU1.
All three weight segments and outputs are exact with max absolute difference
zero.

Artifact:

```text
/root/competition/bench_runs/20260715_E_ATTN_03/runtime-rank2.json
```

## Impact and status

The packed candidate supersedes E-ATTN-02 when fixed TP4 is available. Its
absolute T=1 saving is about 0.0237 ms per full-attention layer, or about
0.237 ms across the model's ten full-attention layers per decode token. That
is likely below 1% of observed end-to-end inter-token latency, so this is a
small cumulative optimization rather than a solution to the Output TPS gap.

`INTEGRATION CANDIDATE, NOT YET QUALIFIED`. GPU0 still times out while GPU1-3
pass, so no TP4 service was launched. Keep the candidate on its experiment
branch. Once GPU0 is recovered, require TP4 checkpoint loading, full smoke,
greedy and 1,000-token hash equality, 235K cold/warm context, and paired service
A/B. Merge only if the service measurement is repeatably positive; retain
E-ATTN-02 as the simpler fallback if packed loading exposes a TP4-only issue.
