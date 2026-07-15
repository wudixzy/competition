# E-ATTN-02: Merge replicated full-attention K/V projections

## Scope correction

The authoritative checkpoint config for Qwen3.6-35B-A3B is:

```text
hidden_size=2048
num_attention_heads=16
num_key_value_heads=2
head_dim=256
tensor_parallel_size=4
```

Therefore each TP4 rank has a 2048-output sharded q/g projection and two
512-output replicated K/V projections. E-ATTN-01 used stale 5120/24/4 source
comments and is invalid. Its candidate also bypassed the merged path at TP4
and is explicitly retracted on its experiment branch.

E-ATTN-02 keeps the q/g projection unchanged and replaces the two replicated
K/V linears with one 1024-output `ReplicatedLinear`, split into two 512-wide
views. It does not change attention, RoPE, cache layout, or evaluator flags.

## Primitive gate

The actual TP4 rank shapes ran serially on physical GPU1. Nine repeats of 300
iterations produced:

| Tokens | Separate QG/K/V | Merged replicated K/V | Speedup | Exact |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.362245 ms | 0.246215 ms | 1.4713x | yes |
| 64 | 0.134784 ms | 0.090755 ms | 1.4852x | yes |

All q/g, key, and value segments are bit-exact with max absolute difference
zero. The p10/p90 ranges were 0.362063/0.362571 ms versus
0.246007/0.246275 ms for T=1, and 0.134717/0.134828 ms versus
0.090739/0.090777 ms for T=64.

A single mixed-sharding QGKV GEMM was measured only as an unattainable upper
bound: 2.7187x for T=1 and 1.9737x for T=64. Standard vLLM linears cannot
represent sharded QG and replicated K/V in one weight, so that result is not
claimed by this candidate.

Artifact:

```text
/root/competition/bench_runs/20260715_E_ATTN_02/result.json
```

## Checkpoint integration

The fixed TP4 branch creates `kv_proj` only when `tp_size > num_kv_heads`.
The existing independently sharded K/V path remains unchanged for other TP
layouts. Both the base loader and the competition MoE loader map checkpoint
`k_proj` and `v_proj` into disjoint halves of the replicated parameter. Invalid
row counts fail closed.

Remote CoreX gates:

```text
KV loader unit tests             4/4 pass
P0 static suite                  40/40 pass
Python compile                   pass
patch_ops registry verification  Qwen3_5ForCausalLM and MoE class found
GPU1/GPU2 NCCL diagnostic        pass (expected sum 3.0)
```

## Hardware limitation and status

The four-card CUDA preflight still reports GPU0 timeout while GPU1-3 pass.
Consequently no TP4 service was started. A TP2 diagnostic was stopped before
model loading after the authoritative config showed that TP2 does not execute
the replicated-K/V candidate branch. TP2 cannot qualify this experiment.

`INTEGRATION CANDIDATE, NOT YET QUALIFIED`. Keep it on the experiment branch.
After platform-side GPU0 recovery, rerun four-card CUDA and collective
preflights, then TP4 checkpoint load, full smoke, deterministic and sustained
decode hashes, 235K cold/warm context, and three paired service A/B runs. Do
not merge it into `integration/perf-winners` before those gates pass.
