# E-MOE-06: Fuse router and shared-expert gate projections

Date: 2026-07-15

## Hypothesis

Every one of the 40 MoE layers projects the same hidden state through a
replicated 256-output router and a separate replicated scalar shared-expert
gate. Concatenating their weights into one 257-output replicated linear should
remove one GEMM launch per layer without changing routing, expert math,
collectives, model output, or the fixed evaluator contract.

## Manifest

```text
baseline: b1d95009d52135a5b00bbac1c5ccc682c4539644
candidate: 579614eeb5436d57038b3ab4eb585bd7eb5b2937
branch: exp/E-MOE-06-router-shared-gate
model: /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
hardware: 4 x BI-V100-50C-200G, TP=4
seed: 123
fixed evaluator contract changed: no
```

## Change

`qwen3_6_scripts/qwen3_5.py` replaces the two `ReplicatedLinear` modules with
one `ReplicatedLinear(2048, 257)`. Its output order is:

```text
[256 router logits, 1 shared-expert gate score]
```

The checkpoint loader copies `mlp.gate.weight` and
`mlp.shared_expert_gate.weight` into disjoint rows of the combined parameter.
It validates source and target shapes and fails closed on missing parameters.
The helper is called only by the MoE loader. Softmax, top-k, routed experts,
shared expert projections, sigmoid, all-reduce, and evaluator arguments remain
unchanged.

The experiment adds:

```text
tests/bench_moe_router_shared_gate.py
tests/test_moe_router_shared_gate_unit.py
```

## Correctness gates

- fused loader and CPU projection tests: 5/5 pass
- P0 static tests: 41/41 pass
- complete CoreX unit discovery: 128 pass, 1 skipped
- four-GPU primitive parity at T=1 and T=64: exact zero maximum absolute
  difference for router logits, shared gate, selected route weights, and
  sigmoid output; top-k IDs equal; all outputs finite
- real checkpoint loading: 26/26 shards
- deterministic API oracle: 3/3 exact for HTTP status, message, finish reason,
  and usage
- full API smoke: 15/15 pass in 129.04 seconds
- candidate health after qualification: HTTP 200
- candidate service error-log scan: zero fatal matches

The candidate reached HTTP readiness in 421 seconds. It reported 16,878 GPU
blocks and 6,553 CPU blocks, versus 16,871 and 6,553 for the baseline, so there
is no KV-capacity regression.

## Primitive results

The probe used hidden size 2,048, 256 experts, top-k 8, float16, and the real
T=1/T=64 shapes on all four devices. `Projection` measures the two linears
versus the fused linear. `Pipeline` additionally includes float32 softmax,
top-k, route renormalization, dtype conversion, and shared-gate sigmoid.

| Shape | Region | Current (ms) | Fused (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| T=1 | Projection | 0.2008-0.2016 | 0.1171-0.1177 | 1.71-1.72x |
| T=1 | Pipeline | 0.3055-0.3074 | 0.2124-0.2159 | 1.42-1.44x |
| T=64 | Projection | 0.1939-0.1951 | 0.1124-0.1134 | 1.71-1.74x |
| T=64 | Pipeline | 0.3249-0.3372 | 0.2547-0.2679 | 1.26-1.28x |

Artifacts:

```text
bench_runs/20260715_E_MOE_06/gpu0.json
bench_runs/20260715_E_MOE_06/gpu1.json
bench_runs/20260715_E_MOE_06/gpu2.json
bench_runs/20260715_E_MOE_06/gpu3.json
```

## Strict short pairs

Each pair used eight serial streaming requests, 64 generated tokens, seed 123,
prompt repeat 126, and the same salt on baseline and candidate. Every side
reported exactly 14,552 prompt tokens, 1,896 uncached prompt tokens, 12,656
cached tokens, and 512 completion tokens.

| Metric | Pair 1 change | Pair 2 change | Pair 3 change | Median |
| --- | ---: | ---: | ---: | ---: |
| Wall time | -4.88% | +1.26% | +5.68% | +1.26% |
| TTFT P90 | -26.13% | -17.37% | +16.50% | -17.37% |
| Decode TPS P10 | +0.33% | -4.57% | -2.19% | -2.19% |
| Weighted overlap score | +4.18% | -1.89% | -4.79% | -1.89% |
| Disjoint score | +2.80% | -2.84% | -3.89% | -2.84% |

Candidate Decode TPS P10 was tightly grouped from 12.9160 to 13.0475. The
baseline ranged from 13.0049 to 13.5338 across startups, recording material
device variability, but the candidate only beats the slowest baseline by
0.33% and loses the other two strict pairs.

Artifacts:

```text
bench_runs/20260715_E_MOE_06/candidate-P1.json
bench_runs/20260715_E_MOE_06/candidate-P2.json
bench_runs/20260715_E_MOE_06/candidate-P3.json
bench_runs/20260715_E_ATTN_PROJ_01/baseline-P1.json
bench_runs/20260715_E_ATTN_PROJ_01/baseline-P2.json
bench_runs/20260715_E_ATTN_PROJ_01/baseline-P3.json
```

## Strict long-prefill sample

Both sides used four serial requests, prompt repeat 600, 16 generated tokens,
seed 123, and the same salt. Token and cache totals match exactly: 33,824
prompt, 8,528 uncached prompt, 25,296 cached, and 64 completion tokens.

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Success rate | 100% | 100% | equal |
| Wall time (s) | 21.0755 | 23.8498 | +13.16% |
| TTFT P90 (s) | 7.9040 | 9.4944 | +20.12% |
| Decode TPS P10 | 13.5947 | 13.5503 | -0.33% |
| Cache hit rate | 74.7871% | 74.7871% | equal |
| Weighted overlap score | 5,392.5828 | 4,791.1061 | -11.15% |
| Disjoint score | 2,033.0676 | 1,822.3865 | -10.36% |

Artifacts:

```text
bench_runs/20260715_E_MOE_06/candidate-LONG.json
bench_runs/20260715_E_ATTN_PROJ_01/baseline-LONG.json
bench_runs/20260715_E_MOE_06/oracle-comparison.json
bench_runs/20260715_E_MOE_06/smoke-full.json
```

## Decision

`REJECT` the production-path change. The fused primitive is exact and improves
isolated route preparation by 1.26x to 1.44x, but the endpoint evidence is
negative: median short-pair Decode TPS P10 changes by -2.19%, median weighted
score by -1.89%, and the strictly matched long-prefill score by -11.15%. The
single +0.33% decode pair is below the 1% noise floor and does not offset the
other regressions. Do not spend another startup on the 99,500-token boundary.
Keep the implementation and benchmark on the experiment branch as evidence;
do not merge it into `integration/perf-winners`.
