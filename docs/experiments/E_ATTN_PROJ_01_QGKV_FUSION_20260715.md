# E-ATTN-PROJ-01: Fuse full-attention QGKV projections

Date: 2026-07-15

## Hypothesis

Each full-attention layer launches separate tensor-parallel Q/G, replicated K,
and replicated V projections. A vendor `QKVParallelLinear` can emit the gated
query and the rank-local K/V heads in one operation, reducing launch and GEMM
overhead while preserving the checkpoint layout and evaluator contract.

## Manifest

```text
baseline: b1d95009d52135a5b00bbac1c5ccc682c4539644
candidate: ed61f519abdd6291844057ddccf0a8f989ce7a02
branch: exp/E-ATTN-PROJ-01-qgkv-fusion
model: /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
hardware: 4 x BI-V100-50C-200G, TP=4
seed: 123
fixed evaluator contract changed: no
```

## Change

`qwen3_6_scripts/qwen3_5.py` replaces `q_proj`, `k_proj`, and `v_proj` in
`Qwen3_5FullAttention` with one `QKVParallelLinear`. The gated query uses 32
virtual query heads so its local output remains 2,048 elements. The vendor
layer emits one 256-element K head and one 256-element V head per TP rank,
instead of computing two replicated 512-element K/V tensors and selecting a
rank-local half afterward.

The model loaders map checkpoint tensors ending in `q_proj`, `k_proj`, and
`v_proj` to the `q`, `k`, and `v` shards of `qkv_proj`. Dense and MoE model
loaders use the same helper. Cache layout, attention math, model weights,
sampling behavior, and evaluator arguments are unchanged.

The experiment adds:

```text
tests/test_attention_qgkv_fusion_unit.py
tests/bench_attention_qgkv_projection.py
```

## Correctness gates

- projection loader tests: 4/4 pass
- P0 static tests: 41/41 pass
- complete CoreX unit discovery: 127 pass, 1 skipped
- four-GPU primitive parity: exact zero maximum absolute error on every rank;
  all outputs finite
- quick API smoke: pass
- full API smoke: 15/15 pass
- deterministic API oracle: 3/3 exact for HTTP status, message, finish reason,
  and usage
- candidate health after qualification: HTTP 200
- fixed-argument startup: pass in 329 seconds
- runtime KV cache: 16,903 GPU blocks and 6,553 CPU blocks

The candidate has 32 more GPU blocks than the `b1d9500` baseline (16,871), so
the change does not introduce a KV-capacity regression.

## Primitive results

The probe used the real tensor-parallel rank shapes on all four devices:
hidden size 2,048, 16 query heads, 2 KV heads, head dimension 256, and TP=4.
All fused outputs exactly matched the three-projection reference.

| Shape | Separate projections (ms) | Fused projection (ms) | Speedup |
| --- | ---: | ---: | ---: |
| decode, T=1 | 0.3676-0.3742 | 0.1346-0.1366 | 2.72-2.74x |
| prefill sample, T=64 | 0.1472-0.1489 | 0.0804-0.0814 | 1.82-1.83x |

A control that only removed redundant replicated K/V output, without fusing
the projections, was approximately neutral at T=1 and slower at T=64. The
isolated gain therefore comes from projection fusion rather than output size
alone.

Artifacts:

```text
bench_runs/20260715_E_ATTN_PROJ_01/primitive/gpu0.json
bench_runs/20260715_E_ATTN_PROJ_01/primitive/gpu1.json
bench_runs/20260715_E_ATTN_PROJ_01/primitive/gpu2.json
bench_runs/20260715_E_ATTN_PROJ_01/primitive/gpu3.json
```

## Short strict pairs

Each pair used eight serial streaming requests, 64 generated tokens, seed 123,
prompt repeat 126, and the same salt on baseline and candidate. Every side
reported exactly 14,552 prompt tokens, 1,896 uncached prompt tokens, 12,656
cached tokens, and 512 completion tokens.

| Metric | Pair 1 change | Pair 2 change | Pair 3 change |
| --- | ---: | ---: | ---: |
| TTFT P90 | -26.35% | -14.81% | +14.93% |
| Decode TPS P10 | +0.62% | -3.79% | -1.45% |
| Weighted overlap score | +4.77% | -1.40% | -3.94% |
| Disjoint score | +3.28% | -2.25% | -3.08% |

The same baseline commit varied from 13.0049 to 13.5338 Decode TPS P10 across
startups, which is larger than the candidate effect. The direction reversal
across pairs does not establish an endpoint benefit.

Artifacts:

```text
bench_runs/20260715_E_ATTN_PROJ_01/baseline-P1.json
bench_runs/20260715_E_ATTN_PROJ_01/candidate-P1.json
bench_runs/20260715_E_ATTN_PROJ_01/baseline-P2.json
bench_runs/20260715_E_ATTN_PROJ_01/candidate-P2.json
bench_runs/20260715_E_ATTN_PROJ_01/baseline-P3.json
bench_runs/20260715_E_ATTN_PROJ_01/candidate-P3.json
```

## Robust decode sample

The final decode sample used 16 serial requests and 64 generated tokens per
request. Completion and total prompt tokens match, but the baseline salt had
been partially warmed by an earlier aborted driver. Cache-dependent throughput
and score are therefore excluded from the strict comparison.

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Success rate | 100% | 100% | equal |
| Completion tokens | 1,024 | 1,024 | equal |
| Prompt tokens | 29,104 | 29,104 | equal |
| Cached tokens | 28,928 | 27,120 | not matched |
| Wall time (s) | 90.7664 | 100.0940 | +10.28% |
| TTFT P90 (s) | 0.9625 | 1.0670 | +10.86% |
| Decode TPS P10 | 13.3657 | 13.0887 | -2.07% |

The cache mismatch prevents a score comparison, but post-first-token Decode
TPS P10 regresses on the larger sample.

## Strict long-prefill sample

Both sides used four serial requests, prompt repeat 600, 16 generated tokens,
seed 123, and the exact salt `E-ATTN-PROJ-01-LONG-PREFILL-20260715`. Token and
cache totals match exactly: 33,824 prompt, 8,528 uncached prompt, 25,296 cached,
and 64 completion tokens.

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Success rate | 100% | 100% | equal |
| Wall time (s) | 21.0755 | 22.9772 | +9.02% |
| TTFT P90 (s) | 7.9040 | 9.0042 | +13.92% |
| Decode TPS P10 | 13.5947 | 13.6820 | +0.64% |
| Input TPS | 1,604.8958 | 1,472.0665 | -8.28% |
| Cache TPS | 1,200.2555 | 1,100.9164 | -8.28% |
| Weighted overlap score | 5,392.5828 | 4,966.6305 | -7.90% |
| Disjoint score | 2,033.0676 | 1,885.1656 | -7.27% |

Artifacts:

```text
bench_runs/20260715_E_ATTN_PROJ_01/baseline-P4.json
bench_runs/20260715_E_ATTN_PROJ_01/candidate-P4.json
bench_runs/20260715_E_ATTN_PROJ_01/baseline-LONG.json
bench_runs/20260715_E_ATTN_PROJ_01/candidate-LONG.json
bench_runs/20260715_E_ATTN_PROJ_01/oracle-comparison.json
bench_runs/20260715_E_ATTN_PROJ_01/smoke-full.log
```

## Decision

`REJECT` the production-path change. The fused primitive is exact and 1.82x to
2.74x faster in isolation, but the effect does not survive full-model endpoint
measurement. Decode TPS P10 regresses by 2.07% in the 16-request sample, while
the strict long-prefill sample regresses by 13.92% in TTFT P90 and 7.90% in
weighted score. The candidate's 0.64% long-sample decode gain is too small to
offset the prefill loss and the negative short-pair median. Keep the experiment
branch and benchmark as evidence; do not merge it into `integration/perf-winners`.
