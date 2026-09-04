# M1-179 FP16-QK incremental distribution attribution

## Decision

The experiment is valid and the distribution result is
`incremental_fp16_qk_distribution_drift / inconclusive`. M1-109 control A and
control B were identical at all 256 sampled positions, while M1-162 differed
from M1-109 at 11 top-1 positions, including eight whose M1-109 margin exceeded
the calibrated 0.1-nat threshold. This is a material FP16-QK incremental drift,
not baseline restart noise.

M1-162 promotion stops here. M1-109 remains the retained fused-prefill
candidate. This is a distribution finding, not an operator-numeric or task
capability failure. No capability suite, performance rerun, long
teacher-forced workload, formal 881 run, default-selector change, YAML change
or main merge was performed.

The privacy-safe machine summary is
`docs/experiments/evidence/M1_179_FP16_QK_INCREMENTAL_DISTRIBUTION_20260905/summary.json`.
Private token identities and raw service logs remain only in the remote `/tmp`
experiment root.

## Design and identity

The fixed full model ran three sequential TP4 FP16 services in one orchestrator
lifetime:

1. control A: `m1_109_fp32_qk`;
2. candidate: `m1_162_fp16_qk`;
3. control B: `m1_109_fp32_qk`.

All three arms set `BI100_ATTN_COREX_FUSED_PREFILL=1`. The only attention
algorithm difference was the external extension built from
`corex_fused_paged_prefill_split4.cu` or
`corex_fused_paged_prefill_fp16_qk.cu`. Both extensions used the same CoreX
3.2.3 clang 16.0.6, Torch 2.1.0 ABI and compiler flags. Runtime import
introspection and four-rank dispatch markers recorded the selected variant and
actual `.so` path. The two extension SHA-256 values were retained because these
were the only precompiled cross-process artifacts; no source/report/overlay
tree hash gate was added.

- evidence source: `653b90b483ee6f704481d9a8d32f52c182873cd2`;
- source dirty summary: `clean`;
- instance: `cc-ce19242b-436f-4141-868b-610eb3ac8cee-0`;
- model/tokenizer: `/root/public-storage/models/Qwen/Qwen3.6-35B-A3B`;
- runtime overlay:
  `/tmp/m1-176-focused-runtime.ihZwyu/runtime/site-packages`;
- runtime: Python 3.10.12, CoreX 3.2.3, Torch 2.1.0, vLLM 0.6.3,
  Transformers 4.55.3 and clang 16.0.6;
- reused preflight: `m1-176-focus-preflight-4e2b2e7`, four GPUs, FP16
  matmul and TP4 collective qualified;
- M1-109 extension SHA-256:
  `b7b30f8c3c3af0153c58dde4760159dbdfeeec17fd352b192ae57990ad1a0be8`;
- M1-162 extension SHA-256:
  `3724d6651eed814b84043d0b0155cfb381baab02e9db56665116dc8e016f2f91`.

The service command, model, tokenizer, TP=4, FP16 dtype,
`max_model_len=262144`, block size 16, sampling semantics and all non-variant
environment fields matched. Each arm completed one cold request at 4K, 16K,
32K and 64K, with 64 fixed positions per request, `max_tokens=1`,
`temperature=0`, fixed seed, non-streaming mode and `cached_tokens=0`.
All 12 requests returned HTTP 200 with complete usage and
`finish_reason=length`. Dispatch counts were four for every arm with the
expected variant.

## A/A distribution

M1-109 control A versus independently started M1-109 control B was exact over
the sampled population:

- sampled positions: 256;
- top-1 agreement: 100%;
- top-1 flips and high-margin flips: 0 / 0;
- mutual top-k coverage: 1.0;
- teacher-token and shared-token absolute logprob P99: 0 / 0 nats;
- paired mean NLL difference: 0 nats;
- position-sampling upper diagnostic: 0 nats;
- per-length mean NLL difference: 0 nats at 4K, 16K, 32K and 64K;
- non-finite values: 0.

Thus the actual independent-service A/A envelope is zero for this fixed
workload. It does not support a baseline nondeterminism or measurement-noise
explanation for the candidate differences.

## M1-162 incremental distribution

M1-162 versus M1-109 produced:

- sampled positions: 256;
- top-1 agreement: 95.703125% (245/256);
- top-1 flips: 11;
- high-margin flips above `max(0.1, 4 * A/A P99) = 0.1` nats: 8;
- mutual top-k coverage: 0.909091;
- teacher-token absolute logprob delta P99: 5.776591 nats;
- shared-token absolute logprob delta P99: 4.591725 nats;
- paired candidate-minus-control mean NLL: -0.083481 nats;
- position-sampling one-sided upper diagnostic: +0.043299 nats;
- NLL threshold: `max(0.01, 2 * A/A upper) = 0.01` nats;
- all observed logprobs finite.

| Prompt tokens | Positions | Candidate minus M1-109 mean NLL | Position-sampling upper diagnostic |
| ---: | ---: | ---: | ---: |
| 4,096 | 64 | +0.052221 | +0.270714 |
| 16,384 | 64 | +0.096181 | +0.250308 |
| 32,768 | 64 | -0.164066 | -0.010792 |
| 65,536 | 64 | -0.318260 | +0.071207 |

The negative aggregate mean does not waive the local 4K and 16K regressions,
the positive sampling upper diagnostic or the high-margin flips. The 11 flips
occurred at:

| Length | Position | M1-109 top-1 margin (nats) |
| ---: | ---: | ---: |
| 4,096 | 178 | 0.460938 |
| 16,384 | 451 | 0.328125 |
| 16,384 | 643 | 0.062500 |
| 16,384 | 965 | 0.343750 |
| 16,384 | 1,864 | 0.750000 |
| 16,384 | 3,172 | 0.171875 |
| 16,384 | 14,798 | 0.031250 |
| 32,768 | 515 | 0.062500 |
| 32,768 | 17,969 | 0.328125 |
| 32,768 | 31,710 | 3.500000 |
| 65,536 | 42,281 | 4.906250 |

The first divergent sampled position was length 4K, position 178, with a
0.460938-nat M1-109 margin.

The bootstrap resamples positions inside each of the four fixed length strata.
It is explicitly a position-sampling diagnostic, not a run-to-run or service
population confidence interval: each arm has only one request per length. The
formal attribution is driven primarily by the exact A/A envelope, high-margin
flips, per-length NLL, finite checks and verified variant identity.

## Relation to M1-178

M1-178 compared fused-off with the combined M1-109 plus M1-162 path, so its
drift could not be assigned solely to FP16-QK. M1-179 now shows that the
M1-162 increment over M1-109 independently creates same-order multi-nat
logprob changes and high-margin flips while M1-109 A/A is exact. This is enough
to attribute a material distribution drift to FP16-QK and stop M1-162.

It does not prove that every M1-178 difference came from FP16-QK; a simultaneous
fused-off arm was intentionally outside this three-arm budget. A later audit of
M1-109 versus fused-off remains separate reviewer work. The old M1-178 HMAC
identity was not reused or reconstructed.

## Lifecycle and resource budget

- control A wall: 674.002 seconds;
- candidate wall: 671.330 seconds;
- control B wall: 740.211 seconds;
- total funnel wall: 2203.379 seconds;
- services: 3;
- teacher-forced model requests: 12;
- old 72-plus-4 three-arm path: 228 requests;
- requests eliminated: 216 (94.74%).

Every arm used scoped TERM-first cleanup with a 60-second grace period. No arm
required SIGKILL. Each service parent waited/reaped its child process tree.
Per-arm and final postflight found no API server, worker or GPU process; all
fatal/OOM/segfault/timeout/collective-reset/worker-loss counts were zero.

## Layered conclusions

- Experiment validity: `pass`; identities, request populations, dispatch,
  lifecycle and evidence contracts are complete.
- A/A distribution: `pass`; the observed M1-109 restart envelope is exact zero.
- Incremental distribution: `inconclusive /
  incremental_fp16_qk_distribution_drift`; it requires adjudication and blocks
  M1-162 promotion.
- Operator numerics: M1-176 G2 `pass` retained; not rerun and not overridden.
- Capability: not run; no capability conclusion.
- Performance: historical M1-176/M1-177 gains retained; not rerun.
- Promotion: M1-162 stopped. M1-109 retained as the next candidate; no further
  experiment is automatically authorized by this result.

Work stops here for reviewer review.
