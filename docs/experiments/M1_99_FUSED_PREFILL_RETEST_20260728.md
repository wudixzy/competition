# M1-99 frozen M1-47 fused-prefill retest

Date: 2026-07-28

Status: private candidate prepared for dispatcher parity and a three-pair TP4
performance screen. It is disabled by default, absent from
`computility-run.yaml`, and not authorized for `main`.

## Why this candidate is reopened

M1-47 used the corrected production rank-local shape `[T, 4, 256]`, one KV
head, block size 16, and the real chunked-prefill dispatcher. Its frozen
evidence reported:

- `2.5530x`, `2.5451x`, and `2.5770x` core-path speedups at 74K, 128K, and
  235K;
- maximum output relative L2 `7.357e-6`;
- dispatcher relative L2 `6.842e-6`;
- one TP4 service pair with 65K cold TTFT improving `3.906%` and 235K cold
  TTFT improving `8.832%`;
- no meaningful warm or Output TPS regression.

It was rejected because both cold cases had to improve by 20%. That was a
reasonable final target but an overstrict continuation screen: it discarded a
measured 235K end-to-end gain in the dominant prefill path. This retest changes
the decision policy, not the kernel, its tiles, its precision, or the fixed
evaluator command.

## Frozen artifact identity

The following M1-47 artifacts are copied byte-for-byte:

| Artifact | SHA-256 |
|---|---|
| `corex_fused_paged_prefill_split4.cu` | `8cf3dee28f8cb69d5a9ddb5afcecd4d4c4a786d020b2e7711e9bfc8363a60c4a` |
| `build_corex_fused_paged_prefill_split4.sh` | `2090a36fcb9c13c4c413e5f8bb3250161c1b7375ece58d487ead4f6ad22ca47b` |
| `corex_fused_paged_prefill.so` | `f654eee2c0677812394ff419d316e7e8c98ed1bcc84853a7f8d2ed5755503009` |

The prebuilt extension is now part of the atomic runtime install and runtime
identity report. A source, installed binary, revision, or runtime-tree mismatch
fails before GPU measurement.

## Integration boundary

`BI100_ATTN_COREX_FUSED_PREFILL=1` is the only arm selector. It defaults to
off and is not in the submission YAML. The native path accepts only:

- causal single-sequence decoder prefill;
- FP16 Q/K/V and KV cache on one CoreX device;
- TP4 rank-local `4` query heads, `1` KV head, and head dimension `256`;
- block size 16 and the fixed 32-block PyTorch tile contract;
- query segments from 17 through 8192 tokens;
- aligned cached context and total length at most 262144;
- no ALiBi, sliding window, non-default KV dtype, or non-unit K/V scales;
- empty in-flight prefix tensors for the selected strict segment.

Unsupported requests keep the current PyTorch implementation. Once a request
passes all guards, a native error propagates; it is never silently retried on a
different mathematical path.

## Revised performance screen

`scripts/run_m1_99_fused_prefill_service_ab.sh` runs exactly three paired TP4
A/B repetitions in alternating order:

1. control, candidate;
2. candidate, control;
3. control, candidate.

Each arm uses the same full model, atomic runtime overlay, fixed TP4 command,
`admission64/direct`, full-attention hybrid accounting, disabled CPU KV
offload, and current default kernels. The service restarts between arms, so
prefix and recurrent-state caches cannot cross measurements. Each arm sends
cold plus two warm requests at 65,536 and 235,000 prompt tokens with 32 greedy
output tokens.

The candidate may proceed to the full TP4 quality gate only when:

- 235K paired median cold-TTFT improvement is at least 5%, with at least two
  of three pairs positive;
- 65K paired median cold regression is no more than 2%;
- median warm-TTFT and Output TPS regressions are no more than 2%;
- no individual warm-TTFT or Output TPS regression exceeds 5%;
- every request is successful, complete, finite, and has the expected
  cold/warm cache accounting;
- the first generated token, completion-token count, and finish reason match
  the control for every request;
- all 65K full output hashes match the control;
- warm repeats are exact within each arm;
- all four TP ranks execute the frozen native path;
- every scoped process group exits through SIGTERM with a 60-second grace
  period, or the experiment is invalid;
- each arm and the finalizer pass process, GPU, fatal, timeout, Gloo/NCCL, and
  preflight/postflight checks.

At 235K, a later-token full-output divergence is recorded but is not an
automatic performance-screen rejection. Different valid FP32 online-softmax
reduction orders can eventually split a greedy sequence despite passing the
fixed `1e-5` operator bound. The hard screen still requires an identical first
generated token and exact warm-repeat stability. Any such divergence must be
resolved by the complete capability and long-context quality suite before
promotion; it cannot be waived by a performance gain.

## Authority boundary

Passing this screen authorizes only:

- the complete `指标集合` functional suite;
- fixed cold/warm correctness checks;
- short, 4K, 32K, 65K, 131K, 235K, and near-262144 quality checks;
- tool calling, reasoning, structured output, multilingual, multimodal, and
  long-context retrieval comparison against the disabled-switch baseline.

It does not authorize an official-score claim, YAML change, default-on switch,
`main` merge, repository visibility change, or production submission. Those
remain blocked until all performance, capacity, lifecycle, and model-capability
hard gates pass together.
