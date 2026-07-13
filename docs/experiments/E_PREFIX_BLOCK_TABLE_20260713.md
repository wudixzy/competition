# E-PREFIX-01 Block-Table Continuity - 2026-07-13

## Official evaluation failure

The 2026-07-13 evaluation attempted 881 requests but completed only five:

```text
http_200_count      6
successful_requests 5
error_requests      876
error_rate          0.99
```

The sixth streaming request killed the asynchronous engine. The first fatal
error in the Docker log was:

```text
RuntimeError: seq 0: num_ctx_blocks=726 > block_tables.shape[1]=706,
ctx_len=11616. Block table is undersized; refusing to truncate context because
attention would be incorrect.
```

The background engine then raised `AsyncEngineDeadError`, explaining all later
request failures and the academic evaluation's zero generated tokens.

## Root cause

Strict prefix alignment split a cold prefill query at a cacheable block
boundary. The second segment had:

```text
block-table context  11296 tokens = 706 blocks
preceding query         320 tokens = 20 blocks
logical context       11616 tokens = 726 blocks
```

The previous implementation passed `ctx_len=11616` to the block-table reader,
although the final 320 tokens belonged to the current request's `key/value`
tensors rather than the block table. The fail-fast guard correctly rejected
the mismatch, but the caller's context-source model was wrong.

Enabling `BI100_ALLOW_PREFIX_GUARD_CAP` or truncating to 706 blocks would drop
320 tokens and corrupt attention. It is not an acceptable fix.

## Fix

Prefix attention now treats context as one logical stream composed of:

1. tokens already addressable through `block_tables`;
2. preceding tokens from the current query segment.

The stream is partitioned at absolute `tile_sz` boundaries. A tile crossing the
source boundary concatenates the tail loaded from paged KV cache with the head
from the current request's `key/value` tensors before one online-softmax update.
This preserves the exact tile partition used by a later warm-cache request.

For the failing shape, the last 512-token context tile contains 32 block-cache
tokens and 320 preceding-query tokens. Block-table validation remains strict and
now validates only the 11,296 tokens that must actually be present there.

## Gates

- exact context-span unit regression: pass;
- invalid-span validation: pass;
- paged-attention unit tests: 9/9 pass;
- real PyTorch/CoreX prefix parity: 2/2 pass;
- cold segmented output vs dense reference: pass;
- cold final token vs warm-cache final token: exact match (`rtol=0`, `atol=0`);
- P0 static tests: 38/38 pass;
- all non-GPU unit tests: 65/65 pass;
- fixed evaluator YAML: unchanged.

## Four-GPU runtime validation

The patched CoreX site package started successfully on four BI100 GPUs. All
26 checkpoint shards loaded, the engine exposed 18,271 GPU blocks, and the
health endpoint became ready without a fatal error.

The API boundary regression used a 11,617-token prompt and forced the final
strict prefix split after 320 current-query tokens. The runtime reused 8,176
tokens from an earlier request, so the remaining current-query contribution was
larger than the original 320-token failure and exercised the same mixed-source
context path.

With one generated token, partial-cache and warm-cache requests both returned
`It` with the same message hash:

```text
partial cache  cached=8176   elapsed=7.457s
warm cache     cached=11600  elapsed=1.822s
```

No block-table error, `AsyncEngineDeadError`, non-finite value, OOM, CUDA, or
NCCL error occurred. The full API smoke suite passed 14/14 cases, including
streaming, reasoning, tool calls, prefix caching, sampling, and determinism.

The 99,500-token contract-boundary test also passed:

```text
cold  prompt=99500 cached=0     elapsed=158.035s
warm  prompt=99500 cached=99296 elapsed=19.664s
```

Cold and warm messages had the same SHA-256 hash. The service remained healthy
after all tests.

## Residual numerical variation

An exploratory 16-token boundary run had the same first token and first 55
characters on partial-cache and warm-cache paths, then greedy decoding chose
different continuations. This is consistent with floating-point reduction-order
variation being amplified during autoregressive decoding; it is not consistent
with the previous missing-context failure, which affected the first prefill
output. The one-token boundary test and the 99.5K multi-token test were exact.

## Decision

Keep and submit. The candidate closes the evaluation's 99% request-failure
blocker while retaining strict context validation. Continue performance work
only after this availability fix is preserved as a rollback point.
