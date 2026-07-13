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

## Decision

Candidate keep, pending four-GPU service restart, 11.6K regression, full smoke,
and long cold/warm output comparison. Performance optimization remains paused
until the 881-request availability blocker is closed.
