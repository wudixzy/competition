# M1-14 MRoPE Chunk Alignment - 2026-07-16

## Evaluation evidence

The supplied 881-request evaluation completed only 269 requests successfully:

```text
successful_requests 269
error_requests      612
error_rate          0.69
output_tps_p10      4.03
ttft_p90_ms         29706.12
cache_hit_rate      0.42
```

The Docker log contains an engine-fatal model execution error:

```text
RuntimeError: shape '[26540, -1, 256]' is invalid for input of size 16384
```

The traceback enters `Qwen3_5InterleavedMRotaryEmbedding.forward`. The MRoPE
position tensor describes 26,540 tokens while the current physical query is a
small chunk. The asynchronous engine then propagates the exception to the
stream generator; later requests cannot be treated as independent API 4xx
failures.

## Root cause

The vendor vLLM 0.6.3 model-input builder maintains two position forms:

- `input_positions`, cropped to the scheduled chunk and then cropped again on
  partial or full prefix-cache hits;
- `mrope_input_positions`, generated from the complete multimodal token list.

`MRotaryEmbedding.get_input_positions(..., context_len=N)` returns every
position from `N` through the end of the full request. It does not know the
current `seq_len`. The original builder therefore supplied full-request MRoPE
positions to a chunked physical query. It also cropped normal positions on a
prefix hit without cropping already-created MRoPE positions.

This violates the model invariant:

```text
positions.shape[-1] == physical_query_tokens
```

and fails before attention, MoE, or GDN kernels execute.

## Fix

`patch_model_runner.py` now applies three related changes:

1. Compute the complete multimodal MRoPE map with `context_len=0`, preserving
   the request-level `mrope_position_delta`.
2. Slice that map to the exact physical interval
   `[inter_data.context_lens[i]:inter_data.seq_lens[i]]`.
3. Apply the same suffix/last-token crop when partial/full prefix-cache logic
   further reduces the physical query.

The injected helper verifies all three axes against
`len(inter_data.input_tokens[i])` and raises a host-side error before GPU model
execution if the invariant is violated. The existing block-table continuity
fix remains unchanged.

## Current gates

- fixed 26,540-token/64-token regression: pass;
- partial prefix-hit MRoPE crop: pass;
- full prefix-hit last-token crop: pass;
- mismatch fail-fast: pass;
- patch idempotency: pass;
- local non-GPU suite: 159 pass, 22 environment skips;
- real CoreX vendor `model_runner.py` copy: all anchors apply, second patch is
  byte-idempotent, `py_compile` passes;
- real vendor `MRotaryEmbedding.get_input_positions` semantic probe: a
  synthetic 26,540-token image request produces three 26,540-element axes;
  the injected helper crops interval `[26476:26540]` to exactly `64/64/64`
  while preserving the request delta (`-1240`);
- dedicated streaming `tests/mrope_chunk_api.py` probe: sends an image plus a
  prompt over 8,192 tokens twice and checks usage, cache hit, output hash, and
  post-request health;
- TP4 long-image chunk/prefix API regression: pending.

## Independent long-context decode evidence

Before changing the production runtime, the qualified M1-12 service was swept
with one warm-up request followed by a strictly serial 64-token measured
request at each context length. Every request kept `/health` at HTTP 200 and
the server log contained no fatal, OOM, Gloo, or worker-loss marker.

| Prompt tokens | Cached tokens | 64-token elapsed | Output TPS |
|---:|---:|---:|---:|
| 32,768 | 32,752 | 6.282 s | 10.188 |
| 65,536 | 65,504 | 9.112 s | 7.024 |
| 131,072 | 131,056 | 12.500 s | 5.120 |
| 235,000 | 234,992 | 17.308 s | 3.698 |

The evaluation `Output TPS P10=4.03` is therefore reproducible without an
engine failure and is strongly context-length dependent. Prefix fast-forward
already reduces each measured request to a tiny suffix, so scheduler/YAML
tuning cannot remove the remaining cost. The next performance experiment must
target the `>32768` paged decode implementation itself; the current runtime
falls back to a PyTorch gather/attention path there.

## Decision boundary

Do not merge until a TP4 image request that crosses a prefill chunk completes
cold and warm with HTTP 200, aligned position/query trace, identical output,
and a healthy engine afterward. This correctness fix takes priority over the
M1-14 WMMA/paged-attention capability probe. Long-context decode performance
remains a separate problem: the controlled 32K-235K sweep reproduces the
evaluation P10 and makes a direct paged-decode kernel the next performance
priority after service stability is restored.
