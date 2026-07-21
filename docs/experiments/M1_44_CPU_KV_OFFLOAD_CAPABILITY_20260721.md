# M1-44 CPU KV Offload Transfer Capability

## Objective

M1-41 cannot be qualified without a complete 881-request trace, while the
remaining BI100 attention, GDN, and MoE kernel candidates have exhausted their
predeclared numerical or performance gates. M1-44 tests one structural cache
direction before any scheduler change: preserve content-addressed full-attention
KV blocks in the existing pinned CPU cache after GPU eviction, then promote an
exact contiguous prefix through vLLM's production `swap_blocks` path.

This can increase effective cache capacity without changing model arithmetic.
It is not enabled in `computility-run.yaml`, and this capability probe does not
modify runtime code.

## Fixed Production Shape

The benchmark uses one TP4 rank of Qwen3.6-35B-A3B:

| Field | Value |
| --- | ---: |
| Full-attention layers | 10 |
| KV heads per rank | 1 |
| Head dimension | 256 |
| Block size | 16 tokens |
| Dtype | FP16 |
| Bytes per block per rank | 163,840 |

`tests/bench_cpu_kv_offload_transfer.py` allocates all ten paged-KV layers in
GPU memory and pinned CPU memory with the exact
`PagedAttention.get_kv_cache_shape` layout. It calls the installed CoreX vLLM
`PagedAttention.swap_blocks` implementation for every layer using the same CPU
`int64 [N, 2]` mapping format as `Worker.execute_worker`. D2H is measured before
H2D in the natural lifecycle, with alternating order across three measured
cycles to expose order sensitivity. First, middle, and last blocks are checked
exactly for every layer and both K/V planes.

## Predeclared Gate

The fixed gate mode tests 4K, 16K, 65K, and 131K strict block-aligned prefixes.
It passes only when:

- every representative block is bit-exact after D2H and after destructive
  overwrite followed by H2D;
- every timing is finite and positive;
- 65K median D2H and H2D are each at most 2,000 ms;
- 131K median D2H plus H2D is at most 5,000 ms.

The 131K round-trip bound reserves the competition's five-second TTFT budget
even for a conservative synchronous implementation. A separate 4K smoke mode
can validate the API but can never qualify the candidate. The protocol has no
tile, batch, cache-size, or transfer-threshold scan.

## Integration Contract If The Gate Passes

The next implementation must remain default-off and scheduler-owned:

- CPU identity is the existing chained SHA-256 logical content key, never a
  physical GPU block number;
- GPU eviction first reserves a CPU destination and emits D2H work, then a CPU
  hit reserves a GPU destination and emits H2D work;
- D2H executes before H2D when both maps occur in one engine step so a reused
  GPU slot cannot overwrite its source before preservation;
- only the longest prefix with contiguous KV and a matching recoverable GDN
  state contributes to `cached_tokens`;
- CPU capacity and eviction are fixed before TP4 A/B; no YAML scan is allowed.

## Status

`IMPLEMENTED; REMOTE_CAPABILITY_GATE_PENDING`.

The current a163074c instance has healthy GPU 0-2 but GPU3 still times out, so
M1-44 may run as a rank-local transfer capability test on GPU0. No TP4 service
or qualification conclusion is allowed until all four GPUs pass CUDA and
collective preflight.
