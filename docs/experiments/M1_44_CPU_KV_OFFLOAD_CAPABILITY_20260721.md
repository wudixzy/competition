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

`CAPABILITY_GATE_PASSED; RUNTIME_INTEGRATION_AUTHORIZED`.

The current a163074c instance has healthy GPU 0-2 but GPU3 still times out, so
M1-44 may run as a rank-local transfer capability test on GPU0. No TP4 service
or qualification conclusion is allowed until all four GPUs pass CUDA and
collective preflight.

The first 4K smoke exposed a vendor interface mismatch before any transfer:
vLLM 0.6.3 called `ixformer.functions.swap_blocks`, while CoreX 3.2.3 exposes
the public `vllm_swap_blocks(src, dst, mapping)` function. The latter accepts a
dictionary, whereas the worker emits a CPU `int64 [N, 2]` tensor. The
idempotent `patch_corex_swap_blocks.py` now selects the available public symbol,
validates and normalizes the mapping, and fails if neither symbol exists. It is
installed by `patch_ops.sh`; malformed, duplicate, negative, non-CPU, and
non-int64 mappings are rejected rather than hidden.

## CoreX Result

The fixed gate ran once on healthy GPU0 with `CUDA_VISIBLE_DEVICES=0`; it did
not time out or OOM. Every sampled first, middle, and final block was bit-exact
for all ten layers and both K/V planes after D2H and destructive overwrite plus
H2D.

| Tokens | Bytes/direction | D2H median | H2D median | Round trip | D2H | H2D |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 40 MiB | 29.590 ms | 32.564 ms | 62.154 ms | 1.320 GiB/s | 1.200 GiB/s |
| 16,384 | 160 MiB | 118.196 ms | 132.598 ms | 250.794 ms | 1.322 GiB/s | 1.178 GiB/s |
| 65,536 | 640 MiB | 472.399 ms | 532.578 ms | 1,004.977 ms | 1.323 GiB/s | 1.174 GiB/s |
| 131,072 | 1.25 GiB | 945.000 ms | 1,065.420 ms | 2,010.420 ms | 1.323 GiB/s | 1.173 GiB/s |

The 65K one-way limits pass with at least 1.46 seconds of margin, and the 131K
round trip passes the five-second gate with 2.99 seconds of margin. The result
therefore authorizes a default-off scheduler integration; it does not qualify
TP4 behavior, cache hit improvement, TTFT, or final score.

Evidence:

- `docs/experiments/evidence/M1_44_CPU_KV_OFFLOAD_GATE.json`
  (`99b148cf61d4a0fc2444fa3af4e105360666a795e1b0eac23fa195df05970ddf`)
