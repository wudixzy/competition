# M1-47: Fused paged prefill attention

Status: runtime path and fixed ABI audited; implementation pending M1-45
long-context and M1-46 transfer decisions; no runtime, YAML, or `main` change.

## Corrected scope

The original Stage 3 direction remains valid, but a paged-cache-only kernel
would miss the main cold-TTFT path. On the installed Qwen3.6 TP4 runtime, one
full-attention rank has:

```text
Q: [T, 6, 256] FP16
K: [T, 1, 256] FP16
V: [T, 1, 256] FP16
GQA ratio: 6
```

When there is no cached prefix, the patched xFormers fallback processes fixed
256-query chunks. Each chunk materializes an FP32 score tensor
`[6, 256, T]`, applies a causal mask and softmax, then performs PV. At 235K,
one score tensor is about 1.44 GiB and the mask is about 60 MiB. The ten
full-attention layers account for 68.788% of the measured 235K model time.

The paged prefix path instead reads:

```text
key_cache:   [blocks, 1, 32, 16, 8] FP16
value_cache: [blocks, 1, 256, 16]    FP16
block_table: [logical blocks]        INT32
```

Its current Python loop gathers 512-token paged tiles, computes QK, updates
FP32 online `m/l/o`, and computes PV. The existing
`corex_paged_kv_gather.so` is decode-only: it materializes complete FP32 K/V
and has no Q, causal position, online softmax/LSE, or PV input. Its physical
block indexing can be reused, but it cannot implement the fused target.

## Fixed first ABI

The first candidate supports only one sequence and the immutable Qwen3.6 TP4
shape:

```text
q           [Q, 6, 256]       FP16
k_new       [Q, 1, 256]       FP16
v_new       [Q, 1, 256]       FP16
k_cache     [N, 1, 32, 16, 8] FP16
v_cache     [N, 1, 256, 16]   FP16
block_table [ceil(ctx/16)]     INT32
ctx_len, q_len, scale
out         [Q, 6, 256]       FP16
lse         [Q, 6]             FP32
```

`ctx_len=0` and an empty block table are required supported cases; `k_new` and
`v_new` supply cold-prefill keys and values. Cached prefill concatenates the
logical paged prefix with current K/V without materializing a complete K/V or
logit tensor. LSE is retained by the diagnostic ABI even if production hides
it.

Every other head count, head dimension, dtype, block size, batch size, encoder
mode, or non-causal mode falls back to the current implementation. Decode is
untouched.

## Fixed protocol

No tile, thread, grid, chunk, threshold, compiler-flag, or tolerance scan is
allowed. Both the fused candidate and the one permitted split-reduction
alternative use the same cases:

| Gate | Fixed cases |
| --- | --- |
| Small dense golden | `(ctx,Q)=(0,1),(0,8),(0,256),(240,16)` |
| Paged boundary | `(ctx,Q)=(65520,16),(234992,8)` |
| Core query tile | `Q=256`, total K/V lengths `74K`, `128K`, `235K` |
| Capacity edge | `ctx+Q+max_tokens <= 262144` |
| Service cold | exact 65K and 235K prompts |
| Service warm | same 65K and 235K requests after prefix restoration |

Microbenchmarks use one fixed seed, five warmups, seven CUDA-event trials, and
the installed patched runtime as the reference. Reports persist only shapes,
timings, finite checks, and output/LSE error statistics.

## Gates

- Numerical: no NaN/Inf; output relative L2 at most `1e-5`; LSE relative L2 at
  most `1e-5`; maximum absolute output error at most `1e-3`.
- Core performance: at least `1.5x` at 74K, 128K, and 235K. Passing only a
  gather, QK, softmax, or PV substage does not qualify the end-to-end kernel.
- Service: 65K and 235K cold TTFT improve at least 20%; warm paths regress at
  most 2%; Output TPS P10 remains at least 20.
- Correctness: response finite/deterministic; direct mode retains the existing
  131K equality boundary; aligned mode retains full 235K equality.
- Capacity/stability: 262K remains supported with no OOM, segfault, collective
  failure, or worker loss.

M1-37's persistent online-softmax result cannot be reused: it was fast but
failed long-row numerical parity, including large exponentiated-score and
running-sum errors. The first M1-47 implementation must use a fresh reduction
whose order is explicitly validated against the fixed FP32 streaming
reference. If the fused implementation and the single split-reduction
alternative cannot satisfy both numerical and `1.5x` gates, this direction
stops without launch or tolerance scanning.
