# M1-47: Fused paged prefill attention

Status: runtime path and fixed ABI audited; first compiled-pipeline source is
ready for CoreX qualification but is not installed or bundled; no runtime,
YAML, or `main` change.

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

For the first request chunk, the patched xFormers fallback processes fixed
256-query tiles. Each tile materializes an FP32 score tensor `[6, 256, Q]`,
applies a causal mask and softmax, then performs PV. The fixed
`max_num_batched_tokens=8192` command bounds this initial dense `Q` at 8192;
the runtime does not submit one 235K dense query tensor.

The paged prefix path instead reads:

```text
key_cache:   [blocks, 1, 32, 16, 8] FP16
value_cache: [blocks, 1, 256, 16]    FP16
block_table: [logical blocks]        INT32
```

After the first chunk, chunked prefill supplies an increasingly long paged
context and at most 8192 current query tokens. Its current Python loop gathers
512-token paged tiles, computes QK, updates FP32 online `m/l/o`, and computes
PV. The ten full-attention layers account for 68.788% of the measured 235K
model time, so this paged-prefix loop is the primary service-level target. The
existing
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

`ctx_len=0` and an empty block table are required supported cases for the first
chunk; `k_new` and `v_new` supply its keys and values. Later chunks concatenate
the logical paged prefix with at most 8192 current K/V tokens without
materializing a complete K/V or logit tensor. LSE is retained by the
diagnostic ABI even if production hides it.

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
| Core query tile | `Q=256`, paged contexts `74K`, `128K`, `235K` |
| Service chunk geometry | `(ctx,Q)=(0,8192),(65536,8192),(122880,8192),(229376,5624)` |
| Capacity edge | `ctx+Q+max_tokens <= 262144` |
| Service cold | exact 65K and 235K prompts |
| Service warm | same 65K and 235K requests after prefix restoration |

Microbenchmarks use one fixed seed, five warmups, seven CUDA-event trials, and
the installed patched runtime as the reference. Reports persist only shapes,
timings, finite checks, and output/LSE error statistics.

`tests/bench_fused_paged_prefill_attention.py` encodes the frozen numerical
and core microbenchmark subset and fails closed until a
`vllm.corex_fused_paged_prefill.forward` implementation exists. Service TTFT,
warm-path, decode, and capacity gates remain separate runtime qualifications.
The script is qualification infrastructure, not evidence that a candidate has
passed.

## First candidate

`qwen3_6_scripts/corex_fused_paged_prefill.cu` implements the frozen shape as
one compiled streaming pipeline. It gathers each physical KV tile directly
into fixed FP32 workspaces, runs six-head strided-batched SGEMMs for QK and PV,
and updates FP32 online max, sum, output, and LSE without allocating complete
sequence logits. The reduction uses the installed ATen FP32 max/exp/sum path;
it does not reuse M1-37's numerically rejected persistent reduction. Context
and current causal K/V retain separate 512-token phases to match the installed
reference partitioning. A fixed `65K context + 8192 query` numerical case
guards the real chunk row count in addition to the small boundary cases. Every
paged case uses a deterministic non-identity physical block permutation, and
the native entry point rejects out-of-range block IDs before launching a
paged read.

The first ABI represents only KV already resident in the physical cache plus
the current segment's K/V. It cannot represent the `prefix_key` carried into a
second strict-prefix segment. Eventual runtime dispatch is therefore limited
to segments with empty `prefix_key` and `q_len > 16`; mixed-prefix segments and
the 16-token cold/warm boundary stay on the installed PyTorch path. This guard
is part of the production gate and is not enabled before the core candidate
qualifies.

This source is intentionally absent from `patch_ops.sh` and the prebuilt
manifest. It must first compile once on CoreX 3.2.3 and pass every numerical
case plus the three fixed `1.5x` core cases. Failure proceeds only to the one
predeclared split-reduction alternative; no tile, cuBLAS algorithm, tolerance,
or launch scan is permitted.

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
