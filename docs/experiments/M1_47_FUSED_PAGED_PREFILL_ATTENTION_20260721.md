# M1-47: Fused paged prefill attention

Status: the first candidate passed all numerical cases but failed the fixed
`1.5x` performance gate; the single split-reduction alternative is unlocked.
Neither candidate is installed or bundled; no runtime, YAML, or `main` change.

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
| ABI maximum-query stress | `(ctx,Q)=(0,8192)` |
| Fused service segments | `(ctx,Q)=(0,8176),(65536,8176),(122880,8176),(229376,5616)` |
| Existing-path fallback tails | `Q=16` for full chunks and `Q=8` at 235K |
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

The benchmark accepts `--extension PATH` and the paired
`--expected-extension-sha256 DIGEST` to load a compiled candidate from an
isolated build directory under the exact native module name. It rejects an
artifact identity mismatch before invoking `forward` and records the resolved
path, size, and digest in the result. This keeps the qualification run
independent of the installed vLLM package and prevents an unqualified binary
from changing the production runtime.

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

With block size 16, the installed `_strict_prefix_query_segments` turns each
full 8,192-token scheduler chunk into a fused-eligible 8,176-token segment and
a 16-token mixed-prefix fallback. The 235,000-token tail becomes 5,616 fused
tokens plus an 8-token fallback. The 8,192-token case remains an ABI-limit
numerical stress because the native entry point accepts it, but service-shape
qualification uses the actual 8,176/5,616 dispatch lengths.

This source is intentionally absent from `patch_ops.sh` and the prebuilt
manifest. It must first compile once on CoreX 3.2.3 and pass every numerical
case plus the three fixed `1.5x` core cases. Failure proceeds only to the one
predeclared split-reduction alternative; no tile, cuBLAS algorithm, tolerance,
or launch scan is permitted.

## Isolated compile gate

The first candidate compiled successfully for `ivcore10` in the isolated
remote runtime at source commit `06963b1`. The resulting shared object is
239,728 bytes with SHA-256
`bd2b0e7283718c503e3c2573851abc9fba3755223c78e6207e88af851993dff5`;
dynamic-link and symbol checks returned zero, and an import-only probe found a
callable `forward`. The probe did not create a CUDA tensor or invoke the entry
point, and it did not install the binary into vLLM. Structured evidence is in
`docs/experiments/evidence/M1_47_COMPILE_GATE.json`.

This passes only the compile gate. The candidate remains unqualified until its
GPU numerical cases and frozen three-point `1.5x` performance grid pass.

## First candidate result

The hash-bound isolated benchmark completed on one Iluvatar BI-V100 without
OOM, illegal access, traceback, or process failure. All 14 numerical cases were
finite. Maximum output absolute error was `6.103515625e-05`, maximum output
relative L2 was `5.207140025e-06`, and every LSE relative L2 was zero. The
invalid physical-block probe was rejected as required.

| Case | Reference | Candidate | Speedup | Decision |
| --- | ---: | ---: | ---: | --- |
| 74K | 69.290 ms | 53.053 ms | 1.3061x | fail |
| 128K | 122.752 ms | 93.971 ms | 1.3063x | fail |
| 235K | 221.170 ms | 168.550 ms | 1.3122x | fail |

The first candidate is `PERFORMANCE_REJECTED`: it is numerically sound but
misses all three predeclared `1.5x` gates. The authoritative remote JSON is
6,780 bytes with SHA-256
`8856f4c9df84482155d1fe98a1453cd7bcb67167ede042e5b02ba4433780317e`;
the safe structured summary is
`evidence/M1_47_FIRST_CANDIDATE_GATE.json`.

No tile, threshold, tolerance, compiler flag, or cuBLAS algorithm will be
scanned. The only remaining implementation is a fixed four-way split over the
existing 512-token partitions: QK/PV remain partition-local, the authoritative
ATen FP32 max/exp/sum operations are batched across four splits, and split
statistics/output are merged in original partition order. This preserves the
512-token reduction partition while amortizing the per-partition eager launch
cost. Failure of this one alternative closes M1-47.

## Fixed split4 alternative result

The single permitted alternative compiled for `ivcore10` and completed the
same hash-bound 14-case benchmark. All outputs and LSE values were finite, the
invalid physical-block probe failed closed, maximum output relative L2 was
`6.12289422136783e-06`, and maximum absolute output error was
`6.103515625e-05`.

| Case | Reference | Candidate | Speedup | Decision |
| --- | ---: | ---: | ---: | --- |
| 74K | 69.472 ms | 31.867 ms | 2.1801x | pass |
| 128K | 123.058 ms | 56.317 ms | 2.1851x | pass |
| 235K | 219.958 ms | 99.715 ms | 2.2059x | pass |

This passes the frozen numerical and `1.5x` core-performance gates. The exact
252,208-byte binary has SHA-256
`e0ff112f965de7126c86a57ba2a64549743ee88c55b25a2396b5f808349ef591`.
The authoritative result is 6,819 bytes with SHA-256
`a1fcd70f1a893911f60ade656362b271641e115ae0a3ddefa21ef291a7276b3f`;
the structured summary is `evidence/M1_47_SPLIT4_GATE.json`.

The status is `CORE_GATE_QUALIFIED; SERVICE_GATE_PENDING`. This result unlocks
hash-pinned runtime integration with the previously frozen shape guards. It
does not qualify the candidate for `main`, change the submission YAML, or
replace the required 65K/235K cold-TTFT, warm-regression, Output TPS, 262K
capacity, and full-workload gates.

## Runtime dispatcher parity

The hash-pinned binary was installed into an isolated copy of the patched
CoreX vLLM package. A real Python-dispatch probe used a padded two-dimensional
block table with 263 columns and a non-identity physical-block permutation.
The dispatcher invoked the native extension exactly once and passed only the
256 active blocks required by the 4,096-token context. Against the installed
PyTorch path, output relative L2 was `4.039600743553589e-06`, maximum absolute
error was `3.0517578125e-05`, and all output values were finite.

The authoritative 840-byte remote result has SHA-256
`7244470f7f06d56dd29eb8df182faa9f907da59492e566861e3d78891e134622`;
the structured copy is `evidence/M1_47_DISPATCH_PARITY.json`. This qualifies
the Python/native interface and block-table slicing, but service-level gates
remain pending.

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
