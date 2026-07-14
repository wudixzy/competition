# E-CTX-01: paged-attention guards and 256K qualification

Date: 2026-07-14

## Objective

Localize the unresolved native decode crash without changing the default
attention dispatch, then qualify the approved evaluator change from a
100,000-token service window to the model-native 262,144-token window.

## Baseline evidence

- Winner baseline: `b1d95009d52135a5b00bbac1c5ccc682c4539644`.
- Installed `paged_attn.py` and `qwen3_5.py` hashes matched repository sources.
- The installed model-runner contains the chunked-prefill prefix-hit reset.
- Crash frame line 277 is `seq_lens.max().item()`, a device synchronization
  point rather than the native paged-attention launch.
- Native V1 handles decode through context length 32,768. Longer decode uses
  the pure-PyTorch fallback because BI100 V2 is unavailable.
- The unchanged baseline passed cold/warm prefix-boundary and 99,500-token
  tests without a native fault. The 99,500-token cold/warm output hashes were
  identical.

The incident is therefore confirmed but not reproduced deterministically.

E-GDN-01 was committed at `05:29 UTC`, the crash occurred around `06:30 UTC`,
and its qualification note was committed at `06:59 UTC`. Because another fault
stack was inside ixformer linear, a parent/winner or E-GDN-01 off/on comparison
remains part of root-cause isolation if the crash reproduces. The timeline is
correlation only; the winner has already completed one 99,500-token run.

## Candidate

The candidate adds host-visible checks before cache writes and decode dispatch:

- key/value token-count agreement;
- slot-mapping count agreement;
- key/value cache block-count and KV-head-count agreement;
- query-head mapping count;
- sequence-count and block-table row/width capacity;
- actual decode length bounded by `max_seq_len`;
- GQA divisibility based on the physical KV cache shape.

The candidate also provides one diagnostic-only knob:

- `BI100_PAGED_ATTN_DIAGNOSTICS=1` checks physical slot and block IDs,
  synchronizes immediately after `reshape_and_cache`, and records sparse
  8,192-token decode metadata snapshots.

The expensive checks and cache-write synchronization default off. The default
dispatch threshold, prefix logic, kernels, and fixed evaluator command remain
unchanged.

## 256K capacity analysis

The model declares `max_position_embeddings=262144`. The observed cache has
16,871 GPU blocks with a 16-token block size:

```text
physical token capacity = 16871 * 16 = 269936
blocks for 256000 tokens = 16000
blocks for 262144 tokens = 16384
```

Both logical windows fit nominally. A 235,000-token prompt that reserves
16,384 output tokens totals 251,384 tokens and fits a 256,000-token service
window. A full 256,000-token prompt cannot also reserve output tokens.

This arithmetic does not qualify startup, peak temporary HBM, prefix-cache
correctness, or performance. Contexts above 32,768 use the PyTorch decode
fallback, whose full-context K/V gathering must be tested on hardware.

## Required gates

1. Static and unit suites, including legacy `head_mapping` interface coverage.
2. Diagnostic TP=4 requests at prompt lengths 32,767, 32,768, and 32,769.
3. Partial and warm prefix-cache boundary equivalence.
4. Cold/warm 99,500-token equivalence with full metadata validation.
5. Startup at `max_model_len=262144` with measured GPU block count.
6. Cold/warm 235,000-token request reserving 16,384 completion tokens.
7. Fatal-log scan and final `/health` check.
8. Default-mode performance comparison before merge.

## Status

Diagnostic hardware qualification passed on the four-card BI100 instance
`ssh-a2d0a302` on 2026-07-15. Default-mode API and performance qualification
remain pending; do not merge until those gates complete.

## Diagnostic results

The TP=4 service started with `max_seq_len=262144`, 16,871 GPU blocks, and
6,553 CPU blocks. The 16-token block size provides 269,936 physical KV tokens.

| Prompt | Cold | Warm | Warm cached | Output hash |
| ---: | ---: | ---: | ---: | --- |
| 32,767 | 35.948 s | 3.647 s | 32,704 | `b6bf19869821ca353e02f53fce527a1210473f92a45b1228546602fce465de48` |
| 32,768 | 35.942 s | 4.132 s | 32,704 | same |
| 32,769 | 36.665 s | 4.813 s | 32,704 | same |

The 32,767 case exercised native V1 at an actual decode length of 32,768.
The 32,768 and 32,769 cases exercised the pure-PyTorch path above that
boundary. All cold/warm responses were equivalent.

The prefix-boundary test reported 8,176 cached tokens on the partial hit and
11,600 on the full warm hit. Their output hash was identical. A forced
1,000-token decode completed in 82.188 seconds with HTTP 200 and
`finish_reason=length`.

### Long-prompt checkpoint retention

The first 235,000-token cold/warm run was stable but exposed a separate cache
retention defect:

| Revision | Cold | Warm | Warm cached |
| --- | ---: | ---: | ---: |
| 16 GDN checkpoints | 520.589 s | 510.317 s | 0 |
| 32 GDN checkpoints (`e1ba860`) | 499.359 s | 41.090 s | 234,544 |

With `max_num_batched_tokens=8192`, a model-native 262,144-token prompt spans
32 staged chunks. Keeping only 16 recurrent-state checkpoints caused the
allocator to find matching KV blocks while the scheduler could not select an
exact GDN state for the first warm chunk. The worker therefore recomputed the
whole prompt.

The scheduler and each TP worker now retain 32 checkpoints. Each checkpoint is
about 16 MB of CPU memory, so the four-rank node uses about 1 GB more host RAM.
The test node had more than 350 GiB available. The fixed warm run was about
12.4 times faster and saved 469.227 seconds. Both runs generated eight tokens,
stopped normally, and returned the same hash:

```text
a3dc73d02269b1b3682ed84197c3d2d0ddc39dfdb544f73fb3ea832f1fb30b4d
```

The diagnostic logs contained both native V1 and PyTorch decode dispatches.
They contained no guard failure, cache-write synchronization failure,
SIGSEGV, fatal Python error, OOM, or worker loss. Final `/health` returned 200.
