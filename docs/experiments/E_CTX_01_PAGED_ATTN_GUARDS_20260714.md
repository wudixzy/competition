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

In progress. Do not merge until hardware gates and default-mode performance
qualification complete.
