# T8 GDN Prefix Boundary Correctness Issue - 2026-07-12

## Trigger

While validating `07174fc`, two identical requests used seed 123 and
temperature 0. The second request had a real prefix-cache hit:

```text
prompt tokens:       3678 / 3678
cached tokens:          0 / 3664
completion tokens:     32 / 18
response hashes: different
finish reason:       length / stop
```

The full smoke suite passed because its prefix test checks only that the second
request reports cached tokens; it does not compare uncached and cached output.

Artifacts:
`bench_runs/20260712_025700_07174fc_T8_validation`.

## Root cause

vLLM prefix caching reuses complete 16-token blocks. For a 3,678-token prompt,
the reusable prefix therefore ends at token 3,664.

The current GDN cache saves state after the entire 3,678-token prefill but keys
that state with `total_processed // 16`, i.e. the 229 complete physical blocks
ending at token 3,664. On the next request it restores the state after token
3,678 at context position 3,664, then processes the final 14 tokens again.
The GDN recurrent state no longer matches the KV-cache boundary.

This bug predates `07174fc`: that experiment only skips a restore on a second
8192-token chunk, which this 3,678-token request never has. The unproven T8
experiment was reverted by `42fc9b7` to restore the retained T7 winner.

## Required correction

A cache entry must contain conv and temporal state captured at exactly the same
complete-block boundary represented by its key. Viable implementation work
must capture per-layer GDN state before the final partial block, then prove:

1. uncached and cached requests produce identical token sequences;
2. aligned and unaligned prompt lengths both work;
3. chunked prefill across 8,192-token boundaries works;
4. interleaved prefixes and eviction do not cross-contaminate state;
5. numerical parity and full API smoke remain green.

Disabling prefix caching would restore correctness but discard a workload lever
covering about 65.6% of input tokens, so it is only a diagnostic fallback.

## Boundary-capture follow-up

Commits `c05fd52` and `9f95cb5` added a scheduler checkpoint mirror and captured
all GDN states at strict block boundaries. Unit tests, MoE/GDN parity, hardware,
NCCL, startup, and full smoke passed. A reconstructed 3,663-token case also
matched exactly with 3,648 cached tokens.

The original 8,712-token payload still diverged with an 8,176-token checkpoint:

```text
uncached: completion=18, finish=stop
cached:   completion=32, finish=length
```

DEBUG logs proved that all four ranks restored the intended GDN checkpoint.
The remaining mismatch is the full-attention partition: the uncached request
computes the last 16 tokens inside an 8,192-token current chunk, while cached
replay computes them as 8,176 context tokens plus 16 current tokens. The custom
online-softmax fallback uses a different reduction partition in those paths.

Both follow-up commits were reverted (`d837caa`, `4161d3f`). A complete solution
must partition uncached full attention at the same strict checkpoint boundary as
GDN, so suffix tokens use identical context/current tiling before their state is
cached. This touches the core paged-attention fallback and requires explicit
approval plus dedicated attention parity tests.

## Final resolution

After approval, the boundary changes were reapplied as `0e52374` and `0ec0607`.
`b22fd8f` splits each full-attention query at the strict prefix-cache boundary,
so uncached and cached suffixes use the same context/current online-softmax
partition. The first CoreX parity run exposed an independent fp32 aliasing bug:
`.float().mul_(scale)` modified an already-fp32 query in place. `a63a1ef`
changed this to non-mutating scaling and added input-immutability coverage.

The final gates passed:

- paged-attention unit 7/7 and real CoreX attention parity 1/1, no skips;
- GDN capture 4/4, scheduler 6/6, MoE parity 3/3, GDN parity 2/2;
- CUDA GPU0-3 and NCCL rank0-3;
- exact fixed-command startup without GPU-block or performance overrides;
- full API smoke 14/14 with exact cached/uncached message comparison;
- aligned 10,592-token and unaligned 10,599-token interleaved prefixes;
- 17-prefix LRU pressure, safe miss after eviction, and hit after refresh.

The original 8,712-token payload now gives identical messages, finish reasons,
and token counts for `cached_tokens=0` and `cached_tokens=8176`. Request time was
19.53 s uncached and 6.62 s cached. Stress testing left all four GPU memory
figures unchanged. Process-group RSS rose about 392 MiB while filling the state
LRU, then only 14.9 MiB over the following three benchmark rounds.

Three exact-parameter comparisons against the historical T7 runs require care:
R1 had comparable completion length and improved TTFT P90 by 24.57%, output TPS
P10 by 12.69%, and weighted score by 1.76%, with wall time unchanged. Historical
R2/R3 stopped at 85/141 completion tokens, while corrected T8 generated the full
512 tokens. Their wall/weighted changes are therefore dominated by changed
output length and are not valid prefill regressions. Correct cached/uncached
equivalence is the deciding gate, so T8 is retained.

Remote evidence:
`bench_runs/20260712_084530_a63a1ef_T8_original_replay`.

## Staged usage accounting

The 99,500-token T9 test revealed that API usage reported only the first 8,176
cached tokens even though the request was 7.66x faster. Long prefixes restore a
sequence of strict GDN checkpoints; the scheduler previously assigned
`num_cached_tokens` only during the first prefill stage. `3f3f021` accumulates
each checkpoint jump relative to the already-computed position. This is a
metrics correction and does not change model execution.

After the fix, the 8,712-token sample reports 8,688 cached tokens and the
99,500-token sample reports 99,296. Both remain response-identical to their
uncached runs.
