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
