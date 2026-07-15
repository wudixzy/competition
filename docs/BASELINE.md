# Qualified Baseline

## Immutable reference

- Tag: `t9-qualified-20260712`
- Commit: `3f3f02198a50e581c84eae8a3413be8460a94ec3`
- Commit subject: `fix: report staged GDN cache reuse`
- Fixed contract SHA256:
  `5f07f4377dcdde3bb858012bedc014f60e84a82a61e9696bee830fec1e517c0f`
- Evidence archive: `baseline_artifacts/t9-qualified-20260712.tar.gz`
- Archive SHA256:
  `729fea28eed1b91f5cc9753198cee1cd3a63d2f3a04d4cd88db6a95cd6d8e816`

This is the most recent commit with complete four-GPU T9 qualification
evidence. Later `main` commits contain packaging, diagnostics, API fixes, and
measurement tooling and must not be described as GPU-qualified until the full
qualification suite is repeated.

## Qualification evidence

- exact fixed-contract startup without GPU-block override: pass;
- full API smoke: 14/14 pass;
- non-GPU package tests at qualification time: 100/100 pass;
- CUDA GPU0-3 and four-rank collective preflight: pass;
- aligned/unaligned prefix and 17-entry LRU pressure: pass;
- 8,712-token cold/warm responses: identical;
- 99,500-token cold/warm responses: identical;
- non-finite, OOM, CUDA, and NCCL errors: none in qualification logs.

The archived 8,712-token pair records `cached_tokens=0/8688`, identical content
SHA256 `1b5458b1e8360884a2e4c1dfab0bdeecb92e124a19789d853456d21cb49a2902`,
and latency `19.6888s/6.6605s`.

## Development head policy

Use the tag for rollback and A/B baselines. Use `main` for ongoing development.
Every performance candidate must name this commit or a later fully qualified
winner as `baseline_commit` in its manifest.

## Current qualified development winner

As of 2026-07-15, the 256K development baseline is `3453dc2` and the latest
qualified performance winner is E-MOE-03 commit `7a68a94` on branch
`exp/E-MOE-03-router-shared-gate`. It includes E-MOE-02, keeps the fixed
262,144-token contract, passes full smoke and a 235K cold/warm cache gate, and
improves Output TPS P10 by a median 5.78% over E-MOE-02 across three matched
pairs. See `docs/experiments/E_MOE_03_ROUTER_SHARED_GATE_20260715.md` and the
preceding E-MOE-02 record for evidence.

E-MOE-04 (`exp/E-MOE-04-weighted-reduce`) is rejected. A GEMV weighted
reduction improved the complete routed-expert microbenchmark by 5.3% or more
on all four cards, but changed the forced 1,000-token output hash. See
`docs/experiments/E_MOE_04_WEIGHTED_REDUCE_20260715.md`. E-MOE-03 remains the
qualified winner.

The immutable `t9-qualified-20260712` tag remains the archival rollback point;
this section records the newer GPU-qualified development chain rather than
retagging that archive.
