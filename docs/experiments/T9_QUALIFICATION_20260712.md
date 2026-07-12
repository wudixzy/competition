# T9 Fixed-Contract Qualification - 2026-07-12

## Retained implementation

- `9cb31f3`: stable MoE route grouping.
- `0e52374`: scheduler gate for exact GDN checkpoints.
- `0ec0607`: GDN state capture at strict block boundaries.
- `b22fd8f`: full-attention segmentation at the same boundary.
- `a63a1ef`: non-mutating fp32 query scaling.
- `3f3f021`: staged GDN cached-token usage accounting.

The evaluator command was not changed. `computility-run.yaml` remains
byte-identical to the T7 winner with SHA256
`5f07f4377dcdde3bb858012bedc014f60e84a82a61e9696bee830fec1e517c0f`.

## Gates

- package shell syntax, 53 Python files, and 100/100 non-GPU tests: pass;
- attention 1/1, GDN 2/2, and MoE 3/3 parity: pass without skips;
- CUDA GPU0-3 and NCCL rank0-3: pass;
- exact no-override restart and cold-response determinism: pass;
- final full API smoke: 14/14 pass;
- aligned/unaligned interleaving and 17-prefix LRU eviction: pass;
- three repeated `workers=1` benchmark runs: success 1.0 and stable;
- server logs: no non-finite, OOM, CUDA, or NCCL error.

Docker image construction was not executable because the allocated host has no
Docker CLI. The tracked offline transformers wheel exists at 11,269,669 bytes
with SHA256
`c85e7feace634541e23b3e34d28aa9492d67974b733237ade9eba7c57c0fd1bd`.
Patch inputs, py_compile, imports, and package tests were verified.

## Contract-boundary result

The final 99,500-token test used `max_tokens=16` under the fixed 100,000-token
limit:

```text
uncached: cached=0,     completion=8, latency=158.625 s
cached:   cached=99296, completion=8, latency=19.670 s
speedup:  8.06x
```

The complete messages, finish reasons, completion counts, and SHA256
`a3dc73d02269b1b3682ed84197c3d2d0ddc39dfdb544f73fb3ea832f1fb30b4d`
match. Final process metrics showed no active, waiting, or swapped requests.

Remote evidence:
`bench_runs/20260712_T9_final_3f3f021`.

## Remaining external issue

The supplied dataset describes inputs above 230K, while the immutable launch
command limits the model to 100K. This package qualifies up to the fixed 100K
boundary only. Organizer clarification is still required for longer samples.
