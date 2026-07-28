# M1-100 E-PREFIX-08 high-precision oracle

Date: 2026-07-28

Status: private single-GPU gate prepared but not yet run. It changes no
production runtime, default, YAML, model, tokenizer, or request semantics.

## Re-audit

E-PREFIX-08 reduced its synthetic `Q=8192, context=65536` complete boundary by
`22.15%` and the partial-context boundary by `22.06%`. Its maximum absolute
errors were only `1.19e-7` and `2.38e-7`, but candidate-versus-control relative
L2 was about `3.5e-5`, above the old unconditional `1e-5` gate.

Policy v2 treats a reordered softmax/PV reduction against a production FP32
control as a numerical comparison, not a mathematical oracle. One
high-precision non-inferiority check is therefore permitted.

The re-audit also found that the old candidate activated only at `Q=8192`.
Current strict prefix segmentation presents the production main segment as
`Q=8176` followed by an unchanged 16-token boundary. M1-100 corrects this
proxy mismatch once; it does not scan query lengths or add a configurable
threshold.

## Frozen inputs

The original final E-PREFIX-08 algorithm and reference are copied byte-for-byte
from `exp/E-PREFIX-08-cold-chunk-hybrid@402bde9`:

| Artifact | SHA-256 |
|---|---|
| `bench_prefix_attention_breakdown.py` | `2ab82f69e7833dc2965b03e4cbcebe5beafd9d4954a3e3babda101bb54a0ddd2` |
| `bench_prefix_cold_chunk_hybrid.py` | `e2dffa151c99f4cf28d827877db68bbcb0a0c0bd6433c466017c255df2f3d076` |

The gate fixes:

- FP16 input/output and FP32 production reduction;
- `Q=8176`, four query heads, one KV head, head dimension 256;
- block size 16 and 512-token tiles;
- contexts 65,536 and 65,552;
- primary seeds `20260716` and `20260727`; partial-context seeds add the
  frozen offset 100;
- sixteen predeclared `(query index, head)` samples spanning the segment;
- one warmup and three timing trials;
- paired control/candidate timing with alternating order across trials and
  seeds;
- eight fixed CPU threads for the FP64 oracle;
- the original minimum 15% primary-boundary reduction.

Only the explicit CoreX device and private output path are command-line
arguments.

## Oracle

For each fixed sample, the gate reconstructs the logically ordered paged
context, appends the causal current chunk, and computes full-sequence
attention on CPU using FP64 QK, softmax, and PV. It rounds the result once to
the production FP16 output dtype.

The candidate must be no worse than the production control for:

- aggregate relative L2, with only the predeclared `1e-8` comparison slack;
- maximum per-sample relative L2, with the same slack;
- maximum absolute error, with no slack;
- mismatch count against the rounded oracle;
- finite outputs.

The report retains scalar metrics only. It does not retain query, KV, output,
token, prompt, or model data.

## Decision boundary

A pass authorizes only a fixed greedy next-token integration gate. It does not
authorize production integration, a service-performance claim, a YAML or
default change, a `main` merge, or repository visibility change.

A failure closes E-PREFIX-08 under the revised oracle. No tile, shape,
threshold, seed, dtype, tolerance, or merge-formula scan follows.
