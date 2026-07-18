# M1-31 Stable GDN Prefix State

## Problem

Qwen3.6 interleaves full attention with GatedDeltaNet. A raw KV prefix hit is
not sufficient to skip prefill: the worker also needs the exact recurrent
convolution and temporal state at the same logical boundary. The previous
prototype keyed those states by tuples of physical KV block ids. Those ids are
recycled, differ across allocator lifetimes, and do not identify multimodal
content, so they cannot be a correctness boundary.

The 2026-07-16 evaluator result also shows why this matters operationally:
42% raw cache hit was reported, but 612 of 881 requests failed after an MRoPE
engine crash. Optimization must preserve the 99% success target before cache
TPS or weighted-score claims are meaningful.

## Implemented Contract

Each full block now has a deterministic chained SHA-256 digest. The GDN key is:

```text
(block_count, digest_of_that_chained_prefix)
```

For multimodal requests the first digest includes a canonical namespace over
the input object. Dict/list/tuple/scalars, CPU-normalized tensors, and PIL image
bytes are encoded with type and length framing. Unsupported objects receive a
request-local namespace, so they cannot reuse another request's KV or GDN
state.

The scheduler owns the state index and emits three worker actions through full
and delta sequence metadata and the TP model-input broadcast:

- `gdn_restore_key`: one exact state to restore before the physical suffix;
- `gdn_capture_points`: at most two `(query_offset, stable_key)` captures;
- `gdn_evict_keys`: explicit state removals mirrored on all four workers.

Workers never derive keys from attention block tables. Missing requested state,
invalid digest shape, invalid capture offset, or action use outside a
single-sequence prefill raises immediately.

## Policies

| Policy | Capacity | Admission |
| --- | ---: | --- |
| `off` | 0 | No GDN state reuse; raw KV remains enabled but cannot advance prefill. |
| `fine32` | 32 | Capture the strict state at every scheduler chunk end. |
| `admission64` | 64 | Capture a repeated raw-KV branch and the final strict prompt state. |

`direct` selects the longest exact live state. `aligned` restricts states to
boundaries divisible by the fixed 8192-token scheduler budget and is retained
only as a fallback if direct replay fails a TP4 gate.

## Trace And Simulator

`BI100_CACHE_TRACE=1` emits one redacted version-4 record per completed
allocator lifecycle. It contains only counts, a truncated request-id SHA-256,
one runtime session id, ordinal, and base64 concatenation of exact 32-byte
block hashes. It does not contain prompts, token ids, images, tools, outputs,
or raw request ids.

`scripts/analyze_prefix_cache_trace.py` validates one ordered session and
replays allocator capacity. It reports raw contiguous KV hits separately from
tokens avoidable by the intersection of live KV and resident GDN state. A GDN
state whose KV blocks were evicted counts as zero; the two metrics are never
added together.

## Fixed TP4 Gate

Run one build and change only the two GDN environment variables between cases:

1. `off/direct`: correctness and no-skip control.
2. `fine32/direct`: current candidate default.
3. `admission64/direct`: reduced capture-transfer candidate.
4. `fine32/aligned`: run only if direct fails replay correctness.

For each retained direct case run quick smoke, multimodal cold/warm, 8712-token
boundary cold/warm, 99.5K cold/warm, 235K cold/warm with 1 and 256 output
tokens, then the fixed eight-request short decode benchmark. Required gates:

- cold/warm response equivalence for deterministic requests;
- cached token count equals the selected stable boundary and never exceeds it;
- no worker loss, Gloo reset, MRoPE mismatch, missing state, NaN, OOM, or HTTP 5xx;
- success rate 100% in the qualification matrix;
- no regression greater than 3% in Output TPS P10 or TTFT P90;
- candidate must reduce measured warm wall time or state-capture overhead by at
  least 5% before changing the submission default.

Only after correctness passes should a complete 881-request trace compare
state policies. The simulator is evidence for choosing one fixed A/B candidate,
not evidence of an official score.

## Current Evidence

- 191 local tests pass; 23 tests skip because this machine lacks optional
  runtime dependencies such as CoreX torch/Pillow.
- Submission preflight passes 8/8, including fixed YAML, offline artifacts,
  shell syntax, Python compilation, and diagnostic-env isolation.
- The model-runner patch applies twice to a clean vLLM 0.6.3 source copy and
  broadcasts all three actions in both base and sampling model-input classes.
- TP4 runtime qualification is blocked because the only documented current
  instance is no longer reachable. No performance gain is claimed.
