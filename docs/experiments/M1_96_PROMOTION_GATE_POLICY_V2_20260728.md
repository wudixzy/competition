# M1-96 promotion gate policy v2

Date: 2026-07-28

Branch: `exp/M1-96-gate-policy-v2-20260728`

## Purpose

The previous policy mixed final competition requirements, inexpensive
continuation screens, and numerical correctness oracles. This made some gates
too weak, such as allowing the full functional suite to skip `n=2`, while
making other gates too rigid, such as treating a vendor FP16 reduction as an
exact mathematical reference.

Policy v2 separates those roles. It does not lower any official metric or
model-capability requirement.

## Unchanged final requirements

The following remain mandatory and cannot be relaxed by an experiment:

- Output TPS P10 at least 20;
- TTFT P90 at most 5 seconds;
- effective cache hit rate at least 50%;
- valid-request success rate at least 99%;
- weighted token throughput at least 8000;
- `max_model_len=262144` and the required large output budgets;
- no fatal error, OOM, Gloo/NCCL reset, worker loss, timeout, or residual
  service process;
- full tool calling, reasoning, thinking, structured output, multilingual,
  multimodal, streaming, and long-context behavior;
- no material model-capability regression against the same-environment
  CoreX baseline.

The formal 53-case functional suite must pass every expected outcome. A
negative-parameter case passes only by returning its expected 4xx; a valid
request cannot be converted to a skip. In particular, `n=2` must return two
valid choices. The fixed 11-case Agent matrix must also pass.

## Diagnostic versus promotion evidence

The four-layer real-weight checkpoint is a structural diagnostic. It may
qualify request parsing, HTTP/SSE envelopes, runtime identity, cache namespace
isolation, capacity plumbing, and lifecycle behavior. It cannot qualify
semantic tool selection, code quality, reasoning, or multimodal understanding.

A diagnostic semantic failure must be retained and investigated, but it must
not prevent independent structural stages from running after a clean
postflight. Conversely, a diagnostic pass never substitutes for the full-model
TP4 semantic gate.

## Numerical gates

### Exact state operations

Cache keys, KV/GDN state copies, indexing, serialization, swap, restore, and
other operations intended to preserve state exactly remain bit-exact gates.
Missing or mismatched state must fail closed.

### Elementwise kernels

For an operation with a stable FP32 reference and no reordered reduction:

- no NaN or Inf;
- relative L2 at most `1e-5`;
- fixed-shape maximum absolute error within the predeclared dtype bound;
- deterministic greedy next-token checks before integration.

### Reordered reductions

For matmul, softmax, expert reduction, or another operation whose legal
implementation changes FP accumulation order, vendor FP16 output is a control,
not mathematical truth. The hard numerical oracle becomes a high-precision
calculation rounded to the production output dtype.

On fixed, predeclared seeds and stratified sequence samples, a candidate must
be no worse than the vendor control:

- candidate aggregate relative L2 <= vendor relative L2 + `1e-8`;
- candidate maximum step relative L2 <= vendor maximum + `1e-8`;
- candidate maximum absolute error <= vendor maximum absolute error;
- candidate mismatch count versus the rounded high-precision reference <=
  vendor mismatch count;
- every output finite.

Candidate-versus-vendor relative L2 remains useful diagnostic data, but it is
not an absolute correctness threshold. This avoids rejecting a result merely
because it is more accurate than a different FP16 accumulation order.

Any greedy next-token divergence triggers a full semantic A/B. It is neither
an automatic pass nor an automatic failure. Promotion still requires no
functional or capability regression.

## Performance screens

Microbenchmarks decide whether an implementation deserves an end-to-end run;
they do not authorize production by themselves.

- Preserve each experiment's predeclared hotspot threshold. Expensive fused
  attention work may retain a `1.5x` continuation target; a narrowly scoped
  kernel may use `1.25x`.
- Require stable repeated timings and report all trials.
- Require at least a 5% paired improvement in the targeted end-to-end proxy
  before a compute-kernel candidate proceeds to a full official-style run.
- Evaluate small proxies with at least three paired A/B repetitions and use
  their paired distribution, not one noisy relative percentage.
- A single run cannot establish a 2% regression. The final absolute Output TPS
  P10 requirement remains 20, while relative-regression decisions use paired
  evidence.

The official 881-request result remains authoritative for final throughput and
TTFT. A 13-request smoke result cannot be compared directly with the official
8000-point threshold.

## Cache screens

Cache correctness remains prior to cache performance:

- cold/warm deterministic output and response structure must match;
- effective cached tokens are the intersection of continuous live KV and a
  restorable matching GDN state;
- multimodal content and physical-block reuse must remain isolated.

The final hit-rate requirement remains 50%. Once an attributable run already
exceeds it, a cache candidate does not need an arbitrary additional five
percentage points. It proceeds only if repeated proxy evidence shows either:

- at least two percentage points more effective hit rate with no quality,
  TTFT, or Output TPS regression; or
- at least 3% more weighted proxy with no reduction in effective hit rate.

This prevents optimizing toward the theoretical cache ceiling when success
rate, TTFT, or input compute is the actual score bottleneck.

## Capability non-inferiority

Deterministic protocol and schema cases remain all-or-nothing. Public
code/math/reasoning subsets use paired baseline comparison from fixed revisions
and sample order. A candidate score must be at least:

```text
baseline score - max(1 / sample_count, 0.02)
```

The one-sample allowance handles small fixed subsets; it does not permit a
concentrated regression in tool calling, structured output, multimodal
understanding, or long-context recall. Those capability groups retain their
own all-or-nothing contract cases.

## M1-91 interpretation

M1-91 remains correctly recorded as rejected under its declared v1
candidate-versus-vendor threshold. Policy v2 does not rewrite that result.

Its `5.46x` routed speedup and exact fixed high-precision results justify one
bounded follow-up of the unchanged kernel using stratified high-precision
sequence samples. That is an oracle correction, not a compensation, tile, or
tolerance scan. If the candidate is worse than vendor against the
high-precision reference, the direction closes. If it is non-inferior, it may
proceed only to next-token and full-model quality/performance A/B; it still
cannot change a default directly.

## Fixed M1-91 v2 screen

The follow-up freezes all implementation and timing choices from M1-91. It
adds high-precision sequence references only at the predeclared step indices:

```text
0, 1, 2, 3, 7, 15, 31, 63, 127, 255, 383, 499
```

For both fixed seeds, the benchmark records vendor, production-direct, and
compensated outputs against CPU float64 dot products rounded once to FP16.
The full 500-step candidate-versus-vendor measurements remain in the report as
diagnostics. They are not silently removed or reinterpreted as exact results.

The runner binds the benchmark, qualification, candidate extension, and
production-direct extension hashes before returning a rejected-candidate exit
code. It therefore distinguishes valid negative evidence from an incomplete
or identity-mismatched experiment. A passing single-GPU numerical screen still
authorizes only a full-model A/B, not production integration.

Local validation before the BI100 rerun:

- 1,086 unit tests passed with 25 expected skips;
- submission preflight passed all nine checks;
- the 53-case functional manifest and 11-case Agent manifest qualified;
- Python and shell syntax checks passed.

## Stop rule

Changing a threshold after seeing a failure is prohibited unless the reference
oracle itself was shown to be invalid for the operation class. Any revised
oracle and sample selection must be fixed before the rerun. Failed candidates
cannot be rescued by relaxing tolerances, selecting favorable samples, or
discarding adverse trials.
