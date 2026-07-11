# T5-T9 Fixed-Contract Optimization Plan

## Evaluator contract

`computility-run.yaml` is immutable. In particular, evaluation uses one request
at a time, TP=4, `max_num_seqs=1`, `max_num_batched_tokens=8192`,
`gpu_memory_utilization=0.9`, and `max_model_len=100000`. Valid benchmarks use
`workers=1`. GPU-block overrides and launch-parameter scans are diagnostic only.

## Dataset implications

The supplied workload report contains 881 streaming agent requests. Mean input
is about 25.7K tokens, input/output is about 60:1, and token-weighted prefix
reuse is about 65.6%. Tool definitions or tool messages occur in most requests.
Optimization therefore prioritizes long prefill, GatedDeltaNet, MoE, prefix
state retention, and structured-output correctness rather than request batching.

The report also contains 230K+ inputs and calls for a 256K context. The model
configuration supports 262144, but the evaluator command caps total context at
100000. The organizer must clarify whether longer samples are filtered or the
fixed command will change. Do not claim support above 100K under this contract.

## T5-T9

- T5: fix the no-override synthetic-profile non-finite GDN failure, then create
  the exact-command full-smoke and `workers=1` baseline.
- T6: profile representative 4K-16K and 32K+ uncached prefills; optimize only
  measured GDN hotspots with numerical parity.
- T7: optimize per-request MoE prefill routing/grouped GEMM and retain an
  efficient small-token decode path.
- T8: validate prefix/GDN state keys, restore, eviction, and interleaved-session
  behavior; optimize retention under the fixed memory budget.
- T9: integrate code-level winners and run clean packaging, hardware gates,
  exact startup, full smoke, repeated single-worker benchmarks, and tests up to
  the supported 100K boundary.

Stop before proceeding when a fix requires changing the evaluator command,
zero-filling non-finite values, changing model semantics, or accepting a parity,
tool-call, cache-correctness, or hardware failure.
