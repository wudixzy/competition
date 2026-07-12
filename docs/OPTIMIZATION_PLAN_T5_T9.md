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

## T5 result

Two consecutive no-override startups completed successfully. vLLM allocated
18,275 GPU blocks (about 292K tokens at 16 tokens/block), full smoke passed
14/14, and the valid 8-request `workers=1` baseline recorded:

```text
success_rate=1.0
ttft_p90=2.5836 s
output_tps_p10=5.7213
input_tps=231.4230
cache_tps=201.3813
cache_hit_rate=0.870187
weighted_proxy_score=856.6219
```

One older startup produced a layer-6 GDN non-finite value. It has not reproduced
in the two qualification starts and remains a stability watch item, not a reason
to add a GPU-block override.

## T6 checkpoint

8K/16K profiling showed routed MoE as the largest measured hotspot, followed by
GDN prefill and full attention. The first GDN inverse-vectorization experiment
passed small parity but produced non-finite values at layer 0 in the real startup
profile. It was reverted in `865ec8a`; see
`docs/experiments/T6_GDN_INVERSE_20260712.md`. T6 is paused for review before a
different numerical approach is selected.

## T7 result

Commit `9cb31f3` groups MoE prefill routes once with a stable sort. All parity,
hardware, exact-startup, and full-smoke gates passed. Across three strict seeded
A/B pairs, weighted proxy score improved by 7.67% on average, TTFT P90 improved
by 16.01%, and input/cache TPS improved by 7.96%. The change is retained; see
`docs/experiments/T7_MOE_GROUPING_20260712.md`.

## T8 result

The current GDN prefix-state cache saves state after the whole prompt while its
key covers only complete 16-token KV blocks. An unaligned 3,678-token prompt was
stored under the 3,664-token boundary, and a cached replay produced different
output from the identical uncached request. The initial T8 experiment was
reverted by `42fc9b7` while a boundary-exact design was developed.

The first boundary-capture implementation proved that GDN state restoration
alone is insufficient: the uncached and cached full-attention paths partition
the online-softmax reduction differently. Those experimental commits were
reverted. The retained implementation uses exact scheduler checkpoints
(`0e52374`), captures recurrent state at that boundary (`0ec0607`), and splits
the full-attention fallback at the same point (`b22fd8f`). `a63a1ef` also makes
query scaling non-mutating for fp32 inputs.

The original 8,712-token failure now produces identical responses with staged
`cached_tokens=0/8688`; request time fell from 19.69 s to 6.66 s. Attention,
GDN, and MoE parity, four-card CUDA/NCCL, exact startup, full smoke 14/14,
aligned/unaligned interleaving, and 17-entry LRU eviction all pass. GPU memory
stayed flat, and RSS stabilized after the expected CPU state-cache allocation.
T8 is retained; full evidence is in
`docs/experiments/T8_GDN_PREFIX_BOUNDARY_ISSUE_20260712.md`.

## T9 result

The final fixed-contract qualification passed package syntax/compile checks,
100/100 non-GPU tests, four-card CUDA/NCCL, exact no-override restart, cold
determinism, and full API smoke 14/14. Three repeated `workers=1` benchmark
runs completed successfully and remained stable. `computility-run.yaml` stayed
byte-identical to the T7 winner at SHA256
`5f07f4377dcdde3bb858012bedc014f60e84a82a61e9696bee830fec1e517c0f`.

At the contract boundary, a 99,500-token prompt returned identical uncached and
cached responses. `3f3f021` accumulates the tokens skipped across staged GDN
checkpoint restores, so usage now reports 99,296 cached tokens instead of only
the first 8,176-token stage. Latency fell from 158.625 s to 19.670 s (8.06x).
GPU memory and process state remained healthy, with no non-finite, OOM, or CUDA
errors. The host has no Docker CLI, so image construction was not executable;
the offline wheel hash, patch inputs, py_compile, imports, and package tests were
verified instead. See `docs/experiments/T9_QUALIFICATION_20260712.md`.
