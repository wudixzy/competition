# E-MOE-05: T=1 route and gather selection

Date: 2026-07-15

## Hypothesis

The fixed evaluator decodes one sequence at a time. The current routed-MoE
path selects eight experts with advanced indexing, performs one flattened W13
GEMM, and performs one W2 batched GEMM. This experiment checks whether
`index_select`, two batched GEMMs, selected-logit softmax, or expert views can
reduce the T=1 cost without changing the model contract.

The production TP-rank shape is used on all four BI100 devices:

```text
experts=256, hidden=2048, intermediate/rank=128, top_k=8, fp16
```

Each result uses five warmups followed by seven repeats of 50 iterations. The
formal service remains running and healthy; no production model code changes in
this experiment.

## Baseline context

Main commit `bd303c0` passed 8/8 fixed-contract benchmark requests:

- TTFT P90: 1.808 s
- decode TPS P10: 12.693
- cache hit rate: 87.16%
- weighted overlap score: 1180.55

The decode target remains 20 TPS, a 36.5% gap from this local baseline.

## Primitive results

Median ranges across GPU0-3:

| Primitive | Median range (ms) | Decision |
| --- | ---: | --- |
| current advanced-index path | 0.4555-0.4559 | reference |
| `index_select` full path | 0.5199-0.5213 | reject, about 12.4% slower |
| double-bmm advanced path | 0.4732-0.4766 | reject, about 4.1% slower |
| advanced gather only | 0.1949-0.1957 | retain |
| `index_select` gather only | 0.2651-0.2663 | reject |
| expert views, fixed CPU IDs | 1.1991-1.2233 | reject |
| expert views, synchronized IDs | 1.4658-1.4875 | reject |
| full softmax then top-k | 0.0520-0.0533 | reference |
| top-k logits then selected softmax | 0.0520-0.0522 | no material gain |

All outputs were finite on every device. Output parity was exact for the
compute variants, route IDs were equal, and the route-weight maximum absolute
difference was `1.49e-8`.

Artifacts:

```text
bench_runs/20260715_E_MOE_05/primitive/gpu0.json
bench_runs/20260715_E_MOE_05/primitive/gpu1.json
bench_runs/20260715_E_MOE_05/primitive/gpu2.json
bench_runs/20260715_E_MOE_05/primitive/gpu3.json
```

## Decision

**Reject all production-path changes.** Advanced indexing plus the flattened
W13 GEMM and W2 bmm remains the fastest supported implementation. Selected
softmax saves at most about 0.001 ms per layer in this probe, which is too small
and inconsistent to justify a full service A/B. Keep `tests/bench_moe_decode.py`
on the experiment branch as reproducible evidence, and move the next
single-variable experiment to GDN input-projection fusion.
