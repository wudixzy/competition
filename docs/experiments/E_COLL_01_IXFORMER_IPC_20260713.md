# E-COLL-01 IxFormer IPC All-Reduce - 2026-07-13

## Hypothesis

The extended decode profile attributed 13.373 ms/token per TP rank (19.91% of
tracked time) to the 40 sequential tensor-parallel all-reduces. CoreX vLLM's
`disable_custom_all_reduce=True` fallback already uses IxFormer collectives,
but IxFormer's same-node CUDA-IPC path is disabled unless:

```text
ENABLE_CUSTOM_IPC=1
```

This experiment changes only the transport used by the existing all-reduce.
The model, weights, fixed evaluator command, scheduler, and cache semantics are
unchanged.

## Runtime Capability

The active call chain is:

```text
tensor_model_parallel_all_reduce
  -> GroupCoordinator.all_reduce
  -> GroupCoordinator._all_reduce_in_place
  -> ixformer.distributed.all_reduce
```

With IPC disabled, IxFormer calls its NCCL communicator. With IPC enabled and
a same-node SUM tensor that fits the shared region, it calls
`cdist.ipc.allreduce`. The runtime initialized a 64 MiB shared communication
region on every rank. No unsupported or external communication library was
introduced.

## Four-GPU Primitive Gate

The benchmark used four processes and the same FP16 tensor sizes needed by the
model. Every rank produced max-abs parity error 0.0 and no rank timed out.

| Elements | IxFormer NCCL | IxFormer IPC | Speedup |
| ---: | ---: | ---: | ---: |
| 2,048 | 0.2270 ms | 0.0212 ms | 10.69x |
| 8,192 | 0.2270 ms | 0.0282 ms | 8.05x |
| 65,536 | 0.2556 ms | 0.0795 ms | 3.22x |

Values are the slowest rank's median over seven repeats, 100 operations per
repeat after ten warmups. Artifacts are under
`bench_runs/20260713_E_COLL_01/full`.

## Service A/B

Both sides used the fixed four-GPU launch command. Three groups used three
sequential requests, one worker, 128 maximum tokens, prompt repeat 126, seed
20260713, and salts `E-COLL-01-{a,b,c}`.

| Metric | Baseline a/b/c | IPC a/b/c | Median change |
| --- | --- | --- | ---: |
| Decode TPS P10 | 8.096 / 8.034 / 8.655 | 11.707 / 12.117 / 11.653 | +44.60% |
| ITL P50 (s) | 0.11866 / 0.11648 / 0.11626 | 0.07625 / 0.07627 / 0.07634 | -34.52% |
| ITL P90 (s) | 0.12914 / 0.12940 / 0.11807 | 0.12060 / 0.12026 / 0.12045 | -6.73% |
| TTFT P90 (s) | 4.554 / 4.512 / 4.527 | 6.643 / 4.438 / 4.396 | -1.96% |
| Overlap score | 658.74 / 661.46 / 680.14 | 811.35 / 899.62 / 815.89 | +23.35% |
| Disjoint score | 353.42 median | 453.07 median | +28.20% |

All requests succeeded. The IPC server log contained no runtime, OOM, CUDA,
NCCL, IPC, or fatal error. The IPC shared region reduced available GPU KV
blocks from 18,271 to 17,943 (-1.80%), which still comfortably supports the
fixed 100K context limit.

## Correctness And Quality

IxFormer IPC changes FP16 reduction order. Fixed greedy summaries were not
byte-identical to the NCCL baseline, although all three retained the same
meaning, returned `finish_reason=stop`, and remained concise and correct. This
is an explicit exception to the earlier bit-exact optimization policy rather
than an unreported parity pass.

The repository's four-conversation, 13-turn quality replay was therefore run
on both modes:

```text
baseline  13/13 non-empty, 1664 tokens, 239.30 s
IPC       13/13 non-empty, 1664 tokens, 176.29 s (-26.33%)
```

Manual key-fact review passed on both outputs: Singapore shopping/food and
chili-crab ingredients, Ming founded in 1368, the Hu Weiyong accusations,
Tang He as a surviving founder, the 23-chicken/12-rabbit solution and program,
the ancient method, and a non-Messi football copy using Cristiano Ronaldo.

Qualification evidence:

- full API smoke retry: 14/14;
- one earlier JSON-object smoke request produced truncated JSON, but the exact
  isolated retry passed and the complete retry suite passed 14/14;
- 99,500-token cold/warm gate: pass;
- cold/warm prompt tokens: 99,500 / 99,500;
- warm cached tokens: 99,296;
- cold/warm completion tokens: 8 / 8;
- cold/warm message SHA-256: identical;
- cold/warm elapsed: 160.928 s / 19.667 s;
- service and collective fatal/OOM/CUDA/NCCL/IPC errors: zero.

## Decision

**Keep as a decode winner with a documented quality-equivalence exception.**
The service-level gain is substantially larger than prior candidates, all
functional and long-context gates pass, and the representative multi-turn
replay preserves key facts and instruction following. The exact-output policy
is relaxed only for this standard reduction-order change; later model-math
optimizations remain bit-exact by default.

Deployment defaults `ENABLE_CUSTOM_IPC=1` in both `Dockerfile` and
`launch_service`. Set it to `0` to restore the prior IxFormer NCCL path. The
fixed `computility-run.yaml` SHA-256 remains:

```text
5f07f4377dcdde3bb858012bedc014f60e84a82a61e9696bee830fec1e517c0f
```
