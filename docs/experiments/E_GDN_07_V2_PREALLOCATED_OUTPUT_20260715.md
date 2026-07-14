# E-GDN-07 v2: Preallocated recurrent output

## Hypothesis

E-GDN-07 v1 returned `torch::empty_like(value)` and validated six tensor
contracts on every decode call. V2 adds an unchecked extension entry point
whose output tensor is preallocated and reused, while keeping the same kernel,
state update, launch-error check, and FP32 arithmetic.

```text
base: 167e0f8
bench: 0b2c591
```

## Result

The real TP-rank decode shape on physical GPU1 passed every numerical gate:

| Gate | Output max abs | State max abs | Close/finite |
| --- | ---: | ---: | --- |
| One step | 3.73e-9 | 2.98e-8 | yes |
| 1,000 repeated inputs | 2.61e-8 | 1.40e-6 | yes |
| 1,000 random inputs | 1.49e-8 | 1.79e-7 | yes |

Nine serial trials with 1,000 recurrent steps per trial produced:

| Path | Median (ms) | P10 (ms) | P90 (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| PyTorch reference | 0.064779 | 0.064624 | 0.064934 | 1.0000x |
| Preallocated fused kernel | 0.050574 | 0.050482 | 0.050615 | 1.2809x |

The candidate latency is effectively unchanged from the checked/allocating v1
runs. Torch allocator reuse and contract checks are not the limiting cost.

Remote evidence:

```text
/root/competition/bench_runs/20260715_E_GDN_07_V2/result.json
/root/competition/bench_runs/20260715_E_GDN_07_V2/bench.log
```

## Decision

`REJECT AS PERFORMANCE WINNER`. V2 remains below the 1.5x recurrent-update
gate and does not justify model integration or TP4 qualification. Future work
must reduce a larger operation boundary or improve device execution; another
output-allocation/dispatch variant is not warranted.
