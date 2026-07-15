# E-MOE-08: Pack shared gate/up as routed W13 expert

## Hypothesis

On fixed TP4, every routed W13 expert and the shared gate/up projection have
the same local shape `(256, 2048)`. Appending shared gate/up as expert 256
would allow decode to gather nine blocks and execute one projection instead of
the routed projection plus a separate shared projection.

## Bare-operator probe

The first probe used standalone tensors and `F.linear` for both paths. It was
bit-exact and appeared promising:

| Region | Baseline | Packed | Speedup |
| --- | ---: | ---: | ---: |
| Gate/up projections | 0.364415 ms | 0.269640 ms | 1.3515x |
| Routed + shared core | 0.597406 ms | 0.503521 ms | 1.1865x |

This result is retained as diagnostic evidence but is not the decision source.

## Actual vLLM layer oracle

The repeated probe replaced the standalone shared projection with the same
vLLM `ReplicatedLinear` method used by the model's local projection math. The
packed candidate remained bit-exact, but the apparent benefit disappeared:

| Region | Actual baseline | Packed | Speedup |
| --- | ---: | ---: | ---: |
| Gate/up projections | 0.265780 ms | 0.269650 ms | 0.9856x |
| Routed + shared core | 0.501676 ms | 0.504542 ms | 0.9943x |

P10/P90 were 0.265469/0.265932 ms versus 0.269594/0.269808 ms for gate/up,
and 0.501372/0.501875 ms versus 0.504243/0.504909 ms for the full core.
Routed gate/up, shared gate/up, and final output all have max absolute
difference zero.

Artifacts:

```text
/root/competition/bench_runs/20260715_E_MOE_08/result.json
/root/competition/bench_runs/20260715_E_MOE_08/actual.json
```

## Decision

`REJECT AS PERFORMANCE WINNER`. Do not replace the FusedMoE W13 parameter or
change checkpoint loading. The bare `F.linear` baseline overstated the shared
projection cost; the actual vLLM-layer baseline is already faster than the
packed path. Future fusion probes must use model-layer or installed-runtime
oracles before any production integration.
