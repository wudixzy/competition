# E-MOE-09: Pack shared down projection as routed W2 expert

## Hypothesis

Routed W2 experts and the shared down projection both have local shape
`(2048, 128)` at fixed TP4. E-MOE-09 appends shared W2 as expert 256 and uses
one nine-way BMM. It preserves the existing arithmetic order by reducing the
first eight routed outputs before adding the ninth shared output separately.
The baseline uses an actual vLLM linear layer for shared down.

## Results

The real-shape probe ran serially on physical GPU1:

| Region | Baseline median | Packed median | Speedup |
| --- | ---: | ---: | ---: |
| Routed BMM + shared down | 0.127906 ms | 0.148046 ms | 0.8640x |
| Down + weighted combine | 0.176145 ms | 0.194967 ms | 0.9035x |

P10/P90 were 0.127850/0.128036 ms versus 0.147966/0.148277 ms for the down
projections, and 0.176045/0.176310 ms versus 0.194748/0.195037 ms for the
complete tail.

Routed down is bit-exact. Shared down differs by `5.9604645e-8`; the final
output happened to be exact for this input after FP16 combination, but that
does not restore a bit-exact intermediate contract.

Artifact:

```text
/root/competition/bench_runs/20260715_E_MOE_09/result.json
```

## Decision

`REJECT`. The candidate is 9.7% slower on the complete tail and changes the
shared intermediate. Do not modify W2 storage or checkpoint loading. Combined
with E-MOE-08, both variants of treating the shared expert as expert 256 are
closed: actual vLLM shared projections outperform the extra gather and larger
batched operation.
