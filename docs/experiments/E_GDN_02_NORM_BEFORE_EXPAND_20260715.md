# E-GDN-02: Normalize q/k before head expansion

> Superseded shape note: this probe followed stale 48-value-head source
> comments and used 12 local heads. E-GDN-09 reran the same exact boundary at
> the checkpoint's real eight local heads and confirmed the rejection.

## Scope

The Gated DeltaNet decode path expands each of four local q/k heads three times
before applying L2 normalization. Since every repeated head is identical,
E-GDN-02 benchmarks normalizing four heads first and then expanding the
normalized values to twelve heads.

```text
base:  ca0857a (qualified E-MOE-03/E-GDN-01 model plus experiment evidence)
bench: c53602a
host:  ssh-a2d0a302.default.gpu.phanthy.com
```

## Primitive gate

`tests/bench_gdn_norm_before_expand.py` covers the complete real-shape TP-rank
decode recurrent step, not only normalization:

```text
batch=1, local key heads=4, local value heads=12
head dimensions=128x128, temporal state=(1, 12, 128, 128)
```

Both paths include q/k normalization, head expansion, state decay, two BMMs,
delta construction, and the in-place `baddbmm_` state update. Each result is
the median of nine trials with 300 steps per trial after 20 warmups on physical
GPU1.

| Case | Median (ms) | P10 (ms) | P90 (ms) | Speedup | Output exact | State exact |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| normalize after expansion | 0.241252 | 0.239087 | 0.244623 | 1.0000x | yes | yes |
| normalize before expansion | 0.235704 | 0.235171 | 0.237156 | 1.0235x | yes | yes |

Both output and mutated temporal state have maximum absolute difference 0.0.
Moving normalization saves work as expected, but improves the complete
recurrent step by only 2.35%.

Remote artifacts are intentionally untracked:

```text
/root/competition/bench_runs/20260715_E_GDN_02/gpu1.json
/root/competition/bench_runs/20260715_E_GDN_02/gpu1.log
/root/competition/bench_runs/20260715_E_GDN_02/gpu1.status
```

## Decision

`REJECT AS PERFORMANCE WINNER`. The complete-path result is below the 1.05x
primitive integration gate, so no production model patch or service A/B is
justified. Keep the qualified E-GDN-01 merged projection implementation.
