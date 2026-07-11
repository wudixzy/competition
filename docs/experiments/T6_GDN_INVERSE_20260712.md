# T6 GDN Chunk-Inverse Experiment - 2026-07-12

## Baseline profile

Run: `bench_runs/20260711_195453_e00de43_T6_profile`

| Prompt | TTFT | GDN prefill | Routed MoE | Full attention |
| --- | ---: | ---: | ---: | ---: |
| 7,879 tokens | 12.3206 s | 11,148.299 ms | 22,859.385 ms | 3,308.114 ms |
| 15,720 tokens | 20.2078 s | 22,833.369 ms | 41,350.518 ms | 9,101.271 ms |

Profile totals aggregate all four TP ranks. Routed MoE is the largest measured
hotspot, followed by GDN prefill and full attention.

## Rejected change

Commit `3e6df10` replaced the 63 sequential row updates used to construct the
GDN chunk inverse with a finite Neumann doubling product. A standalone BI100
microbenchmark reduced that isolated step from 7.05 ms to 0.94 ms. Small parity
tests passed:

```text
batch=1, seq=64, heads=2, k_dim=v_dim=16, chunk=64
output max abs delta=1.49e-08
state max abs delta=2.98e-08
```

The real fixed-contract synthetic profile failed immediately at layer 0:

```text
non-finite values in prefill-norm GatedDeltaNet layer 0
frac by rank: 0.4687, 0.4267, 0.1474, 0.2021
```

Failure artifacts:
`bench_runs/20260711_200313_3e6df10_T6_validation`.

The change was reverted by `865ec8a`. It must not be retried without a
full-dimension long-sequence numerical analysis. GPU-block override and
non-finite zero-fill are not acceptable mitigations.
