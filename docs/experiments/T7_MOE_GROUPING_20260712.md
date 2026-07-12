# T7 MoE Route Grouping - 2026-07-12

## Change

Commit `9cb31f3` replaces one full `(tokens, top_k)` mask/nonzero scan per
active expert with a single stable sort and bincount. Expert GEMMs, routing
weights, expert order, TP reduction, and the single-token decode path are
unchanged.

An isolated BI100 test at `T=2048`, hidden size 2048, 256 experts, top-k 8,
and local intermediate size 128 measured 167.68 ms for the old path versus
130.43 ms for the grouped path (-22.2%). Maximum output delta was 4.77e-7.

## Gates

- P0 static: 35/35 pass
- MoE parity: 3/3 pass, no skips
- GDN parity: 1/1 pass, no skips
- CUDA GPU0-3 and NCCL: pass
- Exact no-override startup: pass, 18,271 GPU blocks
- Full smoke: 14/14 pass
- No non-finite, OOM, CUDA, or NCCL error

Validation run:
`bench_runs/20260712_021519_9cb31f3_MoE_validation`.

## Long-prefill profile

Compared with `bench_runs/20260711_195453_e00de43_T6_profile`:

| Prompt | Routed MoE | TTFT | Wall |
| --- | ---: | ---: | ---: |
| ~8K | -6.77% | +0.47% | +2.41% |
| ~16K | -7.93% | -4.65% | -2.74% |

The unchanged GDN timer varied by +9.15% in the 8K run and masked the MoE
gain. The 16K request improved end to end.

## Strict seeded A/B

Three baseline/candidate pairs used identical prompt salt, seed 123, request
count 8, worker count 1, and empty service cache after restart. Prompt,
completion, and cached token counts matched within every pair.

| Metric | Pair 1 | Pair 2 | Pair 3 | Mean change |
| --- | ---: | ---: | ---: | ---: |
| Weighted proxy | +0.81% | +8.48% | +13.71% | **+7.67%** |
| TTFT P90 | -11.46% | -14.09% | -22.47% | **-16.01%** |
| Output TPS P10 | -4.22% | +10.31% | +9.80% | **+5.30%** |
| Input TPS | +1.63% | +8.45% | +13.80% | **+7.96%** |
| Cache TPS | +1.63% | +8.45% | +13.80% | **+7.96%** |

Artifacts:

- `bench_runs/20260712_023305_T7_strict_ab_A_baseline`
- `bench_runs/20260712_023702_T7_strict_ab_B_candidate`
- `bench_runs/20260712_024302_T7_strict_ab_R23_A_baseline`
- `bench_runs/20260712_024708_T7_strict_ab_R23_B_candidate`

Decision: keep `9cb31f3`.
