# M1-71 Hybrid MoE Exact-Tail Gate

## Scope

M1-71 tested one fixed TP4 rank-local MoE structure on physical GPU1 of
`ssh-73ca29ba`:

- direct selected-expert W13;
- the existing `SiluAndMul` activation;
- vendor `torch.bmm` for W2;
- the existing serial-float routed reduction.

There was no tile, threshold, shape, or accuracy-limit scan. Rejected compute
kernels and submission defaults remained unchanged.

Source revision:
`6b2eac5cabb0d84d7e44bb1edd893c75229026e3`.

Runtime overlays:

- candidate: `/root/m1-69-runtime-37001ed/site-packages`;
- control: `/root/m1-68-runtime-cdb1bc4/site-packages`.

Raw run:
`/tmp/m1-71-hybrid-moe-6b2eac5-20260727T174127Z`.

## Result

The performance gates passed:

| Metric | Observed | Gate |
| --- | ---: | ---: |
| Fixed speedup | 1.4919x | >= 1.25x |
| Routed speedup | 1.3745x | >= 1.25x |
| Routed saving | 0.0910 ms | >= 0.02 ms |

The numerical gate failed:

| Metric | Observed | Gate |
| --- | ---: | ---: |
| Direct W13 relative L2 | 7.3024e-6 | <= 1e-5 |
| Fixed endpoint relative L2 | 5.3364e-5 | <= 1e-5 |
| 500-step sequence relative L2 | 8.1543e-5 | <= 1e-5 |
| Maximum step relative L2 | 3.6255e-4 | <= 1e-5 |
| Exact sequence steps | 50 / 500 | informational |

All 500 sequence steps were finite. The direct W13 endpoint alone met the
limit, but its different accumulation result was amplified by activation and
the exact W2/reduction tail. Operator-local acceptance therefore would have
been insufficient.

## Cleanup

The scoped cleanup, service postflight, fatal scan, timeout scan, and before
versus after GPU preflight all passed. GPU1 free memory was unchanged at
34,057,748,480 bytes, with no API server, worker, or GPU holder left behind.

## Decision

The exact-tail design is rejected and this direction is closed. Its speedup
does not override the model-correctness gate. It must not be enabled by
default, added to `computility-run.yaml`, or promoted to `main`.

This component-only result does not evaluate full-model semantics, TP4 service
performance, long context, or competition score.
