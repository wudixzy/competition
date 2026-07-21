# M1-40 Masked-Neumann GDN Prefill

## Purpose

M1-40 is a bounded capability gate for the remaining GatedDeltaNet prefill
bottleneck. It tests the multiplication-only inverse approximation from
[Zhang et al., arXiv:2606.06034](https://arxiv.org/abs/2606.06034) at the
Qwen3.6 TP-rank production shape. It does not modify the model, runtime,
Docker image, evaluator command, or cache policy.

The branch starts from `124f836`, after the M1-38 decision and before the
rejected M1-39 deferred-MoE implementation.

## Why This Is Not T6

T6 used an unmasked finite Neumann-doubling product. Its isolated inverse was
fast, but real model activations became non-finite at layer 0 because high
matrix powers produced heavy-tailed values.

The new paper addresses that exact failure mode with three fixed operations:

1. truncate the initial Neumann series at order `N=3`;
2. retain only the first three lower sub-diagonals;
3. recover the tail with `S=8` residual terms.

This is a new algorithmic boundary, not a retry or parameter scan of T6. The
configuration follows the paper's 64-by-64 result and is not exposed as a CLI
or YAML knob. The existing exact row-loop remains the reference.

## Fixed Gate

`tests/bench_gdn_masked_neumann_prefill.py` measures both the inverse and the
complete chunk-rule path at lengths 64, 1024, 4096, and segmented 7800. It
also runs 100 independent model-shaped 64-token numerical samples.

The decision was fixed before observing BI100 results:

- chunk size `64`, Neumann order `3`, residual terms `8`;
- all inverse, output, and final-state values finite;
- maximum absolute error no greater than `1e-3`;
- relative L2 error no greater than `1e-5`;
- 4096-token inverse speedup at least `2.0x`;
- complete 4096-token chunk-rule speedup at least `1.5x`;
- candidate peak allocated memory no greater than the exact baseline;
- no order, residual-depth, chunk-size, tolerance, seed, or launch scan.

The `1.5x` complete-path gate preserves the earlier E-GDN-15/E-GDN-16
contract. If any gate fails, M1-40 is rejected without model integration. If
all gates pass, the next step is a default-off model A/B with the existing
long-state and 256K capacity checks; passing this microbenchmark alone does
not authorize integration.

## Repository Contract

- private ModelHub experiment branch only;
- no `main` merge before all numerical, service, and score gates pass;
- no repository visibility changes;
- `qwen3_6_scripts/qwen3_5.py`, prebuilt extensions, Dockerfile,
  `computility-run.yaml`, and running services remain unchanged.

## Status

`PERFORMANCE_REJECTED`.

The fixed gate ran once on physical BI100 GPU0. The process returned `1`
because one predeclared qualification gate failed; it completed normally and
wrote the full report. All 100 stress samples were finite and passed parity.
The worst stress errors were:

| Boundary | Max absolute | Relative L2 |
| --- | ---: | ---: |
| Inverse | `2.98e-8` | `1.18e-8` |
| Output | `1.40e-9` | `5.99e-8` |
| Final state | `2.24e-8` | `7.59e-8` |

The multiplication-only inverse is materially faster, but the gain is
diluted by the unchanged remainder of the chunk rule:

| Tokens | Inverse baseline | Candidate | Speedup | Complete baseline | Candidate | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | `3.999 ms` | `0.527 ms` | `7.585x` | `4.712 ms` | `1.158 ms` | `4.068x` |
| 1024 | `4.566 ms` | `0.761 ms` | `5.998x` | `9.016 ms` | `5.082 ms` | `1.774x` |
| 4096 | `5.747 ms` | `1.548 ms` | `3.712x` | `22.123 ms` | `17.801 ms` | `1.243x` |
| 7800 | `7.219 ms` | `3.467 ms` | `2.082x` | `42.148 ms` | `33.486 ms` | `1.259x` |

Peak allocated memory was equal to the exact baseline at every length. The
4096-token complete-path result misses the fixed `1.5x` gate by a wide margin,
so no model integration, service A/B, prebuilt artifact, or TP4 run is
authorized. Do not scan `N`, `S`, chunk size, precision, tolerance, or launch
configuration. A future GDN design must fuse a larger complete-layer boundary;
accelerating this inverse alone has insufficient score leverage.

Evidence:

- `docs/experiments/evidence/M1_40_MASKED_NEUMANN_GDN_RESULT.json`
- SHA-256 `3ee2cbaddd727ecf9fc740c95761ede5d4367374177d2416d48a7493ef2f194a`
- remote source `/root/M1_40/result.json`
