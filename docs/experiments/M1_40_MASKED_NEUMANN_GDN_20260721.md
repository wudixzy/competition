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

`READY_FOR_FIXED_BI100_GATE`. Local static tests are available without Torch;
the numerical and performance decision requires one healthy BI100 card.
