# M1-35 Canonical GDN State Retention - 2026-07-21

## Objective

M1-34 made `admission64/direct` correct for the known single-token replay case
and raised effective cache hit by 11.137 percentage points, but its fixed
matrix score improved only 2.6945%. Per-request decomposition showed that nine
warm requests took 0.251 seconds longer in aggregate than the frozen baseline.

The scheduler was asking workers to capture the final recurrent state even
when that exact content key was already resident and selected for restore.
Every rank consequently copied approximately 16 MiB of unchanged state from
GPU to CPU and cloned the CPU tensor again before returning the first token.

## Change

`GdnPrefixStatePolicy.should_capture_final` makes the capture contract
explicit:

- `admission64` captures a final content key only while it is absent;
- a warm restore refreshes scheduler LRU through `select_restore`, but retains
  the first cold-captured canonical worker state;
- `fine32` continues its rolling per-step captures unchanged;
- `off` does not request a final capture.

This removes a redundant transfer and avoids replacing a canonical state with
a state recomputed from a restored suffix. Capacity, eviction, restore keys,
worker metadata, YAML, and the default `fine32/direct` policy are unchanged.

`scripts/run_m1_35_canonical_matrix.sh` compares repository and installed
runtime SHA-256 values for both policy and scheduler modules, executes the
capture contract, and only then delegates to the same guarded `m1_32_ab`
matrix used by M1-34.

## Validation

Local discovery passed 226 tests with 24 optional-dependency skips;
submission preflight passed 8/8. Remote runtime preflight passed with empty
stderr:

| Installed module | SHA-256 |
| --- | --- |
| `vllm/gdn_prefix.py` | `a48da42a...b37a0` |
| `vllm/core/scheduler.py` | `f44f1903...c33b` |

The fixed matrix completed 18/18 requests. Startup, capacity, smoke, matrix,
and summary exit codes were zero, and the service log contained no fatal,
CUDA, OOM, Gloo, or worker-loss event.

| Metric | Frozen baseline | M1-34 | M1-35 | M1-35 vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Success | 100% | 100% | 100% | 0 pp |
| Effective cache hit | 49.9301% | 61.0671% | 61.0671% | +11.1370 pp |
| Output TPS P10 | 21.6563 | 21.3347 | 21.7783 | +0.56% |
| Input TPS | 741.4479 | 841.9203 | 841.2359 | +13.46% |
| Cache TPS | 7,607.9233 | 7,437.7376 | 7,600.5593 | -0.10% |
| TTFT P90, all | 20.8748 s | 18.2191 s | 18.0882 s | -13.35% |
| Warm TTFT sum | 10.9843 s | 11.2357 s | 10.9950 s | +0.10% |
| Weighted proxy | 6,699.4888 | 6,880.0051 | 6,976.7204 | +4.14% |

Canonical retention recovered `0.2407s` of M1-34 warm time and `96.7153`
weighted points. It did what the design predicted, but the predeclared stage
threshold is `7034.4632`; M1-35 remains `57.7428` points short.
`compare.rc=1` and `qualification.rc=1`, so long-context gates were not run.

## Decision

Status: `PERFORMANCE_REJECTED`.

Do not repeat the matrix until random variation crosses the threshold. The
next admissible investigation is a bounded operator test of the restore copy
itself. The current worker first materializes a temporary GPU tensor with
`saved_state.to(device)` and then copies that tensor into the live Mamba
buffer. A direct CPU-to-destination `copy_` can remove the allocation and
second device copy without extra GPU memory. It may be integrated only if a
fixed-shape CoreX microbenchmark is exact and saves enough absolute time to
matter; otherwise the cache micro-optimization direction stops.
