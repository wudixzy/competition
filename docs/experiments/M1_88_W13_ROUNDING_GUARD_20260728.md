# M1-88: W13 rounding-risk guard

## Status

`READY FOR SINGLE-GPU EXECUTION`.

The diagnostic harness and fail-closed qualifier are implemented on
`exp/M1-88-w13-rounding-guard-20260728`. No BI100 result exists yet because
the available host's SSH TLS/ProxyCommand forwarding layer did not recover.
This document does not qualify a kernel, runtime default, submission YAML, or
main-branch change.

## Motivation

The direct routed W13 kernel removes the selected-weight gather and is much
faster than the vendor `F.linear` path. M1-60 recorded a fixed-case relative
L2 error of `7.302e-6` for seed `20260716`. The later private M1-61 branch
(`exp/M1-61-exact-w2-hybrid-20260727`, gate commit `c039504`) kept the same
fixture construction and measured direct W13 at `2.452e-5` for seed
`20260727`. The direct path therefore does not satisfy the current
`relative L2 <= 1e-5` quality contract across fixed seeds. Earlier reduction
experiments also showed that changing a fast matvec's accumulation order can
move FP16 outputs across rounding boundaries.

M1-88 tests one bounded hypothesis:

1. Run the production direct W13 accumulation order and one fixed reverse
   order in the same diagnostic kernel.
2. Flag only rows whose two FP32 sums round to different FP16 values.
3. Recompute flagged rows with a CPU float64 dot-product oracle and round the
   result to FP16.
4. Compare both the detector and the oracle against the existing vendor
   `F.linear` result.

This is a feasibility probe, not a production correction path.

## Fixed contract

- Shape per TP4 rank: `E=256`, `top_k=8`, `H=2048`, `I=128`, FP16.
- Seeds: `20260716` and `20260727`.
- Sequence: 500 steps per seed.
- Fixture generation: hidden state, router logits, W13 weights, then sequence
  inputs. This preserves the fixed W13 fixture order used by the historical
  direct-kernel benchmark.
- Forward probe output must be bit-exact with the production direct W13
  extension.
- Every vendor mismatch must be flagged.
- Float64-rounded correction must match vendor output on every flagged row.
- Corrected relative L2 must not exceed `1e-5`.
- Aggregate flagged rows must not exceed 5%; no step may exceed 10%.
- The existing direct-kernel numerical gap must be reproduced.

Any failed condition rejects this direction. A pass authorizes only one later
bounded correction-kernel prototype. It does not authorize production
integration.

## Lifecycle

`scripts/run_m1_88_w13_rounding_guard.sh`:

- requires a clean source tree and immutable runtime identity;
- creates private artifacts outside the repository under `/tmp`;
- executes build and benchmark commands in attested process sessions;
- sends `SIGTERM` to only the recorded process group and waits 60 seconds;
- sends `SIGKILL` only to surviving recorded processes;
- waits/reaps the leader and performs recorded-session recovery;
- runs API/worker/GPU-holder postflight checks;
- runs GPU preflight before and after the experiment and compares them;
- scans logs for fatal GPU, Gloo, NCCL, worker-loss, and timeout evidence.

Postflight, cleanup, fatal scan, timeout scan, or GPU comparison failure makes
the experiment invalid even if the numerical qualifier passes.

## Execution

On a healthy BI100 host with a runtime overlay built from the exact source
revision:

```bash
BI100_RUNTIME_SITE_PACKAGES=/path/to/runtime/site-packages \
  scripts/run_m1_88_w13_rounding_guard.sh \
  0 INSTANCE_LABEL /tmp/m1-88-w13-rounding-guard-RUN_ID
```

The run root must be new. The runner never changes `computility-run.yaml`,
Docker inputs, runtime defaults, Git visibility, or `main`.

## Local validation

Before the first GPU run:

- focused M1-88 unit and runner tests: 15 passed;
- full `tests/` discovery: 1009 passed, 25 dependency skips;
- submission preflight: passed;
- quality data manifest validation: passed;
- quality metric manifest validation: passed;
- shell and Python syntax checks: passed.

CUDA compilation, BI100 numerical output, and performance were not tested on
the local non-GPU environment.
