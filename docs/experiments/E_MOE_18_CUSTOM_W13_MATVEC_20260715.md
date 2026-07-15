# E-MOE-18: Shape-specific W13 matvec

## Hypothesis

E-MOE-16 identified the contiguous `2048x2048` W13 linear as 39.4% of the
current routed boundary, and E-MOE-17 showed that cuBLAS algorithm selection
cannot improve the complete path. E-MOE-18 tested a shape-specific CoreX
matrix-vector kernel intended to read the 8 MiB W13 matrix once and avoid the
general GEMM dispatch.

## Variants

The probe implemented:

- one-warp-per-row FP32 tree reduction;
- 32/64/128/256-thread FP32 segmented reductions;
- one-thread-per-row serial FP32 accumulation;
- 32/64/128/256-thread Kahan FP32 variants;
- 32/64/128/256-thread FP64 segmented variants.

All variants retained FP16 inputs and outputs. No production source was
changed.

## Results

GPU1 baseline:

```text
F.linear W13: 0.136815 ms
fixed full:   0.265840 ms
```

The fastest valid-execution FP32 variant was the warp kernel:

```text
custom W13: 0.023433 ms (5.84x)
fixed full: 0.152229 ms (1.746x)
```

However, every FP32 and Kahan variant changed W13 values by up to
`0.000244-0.000488` after FP16 output rounding. Final routed-expert output
differences were `3.0518e-5` or `6.1035e-5`. The fully serial FP32 variant had
the same mismatch while being much slower (`1.6618 ms`).

The FP64 segmented variants produced grossly incorrect values on this CoreX
backend (`W13 max_abs=3.3887`, final `max_abs=0.17896`) and are invalid.

No mode was bit-exact, so random sequence qualification and cross-device runs
were intentionally skipped. Raw artifacts:

```text
/root/competition/bench_runs/20260715_E_MOE_18/gpu1.json
/root/competition/bench_runs/20260715_E_MOE_18/gpu1-precision-scan.json
```

## Decision

`REJECT FOR PRODUCTION`. The performance opportunity is real, but changing
the W13 reduction order violates the exact-output contract at the same scale
that previously changed the 1,000-token hash in E-MOE-04. Do not integrate the
custom kernel or relax tolerance. A future fused expert kernel must reproduce
the vendor GEMM accumulation result before performance qualification.
