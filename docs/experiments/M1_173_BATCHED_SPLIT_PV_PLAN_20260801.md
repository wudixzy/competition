# M1-173 batched split-PV plan

## Motivation

M1-156 attributed roughly one third of fused-prefill time to PV. The M1-162
FP16-QK candidate improved the operator by 15.4-17.1%, but retained up to four
FP32 `cublasSgemmStridedBatched` PV calls per 2048-token group. M1-172 proved
that mixed FP16-value/FP32-probability GEMM is unsupported by this CoreX cuBLAS,
so M1-173 keeps the full PV input and accumulation contract in FP32.

## Candidate

The candidate duplicates each gathered FP32 value tile over the four query
heads and presents split and head as one contiguous batch dimension. One SGEMM
call with `batchCount=active_splits*4` replaces the per-split call loop. QK,
online normalization, merge order, output division, and all model behavior are
unchanged. The implementation is behind `BI100_BATCHED_SPLIT_PV`; the default
source path and production overlay do not enable it.

The tradeoff is four value-tile writes instead of one. M1-156 measured gather
below 0.6% of operator time, while the candidate removes up to three host API
and device launch boundaries per group. This experiment tests that structural
tradeoff rather than scanning parameters.

## Frozen screen

Compare against the M1-162 FP16-QK extension on physical GPUs 1, 2, and 3 at
the fixed 16K, 32K, and 64K calibrated cells. Require finite results, calibrated
numeric qualification, exact repeatability, no cell below 0.98x, and median
speedup of at least 1.08x. A passing screen authorizes only real-activation
replay. It does not authorize TP4, production overlay, YAML, or `main` changes.

If compilation fails, any cell fails numerical qualification, or the performance
gate fails, close this exact route without tile or threshold scanning.
