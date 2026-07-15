# E-NORM-01 exact Gemma RMSNorm elementwise fusion

## Scope

This experiment replaced the FP16 elementwise portions of the Qwen3.6 Gemma
RMSNorm path with a CoreX extension. The PyTorch `mean` and `rsqrt` reduction
sequence remained unchanged so the candidate preserved the qualified output
hash. The fast path was restricted to contiguous CUDA FP16 decode tensors with
shape `(1, 2048)`; all other inputs used the existing vLLM implementation.

The TP4 service used the production configuration, including 262,144 context,
prefix caching, chunked prefill, and `ENABLE_CUSTOM_IPC=1`.

## Microbenchmark

All four GPUs passed 1,000/1,000 exact comparisons for both residual and
non-residual calls, with zero maximum absolute error. The paired decode-shape
microbenchmark reported:

| GPU | Reference pair ms | Candidate pair ms | Speedup |
| ---: | ---: | ---: | ---: |
| 0 | 0.180573 | 0.086504 | 2.087x |
| 1 | 0.181835 | 0.085547 | 2.126x |
| 2 | 0.181897 | 0.085649 | 2.124x |
| 3 | 0.181802 | 0.086863 | 2.093x |

## Correctness gates

| Gate | Result |
| --- | --- |
| Service startup | HTTP health 200 |
| Quick API smoke | 8/8 pass |
| Forced decode | 1,000 tokens, finish=length |
| Forced-decode SHA256 | `1766c3c44bfb672e32b2e35419c5e06490e539e54250ab2fc1012c539e68835f` |
| Qualified-hash equality | exact |

## End-to-end paired result

The comparison used the same binary and service arguments. The baseline set
`BI100_GEMMA_COREX_RMS_NORM=0`; the candidate used the default enabled path.
Each pair reused the same prompt salt, seed, request count, and token limits.

| Pair | Candidate Output TPS P10 | Baseline Output TPS P10 | Change |
| ---: | ---: | ---: | ---: |
| 1 | 15.6920 | 15.5548 | +0.88% |
| 2 | 15.3442 | 15.5085 | -1.06% |
| 3 | 15.5825 | 15.7213 | -0.88% |
| Mean | 15.5396 | 15.5948 | -0.35% |

The first pair included more baseline warm-up cost. Across the two later
pairs, candidate throughput was approximately 0.97% lower. The candidate did
not demonstrate a stable end-to-end decode gain despite the isolated kernel
speedup.

## Artifacts

```text
/root/e_norm_01/results/gpu0.json
/root/e_norm_01/results/gpu1.json
/root/e_norm_01/results/gpu2.json
/root/e_norm_01/results/gpu3.json
/root/e_norm_01/smoke_quick.json
/root/e_norm_01/sustained_1000.json
/root/e_norm_01/bench_candidate_1.json
/root/e_norm_01/bench_candidate_2.json
/root/e_norm_01/bench_candidate_3.json
/root/e_norm_01/bench_baseline_1.json
/root/e_norm_01/bench_baseline_2.json
/root/e_norm_01/bench_baseline_3.json
```

## Decision

`REJECTED FOR MAIN`: correctness passed, but the paired production-path
benchmark showed no stable throughput improvement. Keep the implementation on
the experiment branch as evidence and do not add its build or runtime path to
the submission baseline.
