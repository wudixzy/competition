# E-NORM-02 exact attention head RMSNorm elementwise fusion

## Scope

E-NORM-02 targets the full-attention q/k head norms at the TP4 single-token
decode shape:

```text
Q: (1, 4, 256)
K: (1, 1, 256)
```

The CoreX extension fuses FP16-to-FP32 conversion with square generation and
fuses inverse scaling, Gemma weight application, and FP16 conversion. PyTorch
still performs `mean` and `rsqrt`, preserving the qualified reduction order.
The production path requires contiguous FP16 input, FP16 weight, no residual,
head dimension 256, and `x.shape[0] == 1`; prefill and unsupported layouts use
the original `GemmaRMSNorm.forward_cuda` path. The opt-out is
`BI100_ATTN_COREX_HEAD_RMS_NORM=0`.

## Four-GPU microbenchmark

Each GPU ran 1,000 random exactness steps for Q and K, followed by nine timing
trials of 1,000 iterations. All outputs were bit-exact with maximum absolute
error zero.

| GPU | Q exact | K exact | Reference Q+K ms | Candidate Q+K ms | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1000/1000 | 1000/1000 | 0.14582 | 0.07525 | 1.9378x |
| 1 | 1000/1000 | 1000/1000 | 0.14585 | 0.07536 | 1.9354x |
| 2 | 1000/1000 | 1000/1000 | 0.14838 | 0.07943 | 1.8682x |
| 3 | 1000/1000 | 1000/1000 | 0.14566 | 0.07557 | 1.9274x |

The absolute saving is approximately 0.07 ms per full-attention layer, or
0.70 ms per decode token across ten layers.

## Correctness and stability

The TP4 service used the production 262,144-context command with chunked
prefill, prefix caching, and `ENABLE_CUSTOM_IPC=1`.

| Gate | Result |
| --- | --- |
| Service health | HTTP 200 |
| Full API/multimodal/tool smoke | 15/15 pass |
| Forced decode | 1,000 tokens, finish=length |
| Forced-decode elapsed | 68.291 s |
| Forced-decode SHA256 | `1766c3c44bfb672e32b2e35419c5e06490e539e54250ab2fc1012c539e68835f` |
| Qualified-hash equality | exact |
| Continuous healthy runtime before smoke | more than 2 h 37 min |
| Service log | no ERROR/Traceback/Gloo/NCCL/OOM/native crash |

Long-prefill math is unchanged because the fast path requires one token in
the leading dimension. Existing 99.5K and 235K qualification therefore
continues to exercise the original Gemma RMSNorm implementation during
prefill.

## Strict paired TP4 result

Candidate and baseline used the same binary, service command, prompt salts,
seed, request count, and cache setup. The baseline changed only
`BI100_ATTN_COREX_HEAD_RMS_NORM=0`.

| Pair | Candidate Output TPS P10 | Baseline Output TPS P10 | Change |
| ---: | ---: | ---: | ---: |
| 1 | 15.8218 | 15.6769 | +0.92% |
| 2 | 15.7884 | 15.7003 | +0.56% |
| 3 | 15.8515 | 15.7029 | +0.95% |
| Mean | 15.8206 | 15.6934 | +0.81% |

Candidate success rate was 100% in all three runs. Candidate TTFT P90 median
was 2.135 s versus 2.159 s for baseline. The weighted local sample mean rose
from 1311.06 to 1335.87, but that short-prompt value is not comparable with
the official 8,000 target.

## Artifacts

```text
/root/e_norm_02/results/gpu0.json
/root/e_norm_02/results/gpu1.json
/root/e_norm_02/results/gpu2.json
/root/e_norm_02/results/gpu3.json
/root/e_norm_02/sustained_1000.json
/root/e_norm_02/bench_candidate_1.json
/root/e_norm_02/bench_candidate_2.json
/root/e_norm_02/bench_candidate_3.json
/root/e_norm_02/bench_baseline_1.json
/root/e_norm_02/bench_baseline_2.json
/root/e_norm_02/bench_baseline_3.json
/root/e_norm_02/smoke_full.json
/root/e_norm_02/service_final.log
```

## Decision

`KEEP AND MERGE`: all numerical and API gates pass, and all three strict
paired runs improve Output TPS P10. The end-to-end gain is small but stable,
the prefill path is untouched, and the environment switch provides an
immediate rollback.
