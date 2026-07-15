# M1-09 current TP4 decode coverage - 2026-07-15

## Scope

This follow-up profiles the fully qualified `d7f28cb` runtime after the E-MOE,
E-GDN, and E-ATTN winners were combined. Instrumentation lives only on
`exp/M1-09-current-tp4-profile`; it is not part of `main` or the submission
image. The branch adds filtered, mutually exclusive timers for:

- the two decoder residual/RMSNorm calls;
- complete GDN and full-attention layers;
- MoE router, routed experts, shared expert, combine, and all-reduce.

`BI100_PROFILE_FILTER` suppresses the older nested timers so their CUDA
synchronizations do not double-count the same region. The summary tool groups
records by process and every 40 input norms, then drops the prefill forward.

## Stable coverage run

The first request exposed lazy-initialization outliers and was retained only as
warm-up. A second independent request used 1,808 prompt tokens and eight output
tokens. It produced seven decode forwards on each of four TP ranks, for 28
complete samples with no outlier or incomplete forward.

The coverage service did not explicitly enable `ENABLE_CUSTOM_IPC`, so the
all-reduce portions below use IxFormer NCCL. Timers synchronize every region;
the values rank hotspots but must not be compared with formal end-to-end TPS.

| Region | Mean ms/token/rank | Share of tracked time |
| --- | ---: | ---: |
| Complete GDN, 30 layers | 29.215 | 25.94% |
| Routed experts, 40 layers | 26.349 | 23.40% |
| Complete full attention, 10 layers | 13.403 | 11.90% |
| MoE all-reduce, 40 calls | 13.167 | 11.69% |
| Input residual/RMSNorm, 40 calls | 8.736 | 7.76% |
| Post-attention residual/RMSNorm, 40 calls | 8.530 | 7.57% |
| Shared expert, 40 layers | 6.358 | 5.65% |
| Router, 40 layers | 5.246 | 4.66% |
| Routed/shared combine, 40 layers | 1.622 | 1.44% |
| **Tracked total** | **112.627** | **100%** |

The two `GemmaRMSNorm` regions total 17.266 ms, or 15.33% of synchronized
tracked time. Source inspection confirms that CoreX vLLM's Gemma variant calls
`forward_native`: it does not use the installed IxFormer fused RMSNorm path.

## Formal IPC baseline

The submission image already defaults `ENABLE_CUSTOM_IPC=1`. On the same
healthy host, the 2,048-element four-rank primitive remained exact and improved
from 0.2431 ms with IxFormer NCCL to 0.02744 ms with IPC. A production-code,
non-profile service with the Docker environment reached health 200 and passed
the fixed eight-request sample:

| Metric | Result | Competition gate |
| --- | ---: | ---: |
| Request success | 100% | >=99% |
| TTFT P90 | 2.628 s | <=5 s |
| Output TPS P10 | 16.0524 | >=20 |
| Cache hit rate | 86.77% | >=50% |
| Input TPS | 321.1503 | dataset-dependent |
| Cache TPS | 278.6738 | dataset-dependent |

IPC is therefore active and useful but does not close the current decode gap.
Reaching 20 from 16.0524 still requires 24.59% relative improvement.

The 262,144-token synthetic capacity profile is slow: the production IPC
service completed `init_device` in nine seconds and loaded weights in about one
minute, but the first synthetic model forward required about six minutes 27
seconds. It eventually initialized cache and served health 200; this is startup
latency, not the earlier permanent Gloo failure.

## Decision

Start `E-NORM-01` before another routed-expert approximation. Preserve
PyTorch's FP32 square/mean/rsqrt reduction order and first fuse only the exact
elementwise boundaries around it. Required gates are:

1. bit-exact normalized output and residual for residual/no-residual modes;
2. real decode shape `(1, 2048)`, FP16, checkpoint Gemma weights;
3. at least 1.3x combined two-norm microbenchmark speedup;
4. 1,000-token qualified hash, full smoke, and three fixed paired IPC A/B runs.

Do not merge the profile instrumentation into production. The fixed evaluator
YAML remains unchanged.

## Remote artifacts

```text
/root/profile_2f9312e/service_run2.log
/root/profile_2f9312e/request_run2.json
/root/profile_2f9312e/profile_summary_run2.json
/root/profile_2f9312e/allreduce_nccl.json
/root/profile_2f9312e/allreduce_ipc.json
/root/profile_2f9312e/bench_fixed_ipc_formal.json
/root/competition-candidate/service_ipc_formal.log
```
