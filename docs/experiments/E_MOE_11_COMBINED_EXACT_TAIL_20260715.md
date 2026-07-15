# E-MOE-11: Combined exact decode tail

## Scope

E-MOE-11 combines two independently exact T=1 routed-expert changes:

- vLLM `SiluAndMul` for the routed SwiGLU activation from E-MOE-05;
- the CoreX FP16-product/FP32-sum reduction from E-MOE-10.

The prefill path, routing, selected weights, GEMMs, evaluator command, and
checkpoint layout remain unchanged. `BI100_MOE_FUSED_ACTIVATION=0` and
`BI100_MOE_COREX_EXACT_REDUCE=0` provide independent service fallbacks.

## Method

`tests/bench_moe_combined_exact_tail.py` measures four complete routed decode
paths with the checkpoint's TP4 rank-local dimensions:

```text
experts=256, top_k=8, hidden=2048, local_intermediate=128, dtype=float16
```

The test uses the runtime `SiluAndMul` implementation and the production-form
`serial_float` CoreX extension. Each GPU ran 30 warmups, 9 repeats of 300
iterations, and 1,000 random full-path exactness steps.

## Results

| GPU | Native ms | Fused activation ms | Exact reduce ms | Combined ms | Combined speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.504801 | 0.490465 | 0.478599 | 0.459196 | 1.0993x |
| 2 | 0.505399 | 0.490873 | 0.479035 | 0.459749 | 1.0993x |
| 3 | 0.505893 | 0.491247 | 0.479304 | 0.459968 | 1.0998x |

All fixed-path outputs and all 1,000 random full-path outputs on every device
were bit-exact (`max_abs=0`). The median complete-path saving is 0.04565 ms per
MoE layer, or about 1.83 ms/token across 40 MoE layers. Relative to E-MOE-10
alone, the fused activation contributes another approximately 0.78 ms/token.

Raw artifacts are intentionally not committed:

```text
/root/competition/bench_runs/20260715_E_MOE_11/gpu1.json  caa50495...f347fea
/root/competition/bench_runs/20260715_E_MOE_11/gpu2.json  6074be83...690e5b9
/root/competition/bench_runs/20260715_E_MOE_11/gpu3.json  a4d1d0f1...15e495
```

## Decision

`QUALIFY FOR TP4 SERVICE A/B`. The stable three-device gain is approximately
9.9%, exceeds the 5% full-boundary gate, and preserves the exact-output
contract. This supersedes E-MOE-10 in the combined candidate stack; E-MOE-10
remains the independently switchable reduction component.

GPU0 on `ssh-a2d0a302.default.gpu.phanthy.com` remains at 257 MiB and 100%
utilization with no visible process, while GPU1-3 pass CUDA probes. TP4 service
qualification still requires a host-side reset or a healthy four-card host.
