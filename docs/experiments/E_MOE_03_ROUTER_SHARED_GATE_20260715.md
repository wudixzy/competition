# E-MOE-03: Fuse router and shared-expert gate

Date: 2026-07-15

## Hypothesis

Every MoE layer applies a 256-output router linear and a separate one-output
shared-expert gate linear to the same hidden state. Storing both checkpoint
tensors in one 257-row replicated weight should remove one GEMM and launch per
layer without changing router logits or the shared-expert sigmoid gate.

## Manifest

```text
baseline commit: e21004eaef875300540a63b7da0fbeab6b976a49
candidate commit: 7a68a9424116ea793fdf99c3789ea68fefdb9ce1
branch: exp/E-MOE-03-router-shared-gate
model: /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
hardware: 4 x BI-V100-50C-200G, TP=4
max model length: 262144
```

## Change

`Qwen3_5MoeSparseBlock` replaces its separate `gate` and
`shared_expert_gate` modules with one `router_shared_gate` replicated linear.
Forward slices rows 0-255 as router logits and row 256 as the scalar shared
gate. A fail-closed loader maps the original checkpoint tensors into disjoint
rows and rejects unknown shards or mismatched shapes.

The fixed evaluator command and environment are unchanged.

## Primitive results

`tests/bench_moe_router_fusion.py` used FP16 weights with hidden size 2,048,
256 router outputs, and one shared-gate output. Both T=1 and T=64 outputs were
bit-exact on all four devices.

| GPU | T=1 separate/fused (ms) | Speedup | T=64 separate/fused (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.202989 / 0.115994 | 1.750x | 0.190406 / 0.103905 | 1.833x |
| 1 | 0.201881 / 0.111428 | 1.812x | 0.189857 / 0.103564 | 1.833x |
| 2 | 0.200826 / 0.113669 | 1.767x | 0.188788 / 0.103194 | 1.829x |
| 3 | 0.200692 / 0.115104 | 1.744x | 0.189316 / 0.103034 | 1.837x |

Artifacts are under `bench_runs/20260715_E_MOE_03/gpu*.json`.

## Correctness gates

- fused loader unit tests: 3/3 pass
- MoE parity tests: 4/4 pass
- static tests: 40/40 pass
- all 26 checkpoint shards loaded on all four ranks
- full API smoke: 15 passed, 0 failed, 0 skipped
- candidate service: HTTP 200, max context 262,144
- baseline blocks: 16,871 GPU / 6,553 CPU
- candidate blocks: 16,878 GPU / 6,553 CPU
- no loader error, traceback, OOM, fatal error, or worker loss

The initial implementation attempted to install a second `weight_loader`
attribute with `set_weight_attrs`, which correctly failed code review because
`ReplicatedLinear` already owns that attribute. It was changed to an explicit
loader replacement before site-package installation or candidate startup; the
published candidate commit contains only the corrected implementation.

## Strict performance pairs

Each side used eight serial streaming requests, 64 generated tokens, and the
same salt within each pair. Every run reported 14,440 prompt tokens, 1,896
uncached prompt tokens, 12,544 cached prompt tokens, 512 completion tokens,
100% success, and an 86.8698% cache hit rate. Ordering was
baseline-to-candidate for A, candidate-to-baseline for B, and
baseline-to-candidate for C.

| Pair | Metric | Baseline | Candidate | Change |
| --- | --- | ---: | ---: | ---: |
| A | Output TPS P10 | 12.8697 | 13.2554 | +3.00% |
| A | ITL P90 (ms) | 78.2264 | 76.7808 | -1.85% |
| A | Weighted short score | 1138.6272 | 1174.7513 | +3.17% |
| B | Output TPS P10 | 12.5925 | 13.3203 | +5.78% |
| B | ITL P90 (ms) | 77.9958 | 76.5694 | -1.83% |
| B | Weighted short score | 1101.5378 | 1176.8092 | +6.83% |
| C | Output TPS P10 | 12.7909 | 13.5399 | +5.86% |
| C | ITL P90 (ms) | 76.9776 | 74.6198 | -3.06% |
| C | Weighted short score | 1148.8578 | 1158.0689 | +0.80% |

Median changes are +5.78% Output TPS P10, -1.85% ITL P90, and +3.17%
weighted short score. Output TPS, ITL, and weighted score improve in every
pair. TTFT remains startup-sensitive (-26.59% to +26.30%) and is not claimed
as a benefit. The weighted value is a local short-test diagnostic, not the
official competition score.

## Long-context and sustained-decode gates

| Request | Elapsed (s) | Prompt | Cached | Completion |
| --- | ---: | ---: | ---: | ---: |
| 235K cold | 503.270 | 235,000 | 0 | 8 |
| 235K warm | 43.172 | 235,000 | 234,544 | 8 |

Both 235K requests returned `FINAL-99500`, stopped normally, and matched the
qualified message SHA256:

```text
a3dc73d02269b1b3682ed84197c3d2d0ddc39dfdb544f73fb3ea832f1fb30b4d
```

A request with `min_tokens=max_tokens=1000` returned HTTP 200 in 76.278
seconds with exactly 1,000 completion tokens and `finish_reason=length`. Its
message SHA256 also matches E-MOE-02:

```text
1766c3c44bfb672e32b2e35419c5e06490e539e54250ab2fc1012c539e68835f
```

The service remained healthy with no fatal error, OOM, or worker loss.

## Decision

`KEEP AS PERFORMANCE WINNER`. The fused gate is bit-exact at real shapes,
loads the complete checkpoint, improves Output TPS P10 in all three matched
pairs, and passes full smoke, 235K cache reuse, and 1,000-token sustained
decode. Integrate `7a68a94` on top of E-MOE-02. Observed Output TPS remains
13.26-13.54, below the competition target of 20, so further expert-kernel and
collective optimization is still required.
