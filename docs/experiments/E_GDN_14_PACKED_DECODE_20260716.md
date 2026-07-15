# E-GDN-14: Packed CoreX GDN decode

## Scope

E-GDN-14 tests an algorithmic boundary taken from the upstream Qwen3.5
packed-decode design: one shape-specific CoreX kernel covers FP16 q/k
normalization, key-head to value-head mapping, beta/decay preparation, FP32
state decay, delta update, and output reduction.

The prototype is isolated to `tests/` and is not wired into the model. It does
not change `computility-run.yaml`, state layout, causal convolution, gated
RMSNorm, or either projection.

Production TP4 rank-local shape:

```text
B=1, key_heads=4, value_heads=8, K=V=128
mixed_qkv=[1,2048] FP16
state=[1,8,128,128] FP32
```

## Implementation

Each `(batch,value_head)` launches one 128-thread block. Adjacent threads own
adjacent value columns, while each thread walks the 128 key rows. The kernel
uses a shared-memory FP16 reduction for q/k norms, reproduces the qualified
beta FP16 round trip, computes decay in FP32, and updates the existing `[K,V]`
state in place.

This preserves the current state allocation and prefix-cache contract. The
upstream `[V,K]` state migration is deliberately excluded because our tile is
symmetrical (`128x128`) and would require a much broader prefill/cache change.

## Results

Fixed protocol: 50 warmups, nine trials of 500 iterations, serial execution.
The quality sequence contains 1,000 random packed inputs and retains state
between steps.

| Physical GPU | Reference median (ms) | Candidate median (ms) | Speedup |
| --- | ---: | ---: | ---: |
| GPU1 | 0.296823 | 0.108354 | 2.739x |
| GPU2 | 0.317750 | 0.110410 | 2.878x |
| GPU2 serial repeat | 0.296469 | 0.109385 | 2.710x |
| GPU3 | 0.298492 | 0.108397 | 2.754x |

GPU2's first candidate median was `0.110410 ms`, just above the predeclared
`0.110 ms` absolute boundary. The untouched serial repeat reached
`0.109385 ms`; no launch or block parameter was changed.

The test reference intentionally expresses the complete boundary in PyTorch
and therefore measures about `0.297 ms`, not the already optimized production
stage sum. E-GDN-13 measured the relevant production sum as `0.165235 ms` per
layer. Comparing the candidate's approximately `0.109 ms` absolute latency to
that production profile projects about `0.056 ms/layer`, or `1.68 ms/token`
over 30 GDN layers. This projection is only a prioritization signal.

## Numerics

All three physical GPUs reproduced the same deterministic bounds:

| Gate | Result |
| --- | ---: |
| One-step output max abs | 2.4252e-5 |
| One-step state max abs | 2.0212e-4 |
| Random-sequence finite | 1000/1000 |
| Random-sequence output max abs | 6.4753e-5 |
| Random-sequence output mean abs | 1.4640e-6 |
| Final state max abs | 4.4093e-4 |
| Final state mean abs | 1.2445e-5 |

The drift is expected from the custom FP16 reduction tree and changed FP32
accumulation order. It is small and bounded in this synthetic sequence, but it
is not sufficient evidence of model quality.

Remote artifacts:

```text
/root/E_GDN_14/result_gpu1.json
/root/E_GDN_14/result_gpu2.json
/root/E_GDN_14/result_gpu2_repeat.json
/root/E_GDN_14/result_gpu3.json
```

## Decision

`QUALIFIED TP4 WINNER`.

The larger boundary and cross-device stability justify one guarded production
integration after E-MOE-20 qualification. Do not merge it based on this
microbenchmark. The production candidate must pass the Agent workload matrix,
multimodal smoke, deterministic repeated decode, and 99.5K/235K cold-warm
requests before a three-pair TP4 service A/B. A service gain below 5% in any
clean pair closes the candidate without block-size tuning.

## Production boundary and v2 dataflow

The guarded production integration is based on `main@101f0d7` and remains
default-off behind `BI100_GDN_COREX_PACKED_DECODE=0`. It matches only one
decode token with FP16 contiguous post-convolution input, four local key heads,
eight local value heads, 128-wide heads, and contiguous FP32
`[1,8,128,128]` state. Unsupported inputs retain the existing E-GDN-10/12 and
`bmm/baddbmm_` path.

The first production comparison used the qualified beta/decay and q/k-map
extensions as its baseline rather than pure PyTorch. It passed the relative
`1.5x` gate on all cards, but GPU1 and GPU2 narrowly missed the absolute
`0.110 ms` gate:

| GPU | Current boundary (ms) | Packed v1 (ms) | Speedup | Strict pass |
| --- | ---: | ---: | ---: | --- |
| GPU1 | 0.19509 | 0.11091 | 1.759x | no |
| GPU2 | 0.17551 | 0.11015 | 1.593x | no |
| GPU3 | 0.19795 | 0.10904 | 1.815x | yes |

The result was not accepted by relaxing the threshold. A dataflow audit found
that v1 wrote the complete decayed state and immediately read and wrote it
again for the rank-one update. V2 keeps the original state through the memory
reduction, recomputes the identical FP32 decay multiplication in the update
pass, and writes only the final state. It also computes normalized q/k once
per block in shared memory instead of once per output column. No launch shape
or service parameter changed.

| GPU | Current boundary (ms) | Packed v2 (ms) | Speedup |
| --- | ---: | ---: | ---: |
| GPU1 | 0.17306 | 0.03673 | 4.712x |
| GPU2 | 0.19798 | 0.03776 | 5.243x |
| GPU3 | 0.17459 | 0.03674 | 4.752x |

All v2 runs passed 1,000/1,000 finite steps. Output max/mean abs remained
`6.4753e-5/1.4640e-6`; final-state max/mean abs remained
`4.4093e-4/1.2445e-5`. The candidate latency now has substantial margin below
`0.110 ms`. Relative to the measured current boundary, the saved
`0.136-0.160 ms/layer` projects to about `4.1-4.8 ms/token` over 30 GDN
layers. This is still a projection, not a service result.

Production artifacts are in private branch
`exp/E-GDN-14-production-integration`; remote evidence is under:

```text
/root/E_GDN_14_prod/results/production_gpu{1,2,3}.json
/root/E_GDN_14_prod/results/production_v2_gpu{1,2,3}.json
```

## TP4 service qualification

The service test used one installed binary and changed only
`BI100_GDN_COREX_PACKED_DECODE=0/1`. E-MOE-20 and custom IPC remained enabled;
all service arguments, request seeds, prompt salts, and cache conditions were
identical. The flag0 source fallback passed full smoke `15/15`, Agent workload
`9/9`, and the historical 1,000-token qualification hash before performance
measurement.

| Pair | Flag0 Output TPS P10 | Flag1 Output TPS P10 | Change | Wall change |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20.28395 | 21.95640 | +8.25% | -4.72% |
| 2 | 20.22108 | 21.75182 | +7.57% | -4.10% |
| 3 | 20.34747 | 21.93097 | +7.78% | -4.19% |

Flag1 TTFT P90 was `2.1945/2.2064/2.2019s`, success was 100% in every pair,
and cache hit rate remained `86.82%`. Local overlap-weighted scores increased
from `1610.39/1611.65/1628.43` to `1701.43/1691.79/1711.25`; this local short
prompt score is not directly comparable with the official 8,000 requirement.

Candidate quality passed full smoke `15/15`, Agent workload `9/9`, and two
independent 1,000-token runs. Both candidate runs remained finite, stopped at
1,000 tokens, and reproduced the historical qualification hash
`1766c3c44bfb672e32b2e35419c5e06490e539e54250ab2fc1012c539e68835f`.

The final long-context gate used new run IDs:

| Prompt | Cold (s) | Warm (s) | Warm cached | Result |
| --- | ---: | ---: | ---: | --- |
| 99,500 | 159.161 | 17.368 | 99,296 | cold/warm hash equal |
| 235,000 | 567.031 | 57.019 | 234,544 | cold/warm hash equal |

Both requests completed eight tokens with `finish_reason=stop`; health stayed
200 and the service log contained no fatal, OOM, or traceback. E-GDN-14 meets
the predeclared requirement of at least 5% Output TPS improvement in all three
pairs and is enabled in `computility-run.yaml`. The exact shape guard and
default-off code fallback remain mandatory.

TP4 evidence is under:

```text
/root/e_gdn_14_tp4/flag0
/root/e_gdn_14_tp4/flag1
/root/e_gdn_14_tp4/service_flag{0,1}.log
```
