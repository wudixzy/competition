# E-GDN-06: Normalize GDN key heads before expansion

Date: 2026-07-15

## Hypothesis

In decode, each local Q/K head is repeated three times to match the value-head
count. The current path repeats first and then performs the same L2 reduction
three times. Normalizing the four local key heads before expanding them to 12
value heads should reduce repeated reductions without changing any value.

## Manifest

```text
baseline: b1d95009d52135a5b00bbac1c5ccc682c4539644
candidate: 1fe7660b9e65f18ddf79d4406506fa74766266bc
branch: exp/E-GDN-06-normalize-before-expand
hardware: 4 x BI-V100-50C-200G, TP=4
fixed evaluator contract changed: no
```

## Change

`qwen3_6_scripts/qwen3_5.py` moves Q/K L2 normalization before
`repeat_interleave` in the decode-only path. Expansion occurs after conversion
to float32. Prefill, recurrent state math, weights, cache layout, and evaluator
arguments are unchanged.

The experiment also adds `tests/bench_gdn_decode_prep.py` and a scoped P0
static assertion. The benchmark covers the real TP-rank shape:

```text
B=1, Hk=4, Hv=12, K=128, V=128, q/k=float16, state=float32
```

## Correctness gates

- P0 static tests: 41/41 pass
- GDN projection loader tests under CoreX: 4/4 pass
- four-GPU operator parity: exact zero max-absolute error for Q, K, recurrent
  output, and final state; all finite
- deterministic API oracle: 3/3 exact for message, finish reason, and usage
- quick API smoke under CoreX: 8/8 pass in 72.00 seconds
- candidate health after tests: HTTP 200
- candidate service fatal-log scan: zero matches

The first smoke invocation omitted the CoreX package path and could not import
Pillow. It did not issue a model request and is excluded. The corrected CoreX
invocation exited 0.

## Primitive results

All timings are medians in milliseconds. `full` includes normalization,
expansion, state clone/decay, both bmm operations, and the in-place baddbmm
state update.

| GPU | Current prep/full | FP32 expand prep/full | Prep speedup | Full speedup |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.1251 / 0.2187 | 0.1220 / 0.2141 | 1.025x | 1.021x |
| 1 | 0.1338 / 0.2393 | 0.1247 / 0.2340 | 1.073x | 1.023x |
| 2 | 0.1335 / 0.2381 | 0.1240 / 0.2334 | 1.077x | 1.020x |
| 3 | 0.1352 / 0.2408 | 0.1269 / 0.2346 | 1.065x | 1.026x |

The primitive gain is consistent but small: 2.0%-2.6% for the complete step.

Artifacts:

```text
bench_runs/20260715_E_GDN_06/prep/gpu0.json
bench_runs/20260715_E_GDN_06/prep/gpu1.json
bench_runs/20260715_E_GDN_06/prep/gpu2.json
bench_runs/20260715_E_GDN_06/prep/gpu3.json
```

## Exploratory service A/B

Both runs used eight serial streaming requests, 64 completion tokens per
request, seed 123, and salt `E-GDN-06-AB-20260715`.

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Success rate | 100% | 100% | equal |
| Prompt tokens | 14,504 | 14,504 | equal |
| Completion tokens | 512 | 512 | equal |
| Decode TPS P10 | 13.0338 | 12.9996 | -0.26% |
| ITL P50 (s) | 0.07739 | 0.07711 | -0.36% |
| ITL P90 (s) | 0.07871 | 0.07821 | -0.63% |

The baseline prompt was already almost entirely warm (`14,464` cached tokens),
while the freshly restarted candidate used the expected mixed cold/warm pattern
(`12,656` cached tokens). Therefore TTFT, input/cache throughput, E2E rate, and
weighted score are not comparable and are deliberately omitted. Decode TPS
uses the post-first-token window and does not show the required 1% gain.

Artifacts:

```text
bench_runs/20260715_E_GDN_06/baseline.json
bench_runs/20260715_E_GDN_06/candidate.json
bench_runs/20260715_E_GDN_06/oracle-candidate.json
bench_runs/20260715_E_GDN_06/smoke-quick-corex.log
```

## Decision

`REJECT` the production-path change. The primitive optimization is exact and
consistently faster in isolation, but its 2.0%-2.6% recurrent-step gain is too
small to move full-model decode and the exploratory endpoint result regresses
Decode TPS P10 by 0.26%. Do not spend two additional cold startups on strict
paired qualification. Keep the experiment branch and benchmark as evidence;
return integration to `b1d9500` and select a larger decode hotspot.
