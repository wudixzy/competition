# M1-67: Exact GDN q/k mapping plus value cast

## Hypothesis

The strict M1-65 decode path maps normalized FP16 q/k heads into expanded FP32
heads, then launches a separate FP16-to-FP32 cast for the value heads. One
production-shape kernel can perform those independent conversions in a single
allocation and launch without changing arithmetic.

This is a bounded component experiment. It does not change the model path,
prebuilt submission extension, runtime defaults, `computility-run.yaml`, or
request semantics.

## Fixed boundary

The only candidate operation is:

```text
reference: qk_map(normalized_q, normalized_k) + value.float()
candidate: qkv_map(normalized_q, normalized_k, value)
```

The shape is fixed to one TP4 rank of Qwen3.6-35B-A3B decode:

- batch: 1
- key heads: 4
- value heads: 8
- head dimension: 128
- input dtype: FP16
- output dtype: FP32

The q/k instructions are the existing mapping instructions. The additional
value conversion is `__half2float`, which is exactly representable in FP32.
There are no tile, block, threshold, dtype, or YAML scans.

## Component gate

`tests/bench_gdn_exact_qkv_map.py` compares two fixed seeds and 1,000
multi-scale random steps. Every complete q/k/v output must be finite and
bit-exact. Relative L2 must also be at most `1e-5`.

The alternating paired benchmark requires both:

- median and paired-median speedup at least `1.25x`;
- absolute saving at least `0.02 ms` per GDN layer.

Failure of either performance limit closes this direction without production
integration. A pass only permits a later production-boundary experiment; it
does not authorize a default switch or promotion.

## Lifecycle

`scripts/run_gdn_exact_qkv_map_gate.sh` requires a clean source tree and an
immutable runtime overlay. Build and benchmark commands run in process groups
created by the invocation. Cleanup sends `SIGTERM`, waits 60 seconds, sends
`SIGKILL` only to surviving members, and reaps the leader.

The exit trap then checks residual API/worker and GPU processes, scans
fatal/OOM/Gloo/NCCL/worker-loss and timeout signatures, repeats the selected
GPU preflight, and compares pre/post GPU state. Any cleanup or postflight
failure invalidates the component result.
The residual scan requires three consecutive clean observations within 30
seconds and preserves every failed observation. A short platform health query
can settle, while a persistent or repeatedly reappearing holder still fails.

## Current status

`REJECTED: ABSOLUTE SAVING BELOW GATE`.

The fixed gate ran on physical GPU1 of `ssh-73ca29ba`:

```text
source revision: 79bdd95f69327dd3e165360ad83a0526dc387ccc
runtime tree:    05d720faf7a6298946b6ab70d0ab73b8e88f0d7b90c7a54cf50c2c19e0273b7b
extension:       b8091fddc4bc6571e25e3ef43e994d82b5f56055db7bc1d2fcf2d2bbc9669c68
```

| Metric | Reference | Candidate | Result |
| --- | ---: | ---: | ---: |
| Median q/k map plus value cast | 0.014316 ms | 0.007265 ms | 1.9705x |
| Paired median speedup | - | - | 1.9717x |
| Saving per GDN layer | - | - | 0.007051 ms |
| Projected saving across 30 layers | - | - | 0.211522 ms/token |
| Fixed-input relative L2 | 0 | 0 | exact |
| 1,000-step relative L2 | 0 | 0 | exact |
| Exact sequence steps | - | 1,000/1,000 | pass |

The speed ratio passes, but the absolute saving is below the frozen
`0.02 ms/layer` gate. `qualification.rc=1` and
`production_promotion_authorized=false`. Do not add this function to the
production extension, prebuilt binary, model path, or default environment.
The value-cast boundary is closed and must not be revisited as a parameter
scan.

All infrastructure gates other than the intentionally failed qualification
returned zero: build, benchmark, runtime identity, cleanup, stable process/GPU
postflight, fatal scan, timeout scan, before/after GPU1 preflight, and GPU
comparison. GPU1 retained exactly `34057748480` free bytes before and after.
No model or API server was started. This remains a one-rank component result
and is not a TP4 or model-throughput result.

Structured evidence is under
[`evidence/M1_67_EXACT_QKV_MAP`](evidence/M1_67_EXACT_QKV_MAP):

```text
benchmark          b754a2bab55163837a4d14f748a2427a590c265b3fea5f3fac0a757d0d49711d
qualification      3b7f6895823542931350cc6f64fc53959738c0809e089d1b8ad1443a0feec1f2
runner status      7043f2dadd33bfe4dcbd864a754a60f656e6daf13cae2fab5a05cb581c84cad7
runtime identity   5524184e2f64a90c4275b719a3388aef201cd0f81b55f9ff33db01128597e5b0
service postflight f9dbb8899cb86e3fc185423d1938e45777f13f101062d5e28cf0cb5659b1f144
GPU comparison     5d09848a67c056014aa0410eae6484b0cc273a9a6366e1633410244632e650c9
```
