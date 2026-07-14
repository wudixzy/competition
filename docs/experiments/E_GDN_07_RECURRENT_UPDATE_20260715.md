# E-GDN-07: Fused BI100 GDN recurrent update

Date: 2026-07-15

## Hypothesis

The decode recurrent update launches separate state decay, batched key-state
matmul, delta, rank-one state update, and query-state matmul operations in all
30 GDN layers for every generated token. A CoreX CUDA extension can fuse this
sequence into one kernel while keeping the temporal state in float32.

## Manifest

```text
baseline: b1d95009d52135a5b00bbac1c5ccc682c4539644
candidate: 54b5146ce517a7e95ae8d714494ae4a13b271d49
branch: exp/E-GDN-07-recurrent-update
hardware: 4 x BI-V100-50C-200G, TP=4
model: /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
fixed evaluator contract changed: no
```

## Change

The candidate adds a source-built `CUDAExtension` using the CoreX compiler.
One block handles one value head and 128 threads handle its output columns. The
kernel reads the 128x128 state twice and writes it once while computing:

```text
decayed_state = state * decay
key_state = key @ decayed_state
delta = (value - key_state) * beta
state = decayed_state + outer(key, delta)
output = query @ state
```

`BI100_GDN_RECURRENT_EXT=0` retains the original Torch path. The extension
requires float32 contiguous query, key, value, beta, decay, and state tensors;
the state remains float32 and is the only mutated input.

The base image has no `ninja`. `setup.py` therefore uses
`BuildExtension.with_options(use_ninja=False)`, and `patch_ops.sh` performs a
fully offline `pip install --no-index --no-deps --no-build-isolation` followed
by a real import gate.

## Build and hardware gates

- minimal CoreX compile, dynamic load, and GPU launch probe: pass
- offline pip wheel build/install/import/GPU launch: pass
- `patch_ops.sh` first and second idempotent runs: pass
- P0 static tests: 40/40 pass
- extension static tests: 4/4 pass
- unit discovery with CoreX package path: 75 pass, 4 skipped
- per-device tensor preflight: 4/4 pass
- four-rank collective preflight: 4/4 pass, value `10.0`
- input immutability: query/key/value/beta/decay unchanged
- candidate installed model SHA equals repository model SHA

The candidate service started in 7 minutes 6 seconds, exposed 16,871 GPU and
6,553 CPU blocks, and returned HTTP 200. No real ERROR, Traceback, non-finite,
or CUDA error appeared in the service log.

## Primitive results

The committed benchmark uses the real per-rank decode shape
`B=1, H=12, K=128, V=128`, 1,000 sequential tokens, seven alternating repeats,
and float32 state. All devices pass `rtol=1e-5, atol=1e-6`, but none are
bitwise equal.

| GPU | Speedup | State max abs | Output max abs | Exact |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 1.8111x | 1.49e-8 | 9.31e-10 | no |
| 1 | 1.8095x | 1.49e-8 | 9.31e-10 | no |
| 2 | 1.5685x | 1.49e-8 | 9.31e-10 | no |
| 3 | 1.8109x | 1.49e-8 | 9.31e-10 | no |

An earlier clean isolated run measured 1.9820x to 1.9928x. GPU 2 changed
clock behavior during the repository benchmark; its median still clears the
1.5x operator threshold.

## Service qualification

- deterministic candidate oracle: 3/3 HTTP 200
- full API smoke: 15/15 pass, including image and tool cases
- performance requests: 8/8 success
- prompt/cached/completion tokens: 14,504 / 14,464 / 512 on both paths

Both services used one excluded warmup request, then eight serial streaming
requests with 64 completion tokens, seed 123, and salt
`E-GDN-07-AB-20260715`.

| Metric | Torch baseline | Fused candidate | Change |
| --- | ---: | ---: | ---: |
| Decode TPS P10 | 12.9838 | 13.2723 | +2.22% |
| ITL P50 | 77.008 ms | 75.822 ms | -1.54% |
| ITL P90 | 78.100 ms | 77.245 ms | -1.09% |
| TTFT P90 | 0.8200 s | 0.8476 s | +3.36% |
| Score overlap | 1285.92 | 1303.92 | +1.40% |
| Wall time | 45.603 s | 45.048 s | -1.22% |

## Correctness failure

The code and tool oracle messages, finish reasons, usage, and hashes are exact.
The long-prefix oracle has equal usage and finish reason, but the reasoning
message diverges near decode token 127: the baseline continues with
`Platform`, while the fused kernel continues with `Core`. A second cold/warm
run on the same Torch service reproduces the baseline message exactly, so this
is not oracle nondeterminism or a prefix-cache hit difference.

The fused kernel changes GEMM reduction order. Although the per-step error is
small, recurrent state drift eventually crosses a token decision boundary.
This violates the experiment's output-hash stop condition.

For comparison, a single C++ dispatch that invokes the original ATen
operations is bitwise exact for output and state after 1,000 tokens, but its
speedup is only 1.0014x (78.117 ms to 78.011 ms). Python dispatch overhead is
therefore not the meaningful cost.

## Decision

`REJECT` the production candidate. Do not merge it into
`integration/perf-winners`. Keep the branch, source extension, benchmarks, and
artifacts as evidence that a fused recurrent kernel can improve endpoint
decode by about 2.2%, but only if a future implementation reproduces the
vendor GEMM reduction order or otherwise proves full output-hash parity.

Return the runtime to `b1d9500` and move the next experiment to a larger exact
hotspot rather than weakening the output/state correctness gate.
