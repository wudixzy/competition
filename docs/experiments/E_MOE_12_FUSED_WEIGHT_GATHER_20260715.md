# E-MOE-12: Exact fused selected-weight gather

## Scope

The T=1 routed-MoE path previously used two advanced-indexing operations to
copy the selected top-8 W13 and W2 tensors. At the TP4 rank-local checkpoint
shape this moves about 12 MiB per MoE layer and accounted for approximately
36% of the E-MOE-11 decode boundary.

E-MOE-12 replaces those two indexing operations with one CoreX kernel. The
kernel copies both tensors with `__half2` loads, uses a measured 1,024-block
grid cap, and leaves routing, selected expert order, GEMMs, activation, and
weighted reduction unchanged. It is enabled only for contiguous FP16 top-8
weights with the expected W13/W2 layout. `BI100_MOE_COREX_WEIGHT_GATHER=0`
restores native advanced indexing; prefill and unsupported layouts always use
the native path.

## Method

The experiment uses the real TP4 rank-local dimensions:

```text
experts=256, top_k=8, hidden=2048, local_intermediate=128, dtype=float16
```

`tests/bench_moe_weight_gather.py` measures the gather, the fixed-route full
E-MOE-11 boundary, and the full boundary including top-k and softmax. Each
device ran 30 warmups and 9 repeats of 300 iterations, followed by 1,000 random
route exactness steps. `tests/probe_moe_weight_gather_runtime.py` additionally
extracts and executes the production `_pure_pytorch_experts` method with the
production extension API and its runtime feature switch.

## Results

| GPU | Routed native ms | Routed CoreX ms | Routed speedup | Fixed full speedup | Gather speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.452768 | 0.359157 | 1.2606x | 1.3297x | 1.9617x |
| 2 | 0.453333 | 0.359350 | 1.2615x | 1.3294x | 1.9654x |
| 3 | 0.453514 | 0.359575 | 1.2613x | 1.3299x | 1.9654x |

All three devices produced bit-exact W13 copies, W2 copies, and final outputs.
Each device also passed 1,000/1,000 random routed full-path comparisons with
`max_abs=0`.

The separately compiled production extension and production dispatch probe on
GPU1 reported:

```text
native full method: 0.457027 ms
CoreX full method:  0.361852 ms
speedup:            1.2630x
fixed exact:        true, max_abs=0
sequence exact:     1000/1000, max_abs=0
```

The cross-device median saving is about 0.09394 ms per MoE layer, or 3.76
ms/token over 40 MoE layers. Added to the existing 5.1 ms/token short-context
candidate projection, the unqualified stack projection becomes approximately
8.9 ms/token. Against the current 13.3-13.5 TPS baseline this implies roughly
15.1-15.3 TPS. This estimate does not replace a TP4 service A/B.

Raw artifacts are intentionally not committed:

```text
/root/competition/bench_runs/20260715_E_MOE_12/cross-gpu1.json
/root/competition/bench_runs/20260715_E_MOE_12/cross-gpu2.json
/root/competition/bench_runs/20260715_E_MOE_12/cross-gpu3.json
/root/competition/bench_runs/20260715_E_MOE_12/production-dispatch-gpu1.json
```

## Decision

`QUALIFY FOR TP4 SERVICE A/B`. The three-device gain is stable at about 26%,
well above the 5% full-boundary gate, and satisfies the bit-exact output
contract. E-MOE-12 builds on E-MOE-11 and is independently switchable.

Physical GPU0 on `ssh-a2d0a302.default.gpu.phanthy.com` still times out on a
single small CUDA tensor operation while GPU1-3 pass. A healthy four-card host
or host-side reset is still required for TP4 service qualification.
