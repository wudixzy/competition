# E-MOE-14: Cast router logits after top-k

## Hypothesis

The current exact router casts all 256 FP16 logits to FP32 before selecting
the top eight. Because the cast preserves every FP16 value exactly, selecting
in FP16 and casting only the eight selected values could reduce conversion
work without changing routing or FP32 softmax arithmetic.

Candidate:

```python
selected, ids = torch.topk(router_logits, 8, dim=-1)
weights = torch.softmax(selected.float(), dim=-1).to(torch.float16)
```

Baseline:

```python
selected, ids = torch.topk(router_logits.float(), 8, dim=-1)
weights = torch.softmax(selected, dim=-1).to(torch.float16)
```

## Method

`tests/bench_moe_route_cast_after_topk.py` measures the route alone and the
complete E-MOE-13 routed decode boundary at the real TP4 rank-local shape. The
GPU1 run used 30 warmups and 9 repeats of 500 iterations. Exactness covered
3,003 cases: all-equal logits, top-boundary ties, alternating equal extrema,
and 1,000 random vectors at each of three scales.

## Results

| Boundary | Baseline ms | Candidate ms | Speedup |
| --- | ---: | ---: | ---: |
| Route only | 0.058380 | 0.057423 | 1.0167x |
| Complete routed MoE | 0.337514 | 0.334028 | 1.0104x |

All 3,003 cases produced identical top-8 IDs, FP16 routing weights, and final
outputs (`max_abs=0`). The complete-path saving is only 0.00349 ms per MoE
layer, or approximately 0.14 ms/token over 40 layers.

Raw artifact:

```text
/root/competition/bench_runs/20260715_E_MOE_14/gpu1.json
```

## Decision

`REJECT FOR PRODUCTION`. Correctness passed, but the 1.04% complete-boundary
gain is below the 5% experiment gate and does not justify another production
branch. E-MOE-13 remains unchanged. No cross-device run is required after the
performance gate fails.
