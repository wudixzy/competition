# E-MOE-02 T1 Primitive Microbenchmark - 2026-07-13

## Scope

This experiment tests supported PyTorch alternatives to the current
single-token routed-MoE path. It does not change the evaluator command or the
formal runtime. The benchmark uses the production TP-rank shapes:

```text
experts                    256
hidden size                2048
intermediate per TP rank   128
top-k experts              8
dtype                      float16
```

The complete expert tensors occupy 384 MiB. The selected experts use finite
random weights with scale 0.02 and are copied into the full tensors, so the
current path measures real gather and non-zero compute rather than an all-zero
shortcut.

## Reproduction

The host SSH environment does not export the CoreX Python paths by default.
Use the same runtime paths as the service:

```bash
PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages:/usr/local/lib/python3.10/site-packages \
LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib \
python3 tests/bench_moe_decode.py \
  --out bench_runs/20260713_E_MOE_02_scaled/report.json
```

The formal API service remained running. It returned HTTP 200 before and after
the benchmark; probes can time out while the benchmark saturates GPU 0. No OOM
or CUDA fatal error occurred.

## Parity

```text
all outputs finite             true
flat W13 vs double-bmm max abs 0.0
current gather path max abs    0.0
routing IDs equal              true
routing weight max abs         1.49e-8
```

## Results

Median of five repeats, 20 iterations per repeat after three warmups:

| Primitive | Median | Speed relative to current full path |
| --- | ---: | ---: |
| Current advanced-index path | 0.4533 ms | 1.000x |
| `index_select` full path | 0.5197 ms | 0.872x |
| Double-bmm full path | 0.4704 ms | 0.964x |
| Advanced-index gather only | 0.1951 ms | 2.323x |
| `index_select` gather only | 0.2681 ms | 1.691x |
| Flat compute, preselected | 0.2530 ms | 1.792x |
| Double-bmm compute, preselected | 0.2759 ms | 1.643x |
| Full softmax plus top-k | 0.0556 ms | 8.153x |
| Top-k plus selected softmax | 0.0553 ms | 8.190x |

The speedups in the final two rows are relative to the entire current routed
expert path, not to each other. The routing variants are effectively tied.
Gather accounts for about 43% of this isolated T=1 path.

## Decision

Reject all three code candidates:

- `index_select` is 12.8% slower than advanced indexing.
- Replacing flattened W13 with a second `bmm` is 3.6% slower.
- Selected-logit softmax saves only 0.0003 ms and cannot affect end-to-end TPS.

No candidate approaches the required 1.3x isolated-path gate, so none advances
to model-output A/B testing or `main`. Keep the current flattened-W13 plus W2
`bmm` implementation.

The next decode targets follow the extended profile: shared expert (9.837 ms
per token per rank) and TP all-reduce (13.373 ms). Any change must preserve the
fixed evaluator YAML, exact greedy outputs, smoke 14/14, and 99.5K cold/warm
prefix equality.
