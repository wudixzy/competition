# E-GRAPH-01: Restore CUDA Graph qualification probe

## Hypothesis

The fixed evaluator supplies `--max-seq-len-to-capture 32768`, and the model
implements sequence-agnostic capture inputs for its GDN/Mamba state. However,
the BI100 base vLLM hardcodes `enforce_eager=True` in
`EngineArgs.create_model_config`, so graph capture and asynchronous output
processing are always disabled.

Restoring `enforce_eager=self.enforce_eager` preserves the CLI contract and
does not change model arithmetic. Graph replay can remove Python and repeated
kernel-launch overhead across the full decode graph, making this a materially
larger candidate than the rejected 3% fused-activation micro-optimization.

```text
base:   c728d7e (E-MOE-03 model winner plus experiment evidence)
branch: exp/E-GRAPH-01-cudagraph-probe
probe:  97440b0
```

## Change

`qwen3_6_scripts/patch_cuda_graph.py` is an idempotent, fail-closed patch. It
only replaces the vendor hardcode:

```python
enforce_eager=True
```

with the existing `EngineArgs.enforce_eager` value. `patch_ops.sh` applies it
during image construction. The fixed command does not pass `--enforce-eager`,
so a candidate image will use vLLM's normal CUDA Graph path.

Local gates completed:

```text
patch idempotence/fail-closed unit tests  2/2
Python compilation                       pass
patch_ops.sh syntax                      pass
diff whitespace                         pass
P0 static                               40/40
remote CoreX client tests               10/10
```

Local broad unit discovery ran 77 tests: 69 passed, seven Torch-dependent
tests skipped because the local Conda environment has no Torch, and the only
collection error was the same environment's missing Pillow. The affected
client suite passed 10/10 under the authoritative remote CoreX environment.

## Runtime stop conditions

CUDA Graph is not qualified merely because a toy graph captures. Execute the
following gates after an instance-level restart, in this order:

1. Four independent Torch allocation/synchronize preflights.
2. Existing TP=4 tensor and collective preflights.
3. `bi100_cuda_graph_preflight.py` on each GPU. It captures the real routed-MoE
   shapes and a mutating GDN recurrent state step; all outputs and state must be
   bit-exact with eager execution.
4. `bi100_cuda_graph_collective.py` under TP=4 with IPC enabled. All ranks must
   complete before the hard timeout and produce exact all-reduce values.
5. Apply `patch_cuda_graph.py` twice and verify the second run is an idempotent
   skip.
6. Start the full service. Graph capture must finish, four workers must remain
   healthy, and GPU block capacity must still support 262,144 tokens.

Any timeout, GPU unhealthy state, collective mismatch, state mismatch, capture
exception, or material KV-capacity loss rejects the candidate before API A/B.

## Qualification gates

If startup succeeds, the candidate still requires:

- full API smoke 15/15;
- deterministic oracle and forced 1,000-token hash equality;
- three strict token-matched eager/graph service pairs;
- Output TPS P10 improvement on every pair and at least 5% median;
- 235K cold/warm output equality and prefix-cache reuse;
- zero fatal/OOM/non-finite/worker-loss/segfault entries.

## Single-card runtime evidence

GPU1 passed a basic Torch allocation/synchronize preflight. The first graph run
used unrealistically large random expert weights, produced non-finite MoE
values, and is excluded. Commit `cca70a7` changed the probe to model-scale
weight initialization and made finite values an explicit requirement.

The corrected run was finite and bit-exact for routed MoE output, mutating GDN
output, and GDN state. Performance was negative:

| Probe | Eager | Graph | Speedup |
| --- | ---: | ---: | ---: |
| one routed-MoE subgraph | 0.4872 ms | 0.5609 ms | 0.8686x |
| 40 repeated routed-MoE subgraphs | 19.3877 ms | 21.0799 ms | 0.9197x |

The 40-layer probe amortizes graph replay overhead across a model-scale number
of launches and still regresses by 8.0%. All max-abs differences were 0.0.
Artifacts are the sibling regular files:

```text
bench_runs/20260715_E_GRAPH_01/gpu1-r2.json
bench_runs/20260715_E_GRAPH_01/gpu1-r3.json
```

## Decision

`REJECT AS PERFORMANCE WINNER`. CUDA Graph replay is numerically compatible
with the tested BI100 MoE and stateful GDN paths, but it is slower both for a
small graph and after 40-layer amortization. Skip the TP collective capture and
full-service graph experiment. Do not merge `patch_cuda_graph.py` or its
`patch_ops.sh` call into `integration/perf-winners`.

GPU0 on the current instance separately remains unusable after stale host PID
`7093` prevented `ixsmi --gpu-reset`; an instance restart is still required
before later TP=4 experiments.
