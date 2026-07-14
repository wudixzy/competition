# E-MOE-05: Fused routed-expert activation

## Scope

The routed T=1 expert path computes SwiGLU as separate `F.silu(gate) * up`
operations, while the shared expert already uses vLLM's CoreX-backed
`SiluAndMul`. E-MOE-05 benchmarks reusing that existing fused operator without
changing model weights, routing, the evaluator command, or context settings.

```text
base:   c1e7e7c (E-MOE-03 model winner plus diagnostic docs/tests)
branch: exp/E-MOE-05-fused-activation
bench:  a88c2ca
```

## Primitive gate

`tests/bench_moe_activation.py` uses the real per-rank decode dimensions:

```text
experts=256, top_k=8, hidden=2048, local_intermediate=128, dtype=float16
```

GPU1-3 completed. On every completed device the fused activation and complete
routed output were bit-exact with the native implementation.

| GPU | Activation speedup | Full routed-path speedup | Exact | Max abs |
| ---: | ---: | ---: | --- | ---: |
| 1 | 1.6515x | 1.0320x | yes | 0.0 |
| 2 | 1.6439x | 1.0322x | yes | 0.0 |
| 3 | 1.6925x | 1.0299x | yes | 0.0 |

The fused activation is valid, but activation is too small a fraction of the
complete path. Its 3% endpoint microbenchmark gain is below the experiment's
5% minimum integration threshold and far below the broader 1.3x T=1 routed
expert target. No model patch or service A/B was attempted.

## GPU0 incident

GPU0 stalled during framework initialization before writing a result. A
20-second one-tensor Torch preflight also timed out. `ixsmi` reported GPU0 at
18,164 MiB and 100% utilization with host PIDs `15445` and `7093`; neither PID
exists in the container namespace. GPU1-3 were idle and healthy.

An `ixsmi -i 0 --gpu-reset` attempt was refused because the stale host
processes still own the device. This requires an instance-level restart or a
host-side reset; further container-level retries are not useful. The three
completed devices already establish that the candidate misses the performance
gate, so GPU0 was not retried.

Artifacts remain untracked on the remote instance:

```text
bench_runs/20260715_E_MOE_05/gpu1.json
bench_runs/20260715_E_MOE_05/gpu2.json
bench_runs/20260715_E_MOE_05/gpu3.json
bench_runs/20260715_E_MOE_05/gpu0.log
```

## Decision

`REJECT AS PERFORMANCE WINNER`. Keep the benchmark as capability evidence, but
do not add an extra model dispatch for a 3% routed microbenchmark gain. The
qualified model remains E-MOE-03. Restart the instance before any further
four-card service or benchmark work.
