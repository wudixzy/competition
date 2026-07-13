# E-GRAPH-01 CLI-Controlled Decode Capture - 2026-07-14

## Hypothesis

The CoreX vLLM image ignores the CLI value and passes `enforce_eager=True` when
constructing `ModelConfig`. Restoring `enforce_eager=self.enforce_eager` could
activate vLLM's existing decode-only CUDA Graph path and reduce launch overhead
across the model's 40 layers without changing the fixed evaluator command.

The experiment is isolated on `exp/E-GRAPH-01-cli-enable`. Its patch is strict,
idempotent, and includes a vendor-eager restoration mode.

## Gates

- Patch unit tests: 2/2 passed.
- Static tests: 39/39 passed.
- Four-device CUDA preflight: passed on every device.
- Four-rank NCCL all-reduce preflight: passed; every rank returned 10.0.
- A preliminary direct `torch.distributed` collective capture did not complete
  before its 90-second timeout. This is supporting evidence only because vLLM
  uses its PyNccl wrapper during graph capture rather than that direct path.

## Runtime Result

The exact evaluator service command reached the following configuration:

```text
enforce_eager=False
tensor_parallel_size=4
max_num_seqs=1
max_seq_len_to_capture=32768
use_async_output_proc=True
```

All four ranks loaded 16.7280 GiB of model weights and completed all 26
safetensor shards. After weight loading reached 100%, the process made no
further log progress for more than three minutes. It never printed the CUDA
Graph capture banner, never reported GPU block allocation, and never opened the
HTTP server. At 5 minutes 46 seconds the parent process was still alive but
`/v1/models` was unavailable.

Artifact:

```text
bench_runs/20260714_E_GRAPH_01/server.log
```

## Decision

**Reject from `main`.** Restoring the normal vLLM setting exposes a CoreX
startup hang before graph capture or cache initialization. The gain cannot be
measured and the startup regression violates the qualification gate. The
installed runtime was restored to vendor-forced eager execution.

Two subsequent eager-mode recovery starts also stopped after all four ranks
loaded 16.7280 GiB. A fresh four-device CUDA preflight and four-rank NCCL
preflight still passed between those starts, so the simple hardware gates do
not detect this residual runtime state. The instance requires a restart before
full-model testing resumes. The qualified `main` source and evaluator YAML were
not changed.

Do not retry by changing evaluator CLI parameters. A future attempt requires a
bounded worker-stage trace that identifies the exact post-load call and proves
that vLLM's PyNccl graph path is supported on this CoreX build.
