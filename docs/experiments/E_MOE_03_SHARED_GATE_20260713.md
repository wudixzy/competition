# E-MOE-03 Shared-Expert Gate Fusion - 2026-07-13

## Hypothesis

Each of the 40 MoE layers launches separate `2048 -> 256` local gate/up and
replicated `2048 -> 1` shared-gate projections. Concatenating their immutable
weights after model loading can remove one small GEMM per layer.

The experiment is opt-in with:

```text
BI100_MOE_FUSE_SHARED_GATE=1
```

The fixed evaluator YAML was not changed.

## Primitive Gate

Production TP-rank shape: hidden 2048, shared intermediate 128, FP16, T=1.
The fused output's gate/up slice must be made contiguous for ixformer's
`SiluAndMul` custom op.

| Path | Median | Speedup |
| --- | ---: | ---: |
| Existing separate projections | 0.2495 ms | 1.000x |
| Fused projection, post-down gate | 0.1651 ms | 1.511x |
| Separate projection, pre-down gate | 0.2488 ms | 1.003x |
| Fused projection, pre-down gate | 0.1646 ms | 1.516x |

The post-down candidate had max-abs and mean-abs output error 0.0 in the
synthetic primitive test. Pre-down gating changed FP16 rounding and was
discarded.

The first model startup exposed a runtime-only constraint: slicing 256 columns
from the 257-column fused result produces a non-contiguous view. ixformer
raised `silu_and_mul expects gpu tensor and be contiguous`. Adding one explicit
`.contiguous()` fixed startup while retaining 1.511x primitive speedup.

## Model Runtime

The corrected candidate started successfully with:

```text
health       HTTP 200
GPU blocks   18271
CPU blocks   6553
fatal/OOM    0
```

The KV block count was identical to the formal baseline. The cached fused
weights add about 40 MiB per TP rank across 40 layers.

Three paired groups used three sequential requests, one worker, 128 maximum
tokens, prompt repeat 126, seed 20260713, and salts `E-MOE-03-{a,b,c}`.

| Metric | Baseline a/b/c | Candidate a/b/c | Median change |
| --- | --- | --- | ---: |
| Decode TPS P10 | 8.266 / 8.165 / 8.186 | 8.383 / 8.371 / 8.394 | +2.41% |
| ITL P50 (s) | 0.11589 / 0.11667 / 0.11678 | 0.11994 / 0.11949 / 0.11961 | +2.52% |
| ITL P90 (s) | 0.12850 / 0.12779 / 0.12841 | 0.12263 / 0.12261 / 0.12182 | -4.52% |
| TTFT P90 (s) | 6.933 / 4.754 / 4.428 | 6.213 / 4.501 / 4.378 | -5.33% |
| Overlap score | 628.38 / 633.40 / 646.00 | 636.64 / 660.65 / 674.20 | +4.30% |
| Disjoint score | 343.06 median | 356.40 median | +3.89% |

All requests succeeded. The candidate2 server log had no runtime, OOM, CUDA,
NCCL, or fatal errors.

## Output Gate

Non-streaming fixed-prompt captures compared full message objects:

| Group | Exact message | Text similarity | Baseline/candidate tokens |
| --- | ---: | ---: | ---: |
| A | yes | 1.0000 | 69 / 69 |
| B | no | 0.8848 | 74 / 69 |
| C | no | 0.8745 | 73 / 67 |

All outputs were concise, correct summaries of the same prefix-cache material,
and all finished with `stop`. Nevertheless, the fused GEMM changes CoreX FP16
rounding enough to alter greedy generation in two of three paired prompts.

## Decision

**Reject from `main` under the current exact-output policy.** The candidate is
a real performance improvement and remains available on
`exp/E-MOE-03-shared-gate`, but it does not pass the byte-identical greedy
output gate. Do not claim this as a qualified competition-score win without a
representative quality evaluation that justifies relaxing that gate.

The installed runtime was restored to the `main` source after the experiment;
the restored service returned HTTP 200 with 18271 GPU blocks.
