# M1-98 compensated W13 integration

Date: 2026-07-28

Branch: `exp/M1-98-compensated-w13-integration-20260728`

## Purpose

M1-96 qualified the compensated W13 arithmetic against a CPU float64 dot
product rounded once to FP16. M1-98 integrates that exact algorithm into the
existing production `corex_moe_direct_routed` ABI so it can be evaluated in a
full Qwen3.6 decode path.

This experiment does not change weights, dtype, routing, activation, W2,
sampling, tokenizer, context, output budget, or model structure.

## Authority boundary

The combined native module retains the existing functions:

```text
w13
w2_reduce
```

and adds:

```text
w13_compensated
```

The model calls the new function only when both variables are true:

```text
BI100_MOE_COREX_DIRECT_ROUTED=1
BI100_MOE_COREX_COMPENSATED_W13=1
```

The new selector defaults to false and is absent from
`computility-run.yaml`. Requesting it without the direct path or without the
new native symbol fails during model import. Unsupported tensor shapes retain
the existing Python fallback before either native W13 function is called.

## Fixed gates

### Compile and ABI

- compile once for CoreX 3.2.3 `ivcore10`;
- bind the exact source revision and extension SHA-256;
- require callable `w13`, `w13_compensated`, and `w2_reduce`;
- no compiler flag, tile, launch, or arithmetic variant scan.

### Single-GPU integrated MoE

Use seeds `20260716` and `20260727`, 500 routed steps per seed, 30 warmups,
300 iterations, and nine alternating timing repetitions.

Compare three paths:

- strict current PyTorch/gather/exact-reduce reference;
- current direct W13 plus direct W2/reduce control;
- compensated W13 plus the same direct W2/reduce candidate.

The candidate must:

- produce only finite outputs;
- be deterministic on repeated fixed inputs;
- be non-inferior to the direct control against the strict reference for
  aggregate relative L2, maximum-step relative L2, maximum absolute error,
  and mismatch count;
- regress the complete routed boundary by no more than 2%.

M1-96's high-precision W13 evidence remains a required parent artifact. The
integrated comparison does not replace it.

### Full-model TP4

Control and candidate use the same combined extension and differ only in
`BI100_MOE_COREX_COMPENSATED_W13=0|1`. Run order alternates across at least
three paired repetitions with a clean restart between arms.

Before performance qualification:

- fixed greedy next-token checks must pass;
- all 53 functional cases and 11 Agent cases must meet their expected
  outcomes;
- tool arguments, reasoning/content separation, structured output,
  multimodal behavior, long-context recall, and cache cold/warm consistency
  must not regress;
- fatal, OOM, Gloo/NCCL, worker-loss, timeout, and residual scans must be
  empty.

The candidate needs at least 5% paired improvement in its targeted end-to-end
proxy to justify an official-style run. Final competition thresholds and the
no-capability-regression rule remain unchanged.

## Stop rule

If the combined ABI fails compilation, the integrated candidate is worse than
the current direct control, next-token or capability behavior regresses, or
the paired end-to-end gain is below 5%, close M1-98. Do not alter compensation
arithmetic, tolerances, launch geometry, YAML, or request semantics to rescue
it.
