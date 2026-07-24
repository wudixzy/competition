# Commit-bound functional baseline (2026-07-25)

## Scope

The current private candidate is materially newer than the platform `main`
result supplied on 2026-07-24. This run therefore establishes a fresh,
commit-bound functional baseline instead of attributing the old 881-request
aggregate to the current branch. It does not measure official throughput,
TTFT, cache hit rate, or weighted score.

The run used source `3cbb98d`, atomic overlay
`84c27dacebce52620084cb2314a535cc9d409ac20ca98a6c8bbd4b7503188001`,
four BI100 GPUs, TP4, a 262144 model length, 8192-token chunked prefill,
`fine32/direct`, full-attention hybrid KV accounting, LRU eviction, and no
fused-prefill candidate.

## Named-tool SSE fix

Two earlier bound runs isolated a protocol defect without changing model
semantics:

- `16da895` reported 52 passes plus the documented `n=2` skip in the 53-case
  functional suite, but both forced and automatic streaming Agent tool cases
  were rejected by an overly strict test parser.
- `6015ff8` accepted valid continuation deltas and made automatic streaming
  pass. Forced named-tool streaming still failed because every
  `DeltaToolCall` relied on Pydantic defaults for `id` and `type` while the
  response serializer used `exclude_unset=True`; those fields never reached
  the client. The same path also repeated the function name on every token.
- `3cbb98d` pre-generates one stable call ID per choice, explicitly emits
  `id`, `type`, and name on the first named-tool delta, and emits only argument
  fragments afterward. Non-streaming responses, tool arguments, sampling,
  tokenizer behavior, chat templates, model execution, and cache behavior are
  unchanged.

## Result

| Gate | Result |
|---|---:|
| Functional contract | 52 pass, 0 fail, 1 documented skip |
| Executed functional pass rate | 100% |
| Agent workload matrix | 11 pass, 0 fail |
| Runtime identity and startup contracts | pass |
| Prefix allocator and GDN action broadcast | pass |
| Four-GPU preflight before/after/comparison | pass |
| Fatal/OOM/Gloo/worker-loss/segfault scan | 0 findings |
| Cleanup and residual API processes | pass, 0 residual |
| Overall RC | 0 |

The sole skip is `n_2`, accepted only for the documented direct-engine
limitation under the fixed `--max-num-seqs 1` contract. Every executed case in
basic chat, streaming and usage, tools, reasoning, thinking, multimodal,
structured output, stop handling, sampling boundaries, multilingual output,
long output, validation, determinism, and cache reporting passed.

The separate Agent matrix passed forced and automatic tools in non-streaming
and SSE modes, tool-result round trips, long history, a large tool schema, and
multiple system messages. Reports contain hashes, counts, usage, and validation
facts only; raw requests, outputs, tool arguments, credentials, and the service
log were not committed.

## Decision

The commit-bound functional baseline passes and unlocks the fixed
`fine32/direct` long-context baseline. It does not authorize `admission64`, a
`main` merge, a `computility-run.yaml` change, or any claim about the current
branch on the official 881-request score. The unbound platform `main` result
remains historical evidence only.

Structured evidence:
`docs/experiments/evidence/QUALITY_FUNCTIONAL_BASELINE_20260725.json`.
