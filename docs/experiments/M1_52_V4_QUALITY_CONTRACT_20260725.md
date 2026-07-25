# M1-52 Long-Context Quality Contract V4

## Purpose

Matrix v4 is a test-contract correction following the bound v3 diagnostic. It
does not change model weights, dtype, tokenizer, chat template, request sampling
semantics, cache policy, scheduler behavior, runtime kernels,
`computility-run.yaml`, or any default optimization switch.

The implementation commit is
`41c770327139fe9800a62dbcbaf6027a01beec33`. The v3 diagnostic source was
`6dfdab10524d71435dd5d60d2ac80135237e5ccf`; its result and artifact hashes are
frozen in
`docs/experiments/evidence/M1_52_V3_TARGETED_QUALITY_20260725.json`.

## Version Separation

The user-supplied platform result is an unbound historical `main` result. Its
source revision, image digest, runtime overlay, and request-level trace are not
known. At the v4 implementation commit, this branch is 134 commits ahead of
local `main` and changes 247 files. Those results cannot establish a gain,
regression, or score for M1-52.

Matrix v2, v3, and v4 reports also cannot form an A/B pair. A valid quality or
performance A/B requires the same exact source, overlay, tokenizer, model,
matrix, request order, instance, and TP4 topology; only a declared optimization
switch may differ.

## V3 Findings

- `65k_multiturn_large_tools` passed, including exact arguments, a 92-tool
  schema, warm cache accounting, and cold/warm equality.
- `131k_reasoning_recall` returned HTTP 200 but used all 512 completion tokens
  in separated reasoning and ended with `finish_reason=length` before emitting
  final content.
- `235k_agent_large_output_budget` returned a structurally valid JSON tool call
  in both cold and warm requests. The automatic tool response also contained
  protocol-valid assistant content, which the v3 helper incorrectly rejected.
- Startup, TP4 synchronization, preflight, cleanup, fatal scan, port cleanup,
  and four-GPU memory comparison all passed.

These findings diagnose test-contract defects. They do not authorize a quality
baseline, cache-policy experiment, performance claim, main merge, or YAML
change.

## V4 Contract

- Frozen matrix: `quality/long_context_matrix.v4.json`
- Matrix SHA-256:
  `242670609fc23668607e5c602ab792a041fff7fcba13db7917cf88fc8281b818`
- Result schema: `bi100-long-context-quality-result-v4`
- Comparison schema: `bi100-long-context-quality-comparison-v3`
- Required base image:
  `harbor.4pd.io/modelhubxc/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3`
- Maximum model length: `262144`
- Final topology: four BI100 GPUs with tensor parallel size four

The 131K reasoning case now has a 1024-token output budget and must finish with
`stop` before consuming the cap. The answer, marker order, and separated
reasoning checks remain mandatory.

Only the 235K automatic Agent case permits optional string or null assistant
content. It still requires the exact tool name and JSON arguments, a nonempty
separated reasoning field, `tool_calls` finish, natural completion before 8192
tokens, valid usage/cache accounting, and exact normalized cold/warm output.
Named and forced tool tests continue to require empty content.

## Local Validation

- focused v4 and evidence tests: 41 passed;
- full unit discovery: 587 passed, 25 optional-dependency skips;
- submission preflight: 9/9 passed;
- quality-data manifest validation: passed with the exact v4 SHA-256;
- Python syntax and Git diff checks: passed.

## Remote Gate Order

1. Install an atomic overlay from the exact clean source containing v4.
2. On a fresh fine32/direct TP4 service, run only the 131K reasoning and 235K
   Agent cases. This explicit diagnostic is never baseline-eligible.
3. If both pass, run the complete functional plus Agent workload gate on a
   fresh service.
4. If functional quality passes, run all 12 v4 long-context cases on another
   fresh service.
5. Only then run one same-source fine32 versus admission64 A/B. Do not tune the
   cache policy if its predefined stage threshold fails.

No result in this document authorizes `main`, `computility-run.yaml`, a default
switch, repository visibility, or an official-score claim.
