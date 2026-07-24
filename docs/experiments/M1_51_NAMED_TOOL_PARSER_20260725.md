# M1-51 Named Tool Parser Compatibility

## Scope

M1-51 fixes one OpenAI serving compatibility defect in the CoreX vLLM 0.6.3
backport. It does not change model weights, dtype, tokenization, chat templates,
sampling parameters, cache keys, scheduler behavior, or model kernels.

The bound `62b8b83` long-context run returned non-JSON `arguments` for the
235K named-tool Agent case. The Qwen model emits XML-style tool calls, and the
configured `qwen3_coder` parser already converts that format to OpenAI tool
calls. The old named-tool non-streaming branch bypassed the parser and copied
the post-reasoning model text directly into `FunctionCall.arguments`.

## Repair Contract

Implementation commit: `69ac3da25e72df3ea7ccd5b2c7a1bf5981333a8c`.

The named-tool response path now applies these rules:

1. A raw JSON object remains byte-for-byte unchanged.
2. Invalid raw JSON may be recovered only from exactly one parser result.
3. The parsed function name must equal the explicitly requested function.
4. Parsed arguments must decode to a JSON object.
5. Missing, ambiguous, wrong-name, or invalid parser output preserves the raw
   value so the quality gate still fails visibly.

This is a response serialization repair, not a mechanism for changing model
output or hiding a capability failure. The approach matches the current vLLM
serving architecture's use of the configured tool parser for model-native tool
syntax, without upgrading the CoreX runtime wholesale.

Primary upstream reference:

- <https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/chat_completion/serving.py>

## Diagnostic Contract

Harness commit: `8c00ba6c6a916b68d1d020330ccf0e4d7fb0800c`.

The long-context validator now reports only argument key differences or the
names of fields whose values differ. It never records argument values. A failed
case retains completed privacy-safe request summaries, including status,
usage, finish reason, hashes, and elapsed time, but not prompts or model text.

## Bound Trigger Evidence

The exact `62b8b83` `fine32/direct` baseline is recorded in:

- `docs/experiments/evidence/QUALITY_LONG_CONTEXT_BASELINE_62B8B83_20260725.json`

It passed 9 of 12 cases. In particular, 235K partial-prefix reuse and both
262144 boundary checks passed. The three failures were:

- 65K multi-turn large tools: generic argument mismatch; exact cause was lost
  by the old runner and requires the new privacy-safe diagnostic.
- 131K reasoning recall: marker recall rule failed after the final-answer rule
  passed; this remains a separate model-quality/test-contract investigation.
- 235K Agent named tool: arguments were not valid JSON; this is the M1-51
  serving defect addressed above.

The run had clean service cleanup, an empty authoritative fatal scan, passing
four-GPU preflight before and after, and zero free-memory drop on every GPU.
It is not a passing quality baseline and authorizes no default change.

## Validation State

Local validation at harness commit `8c00ba6`:

- 559 unit tests passed; 25 optional-dependency tests skipped.
- Submission preflight passed 9/9.
- Quality data manifest validation passed.
- The real Qwen XML parser test produced same-name JSON-object arguments.
- Valid raw JSON, wrong-name, ambiguous, and invalid recovery paths are covered.

Remote runtime overlay:

- source: `8c00ba6c6a916b68d1d020330ccf0e4d7fb0800c`
- branch: `run/quality-m1-51-8c00ba6`
- overlay SHA-256:
  `373d5d28818b8c7b42f0a169d6eac8649fc7162a4d4641e615bde166ac29a9b0`
- Transformers: `4.55.3`
- system site-packages modified: false

The exact TP4 functional and long-context runtime gates remain required. Until
both pass, M1-51 is not qualified for `main`, `computility-run.yaml`, or any
official score claim.

## Version Separation

The user-supplied platform `main` result has no source revision or runtime
overlay identity. It remains an unbound historical failure reference. The
current formal source is more than 120 commits ahead of local `main`, so no
gain, regression, or official score may be inferred across those results.
