# M1-123 Quality 4xx And Partial Lifecycle Repair

## Trigger

M1-116 run
`/tmp/m1-116-fused-quality-40ea9d7-20260729-v2` used source
`40ea9d79794f6ca0816705b069694e8fb8933545` and runtime overlay
`e76396bf3c27303b882271d7c984b4d9308f174a993553754f84a3bae2e82bae`.
The control service produced valid model-quality evidence:

- functional quality: 53/53 passed;
- Agent workload: 11/11 passed;
- 65K and 235K cold/warm diagnostics: HTTP 200 and exact output identity;
- service recovery, postflight, four-GPU postflight, fatal scan, and timeout
  scan: passed.

The control arm still returned nonzero because the 4xx summary contained four
`unclassified_chat_error` records. The fixed request order and elapsed time
identify them as the accepted boundary cases for `top_p=0`, `top_p>1`, negative
`max_tokens`, and a completion budget above the context limit. The remaining
four 400 responses were already classified as three request-validation cases
and one empty-message case.

The candidate arm did not start. `run_arm` restored shell `errexit` before
returning the child status, so a nonzero control child exited the outer runner
before `control.rc` was persisted. Final recovery then correctly found the two
control identities, but its qualifier incorrectly required four identities
from a complete A/B and reported a secondary lifecycle failure.

## Repair

M1-123 changes diagnostics and experiment evidence only:

1. Exact vLLM-generated error prefixes are mapped to fixed privacy-safe reason
   codes: `invalid_top_p`, `invalid_max_tokens`, and
   `context_length_exceeded`. Unknown messages remain
   `unclassified_chat_error` and continue to fail closed. No message, numeric
   field value, request content, or response content is logged.
2. `run_arm` captures the child return code without mutating the caller's
   `errexit` state.
3. Recorded-session recovery qualifies exactly the identity files created by
   the partial or complete run. Zero identities still fail closed.
4. The quality status comparator now matches the emitted optional
   `fused_output_diagnostic` gate and artifact.
5. The fused-prefill comparator explicitly distinguishes M1-112 and M1-116
   labels, requires M1-116 diagnostic evidence, and binds its artifact hashes.

No model code, sampling semantics, tokenizer, chat template, cache policy,
attention selector, `computility-run.yaml`, or default runtime setting changes.

## Validation

- focused unit tests: 60 passed;
- complete unit suite: 1227 passed, 26 skipped;
- submission preflight: 9/9 passed;
- quality-data and metric manifests: passed;
- Python and shell syntax: passed;
- `git diff --check`: passed.

## Status

This repair does not authorize M1-109 fused-prefill promotion. M1-116 must be
rerun with an exact M1-123 source and runtime overlay so that both control and
candidate arms complete. Functional, Agent, 4xx, output-diagnostic, lifecycle,
and aggregate results must be reported separately.
