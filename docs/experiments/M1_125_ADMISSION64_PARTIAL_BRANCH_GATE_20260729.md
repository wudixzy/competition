# M1-125 admission64 partial-branch quality gate

## Problem

The frozen v5 long-context gate sent A cold, A warm, and a first B sibling,
then required B to report a strict partial cache hit. That contract is valid
for `fine32`, which already captures recurrent checkpoints throughout A. It is
not valid for the sparse `admission64` policy.

The M1-117 control trace showed the expected sparse-policy sequence:

- A cold had no raw or effective hit and admitted its final prefill state.
- A warm restored the final A state.
- The first B sibling found a long raw KV prefix but no matching GDN state.
  Its effective hit and API `cached_tokens` correctly remained zero.
- B admitted a `repeated_branch` recurrent checkpoint for a later sibling.

Reporting B as cached at that point would skip tokens without a recoverable
recurrent state and violate the effective-cache correctness contract.

## V6 contract

`quality/long_context_matrix.v6.json` preserves every v5 capability case and
changes only the two partial-branch cases. Both now construct deterministic
A/B/C sibling requests and require cache trace v4:

- A cold and A warm must remain output-identical with cold/warm accounting.
- All partial-branch trace records must belong to one service trace session.
- A, B, and C must return their own branch marker with no leakage.
- Under `fine32`, both B and C must be strict partial hits backed by restored
  GDN states.
- Under `admission64`, B must show a raw KV prefix but zero effective GDN hit,
  zero API `cached_tokens`, and a positive `repeated_branch` admission.
- The subsequent C sibling must restore a GDN state and report a strict
  partial hit.
- The 235K case still repeats B and requires exact output plus a cache hit.

The report retains only policy-specific Boolean proof. It does not retain
request text, model output, token IDs, media, block hashes, or GDN restore
digests.

## Local verification

- Focused manifest, runner, and comparator tests: 52 passed.
- Complete unit suite: 1104 passed, 13 skipped.
- Quality-data manifest validation: qualified.
- Submission preflight: 9/9 passed.
- `git diff --check`: passed.

This change repairs the quality harness. It does not alter the runtime,
`computility-run.yaml`, default optimization switches, or model semantics.
The v6 contract still requires a fresh TP4 A/B before it can authorize any
candidate.

## Targeted diagnostic comparison

The strict comparator accepts explicit, repeated `--case` arguments for
short diagnostic A/B runs. A targeted report must declare the same explicit
case selection and remains ineligible as a complete quality baseline. A
successful targeted comparison sets `targeted_diagnostic_qualified=true` but
always keeps `long_context_quality_non_regression_authorized=false` and
`overall_promotion_authorized=false`.

This mode exists only to adjudicate a focused runtime question before paying
the cost of the complete twelve-case TP4 matrix. Omitting `--case` preserves
the complete-matrix contract and is the only mode that can authorize the
long-context non-regression gate.
