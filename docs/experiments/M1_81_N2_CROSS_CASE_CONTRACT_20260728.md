# M1-81 n=2 cross-case diagnostic contract

Date: 2026-07-28

## Objective

Strengthen the single-BI100 real-HTTP qualification for M1-73. A successful
standalone `n=2` response does not by itself prove that sequential fanout
preserves deterministic output and OpenAI-compatible usage accounting.

## Change

The diagnostic quality contract now runs both frozen `n=1` and `n=2` cases.
It fails closed unless:

- both individual cases pass without a documented skip;
- prompt usage is positive and exactly equal;
- `n=2` completion usage is exactly twice `n=1`;
- the normalized first-choice SHA-256 is canonical and exactly equal;
- both cases retain their exact-index, usage, and deterministic-choice facts.

Only SHA-256 digests and aggregate token counts enter the report. Raw prompts,
model output, endpoint details, and credentials remain excluded. The runner
copies the cross-case result into its final structured status.

The full TP4 quality-report comparator applies the same output-digest
relationship whenever a report contains the new evidence. Legacy v1 reports
that contain neither digest remain readable; partially populated, malformed,
or mismatched new evidence fails closed.

No runtime model code, model weight, dtype, tokenizer, chat template, request
sampling value, cache policy, submission YAML, or default switch changed.

## Qualification status

Local results:

- 906 `unittest` cases passed; 25 optional-dependency cases skipped;
- 49 focused quality-contract, comparator, and runner cases passed;
- the frozen 53-case official metric manifest qualified;
- all seven quality-data provenance sources qualified;
- submission preflight passed 9 of 9 checks;
- shell syntax, Python compilation, and `git diff --check` passed.

A real single-GPU HTTP run remains pending because the current BI100 instance
failed at the SSH handshake. This change does not establish model capability,
CoreX execution, TP4 correctness, or a performance gain.

M1-73 remains ineligible for `main` or formal submission until the real
single-GPU diagnostic and full-model TP4 gates pass.
