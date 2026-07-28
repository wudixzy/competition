# M1-120 quality contract repair

## Purpose

M1-120 repairs two independent failures observed in the first M1-116 control
arm. It does not enable a performance candidate, change sampling semantics, or
relax the production request schema.

The diagnostic source run was:

- source: `c0bac0cefd278fc2b8d4e8842823574762962161`
- overlay: `/root/m1-118-runtime-c0bac0c/site-packages`
- run: `/tmp/m1-116-fused-quality-c0bac0c-20260729-v1`
- instance: `ssh-73ca29ba`

The control arm passed 52 of 53 quality cases and 10 of 11 Agent cases. The
failed cases were `max_tokens_1` and `stream_forced_terminal`. The run did not
start its candidate arm and remains invalid as an A/B.

## Named-tool streaming runtime fix

The named-tool streaming path used `previous_num_tokens[i] == 0` to decide
whether to emit the tool call ID and function name. An engine delta can contain
zero tokens. After such a delta, the next delta also satisfied that predicate,
so a strict SSE client could concatenate the function name twice.

Commit `f39fd69cf6386f6c794e6605775f328b5f40f18c` tracks header emission
independently for each choice. The first emitted named-tool delta carries the
ID and name exactly once, even when it contains zero tokens. Later deltas carry
arguments only. Tool choice, arguments, request sampling, and model execution
are unchanged.

## max_tokens=1 gate contract

The previous gate required `finish_reason == "stop"` for `max_tokens=1`.
OpenAI-compatible generation may correctly return `finish_reason == "length"`
when the one-token limit is reached. Requiring a natural stop therefore
misclassified a valid bounded response.

Commit `ce5045b4567c6281aece33a41e24d6e10980a7e1` keeps the strict token limit:

- completion usage must be exactly one token;
- finish reason must be exactly `stop` or `length`;
- all other finish reasons fail;
- control and candidate must still have identical finish-reason and usage
  sequences in the cross-arm comparison.

This changes only the test oracle. It does not increase or reduce
`max_tokens`, alter service behavior, or weaken the A/B equality requirement.

## Local verification

- named-tool, quality-contract, and Agent focused tests: 63 passed
- patch/static/tool-parser tests: 70 passed
- full suite: 1212 passed, 26 skipped
- Python compile check: passed
- all 52 tracked shell scripts passed `bash -n`
- submission preflight: 9 of 9 checks passed
- `git diff --check`: passed

## Remaining gate

The commits require an immutable exact-source runtime overlay and a real TP4
targeted replay of both failed cases. If that replay passes, M1-116 must be
rerun from a clean service for both control and candidate. M1-120 alone does
not authorize a default-policy change, a production YAML change, or a merge to
`main`.
