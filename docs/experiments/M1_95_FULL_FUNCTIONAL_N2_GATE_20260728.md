# M1-95 full functional n=2 gate

Date: 2026-07-28

Branch: `fix/M1-95-full-functional-n2-gate-20260728`

## Problem

The historical platform report failed the deterministic `n=2` case. The
runtime now contains the M1-73 sequential greedy fan-out path for the fixed
`max_num_seqs=1` submission command, and the single-GPU diagnostic gate
requires that path to return two valid choices.

The formal TP4 functional runner still passed
`--allow-bare-engine-n2-skip` to `quality_gate_api.py`. That compatibility
escape hatch permits the direct endpoint to record the old HTTP 400 as an
explicit skip. A report could therefore complete without proving that M1-73
works on the full model, which conflicts with the no-capability-regression
promotion contract.

## Change

`scripts/run_quality_functional_gate.sh` no longer enables the n=2 skip.
The extended 53-case functional suite must now execute `n_2` and require its
normal success contract:

- HTTP 200;
- exactly two choices with indices zero and one;
- deterministic normalized outputs for the fixed greedy request;
- valid finish reasons and usage accounting;
- no documented bare-engine skip.

The generic API harness retains its explicit skip option for historical
baseline analysis, but the formal competition-quality runner cannot use it.
A static regression test enforces that boundary.

This change affects validation only. It does not change the model, runtime,
weights, dtype, tokenizer, chat template, sampling parameters, request limits,
cache policy, `computility-run.yaml`, Dockerfile, or any serving default.

## Validation

Before a TP4 run:

- quality service, API, and report-comparison focused tests: 43 passed;
- shell syntax and `git diff --check`: passed.

The full local test suite, submission preflight, manifests, and exact-runtime
installation must pass before remote execution.

## Required evidence

Run the complete TP4 functional and Agent gate from an immutable overlay built
from the exact committed M1-95 revision. The result is valid only if `n_2` is
`pass`, `allowed_skip_ids` is empty, all other functional and Agent cases
pass, cleanup and four-card postflight pass, and fatal/timeout scans are empty.

A passing single-GPU diagnostic result is useful plumbing evidence but cannot
replace this full-model TP4 gate. No `main`, YAML, or production-default
promotion is authorized by this harness correction alone.
