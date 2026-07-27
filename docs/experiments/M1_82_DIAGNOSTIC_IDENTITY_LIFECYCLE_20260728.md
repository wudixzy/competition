# M1-82 diagnostic identity and lifecycle gate

Date: 2026-07-28

## Objective

Make the next single-BI100 or TP2 diagnostic run attributable before spending
GPU time. M1-73 cannot be qualified by a healthy HTTP response if the active
runtime overlay, physical GPU set, command, or postflight state is ambiguous.

## Findings

The prior runner had three evidence gaps:

- TP2 accepted a repeated physical index such as `0,0`;
- it hashed `install.json` but did not compare the active overlay tree with
  that report and the current source revision;
- before and after GPU preflights could each pass independently without
  checking topology identity or a persistent free-memory drop.

These gaps did not prove an existing model failure, but they could make a
future result non-attributable or hide a service-lifetime GPU leak.

## Change

`scripts/run_qwen36_diagnostic_gate.sh` now:

- rejects duplicate physical GPU indices and unsafe instance labels before
  touching the model or GPU;
- runs `verify_bare_host_runtime_identity.py`, requiring the complete active
  overlay tree, install report, direct source files, generated files, and
  source revision to match;
- records an allowlisted, credential-free service command and environment
  contract;
- records rc files for checkpoint, overlay, runtime, initial GPU/NCCL, GDN
  broadcast, service contract, startup, HTTP, cleanup, and postflight gates;
- after scoped process-group shutdown, scans for residual processes, repeats
  the selected physical-GPU preflight, and compares it with the initial
  preflight using a fixed 1 GiB maximum free-memory drop;
- includes overlay and preflight-comparison summaries and artifact SHA-256
  values in the final status;
- returns nonzero if any required rc is missing or nonzero.

Cleanup still sends SIGTERM first, waits 60 seconds, uses SIGKILL only after
that grace period, waits/reaps the leader, and refuses evidence with residual
API workers, GPU processes, fatal errors, Gloo/NCCL failures, or timeouts.

No model source, weight, dtype, tokenizer, chat template, request semantics,
cache policy, formal YAML value, Dockerfile, or default switch changed.

## Local gates

- 906 `unittest` cases passed; 25 optional-dependency cases skipped;
- 20 focused runner, overlay-identity, and preflight-comparison cases passed;
- the frozen 53-case official metric manifest qualified;
- all seven quality-data provenance sources qualified;
- submission preflight passed 9 of 9 checks;
- shell syntax and `git diff --check` passed.

These are local contract results. They do not establish BI100 execution,
single-GPU HTTP correctness, TP4 correctness, model capability, or
performance.

## Next run

The old `d9b87da` overlay is not sufficient for an M1-82 result even though
later commits only changed tests and runners. Reinstall the atomic overlay
from the exact M1-82 HEAD, then run the four-layer real-weight checkpoint on
one healthy physical BI100. The run must produce qualified
`runtime_overlay_identity.json`, `service_contract.json`,
`preflight_comparison.json`, `cleanup_status.json`, and `status.json`.

Only after that single-GPU gate passes may the full Qwen3.6-35B-A3B TP4
functional gate begin. M1-82 does not authorize `main`, formal YAML changes,
or a submission claim.
