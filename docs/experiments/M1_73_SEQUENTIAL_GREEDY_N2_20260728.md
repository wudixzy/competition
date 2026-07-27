# M1-73 sequential greedy n=2

Date: 2026-07-28

## Objective

Close the frozen functional failure for `n=2` while retaining the fixed
submission value `--max-num-seqs 1` and preserving request sampling semantics.
The official quality case is non-streaming, uses `temperature=0`, a fixed
seed, and requests two choices with eight output tokens.

## Candidate

| Item | Value |
| --- | --- |
| Branch | `fix/M1-73-sequential-greedy-n2-20260728` |
| Baseline | `1cf779c5bbb4bbdcf7048ea2f3fa1b5702ee7724` |
| Runtime implementation | `383381cd50f791ce54aff9bf0d33a521bee8ecc4` |
| Quality evidence | `f3a878674ec1c5b325beed2113bce571a97deae3` |
| Current candidate | `d9b87da59e71f89012ab170dcf510fb02e826022` |
| Bundle | `/tmp/m1-79-n2-d9b87da.bundle` |
| Bundle SHA-256 | `7d45b0fe9288f741ecebd4c3bb70cc05d4aee794d3592d28cd8d4627148272c8` |

The bundle is verified, contains exact HEAD `d9b87da`, and requires
`9dcdbdc72fc8be8b8a49767b0c9004a548ee29b9`.

## Change

The API layer recognizes only the exact deterministic shape:

- `max_num_seqs == 1`;
- `n == 2`;
- `temperature == 0`;
- non-streaming;
- no beam search, `best_of`, or prompt logprobs.

It executes two isolated `n=1` children sequentially and merges their choices.
The merged response counts prompt tokens once, sums completion and reasoning
tokens, preserves the first prompt-cache accounting, and assigns choice
indices zero and one. Any child error or response-contract mismatch fails
closed. All other `n > max_num_seqs` requests retain the normalized 400.

Normal `n=1` requests do not query scheduler configuration. This avoids adding
an engine RPC or await to the dominant request path.

No model weight, dtype, tokenizer, chat template, sampling value, output
limit, cache policy, formal YAML value, or default compute switch changed.

## Local gates

- 903 `unittest` cases passed; 25 optional-dependency cases skipped.
- The 53-case official metric manifest is qualified.
- Quality data manifests are qualified.
- Submission preflight passed 9 of 9 checks.
- Python compilation and `git diff --check` passed.
- Unit tests cover the fan-out predicate, child execution, merge accounting,
  contract drift, error propagation, metadata restoration, and the normal
  `n=1` scheduler-query bypass.
- The quality gate requires exact choice indices, deterministic normalized
  choices, positive usage, equal `n=1`/`n=2` prompt usage, exactly doubled
  completion usage, and an exact privacy-safe per-choice output digest.
- The diagnostic runner sets `allow_bare_engine_n2_skip = False`; the historic
  400 cannot be reported as a pass.

These are non-GPU and mocked-contract results. They do not establish real
CoreX HTTP behavior, model quality, or performance.

## Remote status

A single bounded probe of `ssh-73ca29ba` failed during SSH handshake with
`Connection closed by UNKNOWN port 65535`. No remote command ran, so GPU
health is unknown. No service, worker, model load, cleanup, or remote file
change occurred.

The GitHub private experiment branch contains the candidate. The ModelHub
push was attempted once and stopped after the peer reset the connection; no
credential or repository-visibility change was made.

## Required next run

Use the immutable four-layer real-weight diagnostic checkpoint on one healthy
BI100 GPU with `max_model_len=262144`, `max_num_seqs=1`, and the fixed runner
`scripts/run_qwen36_diagnostic_gate.sh`. Require:

- all ten quality boundary cases, including real HTTP `n=1` and `n=2`, to
  pass;
- the cross-case contract to prove one prompt charge, summed completion usage,
  and exact deterministic output digests across `n=1` and each `n=2` choice;
- the compatibility, tool, multimodal, and prefix gates to pass;
- `[BI100 N_FANOUT] choices=2 mode=sequential_greedy`;
- scoped process-group SIGTERM cleanup with at least 60 seconds of grace;
- no residual API server, worker, or GPU process;
- qualified repeated GPU postflight;
- empty fatal, Gloo, NCCL, worker-loss, and timeout scans.

After that result, run the full-model TP4 functional gate before any
performance A/B or promotion decision.

## Decision

`INCONCLUSIVE`

The implementation and local evidence are ready for single-GPU HTTP
qualification. The candidate is not authorized for `main`,
`computility-run.yaml`, formal defaults, or submission until the real
single-GPU and TP4 gates pass.
