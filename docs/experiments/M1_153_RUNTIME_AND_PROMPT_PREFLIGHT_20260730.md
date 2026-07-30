# M1-153 runtime and TTFT-P90 prompt preflight

Date: 2026-07-30

Status: the exact-commit bare-host runtime and the M1-152 tokenizer prompt
construction qualified on `ssh-73ca29ba`. This removes runtime identity and
prompt-length ambiguity before the short TP4 TTFT-P90 screen. It is not GPU
service, TP4 performance, model quality, long-context, `main`, YAML, or
official-score evidence.

## Why this preflight exists

The submitted workload's global TTFT P90 is governed mainly by the upper
16K-32K and lower 32K-64K requests. M1-152 therefore measures cold prompts at
8K, 16K, 24K, 32K, 48K and 64K, plus partial-prefix continuations with exactly
8K residual prefill. Before spending a TP4 model load, the tokenizer must prove
that those intended lengths and block-aligned prefix boundaries are real.

M1-109/M1-151 also use an external fused-prefill extension. The active overlay
must attest the extension loader itself, not only the compiled extension and
the surrounding patched files.

## Runtime identity

- source revision:
  `a0c70d480f00ea43728f2d1d9e8063f7da93ee6b`;
- instance: `ssh-73ca29ba`;
- runtime root:
  `/root/bi100-runtime-cache/a0c70d480f00ea43728f2d1d9e8063f7da93ee6b`;
- runtime tree SHA-256:
  `3a89b9eec39792ac5fb4577ab6b148b1319f5ae1a40b4ca76b869ff39cd57631`;
- source tree: clean;
- immutable-overlay cache result: miss, followed by a qualified build;
- build and verification wall time: 8.740 seconds;
- `bi100_external_extension`: direct source file, `generated=false`,
  `same=true`;
- every other required direct or generated runtime file also reported
  `same=true`.

An earlier invocation with the host's naked system Python failed while
importing the prompt harness because that environment did not contain
`requests`. It did not reach prompt construction and is not a tokenizer or
model failure. The final run used the exact immutable overlay and qualified.

## Prompt construction

The frozen prompt set is `m1-152-tokenizer-smoke-v1`.

| Mode | Target prompt tokens | Cached prefix | Residual prefill |
|---|---:|---:|---:|
| cold | 8,192 | 0 | 8,192 |
| cold | 16,384 | 0 | 16,384 |
| cold | 24,576 | 0 | 24,576 |
| cold | 32,768 | 0 | 32,768 |
| cold | 49,152 | 0 | 49,152 |
| cold | 65,536 | 0 | 65,536 |
| partial | 16,384 | 8,192 | 8,192 |
| partial | 32,768 | 24,576 | 8,192 |
| partial | 49,152 | 40,960 | 8,192 |
| partial | 65,536 | 57,344 | 8,192 |

All target lengths were exact. Each partial pair shared one token beyond the
intended block boundary before block-size-16 rounding, and the effective
cached prefix and residual prefill matched the contract.

The report stores only counts and SHA-256 digests. It records no prompt text,
token IDs, credentials, model output, or raw request.

## GPU state and next gate

At the end of this preflight, GPUs 1, 2 and 3 were idle and healthy. GPU0 still
reported 257 MiB, 100% utilization and no visible process. M1-152 therefore
did not start: Qwen3.6 TP3 is invalid for its attention-head geometry, and a
three-card run cannot substitute for the production TP4 screen.

When all four cards pass preflight, M1-152 can reuse this exact source and
runtime identity and run the paired 8K-64K cold/partial/warm service matrix.
Only an M1-152 pass can authorize long-context confirmation. Formal
`computility-run.yaml`, defaults and `main` remain unchanged.

Privacy-safe evidence is under
`docs/experiments/evidence/M1_153_RUNTIME_AND_PROMPT_PREFLIGHT_20260730`.
