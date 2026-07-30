# M1-154 admission-aware TTFT-P90 preflight

Date: 2026-07-30

Status: the corrected three-sibling prompt construction and the exact-commit
runtime overlay qualified on `ssh-73ca29ba`. This is CPU/tokenizer and runtime
identity evidence. It is not TP4 performance, model quality, long-context,
`main`, YAML, or official-score evidence.

## Corrected experiment semantics

The original M1-152 service sequence used a primer and then expected the first
partial-prefix sibling to report an effective hit. That is not the production
`admission64` contract:

1. sibling A creates the raw KV branch;
2. the first sibling B can observe that branch, but has no matching recurrent
   state yet, so `cached_tokens` must remain zero while the scheduler admits a
   checkpoint;
3. a subsequent sibling C can restore the admitted GDN state and report the
   block-aligned shared prefix;
4. an exact repeat of C must then be a full warm hit with identical output.

Treating B as the measured partial hit would reject a correct sparse admission
implementation and would time a setup request instead of a reusable branch.
The frozen replacement is
`quality/short_tp4_p90_pair.v3.json`, SHA-256
`991da440f7b4f64c624fecf49f57fc4f5b38b1c0cda9539b7fcb7f2dcc51a30e`.
The B request remains in the report as diagnostic-only branch-admission TTFT.

M1-153's recorded token lengths remain valid historical evidence, but its
two-sibling service interpretation is superseded by this contract.

## Exact runtime identity

- source revision:
  `1d2a7b65b320c62572350a8385402fa545ebd3c8`;
- instance: `ssh-73ca29ba`;
- runtime root:
  `/root/bi100-runtime-cache/1d2a7b65b320c62572350a8385402fa545ebd3c8`;
- runtime tree SHA-256:
  `3a89b9eec39792ac5fb4577ab6b148b1319f5ae1a40b4ca76b869ff39cd57631`;
- source tree: clean;
- immutable-overlay result: qualified cache miss built in 8.632 seconds;
- every required direct and generated runtime file reported `same=true`;
- the external fused-prefill loader specifically reported
  `generated=false`, `same=true`.

The runtime tree SHA matches M1-153 because M1-154 changes only experiment
code and the frozen contract, not the installed vLLM overlay.

## Real tokenizer construction

The prompt set is `m1-154-admission-v3-1d2a7b6`, built with the local
Qwen3.6-35B-A3B tokenizer from the exact runtime overlay.

| Mode | Target tokens | Shared prefix | Residual prefill |
|---|---:|---:|---:|
| cold | 8,192 | 0 | 8,192 |
| cold | 16,384 | 0 | 16,384 |
| cold | 24,576 | 0 | 24,576 |
| cold | 32,768 | 0 | 32,768 |
| cold | 49,152 | 0 | 49,152 |
| cold | 65,536 | 0 | 65,536 |
| A/B/C branch | 16,384 | 8,192 | 8,192 |
| A/B/C branch | 32,768 | 24,576 | 8,192 |
| A/B/C branch | 49,152 | 40,960 | 8,192 |
| A/B/C branch | 65,536 | 57,344 | 8,192 |

All six cold targets and all eight B/C sibling targets were exact. Every
A/B/C case shared the intended block-size-16 boundary and left exactly 8,192
tokens for the measured C request. The report records only counts and SHA-256
digests; it contains no prompt text, token IDs, credentials, or model output.

## Current decision

The corrected M1-152 v3 runner is ready for a paired TP4 screen once all four
cards pass preflight and the required L2 activation qualification exists.
GPU0 was still stuck at 257 MiB and 100% utilization with no visible process
during this preflight, so no model service was started. TP3 is not a valid
substitute for this model's attention-head geometry.

Formal defaults, `computility-run.yaml`, and `main` remain unchanged.
Privacy-safe evidence is under
`docs/experiments/evidence/M1_154_ADMISSION_BRANCH_P90_PREFLIGHT_20260730`.
