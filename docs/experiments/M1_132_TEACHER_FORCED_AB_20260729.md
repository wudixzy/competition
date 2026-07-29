# M1-132 teacher-forced fused-prefill A/B

Date: 2026-07-29

Status: valid TP4 observation with a failed teacher-forced numerical screen.
Attribution to M1-109 remains pending one fresh fused-off/fused-off repeat in
M1-134. This result does not authorize a default, YAML, or `main` change.

## Contract

Both arms used source `03e22126c94a2cc61719d41fe15ea33e5d0d0207`,
one immutable runtime tree, TP4, FP16, `max_model_len=262144`, the same model,
the same five fixed token sequences, and the same 64 sampled positions per
length. The only intended A/B selector change was:

```text
BI100_ATTN_COREX_FUSED_PREFILL=0 -> 1
```

The lengths were 4K, 32K, 65K, 131K, and 235K. The comparison retained only
aggregate numerical statistics. It retained no prompts, generated output,
token IDs, private token identities, request data, or credentials.

## Result

Both service arms completed successfully. Tokenization and chat requests were
HTTP 200, runtime identity was qualified and equal, and all service, recovery,
postflight, GPU, fatal, and timeout gates passed. The 320-position comparison
was structurally valid but failed the frozen numerical contract:

| Metric | Result | Frozen limit |
|---|---:|---:|
| Top-1 agreement | 0.940625 | at least 0.98 |
| Top-1 mismatches | 19 | diagnostic count |
| Mutually uncovered mismatches | 3 | 0 |
| Teacher-token logprob max delta | 9.797847 | at most 0.1 |
| Teacher-token logprob p99 delta | 7.684062 | at most 0.02 |
| Shared top-k logprob p99 delta | 4.899000 | at most 0.02 |
| Mean teacher-token NLL regression | 0.103504 | at most 0.005 |
| High-margin top-1 mismatches | 0 | 0 |

Top-1 agreement by prompt length was 0.953125 at 4K, 0.9375 at 32K, 1.0 at
65K, 0.921875 at 131K, and 0.890625 at 235K. The large logprob deltas make
this more than an autoregressive suffix-divergence observation. A semantic
task score cannot waive it.

## Cleanup incident

The old outer cleanup returned `private_observation_cleanup.rc=1` after it had
successfully deleted both private observations. Its second verification array
always contained the two fixed path strings, even when neither file existed.
Direct post-run checks confirmed both private files were absent. M1-134 uses a
tested helper that verifies file existence and temporary-file globs separately.
The false cleanup return code does not invalidate the numerical comparison,
but it is recorded independently instead of being silently rewritten.

## Next adjudication

M1-134 starts two fresh services from the same source and immutable runtime,
with fused prefill disabled in both arms. It uses the identical teacher-forced
matrix and the already frozen M1-132 limits.

- If fused-off/fused-off passes near the expected repeatability floor, the
  M1-132 A/B failure is attributable to the fused full-model path and M1-109
  cannot proceed to semantic non-inferiority as a numerical waiver.
- If fused-off/fused-off also fails materially, the collector or control
  runtime is not repeatable enough for attribution. The harness must be fixed
  before M1-109 is classified.

The A/A result may diagnose baseline variance; it must not be used to tune the
contract after seeing M1-109's result.

Privacy-safe aggregate evidence is in
`docs/experiments/evidence/M1_132_TEACHER_FORCED_AB_20260729/summary.json`.
