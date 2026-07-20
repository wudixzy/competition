# M1-33 GDN Native-Chunk Restore - 2026-07-21

## Decision Context

M1-32 established two facts:

- `admission64/direct` has useful performance potential, raising the
  dataset-shaped effective hit rate from `49.9301%` to `61.0671%`, but it
  fails deterministic replay after 17 competing sessions.
- `admission64/aligned` preserves the tested pressure-case output at an
  8,192-token boundary, but its theoretical hit ceiling on the fixed
  4,096/7,800/16,000 matrix is only `14.68%`.

The failed direct checkpoint contains 10,592 tokens. That boundary is aligned
to the 16-token KV block but is 32 tokens into the model's native 64-token
DeltaNet recurrence chunk. Capturing there forces
`_torch_chunk_gated_delta_rule` to use a different padded chunk partition
than the ordinary cold request. The existing unit test permits a `1e-3`
difference for this kind of non-native split, which is too weak for
deterministic autoregressive generation.

M1-33 tests the narrower hypothesis that exact state reuse requires alignment
to the native 64-token recurrence chunk, not to the much coarser 8,192-token
scheduler step.

## Candidate

`BI100_GDN_RESTORE_MODE=chunk64` is an internal, fail-closed experiment:

- live KV/GDN matches are restricted to content boundaries divisible by 64;
- final and repeated-branch states are captured at the longest strict
  64-token boundary;
- scheduler-owned admission, eviction, and worker fail-fast behavior remain
  unchanged;
- `fine32/direct` remains the code and submission default;
- `computility-run.yaml` is unchanged.

For a 235,000-token prompt, the modes restore at:

| Mode | Restore tokens | Physical suffix |
| --- | ---: | ---: |
| `direct` | 234,992 | 8 |
| `chunk64` | 234,944 | 56 |
| `aligned` | 229,376 | 5,624 |

On same-request pairs alone, 64-token strict boundaries provide
`(4032 + 7744 + 15936) / (2 * (4096 + 7800 + 16000)) = 49.67%` matrix-wide
hits. The intended gain still comes from `admission64` retaining repeated
branches across requests; a full matrix is permitted only after correctness
gates pass.

## Qualification Order

1. Local unit discovery, syntax checks, and submission preflight.
2. `tests/gdn_split_exactness.py` on CoreX: fixed native boundaries at
   64, 128, 2,368, and 4,096 tokens must produce bit-identical outputs and
   final states to unsplit calls. A 2,400-token non-native split is retained
   as a diagnostic control but is not a passing alternative.
3. TP4 smoke and the existing 17-session pressure replay. Reported cached
   tokens, message, finish reason, and completion-token count must agree.
4. Fixed 18-request matrix with evaluator kernels. Required versus
   `fine32/direct`: at least +5 percentage points effective hit, at least
   +5% weighted proxy, Output TPS P10 at least 20 and no more than 2% relative
   regression.
5. 131K/256 exact and 235K/1,000 exact replay, followed by the 256K capacity
   check.

Any failed step stops the candidate. No alignment, capacity, or YAML sweep is
allowed.

`scripts/run_m1_33_chunk64_gates.sh` persists the operator result first and
delegates to the existing timeout/cleanup service harness only after that
result is exact. The M1-32 harness retains its existing default paths,
arguments, and gates; its new fallback overrides only isolate M1-33 artifacts
under `bench_runs/m1_33`.

The offline trace simulator accepts
`--gdn-restore-mode direct|chunk64|aligned`. It keeps `fine32/direct` as
the comparison baseline while applying the selected mode only to
`admission64` and `admission64_m1_29`, so qualification cannot gain points
by weakening the baseline.

## Score Economics

The fixed baseline is:

- Output TPS P10 `21.6563`
- Input TPS `741.4479`
- Cache TPS `7607.9233`
- weighted proxy `6699.4886`

At unchanged cache and output throughput, reaching 8,000 requires Input TPS
`1206.08`, or a `1.6267x` aggregate prefill speedup. If full attention is
68.8% of cold prefill, a `1.5x` attention-path speedup yields only a
`1.2976x` aggregate speedup, Input TPS about `962.09`, and score about
`7317.06`. The attention path would need about `2.272x` speedup under this
simple Amdahl projection if cache remains unchanged. Therefore a correct
cache gain is still required; the original 1.5x kernel gate alone is not
sufficient.

## Research Basis

- Marconi requires exact recurrent-state matches and chooses state admission
  using reuse probability and compute saved per memory:
  https://proceedings.mlsys.org/paper_files/paper/2025/hash/7c180af017258d239bac6248d1eb26ac-Abstract-Conference.html
- Sparse Prefix Caching resumes from the deepest exact sparse checkpoint and
  recomputes the remaining suffix:
  https://arxiv.org/abs/2605.05219
- Current vLLM exposes separate Mamba `all` and scheduler-step `align`
  modes, confirming that recurrent-state granularity is an explicit
  correctness contract:
  https://docs.vllm.ai/en/latest/configuration/engine_args/

## Current Status

`COREX_OPERATOR_AND_TP4_PRESSURE_PASS`. Local discovery passes 214 tests with
24 optional dependency skips, and submission preflight passes 8/8. The local
environment has no Torch, so the dedicated test was run on BI100/CoreX:

| Total | Split | Native aligned | Output max abs | State max abs | Exact |
| ---: | ---: | --- | ---: | ---: | --- |
| 193 | 64 | yes | 0 | 0 | yes |
| 193 | 128 | yes | 0 | 0 | yes |
| 2,401 | 2,368 | yes | 0 | 0 | yes |
| 4,097 | 4,096 | yes | 0 | 0 | yes |
| 2,401 | 2,400 | no | 1.676e-8 | 2.682e-7 | no |

This isolates non-native recurrence splitting as the direct-mode failure
mechanism and supports the 64-token candidate. A deployment preflight also
proved that the installed runtime source matched repository SHA-256
`3aef3903...a01388`, accepted `chunk64`, and returned an alignment of 64. This
preflight was added after the first attempt failed before model loading because
the bare-host site package still contained the older `direct|aligned` module;
that attempt is infrastructure-invalid, not a candidate correctness failure.

The corrected TP4 run passed startup, smoke, and the 17-session pressure gate:

| Case | Cold cached | Reused cached | Completion | Hash relation |
| --- | ---: | ---: | ---: | --- |
| interleaved A | 0 | 10,560 | 16 | identical |
| interleaved B | 0 | 10,560 | 16 | identical |
| eviction target after 17 sessions | 0 | 10,560 | 16 | identical |
| eviction target after refresh | 0 | 10,560 | 16 | identical |

Every response ended with `finish_reason=length`. The eviction target hash was
`d250e642...99fcf` before pressure, after pressure, and after refresh. This is
the expected 64-token floor below the failed direct boundary of 10,592 tokens
and is the first TP4 evidence that the root-cause correction survives state
pressure. The automatic 235K/1,000-token exact gate is running; until it and
the fixed-kernel matrix pass, the mode remains unqualified and cannot be used
in a scoring submission.
