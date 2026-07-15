# E-MOE-02: Normalize only selected decode experts

Date: 2026-07-15

## Hypothesis

The MoE router currently computes a 256-wide softmax, selects eight experts,
and renormalizes those eight values. Softmax is monotonic, and renormalizing
the selected probabilities cancels the full-softmax denominator. Selecting the
top-eight logits first and applying softmax only to them should preserve routing
while reducing decode overhead.

## Manifest

```text
baseline commit: 3453dc27c775222de61246a74704e766cd93a1f7
candidate commit: f11c6f9649844b67536956f73bc08e34cba4a86d
branch: exp/E-MOE-02-decode-primitives
model: /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
hardware: 4 x BI-V100-50C-200G, TP=4
max model length: 262144
```

## Change

`qwen3_6_scripts/qwen3_5.py` now performs `topk` on FP32 router logits and
applies softmax to the selected values. Expert IDs, weights, and expert math
remain unchanged. The same routing path is used by decode and prefill.

`tests/bench_moe_decode.py` reproduces the real per-rank decode shapes:

```text
experts=256, top_k=8, hidden=2048, intermediate_per_rank=128
w13=(256, 256, 2048), w2=(256, 2048, 128)
```

## Contract

- `computility-run.yaml` changed: no
- launch arguments changed: no
- performance environment overrides: none
- debug or profiler enabled for score: no
- GPU blocks: 16,871 baseline and candidate
- CPU blocks: 6,553 baseline and candidate

## Primitive scan

All four devices returned bit-exact output for the winning case.

| GPU | Existing median (ms) | Top-k logits median (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 0 | 0.526887 | 0.503705 | 1.0460x |
| 1 | 0.530154 | 0.502846 | 1.0543x |
| 2 | 0.530882 | 0.504093 | 1.0531x |
| 3 | 0.552331 | 0.503006 | 1.0981x |

The GPU0 scan rejected `index_select` (0.8921x), reusable workspace
(0.8905x), gate `bmm` (0.9626x), and workspace plus top-k logits (0.9261x).
These cases remain in the benchmark as negative evidence and are not used by
the model implementation.

Artifacts are under:

```text
bench_runs/20260715_E_MOE_02/gpu0.json
bench_runs/20260715_E_MOE_02/gpu1.json
bench_runs/20260715_E_MOE_02/gpu2.json
bench_runs/20260715_E_MOE_02/gpu3.json
```

## Correctness gates

- remote CoreX MoE parity: 4/4 pass, including 256 experts and large logits
- remote static tests: 40/40 pass
- Python compile checks: pass
- full API smoke: 15 passed, 0 failed, 0 skipped
- candidate startup: HTTP 200, 262,144 context, 16,871 GPU blocks
- candidate log scan: no fatal error, OOM, traceback, or worker loss
- final service PID/PGID: 18,909 / 18,909

The first full-smoke launch exited before issuing requests because the
non-interactive client shell omitted the CoreX path and could not import PIL.
Rerunning with the documented CoreX environment passed 15/15. This client
environment failure is excluded from model qualification.

## Strict performance pairs

Each side used eight serial streaming requests, 64 generated tokens, one
worker, and the same salt within each pair. Every run reported 14,440 prompt
tokens, 1,896 uncached prompt tokens, 12,544 cached prompt tokens, 512
completion tokens, 100% request success, and an 86.8698% cache hit rate.
Ordering was baseline-to-candidate for A, candidate-to-baseline for B, and
baseline-to-candidate for C.

| Pair | Metric | Baseline | Candidate | Change |
| --- | --- | ---: | ---: | ---: |
| A | Output TPS P10 | 13.0632 | 13.6284 | +4.33% |
| A | ITL P90 (ms) | 78.2859 | 74.3801 | -4.99% |
| A | Weighted short score | 1152.6717 | 1210.2509 | +5.00% |
| B | Output TPS P10 | 13.1365 | 13.7276 | +4.50% |
| B | ITL P90 (ms) | 77.6939 | 73.5299 | -5.36% |
| B | Weighted short score | 1113.0532 | 1219.1514 | +9.53% |
| C | Output TPS P10 | 13.0450 | 13.4546 | +3.14% |
| C | ITL P90 (ms) | 77.9088 | 74.4939 | -4.38% |
| C | Weighted short score | 1147.1538 | 1150.1700 | +0.26% |

Median changes across the three pairs are:

- Output TPS P10: +4.33%
- ITL P90: -4.99%
- weighted short score: +5.00%

Output TPS and ITL improve in every pair. TTFT P90 is noisy across fresh
startups (-32.35% to +27.04%) and is not claimed as an E-MOE-02 benefit. The
weighted score is a fixed short-test diagnostic, not the official competition
score.

## 235K long-context gate

The final candidate constructed exactly 235,000 prompt tokens and generated
eight tokens under the 262,144-token service contract.

| Request | Elapsed (s) | Prompt | Cached | Completion |
| --- | ---: | ---: | ---: | ---: |
| cold | 519.855 | 235,000 | 0 | 8 |
| warm | 48.385 | 235,000 | 234,544 | 8 |

Both requests stopped normally and produced `FINAL-99500`. Their message
SHA256 values are identical and match the prior 256K qualification:

```text
a3dc73d02269b1b3682ed84197c3d2d0ddc39dfdb544f73fb3ea832f1fb30b4d
```

Final health returned HTTP 200, with no fatal error, OOM, or worker loss.

## Sustained decode gate

A non-streaming request set both `min_tokens` and `max_tokens` to 1,000 so EOS
could not end the request early. It returned HTTP 200 in 77.831 seconds with
exactly 1,000 completion tokens and `finish_reason=length`. The complete
message SHA256 was:

```text
1766c3c44bfb672e32b2e35419c5e06490e539e54250ab2fc1012c539e68835f
```

The service returned HTTP 200 afterward, and its log contained no fatal error,
OOM, or worker loss. The artifact is
`bench_runs/20260715_E_MOE_02/decode-1000-forced.json`.

## Decision

`KEEP AS PERFORMANCE WINNER`. E-MOE-02 is bit-exact on all four devices,
improves Output TPS P10 in all three token-matched pairs, passes full smoke and
the 235K cold/warm gate, and does not alter the evaluator contract. Integrate
`f11c6f9` on top of the qualified 256K branch. The observed 13.45-13.73 Output
TPS remains below the competition target of 20, so MoE and collective work
must continue.
