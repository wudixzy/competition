# M1-107 aligned GDN fast-forward

Date: 2026-07-28

Branch: `exp/M1-107-aligned-fast-forward-20260728`

Measured source: `bdea86fd2a053cbdaa149f2bbdfe383d0323f08a`

Status: the focused TP4 pair passed cache correctness and the policy-v2
continuation screen. It does not authorize a default, YAML change, `main`
merge, or official-score claim.

## Change

M1-106 preserved the recurrent GDN segment boundaries inside a restored
physical suffix, but the first 16K session still produced a different output
from `fine32/direct`. Its restore checkpoint was 3,072 tokens, so the first
model call ended at logical token 11,264 instead of the control's 8,192-token
boundary.

M1-107 caps `admission64/hybrid64` fast-forward progress at the next canonical
`max_num_batched_tokens` boundary. For the failing request it now schedules:

```text
checkpoint_tokens=3072
logical_tokens=8192
physical_query_tokens=5120
```

The scheduler still reports logical progress while charging only the physical
suffix against the token budget. Other restore modes retain their existing
behavior.

## Validation

Local focused validation:

- 14 scheduler unit tests passed;
- Python compilation passed;
- `git diff --check` passed.

Remote runtime:

- instance: `ssh-73ca29ba`;
- GPU count: 4;
- model: `/root/public-storage/models/Qwen/Qwen3.6-35B-A3B`;
- runtime: CoreX/vLLM `0.6.3`, FP16, TP4;
- max model length: 262,144;
- immutable overlay:
  `/root/m1-107-runtime-bdea86f-v2/site-packages`;
- run:
  `/tmp/m1-107-aligned-bdea86f-20260728-v3`.

The first GPU0 preflight attempt timed out at `mem_get_info`. A standalone
60-second retry passed with all memory free, and the replacement v3 run passed
its four-GPU preflight, runtime identity, startup, request measurement, health,
fatal scan, and per-arm cleanup checks.

## Focused TP4 result

Both arms completed all 18 requests. All nine cold/warm pairs were exact
within each arm, and all 18 candidate/control output hashes matched. The two
M1-106 mismatches, `16000_pair1_cold` and `16000_pair1_warm`, are resolved.

| Metric | Candidate | Control | Candidate delta |
|---|---:|---:|---:|
| Output TPS P10 | 21.986 | 22.225 | -1.08% |
| Input TPS | 951.604 | 867.322 | +9.72% |
| Cache TPS | 8264.961 | 8486.745 | -2.61% |
| Effective hit rate | 62.78% | 49.93% | +12.85 pp |
| TTFT P90 | 16.690 s | 17.290 s | -3.47% |
| Weighted proxy | 7661.187 | 7553.506 | +1.43% |
| Success rate | 100% | 100% | unchanged |

The effective-hit improvement exceeds the policy-v2 two-point continuation
threshold. Output TPS remains above 20 and within the 2% relative floor, TTFT
improves, and the output-correctness regression from M1-106 is eliminated.

## Decision

Retain the M1-107 cache implementation as the current correctness candidate.
The focused weighted proxy remains below 8,000 and TTFT P90 remains above five
seconds, so it is not a submission candidate by itself.

The next single-variable experiment is M1-108: keep
`admission64/hybrid64` on both arms and toggle only the frozen fused paged
prefill kernel. Historical M1-99 evidence showed median cold-TTFT gains of
7.33% at 65K and 9.96% at 235K, with no median warm or Output TPS regression.
