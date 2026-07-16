# M1-20 Current Score Composition

## Scope

This is a local dataset-shaped proxy for the current qualified `main`, not an
official evaluator score. The existing TP4 service was not restarted or
reconfigured. It served three independent cold/warm pairs at each of 4,096,
7,800, and 16,000 prompt tokens. Every request used 29 tool schemas, diverse
repository material, 64 maximum output tokens, one worker, and streaming usage.

Each pair used a unique early salt for a true cold request, followed by the
identical prompt for the warm request. There were 18 requests in total.

## Validation

- request success: 18/18;
- local and server prompt-token counts matched every target;
- all cold requests reported zero cached tokens;
- warm requests reused all but the final cache block;
- service PID remained 23178;
- `/health` and `/v1/models` remained HTTP 200;
- new log segment contained no traceback, fatal, OOM, Gloo reset, MRoPE shape
  failure, or worker loss.

## Results

| Length | Cold TTFT P50/P90 | Warm TTFT P50/P90 | Cold Input TPS P10/P50 | Warm Cache TPS P10/P50 | Output TPS P10/P50 |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 6.968/6.981 s | 1.247/1.248 s | 586.7/587.8 | 3269.6/3272.5 | 21.73/22.06 |
| 7,800 | 9.784/10.116 s | 0.969/1.005 s | 771.3/797.2 | 7752.1/8040.6 | 21.94/21.97 |
| 16,000 | 22.159/22.394 s | 1.383/1.388 s | 714.5/722.1 | 11515.8/11561.1 | 21.43/21.74 |

Aggregate metrics:

| Metric | Result | Target |
|---|---:|---:|
| Output TPS P10 | 21.506 | >=20 |
| Success rate | 100% | >=99% |
| TTFT P90, balanced cold/warm matrix | 21.454 s | <=5 s |
| Cache hit rate, balanced cold/warm matrix | 49.93% | >=50% |
| Cold Input TPS P10 / aggregate | 587.5 / 719.4 | - |
| Warm Cache TPS P10 / aggregate | 3271.7 / 7716.5 | - |
| Conservative weighted proxy | 3837.9 | >=8000 |
| Aggregate weighted proxy | 6696.0 | >=8000 |

The aggregate proxy uses the published formula and contributes approximately
361 points from Output TPS, 2,013 from Input TPS, and 4,322 from Cache TPS.
It is about 1,304 points below 8,000. With the other components fixed, closing
that gap would require roughly 466 additional Input TPS or 2,329 additional
Cache TPS. The official aggregation and request mix may differ, so these are
directional requirements rather than a score prediction.

The 49.93% hit rate is structurally just below 50% because this deliberately
balanced matrix has one cold and one warm request per pair, while each warm
request still executes the final 16-token block. It does not prove that the
official workload reaches the 50% cache-hit requirement; the prior official
run reported only 42%.

## Decision

Output decode and request availability now have local margin. The next score
work must not trade away the 21.5 Output TPS P10 floor. Long-context attention,
GDN prefill, padded/hybrid MoE, and CoreX WMMA MoE have all reached stop gates.

The remaining evidence-based target is prefix-cache retention and admission:
the published workload has a mean reusable prefix near 16.8K tokens, but the
fixed 262K KV capacity can be displaced by long unique suffixes. Audit whether
the current block-manager LRU can preserve short/frequent shared prefixes
without overstating cached usage or changing model math. Continue only with a
trace/simulation that predicts cache hit above 50% and a weighted gain above
5%; otherwise freeze the current candidate and request an official evaluation.

Raw evidence is outside Git at
`result/20260716/M1_20_current_score_profile/`.
