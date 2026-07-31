# M1-170 cold GDN capture overhead TP1 screen

## Question

The `fb0084f` platform result was 34%-39% slower in the two shortest TTFT
buckets than `503fa7c`. The main default-path change between those revisions
was the admission64 capture-boundary correctness fix. M1-170 asks whether the
cost of producing GDN prefix states is large enough to explain that platform
regression.

This is an attribution screen, not an optimization candidate. `off` disables
effective GDN/KV reuse and is never eligible for production promotion.

## Harness corrections

Three failed startup attempts initially reported an unsupported
`--reasoning-parser`. A fourth run loaded the model but failed its policy
contract. Both failures had the same harness cause: launching from the source
root put the repository's old top-level `vllm` package ahead of the certified
runtime overlay. Commit `d957ef9` moved the service and import contract to
`/tmp`; `0077671` applied the same isolation to the policy contract. The
service, API CLI, tool parser, reasoning parser, and GDN policy imports then
resolved from the overlay.

The first complete A/B was also unsuitable for cold attribution. Although the
salt was moved earlier in the system message, Qwen rendered the 29-tool schema
first. Tokenizer inspection found the first request-specific token at position
3093, leaving 193 complete shared cache blocks. The admission64 "cold" rows
therefore contained 21,504 cached tokens.

Commit `70e6d318408c75265b0934bc355b64a4610dc23e` introduced M1-170 v2:

- only this diagnostic uses `tool_count=0`; historical M1-104/M1-169 defaults
  remain at 29 tools;
- the request discriminator appears at token 6, before the first complete
  16-token block;
- both policies must report zero cached tokens for all nine cold rows;
- cross-policy first-token and complete-output identities are reported as
  numeric observations, separately from timing validity;
- no TP4, semantic-quality, YAML, main, or production decision is authorized.

## Fixed setup

- instance/GPU: `ssh-73ca29ba`, GPU 1;
- source: `70e6d318408c75265b0934bc355b64a4610dc23e`;
- runtime tree:
  `93f65cce49f5455401ae35c97371b0e8dbf0f94e7ab23878d2e7c5991603e849`;
- actual overlay:
  `/root/m1-170-source-70e6d31-exact/bench_runs/m1_49/runtime_overlay`;
- model: four-layer real-weight Qwen3.6-35B-A3B diagnostic checkpoint;
- TP1, max model length 262144, 8192 scheduler chunk, 64 output tokens;
- 4096, 7800, and 16000 prompt-token targets, three identities per length;
- each identity runs cold then warm with greedy sampling and a fixed seed;
- forward order `admission64,off`, then reverse order `off,admission64`;
- no concurrent benchmark ran on the host during either timing pair.

Both orders used request manifest
`d364af0f94d2d242784bafdebaa24d7467c4586f35a1ddbb71cd8f226ded56be`.
All 36 requests per order completed successfully. Both arms independently
passed their cold/warm output contract. Every startup, policy, measurement,
health, scoped cleanup, port-release, postflight, repeated GPU preflight, and
fatal scan gate returned zero.

## Result

| Cold TTFT comparison | Forward | Reverse | Order-balanced geometric estimate |
| --- | ---: | ---: | ---: |
| Overall median overhead | +5.52% | -0.77% | +2.33% |
| Overall P90 overhead | +3.53% | -0.05% | +1.72% |
| 4K median overhead | +11.77% | +3.47% | +7.54% |
| 7.8K median overhead | +5.00% | -0.86% | +2.03% |
| 16K median overhead | +3.59% | -0.12% | +1.72% |

The order reversal changes the overall sign and the 7.8K/16K signs. Only the
4K shape shows positive overhead in both orders. With two service orders and
three cold identities per shape, this is bounded attribution evidence, not a
statistical-significance claim.

Both orders had exact cross-policy first tokens for 18/18 requests. Complete
output identity was 16/18 in both orders, with each differing cold request
matching its own warm replay. This means cache transparency within each policy
passed. It does not establish cross-policy numeric equivalence: teacher-forced
logits and relative L2 were not measured, and no semantic-quality conclusion
is drawn from the four-layer model.

## Attribution

Admission/capture work can contribute a small fixed cost at 4K, but it is not
a credible primary cause of the platform regression:

- the order-balanced median signal is about 2.3%, versus the observed
  34%-39% short-bucket P90 increase;
- 7.8K and 16K cross zero when service order is reversed;
- disabling GDN state caching would remove the effective prefix reuse needed
  for the competition target and is not a valid optimization.

Do not remove the capture-boundary correctness fix, do not enable `off`, and
do not spend another iteration scanning GDN admission thresholds for this
regression. The next investigation should profile other request-level fixed
costs and platform variance with the current compatibility fixes in place.

No `computility-run.yaml`, default policy, `main`, or repository-visibility
change is authorized by M1-170.

Privacy-safe structured evidence is under
`docs/experiments/evidence/M1_170_COLD_CAPTURE_OVERHEAD_TP1_20260801`.
