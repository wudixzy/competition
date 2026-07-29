# BI100 layered inference validation gate

## Why the gate is layered

A fused floating-point implementation can be mathematically valid without
being bitwise invariant. PyTorch explicitly does not guarantee bitwise
identity across mathematically equivalent operation orders or backends, and
its scaled-dot-product attention documentation warns that fused backends can
produce different outputs. This means a later greedy-token divergence is
evidence to investigate, but is not by itself proof of a capability loss.

The opposite shortcut is also invalid. A task score can miss a systematic
numeric error, stale cache state, protocol break, or high-confidence logit
flip. Semantic evaluation therefore cannot override a failed numerical,
cache-transparency, or protocol gate.

The project now uses six independent layers. A layer can authorize only its
own scope.

## Layer 1: protocol and request semantics

This remains hard and exact. Valid and invalid HTTP status classes, SSE
framing, usage accounting, finish reasons, tool-call JSON, structured output,
reasoning/content separation, stop behavior, sampling parameters, tokenizer,
chat template, multimodal handling, and maximum-length behavior must satisfy
their frozen contracts. Output wording need not match across implementations
when the contract does not require it.

The 53 functional rows and eleven Agent rows are deterministic contract tests,
not a statistical sample of general model ability. A baseline-passing
contract may not fail in the candidate.

## Layer 2: cache transparency

This also remains hard and exact. Prefix KV and GDN state reuse are intended
to be observationally transparent. For the same request in the same service,
cold and warm status, structure, generated token sequence, tool calls,
reasoning, finish reason, and effective cache accounting must match. Missing
or mismatched recurrent state must fail fast. Physical block reuse and
different multimodal content must not cross logical identities.

This exact rule applies to cache-on versus cache-off behavior. It must not be
reused as the rule for comparing two valid attention reduction orders.

## Layer 3: operator numerical fidelity

Every custom operator independently compares fixed tensors with the reference
path over all supported production shapes. Non-finite output, shape/ABI
drift, relative L2 above `1e-5`, or absolute error above `1e-3` remains a hard
failure. This catches large or structured numeric errors even if a small task
suite happens to pass.

The limits match the existing M1-109 component contract. Current vLLM
attention tests also compare optimized kernels to a reference with explicit
absolute and relative tolerances instead of bitwise equality.

## Layer 4: teacher-forced distribution fidelity

Teacher forcing removes autoregressive error compounding. Control and
candidate receive the same fixed prompt-token sequence, and the private
harness samples prompt logprobs at fixed positions for 4K, 32K, 65K, 131K,
and 235K inputs.

The CoreX vLLM 0.6.3 API exposes prompt top-logprobs. Before inference, the
harness calls the loopback-only `/tokenize` route with the same messages and
chat-template kwargs. It requires the server-rendered token sequence to match
the locally constructed sequence in memory, then uses the server sequence as
the teacher identity. This diagnostic is cold-only: both the serving path and
collector reject any nonzero cached-token count, because skipped prefix
positions cannot provide teacher-forced logits. A disabled-by-default
diagnostic request field retains only the 64 requested prompt-logprob rows in
the HTTP response; ordinary API responses are unchanged. The private arm
reports may temporarily retain HMAC-keyed token identities under the run
root. The comparison report retains only aggregate counts and numeric deltas,
then both final and interrupted atomic-write arm artifacts are deleted.

The fixed screen checks:

- finite teacher-token and shared top-k logprobs;
- teacher-token identity alignment at every sampled position;
- top-1 agreement rate;
- maximum and p99 teacher-token logprob deltas;
- p99 shared top-k logprob delta;
- mean teacher-token NLL regression;
- no high-margin top-1 flip;
- mutual top-5 support for every low-margin top-1 flip.

The high-margin guard is the larger of `0.05` nats and four times the observed
p99 shared-logprob delta. A low-margin flip that remains mutually supported in
top-5 is recorded rather than automatically called a quality failure. This is
similar in role to vLLM's model tests, which permit generated-token mismatch
when each implementation's token remains in the other's top logprobs.

Top-k evidence is not a full-vocabulary proof and cannot replace the operator
gate. A candidate with acceptable semantic answers but failed numeric limits
still fails.

## Layer 5: autoregressive trajectory

Trajectory identity has two roles:

- Same-arm repeat determinism and cache cold/warm identity are hard.
- Control/candidate full greedy identity is diagnostic unless a specific
  protocol or task contract requires an exact answer.

For a cross-arm divergence, record the first position, baseline top-1 margin,
mutual top-k coverage, and whether the teacher-forced high-margin rule passed.
Do not infer a capability loss solely from suffix divergence after a
low-margin token choice. Do not hide a high-margin flip behind semantic
similarity.

## Layer 6: task capability non-inferiority

Task scoring uses paired samples and reports the four paired counts:
both-pass, baseline-only, candidate-only, and both-fail. Critical functional,
tool, structured-output, multimodal, and cache contracts permit zero
baseline-only failures.

Broader capability datasets use a predeclared one-sided paired bootstrap
lower confidence bound and an exact one-sided McNemar diagnostic. The default
promotion margin is two percentage points at 95% confidence. With zero
observed regressions, at least 149 samples are needed before a two-point claim
is considered powered. A fixed 64-row IFEval subset can support a five-point
screen, whose corresponding zero-event floor is 59 samples, but it cannot by
itself establish a two-point promotion claim.

An underpowered result is `inconclusive`, not a pass or a failure. The paired
CLI uses a distinct exit code (`3`) for that state; callers must inspect the
JSON status instead of converting every nonzero exit into a capability
failure. Dataset revision, split, checksum, selection rule, evaluator, and
request semantics remain frozen.

Harness identity, missing cases, incomplete sample matrices, or zero comparable
teacher-forced positions are `invalid`, not numerical failures. The comparison
CLI uses exit code `2` for this state. A numerical threshold crossing is the
only path to `fail` and exit code `1`.

## Performance and promotion

Performance is evaluated only after protocol, cache, and numerical layers
pass. Kernel and TP4 end-to-end results remain separate. A material TP4 gain
with a different but numerically supported trajectory proceeds to capability
non-inferiority testing instead of being rejected immediately.

Production promotion still requires all project performance targets, complete
functional and long-context gates, the teacher-forced screen, sufficiently
powered task evidence, clean TP4 lifecycle, and no fatal/OOM/Gloo/worker-loss
event. No individual report changes `main`, `computility-run.yaml`, defaults,
or repository visibility.

## M1-109 reclassification

Existing evidence supports reopening M1-109:

- the component path improved by a median `1.939x`;
- maximum output relative L2 was `6.625e-6` and maximum absolute error was
  `2.441e-4`, so the operator layer passed;
- cold TP4 TTFT improved by `17.70%`, `23.38%`, `30.36%`, and `36.72%` at
  32K, 65K, 131K, and 235K;
- each arm independently passed 53/53 functional and 11/11 Agent cases;
- same-arm cold/warm outputs were exact and all first tokens matched;
- the complete M1-125 long-context matrix later passed 12/12;
- cross-arm output first diverged at token budget eight for the focused 65K
  request.

Under the layered gate, the final item is trajectory evidence requiring
teacher-forced and task adjudication, not an automatic rejection. M1-109 is
therefore `reopened-pending-evidence`, not promoted. M1-122 IFEval and the new
teacher-forced TP4 run remain required.

## Primary references

- PyTorch numerical accuracy:
  https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
- PyTorch scaled-dot-product attention:
  https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html
- vLLM attention kernel reference comparison:
  https://github.com/vllm-project/vllm/blob/main/tests/kernels/attention/test_attention.py
- vLLM model logprob comparison:
  https://github.com/vllm-project/vllm/blob/main/tests/models/utils.py
- MLPerf Inference rules and model-equivalence accuracy bounds:
  https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc

The references justify separating numeric equivalence, trajectory invariance,
and task accuracy. Project-specific thresholds remain frozen in
`quality/layered_quality_gate.v1.json` and must be changed before, not after,
observing a candidate result.
