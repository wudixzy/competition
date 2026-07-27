# BI100 model-quality non-regression gate

## Scope

Model capability and output correctness are hard promotion gates. They do not
share a pass/fail result with performance. A candidate needs separate qualified
quality, cache-correctness, long-context, numerical, stability, and performance
reports before it can be proposed for `main` or for a formal YAML change.

The functional gate executes all 53 frozen rows derived from `指标集合`. The
long-context gate executes the 12 deterministic cases in
`quality/long_context_matrix.v5.json`. The selected 13-request performance
sample remains a smoke/proxy dataset and is not treated as an 881-request score
or a model-quality reference.

Matrix v5 supersedes v4 for new runs. Bound v4 evidence confirmed the 235K
automatic Agent contract and showed that the 131K response naturally stopped,
contained the complete ordered marker sequence and correct arithmetic result as
its final suffix, but included additional explanatory content before it. V5
therefore gates the intended long-context semantic capability: the expected
answer must occur exactly once as the final suffix, with ordered markers,
correct arithmetic, separated reasoning, and a natural stop. Strict instruction
following is not discarded or inferred from this case; a frozen IFEval gate is
mandatory before promotion. V2, v3, and v4 files and reports remain immutable
historical evidence, and results from different matrix versions cannot form an
A/B pair.

## Frozen identities

- Functional manifest SHA-256:
  `fe9b958610d9d0df8f54504d9c149442f145226c03cf76668711d2d38ed51d0e`
- Long-context matrix SHA-256:
  `924642ffe55ff8bba66aa42c81889e1c35a231a558a9e1f902619f7c6f0182ac`
- Required base image:
  `harbor.4pd.io/modelhubxc/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3`
- Maximum model length: `262144`
- Final topology: four BI100 GPUs with tensor parallel size four

The matrix is project-generated and contains no external dataset rows. Its
manifest freezes the generator, seed, image identities, tool-schema identities,
token accounting rules, and the `261887`/`261888` prompt boundaries. Runtime
reports additionally bind the exact tokenizer/config files and chat template by
SHA-256.

## Runtime contract

Create one private runtime-contract JSON per service launch. The fixed lifecycle
harness calls `tests/build_quality_runtime_contract.py`; operators must not fill
revision, overlay hash, command, or environment fields by hand. The builder
requires a clean source tree, verifies every capability-critical runtime file,
and recomputes the complete overlay tree SHA-256 before writing the contract.
`quality/runtime_contract.example.json` is documentation only.

The validator rejects the retired ModelHub base image, credential-like fields,
SSH keys, access tokens, and a quality service without `BI100_CACHE_TRACE=1`.
Do not add proxy credentials or repository tokens to the contract. Baseline and
candidate contracts must use the same source revision, overlay tree, instance,
model, tokenizer, command, base image, and TP topology. Only the four explicitly
documented optimization switches below may differ.

## Fixed lifecycle harness

Every invocation starts exactly one fresh TP4 service, runs four-GPU preflight
before and after, validates physical KV-block reuse and both installed
model-input GDN action broadcasts, validates the startup contract, executes
one quality surface, and scans fatal/OOM/Gloo/NCCL/worker-loss/timeout
signatures. Cleanup is limited to the process group created by that invocation:
send `SIGTERM`, wait at least 60 seconds for TP4 workers and collective
runtimes to exit, use `SIGKILL` only for surviving members, and then reap the
leader. Do not use broad `pkill` cleanup.

The `EXIT`/signal trap always performs residual API-server and worker scans,
open-GPU-device scans, and a repeated per-card CUDA preflight. Cleanup,
postflight, fatal scan, timeout scan, final preflight, and preflight comparison
are independent fail-closed gates. A nonzero or missing required result makes
the experiment invalid, even when its request or performance report passed.
Raw logs remain under a private `/tmp` path outside the repository. The
functional run executes all 53 rows. It still sends and validates the `n=2`
request, but records the manifest's sole documented skip only when the fixed
`--max-num-seqs 1` direct engine returns the exact normalized 400 response and
the post-request health probe succeeds. No other skip is accepted. The same
fresh service then executes the separate eleven-case Agent compatibility
matrix covering named and automatic tools in both non-streaming and SSE modes,
tool-role round trips, a 92-tool schema, long history, and multiple system
messages. Its report retains only hashes, counts, usage, and validation facts.

The overlay must be installed from the exact clean experiment commit. Use one
overlay and one instance for all four A/B runs:

```bash
export BI100_RUNTIME_SITE_PACKAGES=/root/private-quality-runtime/site-packages

scripts/run_quality_service_gate.sh \
  functional fine32 direct 0 lru baseline-fine32 private-instance \
  /tmp/bi100-quality/baseline-functional

scripts/run_quality_service_gate.sh \
  long-context fine32 direct 0 lru baseline-fine32 private-instance \
  /tmp/bi100-quality/baseline-long-context

scripts/run_quality_service_gate.sh \
  functional admission64 direct 0 lru candidate-admission64 private-instance \
  /tmp/bi100-quality/candidate-functional

scripts/run_quality_service_gate.sh \
  long-context admission64 direct 0 lru candidate-admission64 private-instance \
  /tmp/bi100-quality/candidate-long-context
```

The only A/B environment differences accepted by the comparison gates are
`BI100_GDN_CACHE_POLICY`, `BI100_GDN_RESTORE_MODE`,
`BI100_ATTN_COREX_FUSED_PREFILL`, and `BI100_KV_EVICTION_POLICY`. Every other
recorded environment value must match.

## Manual execution

Manual execution is diagnostic-only. Use a newly started service with an empty
prefix/GDN cache for every gate and every A/B side. Do not execute functional
and long-context suites back to back against one service. Capture stdout and
stderr in a private diagnostic log for long-context cache-trace proof.

To isolate known long-context failures before spending a full run, set a
comma-separated strict case list on the lifecycle harness. Any explicit case
selection remains ineligible for baseline or promotion regardless of outcome:

```bash
BI100_LONG_CONTEXT_CASES=65k_multiturn_large_tools,131k_reasoning_recall,235k_agent_large_output_budget \
scripts/run_quality_service_gate.sh \
  long-context fine32 direct 0 lru diagnostic-m1-52 private-instance \
  /tmp/bi100-quality/diagnostic-m1-52
```

Run the complete functional contract:

```bash
scripts/run_quality_functional_gate.sh \
  http://127.0.0.1:8000 \
  /root/public-storage/models/Qwen/Qwen3.6-35B-A3B \
  /tmp/bi100-baseline-runtime.json \
  baseline-fine32 \
  overlay-identity \
  private-instance-id \
  /tmp/bi100-quality/baseline-functional.json
```

After another clean service restart, run the complete long-context contract:

```bash
scripts/run_quality_long_context_gate.sh \
  http://127.0.0.1:8000 \
  /root/public-storage/models/Qwen/Qwen3.6-35B-A3B \
  /tmp/bi100-baseline-runtime.json \
  /tmp/bi100-baseline-service.log \
  baseline-fine32 \
  overlay-identity \
  private-instance-id \
  /tmp/bi100-quality/baseline-long-context.json
```

Repeat both runs for the candidate using fresh services. Then compare each
quality surface independently:

```bash
python3 tests/compare_quality_gate_reports.py \
  /tmp/bi100-quality/baseline-functional.json \
  /tmp/bi100-quality/candidate-functional.json \
  --out /tmp/bi100-quality/functional-comparison.json

python3 tests/compare_agent_workload_reports.py \
  /tmp/bi100-quality/baseline-functional/agent_workload.json \
  /tmp/bi100-quality/candidate-functional/agent_workload.json \
  --out /tmp/bi100-quality/agent-comparison.json

python3 tests/compare_long_context_quality_reports.py \
  /tmp/bi100-quality/baseline-long-context.json \
  /tmp/bi100-quality/candidate-long-context.json \
  --out /tmp/bi100-quality/long-context-comparison.json
```

The long-context run requires cache trace v4 proof for same-image identity and
cross-image isolation. Text cases require exact post-template token counts.
The multimodal case records the local post-template count separately from the
server's vision-token expansion, preventing a valid visual expansion from being
misclassified as tokenizer drift.

## Interpretation

`quality_non_regression_authorized` and
`long_context_quality_non_regression_authorized` authorize only their own
quality surfaces. Both comparison reports deliberately keep
`overall_promotion_authorized=false`. Performance still requires the separate
official/proxy report, including Output/Input/Cache TPS, TTFT, success rate,
weighted score, fatal/OOM/worker-loss scan, and before/after GPU state.

Do not retain raw prompts or model outputs in repository evidence. Reports keep
only protocol facts, usage counters, generated-asset identities, normalized
output hashes, request-contract hashes, and runtime identities. The legacy
`long_context_api.py` output is explicitly diagnostic-only and cannot authorize
overall promotion.
