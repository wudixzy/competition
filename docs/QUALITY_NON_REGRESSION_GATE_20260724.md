# BI100 model-quality non-regression gate

## Scope

Model capability and output correctness are hard promotion gates. They do not
share a pass/fail result with performance. A candidate needs separate qualified
quality, cache-correctness, long-context, numerical, stability, and performance
reports before it can be proposed for `main` or for a formal YAML change.

The functional gate executes all 53 frozen rows derived from `指标集合`. The
long-context gate executes the 12 deterministic cases in
`quality/long_context_matrix.v2.json`. The selected 13-request performance
sample remains a smoke/proxy dataset and is not treated as an 881-request score
or a model-quality reference.

## Frozen identities

- Functional manifest SHA-256:
  `fe9b958610d9d0df8f54504d9c149442f145226c03cf76668711d2d38ed51d0e`
- Long-context matrix SHA-256:
  `3217ec047f7b78af6747269c3f85baed6bfdd86c6527aca6335dbfa7d9f0452b`
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

Create one private runtime-contract JSON per service launch from
`quality/runtime_contract.example.json`. Replace both zero placeholders and all
`replace-with-*` values. `source_revision` must equal `git rev-parse HEAD`, and
the command/environment must describe the process that is actually running.

The validator rejects the retired ModelHub base image, credential-like fields,
SSH keys, access tokens, and a quality service without `BI100_CACHE_TRACE=1`.
Do not add proxy credentials or repository tokens to the contract. Baseline and
candidate contracts must use the same model, tokenizer, command, base image,
TP topology, and non-`BI100_*` environment. Only documented `BI100_*`
optimization switches may differ.

## Execution order

Use a newly started service with an empty prefix/GDN cache for every gate and
for every A/B side. Do not execute the functional and long-context suites back
to back against one service. Restart between them. Capture the service stdout
and stderr in a local diagnostic log for the long-context cache-trace proof.

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
