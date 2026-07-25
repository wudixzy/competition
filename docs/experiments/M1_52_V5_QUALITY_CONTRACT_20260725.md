# M1-52 Long-Context Quality Contract V5

## Scope

V5 separates two capabilities that v4 combined in one exact-string check:

- 131K semantic recall and reasoning correctness;
- strict instruction following and output-format compliance.

This is a test-only contract change. It does not change model weights, dtype,
tokenizer, chat template, sampling parameters, runtime kernels, cache policy,
scheduler behavior, `computility-run.yaml`, or default optimization switches.
The implementation commit is
`df5185a9b10b8f776ad027eacc630e204dc98d8e`.

## Bound Evidence

The v4 targeted result at source `57574b7` confirmed the 235K automatic Agent
case and rejected 131K. A second run at source `5a468d1` recorded only frozen
boolean sub-rules. It proved that the 131K response:

- returned HTTP 200 and naturally stopped at 696 of 1024 tokens;
- had nonempty separated reasoning and final content;
- contained every marker in order and the correct arithmetic result;
- contained the complete expected answer exactly as the final suffix;
- failed only because explanatory text preceded that suffix.

The v4 reports remain rejected and immutable. Their exact identities and
artifact hashes are in:

- `docs/experiments/evidence/M1_52_V4_TARGETED_QUALITY_20260725.json`
- `docs/experiments/evidence/M1_52_V4_131K_DIAGNOSTIC_20260725.json`

## Frozen V5 Identity

- Matrix: `quality/long_context_matrix.v5.json`
- Matrix SHA-256:
  `924642ffe55ff8bba66aa42c81889e1c35a231a558a9e1f902619f7c6f0182ac`
- Result schema: `bi100-long-context-quality-result-v5`
- Comparison schema: `bi100-long-context-quality-comparison-v4`
- Maximum model length: `262144`
- Final topology: four BI100 GPUs with tensor parallel size four
- Base image:
  `harbor.4pd.io/modelhubxc/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3`

The 131K prompt now explicitly asks the model to end its final answer with the
expected sequence on the last line. Passing requires that sequence to occur
exactly once as the final suffix, all markers to be present and ordered, the
arithmetic result to be present, reasoning to remain separately nonempty, and
generation to stop naturally before the 1024-token cap. Additional text after
the expected suffix or duplicate expected sequences still fail.

## Independent Instruction Gate

V5 does not treat semantic suffix success as proof of strict instruction
following. Before any cache candidate, performance claim, or promotion, the
project must ingest and freeze a licensed IFEval subset using:

- source: `google/IFEval`;
- organization: Google Research;
- license: Apache-2.0;
- revision: `966cd89545d6b6acfd7638bc708b98261ca58e84`;
- official rule evaluator plus a deterministic OpenAI-chat conversion;
- fixed split, selection rule, manifest, file SHA-256, and download time.

Until that dataset and evaluator are committed and both baseline and candidate
pass the same frozen instruction gate, strict instruction quality remains
unqualified.

The dataset and evaluator are now frozen on the private experiment branch:

- manifest: `quality/external/google_ifeval/manifest.v1.json`;
- manifest SHA-256:
  `578e2233c4a02a06fb35987cebc19fb9f490c06f4949a78d3fdd284c232545c5`;
- subset: 64 rows covering all 25 instruction IDs at least four times;
- subset SHA-256:
  `bdb2e4ec0b0fd19b89c55ebb9ed49e17361706c923ddedeeab429f669e4bdb78`;
- official evaluator revision:
  `e6890f85757dd84e27ca6df2dd30651dafad28e0`.

This completes ingestion only. The exact `fine32/direct` TP4 baseline still
has to execute successfully before the instruction gate is qualified.

## Validation And Order

At the implementation commit:

- full unit discovery: 600 passed, 25 optional-dependency skips;
- quality manifest validation: passed with the exact v5 SHA-256;
- submission preflight: 9/9 passed;
- Git diff and Python/shell syntax checks: passed.

Remote order:

1. Rerun the v5 131K and 235K cases on a fresh fine32/direct TP4 service.
2. If both pass, run the complete functional plus Agent workload gate.
3. Freeze and execute the independent IFEval instruction gate.
4. Run all 12 v5 long-context cases on another fresh service.
5. Only after every quality surface passes may one same-source
   fine32/admission64 A/B run begin.

No v5 diagnostic result alone authorizes `main`, YAML, a default switch,
repository visibility, admission64, or an official performance claim.
