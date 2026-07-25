# M1-52 Agent Quality Contract

## Scope

M1-52 repairs and revalidates three long-context quality contracts before any
new cache or kernel candidate is measured. It does not change model weights,
dtype, tokenizer, chat template, sampling semantics, cache policy, scheduler,
model kernels, `computility-run.yaml`, or the default experiment switches.

The experiment branch is `fix/M1-52-agent-quality-20260725`. Its implementation
and intended runtime anchor is `6dfdab10524d71435dd5d60d2ac80135237e5ccf`,
descended from the exact M1-51 diagnostic source
`8c00ba6c6a916b68d1d020330ccf0e4d7fb0800c`.

## Version Separation

The user-supplied 881-request platform result came from a repository `main`
without a source revision, runtime overlay hash, tokenizer identity, or request
manifest identity. It remains the unbound historical reference recorded in
`docs/experiments/evidence/PLATFORM_MAIN_REFERENCE_20260724.json`.

M1-52 is many commits beyond that platform build and uses a new frozen
long-context matrix. No performance gain, quality gain, regression, or formal
score may be inferred by comparing those two runs. A valid A/B still requires
the same exact source, overlay, model, tokenizer, request manifest, request
order, instance, and four-GPU topology, with only the declared optimization
switch changed.

## Repairs

1. Commit `f929677` reclassifies a valid guided JSON object only when a named
   tool response was otherwise consumed as unterminated reasoning. It does not
   rewrite invalid JSON or apply to automatic tool selection.
2. Commit `ddd4a03` records privacy-safe structural diagnostics for malformed
   tool calls. It stores lengths, types, and SHA-256 digests, never tool names,
   argument values, prompts, images, or model output.
3. Commit `441dab1` introduces `quality/long_context_matrix.v3.json` and fixes
   the three rejected M1-51 contracts:
   - the 65K target tool permits and requires only the expected `key` field;
   - the 131K marker recall is checked in observable final content while
     separated, nonempty reasoning remains independently required;
   - the 235K Agent turn uses explicit `tool_choice=auto`, requires exact tool
     arguments, separated reasoning, cold/warm equality, cache accounting, and
     natural completion before the unchanged 8192-token cap.
4. Commit `6dfdab1` allows strict explicit-case diagnostics through
   `BI100_LONG_CONTEXT_CASES`. Such runs are always ineligible for baseline and
   promotion; the complete 12-case extended run remains mandatory.

## Frozen Identities

- Long-context matrix schema: `bi100-long-context-quality-matrix-v3`
- Matrix SHA-256:
  `a968fbbc37bf2e03b14fcf8cdb4df005e1956b4a93a23f62661860d523a85680`
- Result schema: `bi100-long-context-quality-result-v3`
- Required base image:
  `harbor.4pd.io/modelhubxc/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3`
- Model path: `/root/public-storage/models/Qwen/Qwen3.6-35B-A3B`
- Maximum model length: `262144`
- Final topology: four BI100 GPUs, tensor parallel size four

Matrix v2 and its bound M1-51 reports remain immutable historical evidence.
Matrix v2 and v3 reports cannot form a quality A/B pair.

## Local Validation

At source `6dfdab1`:

- full unit discovery: 578 passed, 25 optional-dependency skips;
- submission preflight: 9/9 passed;
- quality-data manifest validation: passed;
- long-context matrix v3: 12 cases, exact SHA-256 matched;
- Python and shell syntax checks: passed;
- repository worktree after commit: clean.

## Remote Gate Order

1. Build an atomic runtime overlay from exact source `6dfdab1` and record its
   complete tree SHA-256.
2. Run the 65K large-tools, 131K reasoning, and 235K Agent cases as an explicit
   diagnostic on a fresh TP4 service.
3. If all three pass, run the complete functional plus Agent workload gate on
   another fresh service.
4. Run the complete 12-case matrix v3 on another fresh service.
5. Only after all quality gates pass may the same-source `fine32` versus
   `admission64` A/B begin.

No result in this document authorizes `main`, a formal YAML change, repository
visibility change, or an official performance claim.
