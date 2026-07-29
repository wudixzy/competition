# M1-116 Fused Prefill Quality Adjudication

## Decision scope

M1-109 passed its component numerical gate and M1-114 made the final prefill
capture boundary executable. The first M1-114 TP4 performance pair showed:

- identical cold/warm output within each arm at 32K, 65K, 131K, and 235K;
- identical control/candidate first-token identities at all four lengths;
- identical complete control/candidate output at 32K, 131K, and 235K;
- deterministic but different complete control/candidate output at 65K;
- candidate cold TTFT improvements of 17.70%, 23.38%, 30.36%, and 36.72%.

The 65K difference is not treated as a pass or an automatic kernel rejection.
Strict exact-output evidence remains a failed gate until independent functional,
agent, and long-context capability checks adjudicate the floating-point-order
difference. This experiment cannot authorize production promotion by itself.

## Diagnostic contract

`tests/diagnose_m1_116_fused_prefill_output.py` runs one fixed 65K synthetic
request contract per arm:

1. cold and warm `max_tokens=32` requests reproduce the M1-109 pair-1 prompt;
2. two warm requests at each `max_tokens=1,2,4,8,16,32` locate the first stable
   output-length divergence;
3. a second cold/warm `235000`-token request reproduces pair 2, where the
   complete output also diverged while the first token remained exact;
4. each arm must remain internally exact, finite, HTTP 200, and fully cached
   after the cold request;
5. the A/B runtime contracts may differ only in
   `BI100_ATTN_COREX_FUSED_PREFILL`.

The report contains no prompt, model output, token ID, credential, or raw
first-token digest. The orchestrator generates one ephemeral 256-bit HMAC key,
passes it to both diagnostics, and never writes or prints it. The service runner
removes the inherited key before starting vLLM. The diagnostic process removes
it from its environment before model requests. Reports retain only keyed output
identities, aggregate timing, usage, and request-contract hashes.

`tests/compare_m1_116_fused_prefill_output.py` separates three conclusions:

- diagnostic/runtime validity;
- next-token identity;
- strict complete-output identity.

A later-token difference with an exact next token is recorded as
`quality_adjudication_required=true`, while the strict gate and production
promotion remain false. A first-token difference is a direct quality failure.

## Execution order

After the ongoing M1-114 three-pair performance runner releases all four GPUs:

1. install one immutable runtime overlay from this commit and verify its exact
   identity;
2. run `scripts/run_m1_116_fused_prefill_quality_adjudication_ab.sh`;
3. retain the strict diagnostic return code even if functional and agent gates
   pass;
4. run the complete 12-case long-context matrix with fresh control and
   candidate services and the existing strict comparator;
5. inspect functional, agent, 65K diagnostic, long-context, lifecycle, fatal,
   timeout, and preflight evidence separately.

No `computility-run.yaml`, default selector, or `main` change is allowed from
this experiment without all quality gates and a repeatable TP4 performance
result.

## Local verification

Before the TP4 run:

- focused M1-116 and shared quality-runner tests pass;
- complete local unit discovery passes;
- shell and Python syntax checks pass;
- submission preflight passes all nine checks;
- the worktree contains no retained raw requests, outputs, credentials, or
  model artifacts.

## Fresh TP4 result

The fixed-order A/B completed on `ssh-73ca29ba` from exact source
`6eeb65a25ed5475ab0b9f2f6a965a21d707ff89f` and one immutable runtime
overlay:

```text
control:   admission64/hybrid64/lru, fused prefill off
candidate: admission64/hybrid64/lru, fused prefill on
run root:  /tmp/m1-116-fused-quality-6eeb65a-20260729-v3
```

Both arms independently passed all 53 functional cases and all 11 Agent
cases. Each arm also passed startup, runtime identity, prefix/GDN checks,
expected-4xx attribution, scoped cleanup, recovery, fatal and timeout scans,
postflight, and repeated four-GPU preflight. The candidate runtime emitted
the new privacy-safe request-validation dimensions: three validation records
were summarized as `root` or `multiple` fields and `value_error` or
`multiple` types. No validation message, field value, or request body was
retained.

The 32768-token truncation case took `1494.667 s` in the control arm and
`1501.907 s` in the candidate arm. The derived output rates were
`21.923 token/s` and `21.818 token/s`, a candidate delta of `-0.482%`.
This is a quality-runner decode observation rather than the frozen TP4
performance benchmark, but it shows no material Output TPS regression.

## Cross-arm adjudication

The strict cross-arm checks did not qualify:

- the functional comparator qualified 52 of 53 cases;
- `multimodal_input` passed its independent rules in both arms, but its
  normalized output and completion-token usage differed;
- all 11 Agent comparisons qualified;
- the focused 65K and 235K diagnostics were valid and internally cold/warm
  exact in both arms;
- every diagnostic first token was identical;
- 65K output remained identical through `max_tokens=4` and first differed at
  `max_tokens=8`;
- the 65K and 235K 32-token complete outputs differed across arms.

The outer return code is therefore `1` by design, while both arm return codes
and every lifecycle gate are zero. This is evidence of deterministic
later-token floating-point divergence, not evidence that the candidate is
production-qualified and not an arm-level functional failure.

M1-116 authorizes the already-defined M1-117 long-context adjudication only.
It does not authorize a selector default, YAML change, `main` merge,
performance claim, or production promotion. The M1-108 exact-output fused
prefill remains the conservative fallback if the broader capability gates do
not clear M1-109.

Structured privacy-safe evidence:
`docs/experiments/evidence/M1_116_FUSED_PREFILL_QUALITY_ADJUDICATION_20260729/qualification.json`.
