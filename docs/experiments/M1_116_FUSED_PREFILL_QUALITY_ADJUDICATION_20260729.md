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
3. each arm must remain internally exact, finite, HTTP 200, and fully cached
   after the cold request;
4. the A/B runtime contracts may differ only in
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
