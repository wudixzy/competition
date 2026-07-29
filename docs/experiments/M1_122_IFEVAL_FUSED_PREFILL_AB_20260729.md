# M1-122 Fused Prefill IFEval A/B

## Purpose

M1-116 covers the functional and Agent contracts and M1-117 covers the fixed
long-context matrix. M1-122 adds an independent instruction-following gate for
the M1-109 fused-prefill candidate. It does not replace either earlier gate.

The experiment uses the committed Google Research IFEval subset and official
rule evaluator:

- dataset: `google/IFEval`;
- dataset revision: `966cd89545d6b6acfd7638bc708b98261ca58e84`;
- evaluator revision: `e6890f85757dd84e27ca6df2dd30651dafad28e0`;
- license: Apache-2.0;
- frozen subset: 64 requests covering all 25 instruction IDs;
- manifest:
  `quality/external/google_ifeval/manifest.v1.json`;
- manifest SHA-256:
  `07ec4efb5fe7afaacb55723c1d53be4c2f58c840bbd6a54bf944e15cfbca1855`.

Request semantics remain fixed at temperature zero, seed `20260725`,
`max_tokens=8192`, thinking unchanged, and no chat-template override.

## A/B Contract

Two fresh TP4 services use one source, one immutable runtime overlay, the
submission kernel profile, `max_model_len=262144`, admission64/hybrid64, and
LRU eviction. The only runtime environment delta is:

```text
BI100_ATTN_COREX_FUSED_PREFILL=0 -> 1
```

The hard experiment-integrity comparator requires:

- 64/64 HTTP-200, normalized responses and zero request errors;
- zero chat 4xx responses;
- exact runtime, request, evaluator, manifest, and topology identities;
- clean service and outer lifecycle evidence.

Capability is evaluated on paired prompts under the layered gate. Strict and
loose prompt outcomes each use the predeclared one-sided paired-bootstrap
screen at 95% confidence and a five-percentage-point non-inferiority margin.
The 64 prompts exceed the zero-regression power floor of 59 for this screen.
They do not meet the 149-prompt floor required for the final two-point
promotion claim, so M1-122 cannot establish production non-inferiority alone.

The previous zero-stratum policy remains a sensitive diagnostic: aggregate,
instruction-level, instruction-ID, and family regressions are still recorded,
but one small stratum no longer overrules the paired prompt result without
statistical power. A separate comparator records exact semantic-output
identity for every request. Exact drift does not get relabeled as equality,
but it is trajectory evidence rather than an automatic capability failure.
Neither semantic screen can waive the independent operator-numerical,
teacher-forced, cache-transparency, protocol, or lifecycle gates.

## Privacy And Lifecycle

The offline evaluator dependencies are added only to the evaluator process
`PYTHONPATH`; they never enter the vLLM service environment. Its install
attestation and import smoke are checked before startup.

Raw prompts and model outputs exist only in a mode-0600 checkpoint under the
private run root in `/tmp`. The service runner removes that checkpoint from its
`trap` path on success, failure, timeout, or interruption. Committed reports
contain only rule results, counts, usage, timing, and output identities.

Each arm and the outer runner retain process-session identity, use SIGTERM with
a 60-second grace before any SIGKILL, wait/reap scoped children, qualify
recorded-session recovery, scan 4xx/fatal/timeout evidence, run process
postflight, and repeat all four GPU preflights.

## Command

After M1-116 and the higher-priority M1-117 run release TP4:

```bash
BI100_RUNTIME_SITE_PACKAGES=/path/to/site-packages \
BI100_RUNTIME_INSTALL_REPORT=/path/to/install.json \
scripts/run_m1_122_ifeval_fused_prefill_ab.sh \
  INSTANCE /path/to/offline-ifeval-env \
  /tmp/m1-122-ifeval-SOURCE
```

The run is intentionally long because each frozen request permits the official
8192-token completion budget.

## Authorization

M1-122 is a five-point capability screen only. It cannot authorize a final
two-point non-inferiority claim, performance claims,
`computility-run.yaml`, a default selector, `main`, repository visibility, or
production promotion. Those remain blocked on the complete M1-109 component,
M1-114 performance, M1-116 functional/Agent/output, M1-117 long-context,
capacity, lifecycle, and final TP4 evidence.
