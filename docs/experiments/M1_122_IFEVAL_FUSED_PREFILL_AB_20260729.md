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

The hard quality comparator requires:

- 64/64 HTTP-200, normalized responses and zero request errors;
- zero chat 4xx responses;
- no regression in strict or loose prompt pass counts;
- no regression in strict or loose instruction pass counts;
- no regression for any instruction ID or instruction family;
- exact runtime, request, evaluator, manifest, and topology identities;
- clean service and outer lifecycle evidence.

A second comparator records exact semantic-output identity for every request.
Exact drift does not get relabeled as equality. A run may pass the independent
IFEval capability gate with deterministic output drift only when all score
non-regression checks pass and every exact-comparison rejection is exclusively
an output-identity difference. This does not relax the separate M1-116
next-token and numerical gates.

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

## Remote Evaluator Preparation

The private `ssh-73ca29ba` instance cannot fetch the pinned NLTK archive
directly. The archive was therefore downloaded locally from revision
`4f15a3d89eefe9748ec1c05be495d91289197155`, verified against manifest
SHA-256
`e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106`,
and only the four English `punkt_tab` files consumed by
`prepare_ifeval_env.py` were transferred over the private instance channel.
The English-only transfer bundle SHA-256 is
`c2608c30f9bb7f342f3b7794bbc0e9f4ef3e626a9eef326825fb50ce2f68d418`.

The resulting private evaluator directory is
`/root/ifeval-env-m1-123-6eeb65a`. Every committed wheel or source
distribution was rechecked against the manifest before installation, the
sentence/word-count import smoke passed, and `install.json` has SHA-256
`4168d1df55339e67b4e15a981aa543dbf06b85acd422a286b329357bcd586d59`.
Neither the NLTK files nor the environment are committed or added to the
model-service path. This preparation is not an IFEval result and grants no
promotion authorization.

## Authorization

M1-122 is a quality gate only. It cannot authorize performance claims,
`computility-run.yaml`, a default selector, `main`, repository visibility, or
production promotion. Those remain blocked on the complete M1-109 component,
M1-114 performance, M1-116 functional/Agent/output, M1-117 long-context,
capacity, lifecycle, and final TP4 evidence.
