# Frozen Google IFEval quality gate

This directory contains the private, reproducible instruction-following gate
for BI100 candidates. It is not an official competition workload or score
proxy.

## Sources and licenses

- Dataset: `google/IFEval`, Google Research, Apache-2.0, Hugging Face revision
  `966cd89545d6b6acfd7638bc708b98261ca58e84`.
- Rule evaluator: Google Research `instruction_following_eval`, Apache-2.0,
  repository revision `e6890f85757dd84e27ca6df2dd30651dafad28e0`.
- The source dataset, dataset card, evaluator source, upstream README, and
  Google Research license are committed because their licenses permit
  redistribution and their combined size is small.
- Python distribution artifacts are fixed for CPython 3.10 on x86_64. Wheels
  retain their upstream metadata and license files. `langdetect` is bound to
  its PyPI source archive and built inside the isolated target so the gate does
  not depend on a machine-specific, locally produced wheel.
- NLTK `punkt_tab` is not committed because the pinned `nltk_data` index does
  not state a redistribution license. Before a run, download revision
  `4f15a3d89eefe9748ec1c05be495d91289197155` from the URL in
  `manifest.v1.json`, verify SHA-256
  `e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106`,
  and pass the archive to `scripts/prepare_ifeval_env.py`.

The Hugging Face source and the evaluator repository source have one prompt
text difference at key `2785`. The frozen 64-row subset does not select that
key. The Hugging Face revision remains the declared data source.

## Frozen selection

`scripts/freeze_ifeval_subset.py` verifies the 541-row source identity, then
selects 64 rows with seed `bi100-ifeval-v1-seed-20260725`. The greedy phase
covers every one of the 25 instruction IDs at least four times. Remaining rows
are selected by a seeded SHA-256 rank, and requests are emitted in source
order.

- Source SHA-256:
  `6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2`
- Subset SHA-256:
  `bdb2e4ec0b0fd19b89c55ebb9ed49e17361706c923ddedeeab429f669e4bdb78`
- Manifest SHA-256:
  `07ec4efb5fe7afaacb55723c1d53be4c2f58c840bbd6a54bf944e15cfbca1855`

Reproduction command:

```bash
python3 scripts/freeze_ifeval_subset.py \
  --downloaded-at-utc 2026-07-25T05:12:45Z
```

## Run contract

The runner sends each prompt verbatim as one user message. It does not
override the chat template, tokenizer, thinking behavior, `top_p`, penalties,
or model structure. It fixes only the deterministic benchmark request fields:
`temperature=0`, `seed=20260725`, `stream=false`, and `max_tokens=8192`.

The committed report contains rule booleans, usage, finish reason, dimensions,
timings, and response hashes only. Raw responses exist only in a mode-0600
checkpoint under `/tmp` while a run is incomplete and are deleted when the
report is written. A separate progress file contains only public dataset keys,
counts, error types, and error-message hashes; it never contains prompt,
content, or reasoning text.

The first request-contract revision used `max_tokens=4096`. Its unqualified
TP4 probe returned HTTP 200 for every attempted request, but each failed
response consumed exactly all 4096 completion tokens. Because default thinking
must remain enabled, that budget could prevent a usable final content field.
No baseline was established from that probe. Contract v2 changes only the
completion budget to 8192; model, tokenizer, chat template, thinking behavior,
temperature, seed, request order, dataset, and evaluator rules are unchanged.

Prepare the evaluator dependencies with the target CoreX CPython 3.10 binary.
The `punkt_tab` archive must already have the revision and SHA-256 above:

```bash
python3 scripts/prepare_ifeval_env.py \
  --target /root/competition-ifeval-env-<evaluator-revision> \
  --punkt-tab-archive /tmp/punkt_tab-4f15a3d.zip
```

Run the model from a clean, commit-bound runtime tree and run the client from a
separate clean evaluator tree. Evaluator-only Python packages are injected into
the client process and never enter the service or worker `PYTHONPATH`:

```bash
bash scripts/run_ifeval_service_gate.sh \
  /root/competition-runtime-source \
  /root/competition-runtime-overlay/site-packages \
  /root/competition-ifeval-env-<evaluator-revision> \
  <runtime-revision> fine32 direct 0 lru \
  ifeval-fine32-direct <instance> /tmp/ifeval-fine32-direct
```

After the lifecycle finishes, qualify every status, artifact, privacy, and GPU
gate before preserving the summary:

```bash
python3 tests/qualify_ifeval_service_gate.py \
  --run-root /tmp/ifeval-fine32-direct \
  --ifeval-install /root/competition-ifeval-env-<evaluator-revision>/install.json \
  --expected-runtime-revision <runtime-revision> \
  --expected-evaluator-revision <evaluator-revision> \
  --out /tmp/ifeval-fine32-direct/qualification.json
```

A complete fine32/direct run establishes baseline counts. A candidate may not
decrease strict or loose pass counts at prompt, instruction, instruction-ID,
or instruction-family level. Cache candidates additionally use
`--require-exact-output`. A standalone IFEval result never authorizes a main
merge, YAML change, default switch, repository visibility change, or
performance claim.
