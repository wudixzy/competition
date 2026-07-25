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
- Python wheels are fixed for CPython 3.10 on x86_64. Each wheel retains its
  upstream metadata and license files.
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
  `8ac44a97a6f569056415deedb8a59cbc815cbad6577cbb2e713016864cc7f0fa`

Reproduction command:

```bash
python3 scripts/freeze_ifeval_subset.py \
  --downloaded-at-utc 2026-07-25T05:12:45Z
```

## Run contract

The runner sends each prompt verbatim as one user message. It does not
override the chat template, tokenizer, thinking behavior, `top_p`, penalties,
or model structure. It fixes only the deterministic benchmark request fields:
`temperature=0`, `seed=20260725`, `stream=false`, and `max_tokens=4096`.

The committed report contains rule booleans, usage, finish reason, dimensions,
timings, and response hashes only. Raw responses exist only in a mode-0600
checkpoint under `/tmp` while a run is incomplete and are deleted when the
report is written.

A complete fine32/direct run establishes baseline counts. A candidate may not
decrease strict or loose pass counts at prompt, instruction, instruction-ID,
or instruction-family level. Cache candidates additionally use
`--require-exact-output`. A standalone IFEval result never authorizes a main
merge, YAML change, default switch, repository visibility change, or
performance claim.
