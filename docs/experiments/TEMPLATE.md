# Experiment `<ID>`: `<Title>`

## Hypothesis

State one measurable hypothesis. Do not combine unrelated model, scheduler, and
API changes.

## Manifest

```json
{
  "experiment_id": "<ID>",
  "commit": "<candidate SHA>",
  "baseline_commit": "<baseline SHA>",
  "contract_sha256": "5f07f4377dcdde3bb858012bedc014f60e84a82a61e9696bee830fec1e517c0f",
  "image_id": "<image or unavailable>",
  "model_config_sha256": "cb30bf4e6205013f03c30fb0e275a6ffec6ecf2002410d401e35b20e8701e69c",
  "gpu_inventory": [],
  "env": {},
  "prompt_dataset_sha256": "<sha256>",
  "seed": 123,
  "cache_mode": "cold|warm|mixed",
  "start_time_utc": "<ISO-8601>"
}
```

## Change

Describe the exact files, algorithmic change, and rollback command.

## Contract

- `computility-run.yaml` changed: no
- performance environment overrides: none
- debug/profiler enabled for score: no

## Correctness gates

- non-GPU tests:
- operator parity:
- four-GPU preflight:
- collective preflight:
- full smoke:
- cached/uncached output hashes:
- 100K boundary:
- error-log scan:

## Performance protocol

Record prompt dataset, request order, seed, output lengths, cache state, warmup,
and A/B ordering. Run at least three paired comparisons and report P10/P50/P90,
the worst request, overlap score, and disjoint score.

## Results

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| TTFT P90 | | | |
| Decode TPS P10 | | | |
| ITL P90 | | | |
| Prompt TPS total | | | |
| Prompt TPS uncached | | | |
| Cache TPS | | | |
| Score overlap | | | |
| Score disjoint | | | |
| HBM peak | | | |
| RSS peak | | | |

## Decision

`KEEP | REJECT | INCONCLUSIVE`

Explain the decision, residual risks, and next experiment. A performance gain
cannot override a correctness, hardware, or contract failure.
