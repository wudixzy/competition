# T5 Batch Scan - 2026-07-12

## Fixed Protocol

- Remote: `ssh-987372d0.default.gpu.phanthy.com`
- Model: `/root/public-storage/models/Qwen/Qwen3.6-35B-A3B`
- Code checkpoint: `49a789c`
- Code archive: `f66d98595ca5b9db388b2d2e34a42897e76c6ea51c5361467dedd02f9b1a1107`
- `MAX_NUM_BATCHED_TOKENS=8192`
- `GPU_MEMORY_UTILIZATION=0.90`
- `NUM_GPU_BLOCKS_OVERRIDE=18000`
- `BI100_PREFIX_BLOCKS_PER_TILE=32`
- `BI100_DNN_CHUNK=4096`
- `BI100_GDN_ALLOW_NAN_ZERO=0`
- `BI100_PROFILE=0`
- Benchmark: 8 requests, 4 workers, 0.25 s stagger, 64 max tokens

The explicit block override is fixed across all T5 configurations. Without it,
vLLM's synthetic startup `profile_run()` produced non-finite GDN values before
serving any request. The override skips only that synthetic capacity probe;
real smoke and benchmark requests keep the normal fail-fast non-finite checks.

## Hardware Gate

- CUDA GPU0-GPU3: PASS
- NCCL rank0-rank3: PASS
- All-reduce value: `10.0`

The hardware gate was repeated after the failed synthetic profile and remained
fully green.

## S1 - `max_num_seqs=1`

- Remote run: `bench_runs/20260711_190358_nogit_T5_S1_maxseq1_override18000`
- Full smoke: 14/14 PASS
- Success rate: `1.0`
- Wall time: `76.07549745216966 s`
- TTFT P90: `29.295343060791488 s`
- Output TPS P10: `1.7261421889924187`
- Input TPS: `192.2300936539315`
- Output TPS total: `6.730156451778783`
- Cache TPS: `167.8332765162334`
- Cache hit rate: `0.8730853391684902`
- Weighted score: `661.0309511927616`
- Prompt/completion/cached tokens: `14624 / 512 / 12768`
- Errors: none

S1 reproduces the historical correct baseline (weighted score approximately
650) on the clean replacement instance.

## Matrix

| Config | max_num_seqs | Smoke | TTFT P90 | Output P10 | Input TPS | Cache hit | Weighted | Result |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| S1 | 1 | 14/14 | 29.2953 | 1.7261 | 192.2301 | 87.31% | 661.0310 | baseline |

