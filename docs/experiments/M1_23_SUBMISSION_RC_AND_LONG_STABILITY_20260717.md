# M1-23 Submission RC And Long-Stability Gate - 2026-07-17

## Scope

M1-23 freezes the current scoring candidate and verifies the submission
artifact before another official evaluation. It does not change model kernels,
prefix policy, launch arguments, or the `262144` context contract.

The RC adds a dependency-free submission preflight, removes Pillow from the
API smoke client, normalizes the run manifest to LF line endings, and checks
the exact Docker/YAML/offline-wheel contract. The private RC branch reached
`215ca46` after fixing an import collision with CoreX's installed `tests`
package.

## RC gates

- local preflight: 7/7 checks pass;
- local unit discovery: 174 tests pass, 22 environment skips;
- remote CoreX preflight: exit 0;
- remote CoreX unit discovery: 172 tests pass, one environment skip, exit 0;
- production PID remained `35836` and `/health` plus `/v1/models` remained
  HTTP 200;
- no model restart, runtime patch, or YAML semantic change was used for these
  gates.

The authoritative remote outputs are `preflight.real.json`,
`preflight.real.exitcode`, `unittest.real.log`, and
`unittest.real.exitcode`. An older `unittest.exitcode=1` was stale and was
superseded by the timestamped `.real` run.

## Harness incident

The first detached long runner is invalid evidence. Its `timeout` invocation
omitted the duration, so every command returned 125 immediately. A shell
variable named `rc` was then overwritten by the health helper, causing the
runner log to claim success. The raw directory is retained under
`M1_23_LONG_STABILITY_20260717`, but none of its subtests count.

The corrected runner uses explicit durations, local status variables, a
current server-log offset, per-item metadata, and fail-fast health checks. A
second correction gives every context length a unique first block; otherwise
the 65K cold request legitimately reused the preceding 32K prompt and violated
the zero-cache test assumption.

## Three-cycle TP4 matrix

The authoritative directory is
`/root/M1_23_LONG_STABILITY_REAL2_20260717`. All 18 subtests passed with
`exit_code=0`, `postcheck_code=0`, `RUNNER_OK`, and an empty server/stderr
error scan. Each cycle included 25-prefix LRU pressure, four unique
cold/warm lengths, and a 1,000-token decode.

| Prompt | Cold elapsed range | Warm elapsed range | Cold cache | Warm cache | Hash |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 32,768 | 39.010-39.217 s | 2.092-2.143 s | 0 | 32,752 | same |
| 65,536 | 87.249-88.829 s | 2.807-3.086 s | 0 | 65,520 | same |
| 131,072 | 223.149-224.616 s | 3.977-5.148 s | 0 | 131,056 | same |
| 235,000 | 557.967-559.718 s | 5.557-6.293 s | 0 | 234,992 | same |

Every prefix-pressure cycle evicted the first 10.6K request to zero cached
tokens, then restored a 10,608-token hit after refresh without changing its
output hash. The three independent 1,000-token decode requests completed in
`47.899-48.049 s` and produced the same message hash.

The archived matrix is only 936 KiB before compression. Its remote and local
tarball SHA-256 is
`63bebff8ccb9e3387761d1d57408b1f3fd6b71b29fd48f69d69ca5e0795c4f0d`.

## Sustained long-decode finding

A stronger 235K cold/warm request forced 1,000 generated tokens. Both requests
completed with `finish_reason=length`; cold/warm cache was `0/234992`, elapsed
time was `784.583/220.673 s`, and the service remained healthy without native
or worker errors. The output hashes differed, however:

- cold: `35c789137abd0acca6d3cb2102009b3f051a0af8a3e9286a30051f48d24192e5`;
- warm: `daa1d37db0095b15a34cb371607b9ba857c6b391b4dd999708b383c32b34db1d`.

Tokenizer comparison locates the first content divergence at generated token
97. A follow-up matrix forced 256 tokens at 32K, 65K, and 131K; all three
cold/warm pairs were identical. This excludes a simple `>32768` dispatch
boundary and isolates an extreme-context numerical/prefix-replay issue rather
than an engine crash.

The 235K archive SHA-256 is
`089c81edfdc8a7b4ce176d70a68094cfaa09bf01dd4a39521baa88b991544e0c`.
The 32K-131K diagnostic archive SHA-256 is
`0784f31f906539bb3ea8f1d6160c376a31e5c2720d8355f64d61fa2587fbb975`.

## Infrastructure finding

The proposed second four-card instance `ssh-913ffbfe` contains the model and
CoreX userspace files, but its container exposes no GPU PCI function or
`/dev/ix*`, `/dev/bi*`, or `/dev/dri` node. It cannot run parallel GPU tests;
installing Torch would not repair missing device passthrough.

## Decision

`QUALIFY_RC`, because the RC itself changes submission validation and test
portability only. It may be fast-forwarded to main after this evidence is
committed.

Do not claim full 235K cold/warm numerical identity for long generation. The
next runtime experiment is M1-24: preserve the cold request's final scheduler
chunk boundary during direct prefix fast-forward, then require the 235K/1,000
token hash to match while retaining a material warm-speed and cache benefit.
This must be a guarded algorithmic A/B, not a scan of chunk sizes or YAML
parameters.
