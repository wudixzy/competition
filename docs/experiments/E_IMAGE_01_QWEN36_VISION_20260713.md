# E-IMAGE-01: Qwen3.6 native image support

Date: 2026-07-13

## Metric impact

The formal suite requires base64 PNG requests to return HTTP 200 with
recognized content, and the overall request success rate must be at least 99%.
The observed dataset contains 27 image requests among 881 requests. If all of
them return 400, the maximum possible success rate is 854 / 881 = 96.94%.
This was therefore a correctness blocker, not a secondary feature.

The performance targets remain:

- Output TPS P10 >= 20
- TTFT P90 <= 5 seconds
- prefix-cache hit rate >= 50%
- request success rate >= 99%
- weighted token throughput >= 8000

## Root cause

The checkpoint is multimodal:

- `config.json` contains `vision_config` and `image_token_id=248056`.
- `configuration.json` declares `image-text-to-text`.
- 333 checkpoint tensors use the `model.visual.*` prefix.

However, both the official EngineX baseline and our inherited model adapter
declared `Text-only (no VL)` and skipped every `model.visual.*` weight. The
baseline `chat_utils.py` also had no placeholder mapping for
`model_type=qwen3_5_moe`, causing:

```text
TypeError: Unknown model type: qwen3_5_moe
RuntimeWarning: coroutine 'async_get_and_parse_image' was never awaited
HTTP 400
```

The official baseline repository has the same omission, so the failure was
not introduced by the later throughput optimizations.

## Implementation

`qwen3_6_scripts/qwen3_5.py` now provides the native single-image path:

1. Slow `Qwen2VLImageProcessor` preprocessing, avoiding an unavailable
   `torch.compiler.is_compiling` API in the CoreX torch 2.1 runtime.
2. Qwen3.6 patch embedding, learned interpolated position embeddings, 27
   vision blocks, patch merger, and all 333 visual checkpoint tensors.
3. TP-sharded visual QKV/MLP/projection loading with QKV head reordering.
4. Interleaved Qwen3.5 T/H/W MRoPE instead of ordinary text-only RoPE.
5. Expansion of one image placeholder to the exact number of merged visual
   tokens and replacement with visual embeddings during prefill.
6. A deterministic image-content cache marker so different images cannot
   alias to the same token-only prefix-cache key.
7. Prefix-cache-aware suffix merging: if leading image tokens are cached,
   only uncached visual embeddings are merged; a fully cached image skips the
   visual tower.

8. Explicit RGB conversion and channels-last input metadata for both visual
   placeholder sizing and Qwen2VL preprocessing. Keeping those two stages on
   the same normalized image avoids ambiguous channel inference and token versus
   embedding count mismatches for tiny images, while accepting grayscale/RGBA
   PNGs. In vLLM 0.6.3, an exception at the late mapper/model stage terminates
   the async engine, so this normalization protects the request-success metric
   as well as individual image compatibility.

`chat_utils.py` maps Qwen3.5/Qwen3.6 images to the native token sequence:

```text
<|vision_start|><|image_pad|><|vision_end|>
```

The vLLM 0.6.3 blanket prefix-cache disable is bypassed only for the registered
`Qwen3_5MoeForCausalLM` native vision adapter. Other multimodal models retain
the vendor default.

### Guided JSON correctness

Full smoke repeatedly exposed an inherited vLLM/Outlines grammar defect:
`response_format={"type":"json_object"}` could return HTTP 200 but include a
raw control character inside a JSON string. The vendor grammar imported
`UNESCAPED_STRING: /"[^"]*"/`, which admits raw newlines, tabs, and bytes
`0x00-0x1f` even though `json.loads()` must reject them.

The install patch now replaces that terminal with an RFC 8259-compatible JSON
string expression. Escaped control characters remain valid, while raw control
characters are excluded during token generation. It also replaces unlimited
ignored whitespace with explicit structural whitespace slots capped at four
characters. This prevents otherwise valid generation from spending the whole
`max_tokens` budget on whitespace before closing the object. The patch can
upgrade both the vendor grammar and its first strict-string revision, remains
idempotent, and has direct regex unit coverage.

The OpenAI protocol adapter additionally maps `json_object` to the generic
JSON Schema `{"type":"object"}`. In this Outlines version that schema permits
arbitrary object properties but uses the stable regex guide, avoiding the
generic CFG guide's stateful first-request failure. Both Chat Completion and
Completion request conversions use the same mapping.

## Runtime evidence

- All four ranks loaded `vision_items=333`.
- Per-card model weights: 16.7280 GB.
- GPU KV blocks: 16884, versus 17943 for the qualified text-only IPC build
  (-1059, -5.90%).
- Maximum reported concurrency at 100000 tokens: 2.70x.
- Qualified native-vision service startup time: approximately 6 minutes 23
  seconds to 7 minutes 34 seconds; most of this is the fixed 8192-token,
  max-image `determine_num_available_blocks` profile.
- A 256x256 PNG becomes 256 raw patches and 64 merged LLM visual tokens.
- Pixel semantic check with identical prompts:
  - red image -> `红色`
  - green image -> `绿色`
- The prefix-cache test and two consecutive image-semantic rounds passed. The
  second round reused the same images and left the engine healthy, directly
  covering the previously fatal fully cached-image path.
- The 27-request unique-image stress run passed 27/27 (100%): latency p50
  6.807 seconds, p90 6.906 seconds, and maximum 6.940 seconds. No request
  failed and the engine remained healthy.
- The original `Unknown model type`, unawaited coroutine, and HTTP 400 errors
  no longer occur on the native image path.
- The image-channel edge qualification on 2026-07-14 returned HTTP 200 for a
  32x32 RGB PNG (26.71 seconds) and a 1x1 grayscale PNG (8.97 seconds). A final
  health request also returned HTTP 200, PID 10600 remained alive, and the
  service log contained no new errors. The pre-fix grayscale request produced
  70 placeholders for 64 visual embeddings and terminated the async engine.

## Qualification

Automated unit/static suite after the image-cache and guided-JSON fixes: 116
tests passed, with one existing runtime-dependent test skipped.

Final full API smoke: 15/15 passed. This includes native image semantics,
prefix caching, `json_object`, `json_schema`, streaming, forced tool calls,
sampling boundaries, multilingual turns, and deterministic seeds. The final
`json_object` smoke completed in 26.48 seconds; five cold/unique object
requests also returned parseable JSON, including the first guided request
after service startup.

Comparable text benchmark (`3` requests, `1` worker, prompt repeat `126`,
`max_tokens=64`):

- success rate: 100%
- TTFT P90: 4.158 seconds
- Output TPS P10: 12.004
- input TPS: 241.689
- cache hit rate: 66.26%
- weighted score: 967.79

The best of the three preceding qualified text-candidate samples scored
899.62, so this sample is 7.6% higher. Decode TPS is effectively unchanged;
the gain comes from input and cache throughput. This small local benchmark is
useful for regression comparison but is not a substitute for the official
881-request score.

Image stress artifacts are stored under:

```text
bench_runs/20260713_E_IMAGE_03_CACHEFIX/
```

Final schema-JSON, full-smoke, and text-performance artifacts are stored under:

```text
bench_runs/20260713_E_IMAGE_06_SCHEMA_JSON/
```

## Scope

This experiment supports image input, including multiple images flattened in
request order. Video input is not enabled. Image preprocessing is capped at
1280 merged visual tokens (about 1.31 MP at patch size 16 and merge size 2) to
bound startup profiling and per-request memory.

The evaluator command fixes `--max-num-seqs 1`, so each scheduler batch has a
single sequence. Prefix-cache-aware visual suffix merging is qualified under
that fixed constraint; mixed cache states across multiple simultaneous image
sequences are outside this submission's runtime contract.
