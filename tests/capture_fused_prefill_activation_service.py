#!/usr/bin/env python3
"""Issue one synthetic request per length for private activation capture."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from bench_fused_prefill_service import _post_stream
from long_context_api import build_exact_prompt


def main() -> int:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model-path",
        default="/root/public-storage/models/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--targets", default="32768,65536,131072")
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    targets = [int(value) for value in args.targets.split(",")]
    if (
        not targets
        or targets != sorted(set(targets))
        or any(value <= 32 for value in targets)
        or any(value + args.max_tokens > 262144 for value in targets)
    ):
        parser.error("targets must be unique increasing valid prompt lengths")
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("timeout-s must be finite and positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    started = time.monotonic()
    requests = []
    for target in targets:
        content = build_exact_prompt(
            tokenizer, target, f"{args.run_id}-{target}")
        payload = {
            "model": "llm",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": args.max_tokens,
            "min_tokens": args.max_tokens,
            "temperature": 0,
            "seed": 20260730,
            "thinking": False,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        result = _post_stream(
            args.base, payload, args.timeout_s, tokenizer)
        if (
            result["prompt_tokens"] != target
            or result["completion_tokens"] < args.max_tokens
            or result["cached_tokens"] != 0
        ):
            raise RuntimeError(
                f"capture request contract failed for target {target}")
        requests.append({
            "target_prompt_tokens": target,
            "elapsed_s": result["elapsed_s"],
            "ttft_s": result["ttft_s"],
            "completion_tokens": result["completion_tokens"],
            "cached_tokens": result["cached_tokens"],
            "finish_reason": result["finish_reason"],
            "first_token_sha256": result["first_token_sha256"],
            "output_sha256": result["output_sha256"],
        })
    report = {
        "schema": "bi100-fused-prefill-activation-capture-requests-v1",
        "version": 1,
        "run_id": args.run_id,
        "targets": targets,
        "max_tokens": args.max_tokens,
        "elapsed_s": time.monotonic() - started,
        "requests": requests,
        "privacy": {
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "targets": targets,
        "elapsed_s": report["elapsed_s"],
        "qualified": True,
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
