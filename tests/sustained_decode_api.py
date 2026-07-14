#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


Json = dict[str, Any]


def post_chat(base: str, payload: Json, timeout_s: float) -> tuple[Json, float]:
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--tokens", type=int, default=1000)
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--expected-sha256", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = {
        "model": "llm",
        "messages": [{
            "role": "user",
            "content": "Reply with exactly: DECODE-1000",
        }],
        "max_tokens": args.tokens,
        "min_tokens": args.tokens,
        "temperature": 0,
        "seed": args.seed,
    }
    response, elapsed_s = post_chat(args.base, payload, args.timeout_s)
    choice = response["choices"][0]
    message = choice["message"]
    usage = response.get("usage") or {}
    encoded = json.dumps(
        message, ensure_ascii=False, sort_keys=True).encode("utf-8")
    report = {
        "elapsed_s": elapsed_s,
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
        "response": response,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    assert report["completion_tokens"] == args.tokens, report
    assert report["finish_reason"] == "length", report
    if args.expected_sha256:
        assert report["message_sha256"] == args.expected_sha256, report
    print(json.dumps({key: value for key, value in report.items()
                      if key != "response"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
