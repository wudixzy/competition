#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import time
import urllib.request
from typing import Any


# A deterministic 1x1 PNG. The server image processor expands it to its
# configured minimum size, while the repeated text forces chunked prefill.
_TEST_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII=")


def _data_url() -> str:
    return "data:image/png;base64," + base64.b64encode(_TEST_PNG).decode("ascii")


def _stream_chat(base: str, payload: dict[str, Any], timeout: float) -> dict:
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_delta_s = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    saw_done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.status
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if value == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(value)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content")
            reasoning = delta.get("reasoning_content")
            if text:
                content_parts.append(text)
            if reasoning:
                reasoning_parts.append(reasoning)
            if first_delta_s is None and (text or reasoning):
                first_delta_s = time.perf_counter() - started
    elapsed_s = time.perf_counter() - started
    output = "".join(reasoning_parts) + "\n" + "".join(content_parts)
    details = usage.get("prompt_tokens_details") or {}
    return {
        "http_status": status,
        "saw_done": saw_done,
        "elapsed_s": elapsed_s,
        "ttft_s": first_delta_s,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),
        "completion_tokens": usage.get("completion_tokens"),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "output": output,
    }


def _health(base: str, timeout: float = 10) -> int:
    with urllib.request.urlopen(base.rstrip("/") + "/health",
                                timeout=timeout) as response:
        return response.status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--repeat", type=int, default=12000)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.repeat < 3000:
        raise SystemExit("--repeat must be at least 3000 for a chunked prompt")

    long_text = (
        "Keep this numbered reference material unchanged: "
        + "prefix-cache-mrope-token " * args.repeat
        + "\nReply with exactly OK."
    )
    payload = {
        "model": "llm",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": _data_url()},
            }, {
                "type": "text",
                "text": long_text,
            }],
        }],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 8,
        "min_tokens": 8,
        "ignore_eos": True,
        "thinking": False,
        "temperature": 0,
        "seed": 20260716,
    }

    results = []
    error = ""
    try:
        for label in ("cold", "warm"):
            result = _stream_chat(args.base, payload, args.timeout)
            result["label"] = label
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        error = repr(exc)

    health_status = None
    try:
        health_status = _health(args.base)
    except Exception as exc:
        if not error:
            error = f"health: {exc!r}"

    report = {
        "repeat": args.repeat,
        "results": results,
        "health_status": health_status,
        "error": error,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if error or len(results) != 2 or health_status != 200:
        return 1
    cold, warm = results
    if not all(result["http_status"] == 200 and result["saw_done"]
               for result in results):
        return 1
    if not cold["prompt_tokens"] or cold["prompt_tokens"] <= 8192:
        return 1
    if warm["cached_tokens"] <= 0:
        return 1
    if cold["output_sha256"] != warm["output_sha256"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
