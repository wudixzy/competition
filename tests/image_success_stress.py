#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

from smoke_api import _assert_content, _solid_png_data_url, post_chat


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=27)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be positive")

    results: list[dict[str, Any]] = []

    def write_report() -> None:
        if not args.json_out:
            return
        success_count = sum(bool(item["ok"]) for item in results)
        latencies = [float(item["elapsed_s"]) for item in results]
        report = {
            "count": args.count,
            "completed": len(results),
            "success_count": success_count,
            "success_rate": success_count / len(results) if results else 0.0,
            "latency_p50_s": _percentile(latencies, 0.50),
            "latency_p90_s": _percentile(latencies, 0.90),
            "results": results,
        }
        pathlib.Path(args.json_out).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for index in range(args.count):
        rgb = ((index * 47 + 17) % 256,
               (index * 89 + 31) % 256,
               (index * 131 + 53) % 256)
        started = time.perf_counter()
        try:
            data = post_chat(args.base, {
                "model": "llm",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _solid_png_data_url(rgb),
                            },
                        },
                        {
                            "type": "text",
                            "text": "Reply with OK only.",
                        },
                    ],
                }],
                "max_tokens": 4,
                "thinking": False,
                "temperature": 0,
            }, timeout=360)
            content = _assert_content(data)
            result = {
                "index": index,
                "rgb": rgb,
                "ok": True,
                "elapsed_s": time.perf_counter() - started,
                "content": content,
                "error": "",
            }
        except Exception as exc:
            result = {
                "index": index,
                "rgb": rgb,
                "ok": False,
                "elapsed_s": time.perf_counter() - started,
                "content": "",
                "error": repr(exc),
            }
        results.append(result)
        write_report()
        print(
            f"[{index + 1:02d}/{args.count}] ok={result['ok']} "
            f"elapsed={result['elapsed_s']:.3f}s",
            flush=True,
        )

    success_count = sum(bool(item["ok"]) for item in results)
    success_rate = success_count / args.count
    print(f"success={success_count}/{args.count} rate={success_rate:.4f}")
    if success_rate < 0.99:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
