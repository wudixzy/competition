#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import time
from typing import Any

import smoke_api


Json = dict[str, Any]


def _cached_tokens(response: Json) -> int:
    details = response.get("usage", {}).get("prompt_tokens_details") or {}
    value = details.get("cached_tokens", 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"invalid cached_tokens: {value!r}")
    return value


def _response_digest(response: Json) -> str:
    message = response["choices"][0]["message"]
    encoded = json.dumps(
        message,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record(response: Json, elapsed_s: float, expected_color: str) -> Json:
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        raise ValueError(f"invalid elapsed time: {elapsed_s!r}")
    choice = response["choices"][0]
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response has no text content")
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise ValueError(f"invalid prompt_tokens: {prompt_tokens!r}")
    if not isinstance(completion_tokens, int) or completion_tokens <= 0:
        raise ValueError(f"invalid completion_tokens: {completion_tokens!r}")
    return {
        "cached_tokens": _cached_tokens(response),
        "completion_tokens": completion_tokens,
        "elapsed_s": elapsed_s,
        "expected_color_observed": expected_color in content,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": prompt_tokens,
        "response_sha256": _response_digest(response),
    }


def _payload(rgb: tuple[int, int, int]) -> Json:
    material = "内容寻址缓存隔离验证材料。" * 620
    return {
        "model": "llm",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": smoke_api._solid_png_data_url(rgb),
                    },
                },
                {
                    "type": "text",
                    "text": material + "\n图片主体是什么颜色？只回答中文颜色名称。",
                },
            ],
        }],
        "max_tokens": 16,
        "thinking": False,
        "temperature": 0,
        "seed": 20260721,
    }


def _request(base: str, rgb: tuple[int, int, int], color: str) -> Json:
    started = time.perf_counter()
    response = smoke_api.post_chat(base, _payload(rgb), timeout=600)
    return _record(response, time.perf_counter() - started, color)


def run_gate(base: str, run_id: str) -> Json:
    red_cold = _request(base, (255, 0, 0), "红")
    red_warm = _request(base, (255, 0, 0), "红")
    green_isolated = _request(base, (0, 255, 0), "绿")

    red_exact = all(
        red_cold[field] == red_warm[field]
        for field in ("completion_tokens", "finish_reason", "response_sha256")
    )
    prompt_tokens_match = len({
        red_cold["prompt_tokens"],
        red_warm["prompt_tokens"],
        green_isolated["prompt_tokens"],
    }) == 1
    checks = {
        "cold_has_no_hit": red_cold["cached_tokens"] == 0,
        "different_image_isolated": green_isolated["cached_tokens"] == 0,
        "prompt_tokens_match": prompt_tokens_match,
        "red_cold_warm_exact": red_exact,
        "semantic_colors_observed": (
            red_cold["expected_color_observed"]
            and red_warm["expected_color_observed"]
            and green_isolated["expected_color_observed"]
        ),
        "same_image_hits": red_warm["cached_tokens"] > 0,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "qualified": not reasons,
        "reasons": reasons,
        "requests": {
            "different_image": green_isolated,
            "same_image_cold": red_cold,
            "same_image_warm": red_warm,
        },
        "run_id": run_id,
        "schema": "bi100-multimodal-prefix-isolation-v1",
        "version": 1,
    }


def _write_report(path: pathlib.Path, report: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    smoke_api.get_models(args.base)
    report = run_gate(args.base, args.run_id)
    _write_report(pathlib.Path(args.json_out), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
