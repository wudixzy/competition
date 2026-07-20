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


def eviction_target_mods(eviction_count: int, target_mod: int) -> tuple[int, ...]:
    if eviction_count <= 0:
        raise ValueError("eviction_count must be positive")
    if not 0 <= target_mod < 16:
        raise ValueError("target_mod must be in [0, 15]")
    return tuple((target_mod + index) % 16
                 for index in range(eviction_count))


def post_chat(base: str, payload: Json, timeout: float) -> Json:
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def make_payload(tokenizer: Any, marker: str, target_mod: int) -> tuple[Json, int]:
    material = (
        f"测试编号 {marker}。请记住以下材料："
        + "信创模盒 BI100 prefix cache 隔离与驱逐测试材料。" * 620
    )
    for filler_count in range(128):
        content = material + (" x" * filler_count) + "\n问题：材料的测试编号是什么？"
        messages = [{"role": "user", "content": content}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(token_ids) > 8192 and len(token_ids) % 16 == target_mod:
            return ({
                "model": "llm",
                "messages": messages,
                "max_tokens": 16,
                "thinking": False,
                "temperature": 0,
                "seed": 20260712,
            }, len(token_ids))
    raise RuntimeError(f"could not construct prompt modulo {target_mod}")


def response_summary(response: Json, elapsed_s: float) -> Json:
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    choice = response["choices"][0]
    message = choice["message"]
    encoded = json.dumps(message, ensure_ascii=False, sort_keys=True).encode()
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
        "elapsed_s": round(elapsed_s, 3),
    }


def request_once(base: str, payload: Json, timeout: float) -> tuple[Json, Json]:
    started = time.monotonic()
    response = post_chat(base, payload, timeout)
    return response, response_summary(response, time.monotonic() - started)


def assert_equivalent(reference: Json, candidate: Json) -> None:
    reference_choice = reference["choices"][0]
    candidate_choice = candidate["choices"][0]
    assert candidate_choice["message"] == reference_choice["message"]
    assert candidate_choice.get("finish_reason") == reference_choice.get(
        "finish_reason")
    assert (candidate.get("usage") or {}).get("completion_tokens") == (
        reference.get("usage") or {}).get("completion_tokens")


def persist_report(path: Path, report: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model-path",
        default="/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
    )
    parser.add_argument("--eviction-count", type=int, default=17)
    parser.add_argument("--eviction-target-mod", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--run-id", default=str(time.time_ns()))
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.eviction_count < 17:
        parser.error("--eviction-count must be at least 17")
    if not 0 <= args.eviction_target_mod < 16:
        parser.error("--eviction-target-mod must be in [0, 15]")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    report: Json = {"run_id": args.run_id, "interleaved": [], "eviction": []}
    persist_report(args.json_out, report)

    interleaved = [
        make_payload(tokenizer, f"{args.run_id}-aligned", 0),
        make_payload(tokenizer, f"{args.run_id}-unaligned", 7),
    ]
    first_results = []
    for payload, expected_tokens in interleaved:
        response, summary = request_once(args.base, payload, args.timeout_s)
        assert summary["prompt_tokens"] == expected_tokens, summary
        first_results.append(response)
        report["interleaved"].append({"first": summary})
        persist_report(args.json_out, report)
    for index, (payload, expected_tokens) in enumerate(interleaved):
        response, summary = request_once(args.base, payload, args.timeout_s)
        assert summary["prompt_tokens"] == expected_tokens, summary
        assert summary["cached_tokens"] >= 8176, summary
        assert_equivalent(first_results[index], response)
        report["interleaved"][index]["cached"] = summary
        persist_report(args.json_out, report)

    eviction_payloads = []
    eviction_first = []
    target_mods = eviction_target_mods(
        args.eviction_count, args.eviction_target_mod)
    for index, target_mod in enumerate(target_mods):
        payload, expected_tokens = make_payload(
            tokenizer, f"{args.run_id}-evict-{index:02d}", target_mod)
        response, summary = request_once(args.base, payload, args.timeout_s)
        assert summary["prompt_tokens"] == expected_tokens, summary
        eviction_payloads.append(payload)
        eviction_first.append(response)
        report["eviction"].append({"first": summary})
        persist_report(args.json_out, report)

    replay, replay_summary = request_once(
        args.base, eviction_payloads[0], args.timeout_s)
    report["eviction"][0]["after_lru_pressure"] = replay_summary
    persist_report(args.json_out, report)
    assert_equivalent(eviction_first[0], replay)

    cached, cached_summary = request_once(
        args.base, eviction_payloads[0], args.timeout_s)
    report["eviction"][0]["cached_after_refresh"] = cached_summary
    persist_report(args.json_out, report)
    assert cached_summary["cached_tokens"] >= 8176, cached_summary
    assert_equivalent(eviction_first[0], cached)

    persist_report(args.json_out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
