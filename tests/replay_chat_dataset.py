#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def post_chat(base: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("chat_dataset_v0.json"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--timeout-s", type=float, default=360)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text())
    conversations = []
    for conversation_index, item in enumerate(dataset):
        messages: list[dict[str, Any]] = []
        if item.get("system_prompt"):
            messages.append({"role": "system", "content": item["system_prompt"]})
        turns = []
        for turn_index, question in enumerate(item["user_questions"]):
            messages.append({"role": "user", "content": question})
            payload = {
                "model": "llm",
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "thinking": False,
                "seed": args.seed,
            }
            started = time.perf_counter()
            body = post_chat(args.base, payload, args.timeout_s)
            elapsed = time.perf_counter() - started
            choice = body["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            if not content:
                raise RuntimeError(
                    f"empty content at conversation {conversation_index} turn {turn_index}",
                )
            messages.append({"role": "assistant", "content": content})
            turns.append({
                "turn": turn_index,
                "question": question,
                "message": message,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "finish_reason": choice.get("finish_reason"),
                "usage": body.get("usage"),
                "elapsed_s": elapsed,
            })
        conversations.append({
            "conversation": conversation_index,
            "system_prompt": item.get("system_prompt", ""),
            "turns": turns,
        })

    report = {
        "label": args.label,
        "dataset": str(args.dataset),
        "conversation_count": len(conversations),
        "turn_count": sum(len(item["turns"]) for item in conversations),
        "conversations": conversations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "label": args.label,
        "conversation_count": report["conversation_count"],
        "turn_count": report["turn_count"],
        "completion_tokens": sum(
            int(turn["usage"].get("completion_tokens") or 0)
            for item in conversations for turn in item["turns"]
        ),
        "elapsed_s": sum(
            float(turn["elapsed_s"])
            for item in conversations for turn in item["turns"]
        ),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
