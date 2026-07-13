#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--salt-prefix", required=True)
    parser.add_argument("--groups", default="abc")
    parser.add_argument("--prompt-repeat", type=int, default=126)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    results = {}
    for group in args.groups:
        salt = f"{args.salt_prefix}-{group}"
        prompt = (
            f"请基于以下材料回答问题。RUN_ID={salt}\n"
            + "BI100 Qwen3.6 prefix cache benchmark material. "
            * args.prompt_repeat
            + "\n问题：请用一小段话概括材料主题。"
        )
        payload = {
            "model": "llm",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "thinking": False,
            "seed": args.seed,
        }
        request = urllib.request.Request(
            f"{args.base.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=360) as response:
            body = json.load(response)
        choice = body["choices"][0]
        message = choice["message"]
        canonical = json.dumps(message, ensure_ascii=False, sort_keys=True)
        results[group] = {
            "message": message,
            "message_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage"),
        }

    report = {
        "label": args.label,
        "salt_prefix": args.salt_prefix,
        "groups": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        group: {
            "sha256": value["message_sha256"],
            "finish_reason": value["finish_reason"],
            "usage": value["usage"],
        }
        for group, value in results.items()
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
