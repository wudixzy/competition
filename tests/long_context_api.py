#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any

import exact_chat_prompt as exact_prompt
import quality_gate_api as quality

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[1]


def prompt_token_count(tokenizer: Any, content: str) -> int:
    return exact_prompt.chat_template_token_count(
        tokenizer,
        [{"role": "user", "content": content}],
        thinking=False,
        template_kwargs_mode="direct",
    )


def build_exact_prompt(tokenizer: Any, target_tokens: int, run_id: str) -> str:
    content, _ = build_exact_prompt_with_evidence(
        tokenizer, target_tokens, run_id)
    return content


def build_exact_prompt_with_evidence(
    tokenizer: Any,
    target_tokens: int,
    run_id: str,
) -> tuple[str, Json]:
    run_identity = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]

    def recipe(filler: str) -> tuple[list[Json], None]:
        content = (
            f"Long-context contract test {run_identity}. "
            "Remember marker FINAL-99500.\n"
            + filler
            + "\nReply with the marker only."
        )
        return [{"role": "user", "content": content}], None

    try:
        messages, _, evidence = exact_prompt.fit_exact_chat_prompt(
            tokenizer,
            target_tokens,
            recipe,
            seed=20260712,
            namespace="legacy-long-context-" + run_identity,
            thinking=False,
            template_kwargs_mode="direct",
        )
    except exact_prompt.PromptConstructionError as error:
        raise RuntimeError(str(error)) from error
    return messages[0]["content"], evidence


def post_chat(base: str, payload: Json, timeout_s: float) -> tuple[Json, float]:
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        content_type = response.headers.get_content_type()
        if response.status != 200 or content_type != "application/json":
            raise RuntimeError(
                f"invalid HTTP response status={response.status} "
                f"content_type={content_type}")
        result = json.loads(response.read().decode("utf-8"))
    return result, time.monotonic() - started


def summarize(response: Json, elapsed_s: float) -> Json:
    quality._validate_response_schema(response)
    assert_finite(response)
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    choice = response["choices"][0]
    message = choice["message"]
    encoded = json.dumps(message, ensure_ascii=False, sort_keys=True).encode()
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "model": response.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
        "semantic_output_sha256": quality._sha256_json(
            quality._normalized_response(response)),
        "protocol_validated": True,
        "elapsed_s": round(elapsed_s, 3),
    }


def assert_equivalent(first: Json, second: Json) -> None:
    first_choice = first["choices"][0]
    second_choice = second["choices"][0]
    assert second_choice["message"] == first_choice["message"]
    assert second_choice.get("finish_reason") == first_choice.get("finish_reason")
    assert (second.get("usage") or {}).get("completion_tokens") == (
        first.get("usage") or {}).get("completion_tokens")


def assert_finite(value: Any, path: str = "response") -> None:
    if isinstance(value, float):
        assert math.isfinite(value), f"non-finite value at {path}: {value}"
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def persist_report(path: Path, report: Json) -> None:
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
    parser.add_argument("--target-prompt-tokens", type=int, default=99500)
    parser.add_argument("--served-model-name", default="llm")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--min-completion-tokens", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=100000)
    parser.add_argument("--min-cached-tokens", type=int, default=98304)
    parser.add_argument("--max-first-cached-tokens", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--run-id", default=str(time.time_ns()))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--retain-raw-responses",
        action="store_true",
        help="retain raw model responses for local diagnosis only",
    )
    parser.add_argument(
        "--equivalence-mode",
        choices=("exact", "warm-repeat"),
        default="exact",
        help=(
            "exact compares cold and warm; warm-repeat permits a cold/warm "
            "difference but requires two cached warm responses to match"
        ),
    )
    args = parser.parse_args()
    if args.target_prompt_tokens + args.max_tokens > args.max_model_len:
        parser.error("prompt plus max tokens exceeds --max-model-len")
    if not 0 <= args.min_completion_tokens <= args.max_tokens:
        parser.error(
            "--min-completion-tokens must be between zero and --max-tokens")
    if not 0 <= args.max_first_cached_tokens < args.min_cached_tokens:
        parser.error(
            "--max-first-cached-tokens must be nonnegative and less than "
            "--min-cached-tokens")
    if args.retain_raw_responses:
        output_path = args.output_dir.resolve()
        if output_path == ROOT or output_path.is_relative_to(ROOT):
            parser.error(
                "raw responses may only be retained in a directory outside "
                "the repository")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    content, prompt_construction = build_exact_prompt_with_evidence(
        tokenizer, args.target_prompt_tokens, args.run_id)
    payload = {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
        "min_tokens": args.min_completion_tokens,
        "thinking": False,
        "temperature": 0,
        "seed": 20260712,
    }
    request_contract_sha256 = quality._sha256_json(payload)
    tokenizer_metadata = exact_prompt.tokenizer_identity(
        Path(args.model_path), tokenizer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    first, first_elapsed = post_chat(args.base, payload, args.timeout_s)
    if args.retain_raw_responses:
        (args.output_dir / "long_context_response1.json").write_text(
            json.dumps(first, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    second, second_elapsed = post_chat(args.base, payload, args.timeout_s)
    if args.retain_raw_responses:
        (args.output_dir / "long_context_response2.json").write_text(
            json.dumps(second, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    first_summary = summarize(first, first_elapsed)
    second_summary = summarize(second, second_elapsed)
    first_summary["request_contract_sha256"] = request_contract_sha256
    second_summary["request_contract_sha256"] = request_contract_sha256
    report = {
        "schema": "bi100-long-context-api-result-v2",
        "version": 2,
        "privacy": {
            "contains_raw_request": False,
            "contains_raw_model_output": False,
            "raw_response_files_retained": args.retain_raw_responses,
        },
        "evidence_scope": "legacy-long-context-diagnostic",
        "overall_promotion_authorized": False,
        "run_id_sha256": hashlib.sha256(
            args.run_id.encode("utf-8")).hexdigest(),
        "prompt_construction": prompt_construction,
        "tokenizer": tokenizer_metadata,
        "runtime": {
            "model_path": args.model_path,
            "served_model_name": args.served_model_name,
            "max_model_len": args.max_model_len,
        },
        "target_prompt_tokens": args.target_prompt_tokens,
        "max_tokens": args.max_tokens,
        "min_cached_tokens": args.min_cached_tokens,
        "max_first_cached_tokens": args.max_first_cached_tokens,
        "min_completion_tokens": args.min_completion_tokens,
        "equivalence_mode": args.equivalence_mode,
        "first": first_summary,
        "second": second_summary,
    }
    summary_path = args.output_dir / "long_context_summary.json"
    persist_report(summary_path, report)
    assert_finite(first, "cold_response")
    assert_finite(second, "warm_response_1")

    assert first_summary["prompt_tokens"] == args.target_prompt_tokens, first_summary
    assert second_summary["prompt_tokens"] == args.target_prompt_tokens, second_summary
    assert first_summary["model"] == args.served_model_name, first_summary
    assert second_summary["model"] == args.served_model_name, second_summary
    first_cached_tokens = first_summary["cached_tokens"]
    assert (isinstance(first_cached_tokens, int)
            and not isinstance(first_cached_tokens, bool)), first_summary
    assert 0 <= first_cached_tokens <= args.max_first_cached_tokens, first_summary
    assert second_summary["cached_tokens"] >= args.min_cached_tokens, second_summary
    assert first_summary["completion_tokens"] >= args.min_completion_tokens, first_summary
    assert second_summary["completion_tokens"] >= args.min_completion_tokens, second_summary
    if args.equivalence_mode == "exact":
        assert_equivalent(first, second)
    else:
        third, third_elapsed = post_chat(args.base, payload, args.timeout_s)
        if args.retain_raw_responses:
            (args.output_dir / "long_context_response3.json").write_text(
                json.dumps(third, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        third_summary = summarize(third, third_elapsed)
        third_summary["request_contract_sha256"] = request_contract_sha256
        report["third"] = third_summary
        persist_report(summary_path, report)
        assert_finite(third, "warm_response_2")
        assert third_summary["prompt_tokens"] == args.target_prompt_tokens, third_summary
        assert third_summary["model"] == args.served_model_name, third_summary
        assert third_summary["cached_tokens"] >= args.min_cached_tokens, third_summary
        assert third_summary["completion_tokens"] >= args.min_completion_tokens, third_summary
        assert_equivalent(second, third)

    persist_report(summary_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
