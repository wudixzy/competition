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


Json = dict[str, Any]
SchemaVersion = 1
SCHEMA = "bi100-cpu-kv-offload-pressure-api-v1"
DEFAULT_TARGET_PROMPT_TOKENS = 65_536
DEFAULT_MAX_TOKENS = 8
DEFAULT_BLOCK_SIZE = 16
DEFAULT_MIN_CANDIDATE_OFFSET = 32
DEFAULT_MAX_CONTROL_CACHED = 16
DEFAULT_TIMEOUT_S = 900.0
DEFAULT_MODE = "candidate"
DEFAULT_SEED = 20260721
MODEL_NAME = "llm"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CPU KV offload pressure validation using the v1 chat API."
        ))
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model-path",
        default="/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
    )
    parser.add_argument(
        "--target-prompt-tokens",
        type=int,
        default=DEFAULT_TARGET_PROMPT_TOKENS,
    )
    parser.add_argument("--pressure-prompt-tokens", type=int, required=True)
    parser.add_argument("--pressure-count", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--run-id", default=str(time.time_ns()))
    parser.add_argument(
        "--mode",
        choices=("control", "candidate"),
        default=DEFAULT_MODE,
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--min-candidate-cached",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-control-cached",
        type=int,
        default=DEFAULT_MAX_CONTROL_CACHED,
    )
    parser.add_argument("--json-out", type=Path, required=True)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.pressure_count <= 0:
        parser.error("--pressure-count must be greater than 0")
    if args.target_prompt_tokens <= 0:
        parser.error("--target-prompt-tokens must be greater than 0")
    if args.pressure_prompt_tokens <= 0:
        parser.error("--pressure-prompt-tokens must be greater than 0")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than 0")
    if args.block_size <= 0:
        parser.error("--block-size must be greater than 0")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be finite and greater than 0")
    if args.max_control_cached < 0:
        parser.error("--max-control-cached must be non-negative")
    if args.target_prompt_tokens + args.max_tokens > 262_144:
        parser.error("--target-prompt-tokens + --max-tokens exceeds 262144")
    if args.pressure_prompt_tokens + args.max_tokens > 262_144:
        parser.error(
            "--pressure-prompt-tokens + --max-tokens exceeds 262144")
    if args.min_candidate_cached is None:
        args.min_candidate_cached = (
            args.target_prompt_tokens - DEFAULT_MIN_CANDIDATE_OFFSET)
    if not 0 <= args.min_candidate_cached <= args.target_prompt_tokens:
        parser.error(
            "--min-candidate-cached must be within the target prompt")
    if args.max_control_cached > args.target_prompt_tokens:
        parser.error("--max-control-cached exceeds the target prompt")
    return args


def _token_count(tokenizer: Any, content: str) -> int:
    return len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        ))


def _append_repeat(
    tokenizer: Any,
    content: str,
    phrase: str,
    target_tokens: int,
) -> tuple[str, int]:
    base_count = _token_count(tokenizer, content)
    if base_count > target_tokens:
        return content, base_count

    # Phase 1: append as many phrase-chunks as possible by binary search.
    one_count = _token_count(tokenizer, content + phrase)
    delta_one = one_count - base_count
    if delta_one <= 0:
        return content, base_count

    max_possible = (target_tokens - base_count) // delta_one
    if max_possible <= 0:
        return content, base_count
    low = 0
    high = max_possible
    while low < high:
        mid = (low + high + 1) // 2
        count = _token_count(tokenizer, content + phrase * mid)
        if count <= target_tokens:
            low = mid
        else:
            high = mid - 1

    if low <= 0:
        return content, base_count
    content = content + phrase * low
    return content, _token_count(tokenizer, content)


def _build_prompt(
    tokenizer: Any,
    run_id: str,
    phase: str,
    target_tokens: int,
    index: int | None = None,
) -> str:
    suffix = "" if index is None else f"-{index:04d}"
    prefix = f"{run_id} {phase}{suffix} "
    core = (
        "This is a CPU KV offload pressure stability request used for test "
        "instrumentation. Keep only the required context and answer "
        "deterministically when requested by the model."
    )
    content = f"{prefix}{core}"
    if _token_count(tokenizer, content) > target_tokens:
        raise ValueError(f"cannot build {phase} prompt with {target_tokens} tokens")

    fillers = (
        " baseline cache probe payload for deterministic warmup and eviction. ",
        " x",
        " a",
        " b",
        " c",
        " y",
        " z",
    )
    current = _token_count(tokenizer, content)
    for phrase in fillers:
        content, current = _append_repeat(tokenizer, content, phrase, target_tokens)
        if current == target_tokens:
            break

    # Final correction pass: greedy adjustments for small residual error.
    guard = 0
    while current < target_tokens and guard < 20_000:
        guard += 1
        progressed = False
        for phrase in fillers:
            next_count = _token_count(tokenizer, content + phrase)
            if next_count > target_tokens:
                continue
            if not math.isfinite(next_count - current):
                continue
            content = content + phrase
            current = next_count
            progressed = True
            break
        if current == target_tokens:
            break
        if not progressed:
            break

    if current != target_tokens:
        raise ValueError(
            f"could not build exactly {target_tokens} prompt tokens for {phase}")
    return content


def make_request_payload(content: str, max_tokens: int) -> Json:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": DEFAULT_SEED,
        "thinking": False,
    }


def post_chat(base: str, payload: Json, timeout_s: float) -> Json:
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def summarize_response(response: Json, elapsed_s: float) -> Json:
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    choice = response["choices"][0]
    message = choice["message"]
    encoded = json.dumps(message, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
        "elapsed_s": float(elapsed_s),
    }


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def request_once(
    base: str,
    content: str,
    max_tokens: int,
    timeout_s: float,
) -> Json:
    payload = make_request_payload(content, max_tokens)
    started = time.monotonic()
    response = post_chat(base, payload, timeout_s)
    return summarize_response(response, time.monotonic() - started)


def _request_record(name: str, status: str, expected_tokens: int,
                   summary: Json | None = None, error: str = "") -> Json:
    record: Json = {
        "name": name,
        "status": status,
        "expected_prompt_tokens": expected_tokens,
    }
    if summary is not None:
        record["summary"] = summary
    if error:
        record["error"] = error
    return record


def persist_report(path: Path, report: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _find_request(
    requests: list[Json],
    name: str,
) -> Json | None:
    for item in requests:
        if item.get("name") == name:
            return item
    return None


def _target_names() -> tuple[str, str, str, str]:
    return (
        "target_cold",
        "target_immediate_warm",
        "target_after_pressure",
        "target_refreshed",
    )


def _validate_finite_summary(
    summary: Json,
    reasons: list[str],
) -> bool:
    required_numeric = ("prompt_tokens", "cached_tokens",
                       "completion_tokens", "elapsed_s")
    for field in required_numeric:
        if not is_finite_number(summary.get(field)):
            reasons.append(f"request summary field {field} is not finite")
            return False
    return True


def evaluate_validation(
    args: argparse.Namespace,
    requests: list[Json],
) -> Json:
    reasons: list[str] = []
    checks: dict[str, bool | int | str | float | None] = {}
    target_names = _target_names()
    target_records = [_find_request(requests, name) for name in target_names]

    for name in (
        *target_names,
        *(f"pressure_cold_{index:04d}"
          for index in range(args.pressure_count)),
    ):
        record = _find_request(requests, name)
        if record is None:
            reasons.append(f"missing request {name}")
            continue
        if record.get("status") != "ok":
            reasons.append(f"{name} request failed: {record.get('error')}")
            continue
        summary = record.get("summary")
        if not isinstance(summary, dict):
            reasons.append(f"{name} summary missing")
            continue
        if summary.get("prompt_tokens") != record.get("expected_prompt_tokens"):
            reasons.append(
                f"{name} prompt token count mismatch: "
                f"{summary.get('prompt_tokens')} != {record.get('expected_prompt_tokens')}"
            )
        _validate_finite_summary(summary, reasons)

    checks["target_prompt_exact"] = all(
        item and isinstance(item.get("summary"), dict)
        and item["summary"].get("prompt_tokens") == item["expected_prompt_tokens"]
        for item in target_records
    )

    cold_record = _find_request(requests, "target_cold")
    cold_summary = (cold_record or {}).get("summary")
    checks["target_cold_cached_tokens"] = (
        cold_summary.get("cached_tokens")
        if isinstance(cold_summary, dict) else None)
    if (isinstance(cold_summary, dict)
            and cold_summary.get("cached_tokens") != 0):
        reasons.append(
            "target_cold must start with exactly zero cached tokens")

    warm_threshold = max(
        0, args.target_prompt_tokens - args.block_size * 2)
    checks["immediate_warm_threshold"] = warm_threshold
    checks["refreshed_threshold"] = warm_threshold

    for name in ("target_immediate_warm", "target_refreshed"):
        record = _find_request(requests, name)
        summary = (record or {}).get("summary")
        if not isinstance(summary, dict):
            reasons.append(f"{name} summary missing")
            continue
        if summary.get("cached_tokens", 0) < warm_threshold:
            reasons.append(
                f"{name} cached_tokens not near full cache: "
                f"{summary.get('cached_tokens')} < {warm_threshold}")
        if summary.get("finish_reason") not in ("stop", "length", None):
            reasons.append(f"{name} has unexpected finish_reason")

    after_pressure = _find_request(requests, "target_after_pressure")
    after_summary = (after_pressure or {}).get("summary")
    if not isinstance(after_summary, dict):
        reasons.append("target_after_pressure summary missing")
    else:
        if args.mode == "control":
            checks["mode"] = "control"
            if after_summary.get("cached_tokens", 0) > args.max_control_cached:
                reasons.append(
                    "target_after_pressure cached_tokens exceeds "
                    f"control threshold {args.max_control_cached}")
        else:
            checks["mode"] = "candidate"
            if after_summary.get("cached_tokens", 0) < args.min_candidate_cached:
                reasons.append(
                    "target_after_pressure cached_tokens below "
                    f"candidate threshold {args.min_candidate_cached}")

    target_summaries = []
    for item in target_records:
        summary = (item or {}).get("summary")
        if not isinstance(summary, dict):
            continue
        target_summaries.append(summary)

    for index, summary in enumerate(target_summaries):
        completion_tokens = summary.get("completion_tokens")
        if (not isinstance(completion_tokens, int)
                or isinstance(completion_tokens, bool)
                or completion_tokens <= 0):
            reasons.append(
                f"target summary at index {index} has no completion tokens")
        if summary.get("finish_reason") not in ("stop", "length"):
            reasons.append(
                f"target summary at index {index} has unexpected finish_reason")

    if len(target_summaries) >= 2:
        first = target_summaries[0]
        for index, summary in enumerate(target_summaries[1:], start=1):
            if summary.get("message_sha256") != first.get("message_sha256"):
                reasons.append(
                    f"target summary mismatch at index {index}: message hash changed")
            if summary.get("finish_reason") != first.get("finish_reason"):
                reasons.append(
                    f"target summary mismatch at index {index}: finish_reason changed")
            if summary.get("completion_tokens") != first.get(
                    "completion_tokens"):
                reasons.append(
                    f"target summary mismatch at index {index}: completion_tokens changed")
        if target_summaries:
            checks["target_completion"] = target_summaries[0].get("completion_tokens")
            checks["target_finish_reason"] = target_summaries[0].get("finish_reason")
            checks["target_message_sha256"] = target_summaries[0].get("message_sha256")
    else:
        reasons.append("target response consistency cannot be verified")

    checks["pressure_count_expected"] = args.pressure_count
    checks["request_count"] = len(requests)
    checks["target_requests_seen"] = sum(
        1
        for item in target_records
        if item is not None and item.get("status") == "ok")

    qualified = len(reasons) == 0
    return {
        "schema": SCHEMA,
        "version": SchemaVersion,
        "checks": checks,
        "qualified": qualified,
        "reasons": reasons,
    }


def run_benchmark(
    args: argparse.Namespace,
    tokenizer: Any | None = None,
    request_fn: Any | None = None,
) -> Json:
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True, local_files_only=True)
    if request_fn is None:
        request_fn = request_once
    params = vars(args).copy()
    if "json_out" in params:
        params["json_out"] = str(params["json_out"])

    report: Json = {
        "schema": SCHEMA,
        "version": SchemaVersion,
        "params": params,
        "requests": [],
        "validation": {
            "qualified": False,
            "checks": {},
            "reasons": [],
        },
        "qualified": False,
    }

    # Persist at startup so progress is visible even when the first request fails.
    persist_report(args.json_out, report)

    def run_and_record(name: str, content: str, expected_tokens: int) -> None:
        try:
            summary = request_fn(
                args.base, content, args.max_tokens, args.timeout_s)
            if not isinstance(summary, dict):
                raise ValueError("summary type is invalid")
            if not _validate_finite_summary(summary, report["validation"]["reasons"]):
                record = _request_record(
                    name,
                    "invalid_summary",
                    expected_tokens,
                    summary,
                    "summary contains non-finite numeric fields",
                )
            elif summary.get("prompt_tokens") != expected_tokens:
                record = _request_record(
                    name,
                    "prompt_mismatch",
                    expected_tokens,
                    summary,
                    f"prompt tokens mismatch expected {expected_tokens}, "
                    f"got {summary.get('prompt_tokens')}",
                )
            else:
                record = _request_record(name, "ok", expected_tokens, summary)
        except Exception as error:  # pragma: no cover - network/IO path in unit tests
            record = _request_record(
                name,
                "error",
                expected_tokens,
                error=str(error),
            )
        report["requests"].append(record)
        validation = evaluate_validation(args, report["requests"])
        report["validation"] = validation
        report["qualified"] = validation["qualified"]
        persist_report(args.json_out, report)

    def record_build_failure(
        name: str,
        expected_tokens: int,
        error: Exception,
    ) -> None:
        report["requests"].append(
            _request_record(
                name, "prompt_build_failed", expected_tokens,
                error=str(error)))
        validation = evaluate_validation(args, report["requests"])
        report["validation"] = validation
        report["qualified"] = validation["qualified"]
        persist_report(args.json_out, report)

    target_cold_content = None
    try:
        target_cold_content = _build_prompt(
            tokenizer,
            args.run_id,
            "target_cold",
            args.target_prompt_tokens,
        )
        run_and_record("target_cold", target_cold_content, args.target_prompt_tokens)
        run_and_record("target_immediate_warm", target_cold_content,
                       args.target_prompt_tokens)
    except Exception as error:
        record_build_failure("target_cold", args.target_prompt_tokens, error)

    for index in range(args.pressure_count):
        try:
            pressure_content = _build_prompt(
                tokenizer,
                args.run_id,
                "pressure_cold",
                args.pressure_prompt_tokens,
                index=index,
            )
            run_and_record(
                f"pressure_cold_{index:04d}",
                pressure_content,
                args.pressure_prompt_tokens,
            )
        except Exception as error:
            record_build_failure(
                f"pressure_cold_{index:04d}",
                args.pressure_prompt_tokens,
                error,
            )

    if target_cold_content:
        run_and_record("target_after_pressure", target_cold_content,
                       args.target_prompt_tokens)
        run_and_record("target_refreshed", target_cold_content,
                       args.target_prompt_tokens)
    else:
        record_build_failure(
            "target_after_pressure", args.target_prompt_tokens,
            RuntimeError("target prompt unavailable"))
        record_build_failure(
            "target_refreshed", args.target_prompt_tokens,
            RuntimeError("target prompt unavailable"))

    return report


def main() -> int:
    args = parse_args()
    report = run_benchmark(args)
    print(json.dumps(
        {"qualified": report["qualified"], "schema": report["schema"]},
        ensure_ascii=False,
        indent=2,
    ))
    if report["qualified"]:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
