#!/usr/bin/env python3
"""Collect private fixed-position prompt top-logprobs from a TP4 service."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

import exact_chat_prompt
from long_context_api import build_exact_prompt
import quality_runtime_contract as runtime_contract


SCHEMA = "bi100-teacher-forced-topk-observation-v1"
VERSION = 1
TARGETS = (4096, 32768, 65536, 131072, 235000)
TOP_K = 5
POSITIONS_PER_CASE = 64
SEED = 20260729
Json = dict[str, Any]
RUNTIME_MANIFEST_V2_FIELDS = {
    "schema", "version", "source_revision", "runtime_identity", "instance",
    "model_path", "tokenizer_path", "gpu_count", "tensor_parallel_size",
    "max_model_len", "served_model_name", "command", "environment",
}


def parse_targets(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in raw.split(","))
    except ValueError as exc:
        raise ValueError("teacher-forced targets must be integers") from exc
    if (
        not values
        or values != tuple(sorted(set(values)))
        or any(value <= POSITIONS_PER_CASE + 1 or value > 262143
               for value in values)
    ):
        raise ValueError(
            "teacher-forced targets must be unique increasing valid lengths")
    return values


def load_runtime_manifest_v2(path: Path, expected: Json) -> Json:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != RUNTIME_MANIFEST_V2_FIELDS:
        raise ValueError("runtime manifest v2 fields are invalid")
    if (value.get("schema") != "bi100-quality-runtime-manifest-v2"
            or value.get("version") != 2):
        raise ValueError("runtime manifest schema/version is invalid")
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(f"runtime manifest {name} differs from the run")
    command = value.get("command")
    environment = value.get("environment")
    if (not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
            or not isinstance(environment, dict)
            or not all(isinstance(name, str) and name
                       and isinstance(item, str)
                       for name, item in environment.items())):
        raise ValueError("runtime command/environment is malformed")
    blocked = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
    if any(fragment in name.upper()
           for name in environment for fragment in blocked):
        raise ValueError("runtime manifest contains a secret-bearing name")
    selector = environment.get("BI100_ATTN_COREX_FUSED_PREFILL")
    if selector not in {"0", "1"}:
        raise ValueError("runtime manifest fused-prefill selector is invalid")
    if environment.get("BI100_CACHE_TRACE") != "1":
        raise ValueError("runtime manifest must enable cache trace")
    required_optimization = {
        "BI100_GDN_CACHE_POLICY", "BI100_GDN_RESTORE_MODE",
        "BI100_KV_EVICTION_POLICY",
    }
    if not required_optimization.issubset(environment):
        raise ValueError("runtime manifest optimization identity is incomplete")
    return value


def _atomic_write(path: Path, value: Json, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2,
                      sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _spread_values(values: list[int], count: int) -> list[int]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]
    return [
        values[round(index * (len(values) - 1) / (count - 1))]
        for index in range(count)
    ]


def sample_positions(token_count: int, count: int) -> list[int]:
    if token_count <= count + 1:
        raise ValueError("prompt is too short for the fixed position sample")
    uniform = sorted({
        round(1 + index * (token_count - 2) / 31)
        for index in range(32)
    })
    boundary_candidates = sorted({
        position
        for boundary in range(8192, token_count, 8192)
        for position in (boundary - 1, boundary, boundary + 1)
        if 1 <= position < token_count
    })
    selected = set(_spread_values(boundary_candidates, 32))
    selected.update(uniform)
    fill_candidates = sorted({
        round(1 + index * (token_count - 2) / (count * 4 - 1))
        for index in range(count * 4)
    })
    for position in fill_candidates:
        if len(selected) >= count:
            break
        selected.add(position)
    if len(selected) < count:
        for position in range(token_count - 1, 0, -1):
            selected.add(position)
            if len(selected) >= count:
                break
    if len(selected) > count:
        boundary = _spread_values(sorted(
            selected & set(boundary_candidates)), min(32, count))
        remainder = [
            position for position in sorted(selected)
            if position not in set(boundary)
        ]
        selected = set(boundary)
        selected.update(_spread_values(remainder, count - len(selected)))
    result = sorted(selected)
    if len(result) != count:
        raise ValueError("fixed position sampler produced the wrong size")
    return result


def _token_key(key: bytes, token_id: int) -> str:
    return hmac.new(
        key, str(token_id).encode("ascii"), hashlib.sha256).hexdigest()


def _post(base: str, route: str, payload: Json, timeout_s: float) -> Json:
    request = urllib.request.Request(
        f"{base.rstrip('/')}/{route.lstrip('/')}",
        data=json.dumps(payload, ensure_ascii=True).encode("ascii"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            if response.status != 200:
                raise RuntimeError(f"teacher-forced HTTP status {response.status}")
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        error.read()
        raise RuntimeError(
            f"teacher-forced HTTP status {error.code}") from error


def _response_token_ids(response: Any, expected_prompt_tokens: int) -> list[int]:
    if not isinstance(response, dict) or set(response) != {
        "count", "max_model_len", "tokens"
    }:
        raise ValueError("tokenize response contract differs")
    tokens = response.get("tokens")
    if (
        response.get("count") != expected_prompt_tokens
        or response.get("max_model_len") != 262144
        or not isinstance(tokens, list)
        or len(tokens) != expected_prompt_tokens
        or not all(
            isinstance(token_id, int) and not isinstance(token_id, bool)
            for token_id in tokens
        )
    ):
        raise ValueError("server-rendered prompt token sequence is invalid")
    return tokens


def _parse_logprob(value: Any) -> float:
    if not isinstance(value, dict):
        raise ValueError("prompt logprob value is not an object")
    logprob = value.get("logprob")
    if (
        not isinstance(logprob, (int, float))
        or isinstance(logprob, bool)
        or not math.isfinite(logprob)
    ):
        raise ValueError("prompt logprob is not finite")
    return float(logprob)


def summarize_position(
    raw: Any,
    *,
    position: int,
    actual_token_id: int,
    identity_key: bytes,
    top_k: int,
) -> Json:
    if not isinstance(raw, dict) or len(raw) < 2:
        raise ValueError("sampled prompt logprobs are missing")
    values = []
    for raw_token_id, metadata in raw.items():
        try:
            token_id = int(raw_token_id)
        except (TypeError, ValueError) as error:
            raise ValueError("prompt logprob token id is invalid") from error
        values.append((token_id, _parse_logprob(metadata)))
    by_id = dict(values)
    if actual_token_id not in by_id:
        raise ValueError("teacher token is absent from prompt logprobs")
    ordered = sorted(values, key=lambda item: (-item[1], item[0]))
    retained = ordered[:top_k]
    if actual_token_id not in {token_id for token_id, _ in retained}:
        retained.append((actual_token_id, by_id[actual_token_id]))
        retained.sort(key=lambda item: (-item[1], item[0]))
    return {
        "position": position,
        "actual_token_key": _token_key(identity_key, actual_token_id),
        "top_logprobs": [
            {
                "token_key": _token_key(identity_key, token_id),
                "logprob": logprob,
            }
            for token_id, logprob in retained
        ],
    }


def _response_prompt_logprobs(
    response: Any,
    *,
    expected_prompt_tokens: int,
) -> list[Any]:
    if not isinstance(response, dict):
        raise ValueError("response root is not an object")
    choices = response.get("choices")
    usage = response.get("usage")
    prompt_logprobs = response.get("prompt_logprobs")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(usage, dict)
        or usage.get("prompt_tokens") != expected_prompt_tokens
        or usage.get("completion_tokens") != 1
        or not isinstance(prompt_logprobs, list)
        or len(prompt_logprobs) != expected_prompt_tokens
    ):
        raise ValueError("teacher-forced response contract differs")
    details = usage.get("prompt_tokens_details") or {}
    if details.get("cached_tokens", 0) != 0:
        raise ValueError("teacher-forced request was not cold")
    return prompt_logprobs


def collect_case(
    *,
    base: str,
    tokenizer: Any,
    target_tokens: int,
    served_model_name: str,
    timeout_s: float,
    identity_key: bytes,
) -> Json:
    run_id = f"m1-132-teacher-forced-{target_tokens}-v1"
    content = build_exact_prompt(tokenizer, target_tokens, run_id)
    messages = [{"role": "user", "content": content}]
    local_token_ids = exact_chat_prompt.chat_template_token_ids(
        tokenizer,
        messages,
        thinking=False,
        template_kwargs_mode="direct",
    )
    if len(local_token_ids) != target_tokens:
        raise ValueError("locally rendered prompt token count differs")
    template_kwargs = {"enable_thinking": False}
    server_token_ids = _response_token_ids(
        _post(
            base,
            "/tokenize",
            {
                "model": served_model_name,
                "messages": messages,
                "add_generation_prompt": True,
                "continue_final_message": False,
                "add_special_tokens": False,
                "chat_template_kwargs": template_kwargs,
            },
            timeout_s,
        ),
        target_tokens,
    )
    if server_token_ids != local_token_ids:
        raise ValueError("local and server prompt token identities differ")
    sampled = sample_positions(target_tokens, POSITIONS_PER_CASE)
    payload = {
        "model": served_model_name,
        "messages": messages,
        "max_tokens": 1,
        "min_tokens": 1,
        "temperature": 0,
        "seed": SEED,
        "chat_template_kwargs": template_kwargs,
        "stream": False,
        "prompt_logprobs": TOP_K,
        "bi100_prompt_logprobs_sample_positions": sampled,
    }
    response = _post(base, "/v1/chat/completions", payload, timeout_s)
    prompt_logprobs = _response_prompt_logprobs(
        response, expected_prompt_tokens=target_tokens)
    return {
        "id": f"length_{target_tokens}",
        "prompt_tokens": target_tokens,
        "positions": [
            summarize_position(
                prompt_logprobs[position],
                position=position,
                actual_token_id=server_token_ids[position],
                identity_key=identity_key,
                top_k=TOP_K,
            )
            for position in sampled
        ],
    }


def main() -> int:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--served-model-name", default="llm")
    runtime_group = parser.add_mutually_exclusive_group(required=True)
    runtime_group.add_argument("--runtime-contract", type=Path)
    runtime_group.add_argument("--runtime-manifest-v2", type=Path)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--mode", choices=("control", "candidate"),
                        required=True)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument(
        "--targets", default=",".join(map(str, TARGETS)))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not runtime_contract.is_git_revision(args.source_revision):
        parser.error("--source-revision must be a fixed Git object id")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be finite and positive")
    try:
        targets = parse_targets(args.targets)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.out.resolve()
    root = Path(__file__).resolve().parents[1]
    if output == root or output.is_relative_to(root):
        parser.error("private teacher-forced output must stay outside the repo")

    identity_key_hex = os.environ.pop(
        "BI100_TEACHER_FORCED_HMAC_KEY", "")
    if (
        len(identity_key_hex) != 64
        or any(character not in "0123456789abcdef"
               for character in identity_key_hex)
    ):
        parser.error("BI100_TEACHER_FORCED_HMAC_KEY is missing or invalid")
    identity_key = bytes.fromhex(identity_key_hex)

    model_path = str(args.model_path.resolve(strict=True))
    expected_contract = {
        "source_revision": args.source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "model_path": model_path,
        "tokenizer_path": model_path,
        "served_model_name": args.served_model_name,
    }
    if args.runtime_contract is not None:
        contract, _ = runtime_contract.load_runtime_contract(
            args.runtime_contract,
            expected_contract,
            require_cache_trace=True,
        )
    else:
        try:
            contract = load_runtime_manifest_v2(
                args.runtime_manifest_v2, expected_contract)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            parser.error(str(exc))
    environment = contract["environment"]
    expected_selector = "0" if args.mode == "control" else "1"
    if environment.get("BI100_ATTN_COREX_FUSED_PREFILL") != expected_selector:
        parser.error("runtime fused-prefill selector differs")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)
    cases = []
    for target_tokens in targets:
        started = time.monotonic()
        case = collect_case(
            base=args.base,
            tokenizer=tokenizer,
            target_tokens=target_tokens,
            served_model_name=args.served_model_name,
            timeout_s=args.timeout_s,
            identity_key=identity_key,
        )
        case["elapsed_s"] = time.monotonic() - started
        cases.append(case)
    # Elapsed time is useful operationally but is not part of the private
    # numerical observation consumed by the strict comparator.
    for case in cases:
        case.pop("elapsed_s", None)

    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "mode": args.mode,
        "source_revision": args.source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "model_path": model_path,
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "top_k": TOP_K,
        "optimization": {
            "fused_prefill": expected_selector,
            "gdn_cache_policy": environment["BI100_GDN_CACHE_POLICY"],
            "gdn_restore_mode": environment["BI100_GDN_RESTORE_MODE"],
            "kv_eviction_policy": environment["BI100_KV_EVICTION_POLICY"],
        },
        "cases": cases,
        "privacy": {
            "contains_private_hmac_token_keys": True,
            "must_remain_outside_repository": True,
            "contains_raw_prompts": False,
            "contains_raw_model_outputs": False,
            "contains_raw_token_ids": False,
            "contains_credentials": False,
        },
    }
    _atomic_write(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
