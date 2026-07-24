#!/usr/bin/env python3
"""Deterministically construct exact-length chat prompts without raw artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


Json = dict[str, Any]
Recipe = Callable[[str], tuple[list[Json], list[Json] | None]]
TEMPLATE_KWARG_MODES = ("direct", "nested")
MAX_FILLER_SOURCE_VARIANTS = 16


class PromptConstructionError(RuntimeError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PromptConstructionError(reason)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _as_token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (isinstance(value, list) and len(value) == 1
            and isinstance(value[0], list)):
        value = value[0]
    _require(isinstance(value, list), "tokenizer returned no input_ids")
    _require(all(isinstance(item, int) and not isinstance(item, bool)
                 for item in value), "tokenizer returned invalid input_ids")
    return value


def _template_kwargs(mode: str, thinking: bool) -> Json:
    _require(mode in TEMPLATE_KWARG_MODES,
             f"unsupported chat-template kwargs mode: {mode}")
    if mode == "direct":
        return {"enable_thinking": thinking}
    return {"chat_template_kwargs": {"enable_thinking": thinking}}


def _messages_for_template(messages: list[Json]) -> list[Json]:
    """Mirror vLLM's OpenAI-to-HF tool argument normalization."""
    normalized = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(
                tool_calls, list):
            normalized.append(message)
            continue

        normalized_calls = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                normalized_calls.append(tool_call)
                continue
            normalized_call = dict(tool_call)
            function = normalized_call.get("function")
            if isinstance(function, dict):
                normalized_function = dict(function)
                arguments = normalized_function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as error:
                        raise PromptConstructionError(
                            "assistant tool arguments are not valid JSON"
                        ) from error
                    _require(isinstance(arguments, dict),
                             "assistant tool arguments must decode to an object")
                    normalized_function["arguments"] = arguments
                normalized_call["function"] = normalized_function
            normalized_calls.append(normalized_call)
        normalized_message = dict(message)
        normalized_message["tool_calls"] = normalized_calls
        normalized.append(normalized_message)
    return normalized


def chat_template_token_ids(
    tokenizer: Any,
    messages: list[Json],
    *,
    tools: list[Json] | None = None,
    thinking: bool = False,
    template_kwargs_mode: str = "direct",
) -> list[int]:
    kwargs: Json = {
        "tokenize": True,
        "add_generation_prompt": True,
        **_template_kwargs(template_kwargs_mode, thinking),
    }
    if tools is not None:
        kwargs["tools"] = tools
    try:
        value = tokenizer.apply_chat_template(
            _messages_for_template(messages), **kwargs)
    except (TypeError, ValueError) as error:
        raise PromptConstructionError(
            f"chat-template invocation failed in {template_kwargs_mode} mode"
        ) from error
    return _as_token_ids(value)


def chat_template_token_count(
    tokenizer: Any,
    messages: list[Json],
    *,
    tools: list[Json] | None = None,
    thinking: bool = False,
    template_kwargs_mode: str = "direct",
) -> int:
    return len(chat_template_token_ids(
        tokenizer,
        messages,
        tools=tools,
        thinking=thinking,
        template_kwargs_mode=template_kwargs_mode,
    ))


def _text_token_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        value = tokenizer.encode(text, add_special_tokens=False)
    except AttributeError:
        value = tokenizer(text, add_special_tokens=False)
    return _as_token_ids(value)


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        value = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        value = tokenizer.decode(token_ids, skip_special_tokens=False)
    _require(isinstance(value, str), "tokenizer decode returned no text")
    return value


def _filler_source(seed: int, namespace: str) -> str:
    lines = []
    for index in range(512):
        digest = hashlib.sha256(
            f"{seed}:{namespace}:{index}".encode("ascii")
        ).hexdigest()[:20]
        lines.append(
            f"record {index:04d} module_{index % 37:02d}.py "
            f"digest {digest} invariant stable\n"
        )
    return "".join(lines)


def _variant_namespace(namespace: str, variant: int) -> str:
    _require(0 <= variant < MAX_FILLER_SOURCE_VARIANTS,
             "filler source variant is outside the fixed search range")
    return namespace if variant == 0 else f"{namespace}:v{variant}"


def fit_exact_chat_prompt(
    tokenizer: Any,
    target_tokens: int,
    recipe: Recipe,
    *,
    seed: int,
    namespace: str,
    thinking: bool = False,
    template_kwargs_mode: str = "direct",
) -> tuple[list[Json], list[Json] | None, Json]:
    """Fit a recipe to an exact post-template length and return safe evidence."""
    _require(isinstance(target_tokens, int) and target_tokens > 0,
             "target token count must be a positive integer")
    empty_messages, empty_tools = recipe("")
    fixed_tokens = chat_template_token_count(
        tokenizer,
        empty_messages,
        tools=empty_tools,
        thinking=thinking,
        template_kwargs_mode=template_kwargs_mode,
    )
    _require(fixed_tokens <= target_tokens,
             "fixed request structure exceeds target token count")

    initial_requested = target_tokens - fixed_tokens
    closest_delta: int | None = None
    attempts = 0
    for variant in range(MAX_FILLER_SOURCE_VARIANTS):
        source_namespace = _variant_namespace(namespace, variant)
        source_text = _filler_source(seed, source_namespace)
        source_ids = _text_token_ids(tokenizer, source_text)
        _require(bool(source_ids), "deterministic filler produced no tokens")

        def materialize(requested_ids: int) -> tuple[
                int, list[Json], list[Json] | None, str, list[int]]:
            _require(requested_ids >= 0,
                     "requested filler token count became negative")
            repeats = (
                (requested_ids + len(source_ids) - 1) // len(source_ids)
                if requested_ids else 0)
            ids = (source_ids * repeats)[:requested_ids]
            filler = _decode_token_ids(tokenizer, ids)
            messages, tools = recipe(filler)
            rendered_ids = chat_template_token_ids(
                tokenizer,
                messages,
                tools=tools,
                thinking=thinking,
                template_kwargs_mode=template_kwargs_mode,
            )
            return len(rendered_ids), messages, tools, filler, rendered_ids

        def evaluate(requested_ids: int) -> tuple[
                int, list[Json], list[Json] | None, str, list[int]]:
            nonlocal attempts, closest_delta
            attempts += 1
            value = materialize(requested_ids)
            delta = target_tokens - value[0]
            if closest_delta is None or abs(delta) < abs(closest_delta):
                closest_delta = delta
            return value

        def completed(
            requested_ids: int,
            value: tuple[int, list[Json], list[Json] | None, str, list[int]],
        ) -> tuple[list[Json], list[Json] | None, Json] | None:
            actual, messages, tools, filler, rendered_ids = value
            if actual != target_tokens:
                return None
            evidence = {
                "schema": "bi100-exact-chat-prompt-v1",
                "target_prompt_tokens": target_tokens,
                "local_prompt_tokens": actual,
                "fixed_prompt_tokens": fixed_tokens,
                "filler_token_ids_requested": requested_ids,
                "filler_text_sha256": hashlib.sha256(
                    filler.encode("utf-8")).hexdigest(),
                "filler_source_sha256": hashlib.sha256(
                    source_text.encode("ascii")).hexdigest(),
                "rendered_prompt_token_ids_sha256": _sha256_json(rendered_ids),
                "messages_sha256": _sha256_json(messages),
                "tools_sha256": _sha256_json(tools),
                "thinking": thinking,
                "template_kwargs_mode": template_kwargs_mode,
                "attempts": attempts,
            }
            return messages, tools, evidence

        requested = initial_requested
        variant_closest: tuple[int, int] | None = None
        seen = set()
        for _ in range(12):
            if requested in seen:
                break
            seen.add(requested)
            value = evaluate(requested)
            delta = target_tokens - value[0]
            if variant_closest is None or abs(delta) < abs(variant_closest[1]):
                variant_closest = (requested, delta)
            result = completed(requested, value)
            if result is not None:
                return result
            requested = max(0, requested + delta)

        assert variant_closest is not None
        closest_requested, _ = variant_closest
        for offset in range(1, 17):
            for candidate in (closest_requested - offset,
                              closest_requested + offset):
                if candidate < 0 or candidate in seen:
                    continue
                seen.add(candidate)
                value = evaluate(candidate)
                result = completed(candidate, value)
                if result is not None:
                    return result

    assert closest_delta is not None
    raise PromptConstructionError(
        "exact prompt construction failed: "
        f"target={target_tokens} closest_delta={closest_delta} "
        f"attempts={attempts} variants={MAX_FILLER_SOURCE_VARIANTS}"
    )


def tokenizer_identity(model_path: Path, tokenizer: Any) -> Json:
    candidates = {
        "chat_template.jinja",
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    files = []
    for name in sorted(candidates):
        path = model_path / name
        if not path.is_file():
            continue
        payload = path.read_bytes()
        files.append({
            "name": name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    _require(bool(files), "model path contains no tokenizer/config artifacts")
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        chat_template = ""
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "artifact_set_sha256": _sha256_json(files),
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")).hexdigest(),
        "files": files,
    }
