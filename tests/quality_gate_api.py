#!/usr/bin/env python3
"""Run the frozen BI100 API/model-quality contract without retaining outputs."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time
from typing import Any, Callable
import urllib.error
import urllib.request
import zlib

import exact_chat_prompt as exact_prompt
import quality_runtime_contract as runtime_contract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "quality/official_metrics_manifest.v1.json"
RESULT_SCHEMA = "bi100-quality-gate-result-v1"
RESULT_VERSION = 1
TIER_RANK = {"quick": 0, "full": 1, "extended": 2}
SEED = 20260724
EXPECTED_SOURCE_SHA256 = (
    "116e7edc617d8f96fc92caa3e75a3ba4692aae7619026896df1eaf69df12feac"
)
EXPECTED_MANIFEST_SHA256 = (
    "fe9b958610d9d0df8f54504d9c149442f145226c03cf76668711d2d38ed51d0e"
)
Json = dict[str, Any]
Handler = Callable[["Client", "RunConfig"], Json]
ALLOWED_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter"}


class CaseFailure(RuntimeError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise CaseFailure(reason)


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_response_schema(data: Json) -> None:
    require(isinstance(data.get("id"), str) and bool(data["id"]),
            "response id is invalid")
    require(data.get("object") == "chat.completion",
            "response object is invalid")
    require(_nonnegative_int(data.get("created")),
            "response created is invalid")
    require(isinstance(data.get("model"), str) and bool(data["model"]),
            "response model is invalid")
    choices = data.get("choices")
    require(isinstance(choices, list) and bool(choices),
            "response choices are missing")
    for expected_index, choice in enumerate(choices):
        require(isinstance(choice, dict), "response choice is invalid")
        require(choice.get("index") == expected_index,
                "response choice index is invalid")
        finish_reason = choice.get("finish_reason")
        require(finish_reason is None
                or finish_reason in ALLOWED_FINISH_REASONS,
                "response finish_reason is invalid")
        message = choice.get("message")
        require(isinstance(message, dict), "response message is missing")
        require(message.get("role") == "assistant",
                "response message role is invalid")
        content = message.get("content")
        require(content is None or isinstance(content, str),
                "response content type is invalid")
        reasoning = message.get("reasoning_content", message.get("reasoning"))
        require(reasoning is None or isinstance(reasoning, str),
                "response reasoning type is invalid")
        tool_calls = message.get("tool_calls")
        require(tool_calls is None or isinstance(tool_calls, list),
                "response tool_calls type is invalid")
        for call in tool_calls or []:
            require(isinstance(call, dict)
                    and isinstance(call.get("id"), str) and bool(call["id"])
                    and call.get("type") == "function"
                    and isinstance(call.get("function"), dict),
                    "response tool call schema is invalid")
        require(content is not None or bool(tool_calls),
                "response message has neither content nor tool calls")
    usage = data.get("usage")
    require(isinstance(usage, dict), "response usage is missing")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require(_nonnegative_int(usage.get(key)), f"usage {key} is invalid")
    require(usage["total_tokens"]
            == usage["prompt_tokens"] + usage["completion_tokens"],
            "usage total_tokens is inconsistent")
    details = usage.get("prompt_tokens_details")
    if details is not None:
        require(isinstance(details, dict),
                "prompt_tokens_details is invalid")
        cached = details.get("cached_tokens", 0)
        require(_nonnegative_int(cached)
                and cached <= usage["prompt_tokens"],
                "cached_tokens is invalid")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _message(data: Json, index: int = 0) -> Json:
    choices = data.get("choices")
    require(isinstance(choices, list) and len(choices) > index,
            "response choices are missing")
    message = choices[index].get("message")
    require(isinstance(message, dict), "response message is missing")
    return message


def _content(data: Json, index: int = 0) -> str:
    content = _message(data, index).get("content")
    require(isinstance(content, str) and bool(content.strip()),
            "response content is empty")
    return content


def _reasoning(message: Json) -> str:
    value = message.get("reasoning_content")
    if value is None:
        value = message.get("reasoning")
    return value if isinstance(value, str) else ""


def _normalized_tool_calls(message: Json) -> list[Json]:
    normalized = []
    calls = message.get("tool_calls") or []
    require(isinstance(calls, list), "tool_calls must be a list")
    for call in calls:
        require(isinstance(call, dict), "tool call must be an object")
        require(isinstance(call.get("id"), str) and bool(call["id"]),
                "tool call id is missing")
        require(call.get("type") == "function",
                "tool call type is invalid")
        function = call.get("function") or {}
        require(isinstance(function, dict), "tool function must be an object")
        name = function.get("name")
        arguments = function.get("arguments")
        require(isinstance(name, str) and bool(name), "tool name is missing")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise CaseFailure("tool arguments are not valid JSON") from error
        require(isinstance(arguments, (dict, list)),
                "tool arguments must decode to JSON object or list")
        normalized.append({"name": name, "arguments": arguments})
    return normalized


def _normalized_response(data: Json) -> Json:
    choices = data.get("choices") or []
    normalized = []
    for choice in choices:
        message = choice.get("message") or {}
        normalized.append({
            "finish_reason": choice.get("finish_reason"),
            "content": message.get("content"),
            "reasoning_content": _reasoning(message),
            "tool_calls": _normalized_tool_calls(message),
        })
    return {"choices": normalized}


def _usage(data: Json) -> Json:
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),
        "completion_tokens": usage.get("completion_tokens"),
    }


def _observation(
    responses: list[tuple[int, Json]],
    semantic_values: list[Any],
    *,
    facts: Json | None = None,
) -> Json:
    usages = [_usage(data) for status, data in responses if status == 200]
    finish_reasons = []
    for status, data in responses:
        if status != 200:
            continue
        for choice in data.get("choices") or []:
            finish_reasons.append(choice.get("finish_reason"))
    return {
        "status_codes": [status for status, _ in responses],
        "finish_reasons": finish_reasons,
        "prompt_tokens": [row["prompt_tokens"] for row in usages],
        "cached_tokens": [row["cached_tokens"] for row in usages],
        "completion_tokens": [row["completion_tokens"] for row in usages],
        "semantic_output_sha256": _sha256_json(semantic_values),
        "facts": facts or {},
    }


def _expect_200(result: tuple[int, Json]) -> Json:
    status, data = result
    require(status == 200, "expected HTTP 200")
    require(isinstance(data, dict), "response JSON must be an object")
    _validate_response_schema(data)
    return data


def _expect_4xx(result: tuple[int, Json]) -> Json:
    status, data = result
    require(400 <= status < 500, "expected HTTP 4xx")
    require(isinstance(data, dict), "4xx response JSON must be an object")
    require(bool(_error_message(data).strip()),
            "4xx response lacks a structured error message")
    return data


def _error_message(data: Json) -> str:
    error = data.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    message = data.get("message")
    return message if isinstance(message, str) else ""


def _expect_4xx_and_health(
    client: "Client",
    config: "RunConfig",
    result: tuple[int, Json],
) -> None:
    _expect_4xx(result)
    client.models(config.model)


def _parse_sse_payload(raw: bytes) -> Json:
    text = raw.decode("utf-8", "strict").replace("\r\n", "\n")
    require(text.endswith("\n\n"), "stream lacks final SSE frame boundary")
    values = []
    for frame in text.split("\n\n"):
        if not frame:
            continue
        data_lines = []
        for line in frame.split("\n"):
            if line.startswith(":"):
                continue
            require(line.startswith("data:"), "stream contains non-data SSE field")
            data_lines.append(line[5:].lstrip(" "))
        if not data_lines:
            continue
        require(len(data_lines) == 1,
                "stream data event must contain exactly one data line")
        values.append(data_lines[0])
    require(bool(values), "stream contains no data events")
    require(values[-1] == "[DONE]", "stream does not end with DONE")
    require(values.count("[DONE]") == 1,
            "stream must contain exactly one DONE event")

    chunks = []
    for value in values[:-1]:
        try:
            chunk = json.loads(value)
        except json.JSONDecodeError as error:
            raise CaseFailure("stream contains invalid JSON") from error
        require(isinstance(chunk, dict), "stream chunk must be an object")
        require(isinstance(chunk.get("id"), str) and bool(chunk["id"]),
                "stream chunk id is invalid")
        require(chunk.get("object") == "chat.completion.chunk",
                "stream chunk object is invalid")
        require(_nonnegative_int(chunk.get("created")),
                "stream chunk created is invalid")
        require(isinstance(chunk.get("model"), str) and bool(chunk["model"]),
                "stream chunk model is invalid")
        choices = chunk.get("choices")
        require(isinstance(choices, list), "stream choices are invalid")
        usage = chunk.get("usage")
        if usage is not None:
            require(isinstance(usage, dict) and not choices,
                    "stream usage chunk must have empty choices")
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                require(_nonnegative_int(usage.get(key)),
                        f"stream usage {key} is invalid")
            require(usage["total_tokens"]
                    == usage["prompt_tokens"] + usage["completion_tokens"],
                    "stream usage total_tokens is inconsistent")
        else:
            require(bool(choices), "non-usage stream chunk has no choices")
            indexes = []
            for choice in choices:
                require(isinstance(choice, dict)
                        and _nonnegative_int(choice.get("index")),
                        "stream choice index is invalid")
                indexes.append(choice["index"])
                require(isinstance(choice.get("delta"), dict),
                        "stream delta is invalid")
                finish = choice.get("finish_reason")
                require(finish is None or finish in ALLOWED_FINISH_REASONS,
                        "stream finish_reason is invalid")
            require(len(indexes) == len(set(indexes)),
                    "stream choice indexes are duplicated")
        chunks.append(chunk)
    require(bool(chunks), "stream contains no JSON chunks")
    usage_indexes = [
        index for index, chunk in enumerate(chunks)
        if chunk.get("usage") is not None
    ]
    require(usage_indexes == [len(chunks) - 1],
            "stream usage must be the final JSON chunk")
    first_choices = chunks[0].get("choices") or []
    require(len(first_choices) == 1
            and first_choices[0].get("index") == 0
            and first_choices[0].get("delta", {}).get("role") == "assistant",
            "stream first chunk must declare the assistant role")
    require(all(
        choice.get("index") == 0
        for chunk in chunks[:-1]
        for choice in chunk.get("choices") or []
    ), "stream quality request must contain exactly choice index zero")
    terminal_reasons = [
        choice.get("finish_reason")
        for chunk in chunks[:-1]
        for choice in chunk.get("choices") or []
        if choice.get("finish_reason") is not None
    ]
    require(len(terminal_reasons) == 1,
            "stream must contain exactly one final finish_reason")

    usage = chunks[-1]["usage"]
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reasons = []
    tools: dict[int, Json] = {}
    for chunk in chunks[:-1]:
        for choice in chunk["choices"]:
            if choice.get("finish_reason") is not None:
                finish_reasons.append(choice["finish_reason"])
            delta = choice["delta"]
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            reasoning = delta.get("reasoning_content", delta.get("reasoning"))
            if isinstance(reasoning, str):
                reasoning_parts.append(reasoning)
            for call in delta.get("tool_calls") or []:
                require(isinstance(call, dict),
                        "stream tool call delta is invalid")
                index = call.get("index", 0)
                require(_nonnegative_int(index),
                        "stream tool call index is invalid")
                item = tools.setdefault(index, {
                    "id": None,
                    "type": None,
                    "name": "",
                    "arguments": "",
                })
                if "id" in call:
                    call_id = call["id"]
                    require(isinstance(call_id, str) and bool(call_id),
                            "stream tool call identity is invalid")
                    require(item["id"] in (None, call_id),
                            "stream tool call identity changed")
                    item["id"] = call_id
                if "type" in call:
                    call_type = call["type"]
                    require(call_type == "function",
                            "stream tool call identity is invalid")
                    require(item["type"] in (None, call_type),
                            "stream tool call identity changed")
                    item["type"] = call_type
                function = call.get("function")
                require(isinstance(function, dict),
                        "stream tool function delta is invalid")
                if isinstance(function.get("name"), str):
                    item["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    item["arguments"] += function["arguments"]
    normalized_tools = []
    for index in sorted(tools):
        item = tools[index]
        require(isinstance(item["id"], str) and bool(item["id"])
                and item["type"] == "function",
                "stream tool call identity is invalid")
        require(bool(item["name"]), "streamed tool name is empty")
        try:
            arguments = json.loads(item["arguments"] or "{}")
        except json.JSONDecodeError as error:
            raise CaseFailure(
                "streamed tool arguments are invalid JSON") from error
        require(isinstance(arguments, dict),
                "streamed tool arguments must be a JSON object")
        normalized_tools.append({"name": item["name"], "arguments": arguments})
    return {
        "chunks": len(chunks),
        "done": 1,
        "usage_blocks": 1,
        "usage": usage,
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "finish_reasons": finish_reasons,
        "tool_calls": normalized_tools,
    }


class Client:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def post(self, payload: Json, *, timeout: float = 300) -> tuple[int, Json]:
        request = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", "replace")
                data = json.loads(raw) if raw else {}
                return response.status, data
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", "replace")
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                data = {"error_type": "non_json_error"}
            return error.code, data
        except (OSError, TimeoutError) as error:
            raise CaseFailure(
                f"transport failure: {type(error).__name__}") from error

    def models(self, expected_model: str = "llm") -> Json:
        request = urllib.request.Request(f"{self.base}/v1/models", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError) as error:
            raise CaseFailure("model-list endpoint is unavailable") from error
        models = data.get("data") if isinstance(data, dict) else None
        require(response.status == 200 and isinstance(models, list)
                and any(isinstance(model, dict)
                        and model.get("id") == expected_model
                        for model in models),
                f"model-list endpoint does not expose {expected_model}")
        return data

    def stream(self, payload: Json, *, timeout: float = 300) -> tuple[int, Json]:
        request = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            error.read()
            return error.code, {}
        except (OSError, TimeoutError) as error:
            raise CaseFailure(
                f"stream transport failure: {type(error).__name__}") from error

        started = time.perf_counter()
        with response:
            status = response.status
            content_type = response.headers.get_content_type()
            require(content_type == "text/event-stream",
                    "stream content type is not text/event-stream")
            raw_parts = []
            event_times = []
            for line in response:
                raw_parts.append(line)
                stripped = line.strip()
                if (stripped.startswith(b"data:")
                        and stripped[5:].strip() != b"[DONE]"):
                    event_times.append(time.perf_counter() - started)
        stream = _parse_sse_payload(b"".join(raw_parts))
        require(len(event_times) == stream["chunks"],
                "stream event accounting differs")
        stream["event_span_s"] = (
            event_times[-1] - event_times[0] if len(event_times) > 1 else 0.0)
        return status, stream


class RunConfig:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model = args.model
        self.max_model_len = args.max_model_len
        self.truncation_tokens = args.truncation_tokens
        self.endpoint_mode = args.endpoint_mode
        self.allow_bare_engine_n2_skip = args.allow_bare_engine_n2_skip


def _base_payload(prompt: str, *, max_tokens: int | None = 32) -> Json:
    payload: Json = {
        "model": "llm",
        "messages": [{"role": "user", "content": prompt}],
        "thinking": False,
        "temperature": 0,
        "seed": SEED,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def _weather_tool() -> list[Json]:
    return [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return weather for one city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }]


def _solid_png_data_url(rgb: tuple[int, int, int]) -> str:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", checksum))

    width = height = 128
    scanline = b"\x00" + bytes(rgb) * width
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    image = (b"\x89PNG\r\n\x1a\n"
             + chunk(b"IHDR", header)
             + chunk(b"IDAT", zlib.compress(scanline * height, level=9))
             + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


def _basic_chat(client: Client, config: RunConfig) -> Json:
    result = client.post(_base_payload("用一句话回答：你好。"))
    data = _expect_200(result)
    _content(data)
    usage = _usage(data)
    require(isinstance(usage["completion_tokens"], int)
            and usage["completion_tokens"] > 0, "completion usage is invalid")
    return _observation([result], [_normalized_response(data)], facts={
        "schema_content_usage_valid": True,
    })


def _streaming_usage(client: Client, config: RunConfig) -> Json:
    expected_lines = [
        "红色", "橙色", "黄色", "绿色", "蓝色",
        "紫色", "黑色", "白色", "灰色", "粉色",
    ]
    payload = _base_payload(
        "严格按给定顺序逐行输出这十个词，不要编号、标点或解释："
        + "、".join(expected_lines),
        max_tokens=64,
    )
    payload.update({
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": False,
        },
    })
    status, stream = client.stream(payload)
    require(status == 200, "expected HTTP 200 stream")
    require(stream["chunks"] >= 5, "stream has fewer than five data chunks")
    require(stream["done"] == 1, "stream must contain exactly one DONE")
    require(stream["usage_blocks"] == 1,
            "stream must contain exactly one usage block")
    completion_tokens = stream["usage"].get("completion_tokens")
    require(isinstance(completion_tokens, int) and completion_tokens > 0,
            "stream completion usage is invalid")
    require(bool(stream["content"]), "stream content is empty")
    actual_lines = [
        line.strip() for line in stream["content"].splitlines() if line.strip()
    ]
    require(actual_lines == expected_lines,
            "streamed color lines differ from the fixed contract")
    require(stream["event_span_s"] >= 0.01,
            "stream data events were not observed incrementally")
    response = {
        "choices": [
            {"finish_reason": reason}
            for reason in stream["finish_reasons"]
        ],
        "usage": stream["usage"],
    }
    semantic_stream = {
        "content": stream["content"],
        "reasoning_content": stream["reasoning_content"],
        "finish_reasons": stream["finish_reasons"],
        "tool_calls": stream["tool_calls"],
    }
    return _observation([(status, response)], [semantic_stream], facts={
        "chunks": stream["chunks"],
        "done": stream["done"],
        "usage_blocks": stream["usage_blocks"],
        "completion_tokens_positive": True,
        "ten_distinct_color_lines_exact": True,
        "incremental_events_observed": True,
    })


def _forced_tool(client: Client, config: RunConfig) -> Json:
    payload = _base_payload("调用 get_weather 查询北京天气。", max_tokens=128)
    payload["tools"] = _weather_tool()
    payload["tool_choice"] = {
        "type": "function", "function": {"name": "get_weather"},
    }
    return _weather_tool_case(client, payload, "named")


def _auto_tool(client: Client, config: RunConfig) -> Json:
    payload = _base_payload("调用 get_weather 查询北京天气。", max_tokens=128)
    payload["tools"] = _weather_tool()
    payload["tool_choice"] = "auto"
    return _weather_tool_case(client, payload, "auto")


def _weather_tool_case(client: Client, payload: Json, mode: str) -> Json:
    result = client.post(payload)
    data = _expect_200(result)
    calls = _normalized_tool_calls(_message(data))
    require(bool(calls), "weather tool call is empty")
    require(calls[0]["name"] == "get_weather", "weather tool name differs")
    arguments = calls[0]["arguments"]
    require(isinstance(arguments, dict), "weather arguments are not an object")
    city = arguments.get("city")
    require(isinstance(city, str) and (
        "北京" in city or "beijing" in city.lower()),
        "weather city argument is incorrect")
    finish = (data.get("choices") or [{}])[0].get("finish_reason")
    require(finish == "tool_calls", "tool response did not finish as tool_calls")
    return _observation([result], [_normalized_response(data)], facts={
        "tool_calls": len(calls),
        "arguments_valid_json": True,
        "argument_semantics_valid": True,
        "finish_reason_tool_calls": True,
        "tool_choice_mode": mode,
    })


def _reasoning_case(client: Client, config: RunConfig) -> Json:
    payload = _base_payload("请逐步计算 17 乘以 19，并在最后给出答案。", max_tokens=256)
    payload["thinking"] = True
    result = client.post(payload)
    data = _expect_200(result)
    message = _message(data)
    text = _reasoning(message) + (message.get("content") or "")
    require(len(text.strip()) >= 20, "reasoning response is too short")
    require("323" in text, "reasoning answer is incorrect")
    return _observation([result], [_normalized_response(data)], facts={
        "answer_rule_passed": True,
        "reasoning_present": bool(_reasoning(message)),
    })


def _image_payload(rgb: tuple[int, int, int], label: str) -> Json:
    payload = _base_payload("unused", max_tokens=96)
    payload["messages"] = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {
                "url": _solid_png_data_url(rgb),
            }},
            {"type": "text", "text": (
                f"质量门禁 {label}：请用不少于二十个汉字描述图片中心的"
                "颜色和整体构成。")},
        ],
    }]
    return payload


def _image_case(
    client: Client,
    config: RunConfig,
    label: str,
    red_rgb: tuple[int, int, int],
    blue_rgb: tuple[int, int, int],
) -> Json:
    red_payload = _image_payload(red_rgb, label)
    blue_payload = _image_payload(blue_rgb, label)
    red_cold = client.post(red_payload, timeout=600)
    red_warm = client.post(red_payload, timeout=600)
    blue = client.post(blue_payload, timeout=600)
    red_cold_data = _expect_200(red_cold)
    red_warm_data = _expect_200(red_warm)
    blue_data = _expect_200(blue)
    red_cold_content = _content(red_cold_data)
    red_warm_content = _content(red_warm_data)
    blue_content = _content(blue_data)
    require(len(red_cold_content.strip()) > 15
            and len(blue_content.strip()) > 15,
            "multimodal content is too short")
    require("红" in red_cold_content and "蓝" not in red_cold_content,
            "red-image identification rule failed")
    require("蓝" in blue_content and "红" not in blue_content,
            "blue-image identification rule failed")
    red_cold_output = _normalized_response(red_cold_data)
    red_warm_output = _normalized_response(red_warm_data)
    blue_output = _normalized_response(blue_data)
    require(red_cold_output == red_warm_output,
            "same-image cold/warm output differs")
    require(red_cold_output != blue_output,
            "different-image output was not isolated")
    require(_usage(red_cold_data)["completion_tokens"]
            == _usage(red_warm_data)["completion_tokens"],
            "same-image completion usage differs")
    red_cached = _usage(red_warm_data)["cached_tokens"]
    blue_cached = _usage(blue_data)["cached_tokens"]
    require(_usage(red_cold_data)["cached_tokens"] == 0,
            "same-image cold request unexpectedly reports cached tokens")
    require(isinstance(red_cached, int) and red_cached > 0,
            "same-image warm request reports no cached tokens")
    require(blue_cached == 0,
            "different-image cache identity was not isolated")
    return _observation(
        [red_cold, red_warm, blue],
        [red_cold_output, red_warm_output, blue_output],
        facts={
            "content_length_gt_15": True,
            "red_identified": True,
            "blue_identified": True,
            "same_image_cold_warm_exact": True,
            "different_image_isolated": True,
            "cold_and_cross_image_cached_tokens_zero": True,
        },
    )


def _prefix_cache(client: Client, config: RunConfig) -> Json:
    prefix = "请记住以下材料：" + ("BI100 缓存正确性材料。" * 900)
    payload = _base_payload(prefix + "\n问题：材料主题是什么？", max_tokens=64)
    first = client.post(payload, timeout=600)
    second = client.post(payload, timeout=600)
    cold = _expect_200(first)
    warm = _expect_200(second)
    cold_output = _normalized_response(cold)
    warm_output = _normalized_response(warm)
    require(cold_output == warm_output, "cold/warm normalized output differs")
    require(_usage(cold)["prompt_tokens"] == _usage(warm)["prompt_tokens"],
            "cold/warm prompt usage differs")
    require(_usage(cold)["completion_tokens"] == _usage(warm)["completion_tokens"],
            "cold/warm completion usage differs")
    cold_cached = _usage(cold)["cached_tokens"]
    cached = _usage(warm)["cached_tokens"]
    require(cold_cached == 0, "cold request unexpectedly reports cached tokens")
    require(isinstance(cached, int) and cached > 0,
            "warm request reports no cached tokens")
    require(cached <= _usage(warm)["prompt_tokens"],
            "warm cached tokens exceed prompt tokens")
    return _observation([first, second], [cold_output, warm_output], facts={
        "cold_warm_exact": True,
        "cold_cached_tokens_zero": True,
        "warm_cached_tokens_positive": True,
    })


def _reasoning_split(client: Client, config: RunConfig) -> Json:
    payload = _base_payload("请思考后回答：6 乘以 7 等于多少？", max_tokens=512)
    payload["thinking"] = True
    result = client.post(payload)
    data = _expect_200(result)
    message = _message(data)
    require(bool(_reasoning(message).strip()), "reasoning_content is empty")
    require(isinstance(message.get("content"), str)
            and bool(message["content"].strip()), "final content is empty")
    require("42" in message["content"], "final reasoning answer is incorrect")
    return _observation([result], [_normalized_response(data)], facts={
        "reasoning_content_nonempty": True,
        "content_nonempty": True,
        "final_answer_rule_passed": True,
    })


def _thinking(client: Client, config: RunConfig, value: Any) -> Json:
    payload = _base_payload("计算 8 加 9，最后只给出答案。", max_tokens=96)
    if value is None:
        payload.pop("thinking", None)
    else:
        payload["thinking"] = value
    result = client.post(payload)
    data = _expect_200(result)
    reasoning = _reasoning(_message(data))
    enabled = value is None or value is True
    require(bool(reasoning.strip()) == enabled,
            "thinking/reasoning behavior differs from contract")
    if not enabled:
        _content(data)
    return _observation([result], [_normalized_response(data)], facts={
        "thinking_enabled": enabled,
        "reasoning_present": bool(reasoning.strip()),
    })


def _thinking_disabled_top_level(client: Client, config: RunConfig) -> Json:
    payload = _base_payload("计算 8 加 9，最后只给出答案。", max_tokens=96)
    payload["thinking"] = {"type": "disabled"}
    first = client.post(payload)
    responses = [first]
    protocol = "top_level"
    if 400 <= first[0] < 500 and config.endpoint_mode == "direct":
        fallback = dict(payload)
        fallback.pop("thinking")
        fallback["chat_template_kwargs"] = {"enable_thinking": False}
        final = client.post(fallback)
        responses.append(final)
        protocol = "direct_chat_template_fallback"
    else:
        final = first
    data = _expect_200(final)
    reasoning = _reasoning(_message(data))
    require(not reasoning.strip(), "disabled thinking returned reasoning")
    _content(data)
    return _observation(
        responses,
        [_normalized_response(data)],
        facts={
            "thinking_enabled": False,
            "reasoning_present": False,
            "request_protocol": protocol,
        },
    )


def _parameter_case(
    client: Client,
    config: RunConfig,
    field: str,
    value: Any,
    *,
    expect_4xx: bool = False,
    accept_2xx_or_4xx: bool = False,
) -> Json:
    payload = _base_payload("只输出字母 A。", max_tokens=8)
    payload[field] = value
    result = client.post(payload)
    status, data = result
    if expect_4xx:
        _expect_4xx_and_health(client, config, result)
    elif accept_2xx_or_4xx:
        require(status == 200 or 400 <= status < 500,
                "expected HTTP 2xx or 4xx")
        if status == 200:
            data = _expect_200(result)
            _content(data)
        else:
            require(config.endpoint_mode == "direct",
                    "gateway must normalize top_p=0 to a successful request")
            _expect_4xx_and_health(client, config, result)
    else:
        data = _expect_200(result)
        _content(data)
    semantic = [_normalized_response(data)] if status == 200 else [{"status": status}]
    facts = {
        "parameter": field,
        "accepted": status == 200,
        "endpoint_mode": config.endpoint_mode,
    }
    if status != 200:
        facts.update({
            "structured_error": True,
            "post_error_health": True,
        })
    return _observation([result], semantic, facts=facts)


def _n_case(client: Client, config: RunConfig, n: int) -> Json:
    payload = _base_payload("只输出字母 A。", max_tokens=8)
    payload["n"] = n
    result = client.post(payload)
    known_limit = "n=2 exceeds max_num_seqs=1" in _error_message(result[1])
    if (n == 2 and result[0] == 400 and known_limit
            and config.endpoint_mode == "direct"
            and config.allow_bare_engine_n2_skip):
        client.models()
        observation = _observation([result], [{"status": result[0]}], facts={
            "n": n,
            "documented_bare_engine_skip": True,
            "normalized_error": "n_exceeds_max_num_seqs",
            "post_skip_health": True,
        })
        observation["_skip_reason"] = "documented bare-engine n=2 limitation"
        return observation
    data = _expect_200(result)
    choices = data.get("choices") or []
    require(len(choices) == n, "number of choices differs from n")
    for index in range(n):
        _content(data, index)
    normalized = _normalized_response(data)
    normalized_choices = normalized["choices"]
    require(all(choice == normalized_choices[0]
                for choice in normalized_choices[1:]),
            "fixed greedy n choices differ")
    usage = data["usage"]
    require(usage["prompt_tokens"] > 0,
            "n response prompt usage is empty")
    require(usage["completion_tokens"] >= n,
            "n response completion usage is undercounted")
    return _observation([result], [normalized], facts={
        "n": n,
        "choice_indices_exact": True,
        "usage_accounted": True,
        "deterministic_choices_exact": True,
        "choice_output_sha256": _sha256_json(normalized_choices[0]),
    })


def _max_tokens_case(
    client: Client,
    config: RunConfig,
    value: int | None,
    *,
    expect_4xx: bool = False,
) -> Json:
    payload = _base_payload("只输出字母 A。", max_tokens=value)
    result = client.post(payload, timeout=600)
    if expect_4xx:
        _expect_4xx_and_health(client, config, result)
        return _observation([result], [{"status": result[0]}], facts={
            "requested_max_tokens": value,
            "rejected_without_5xx": True,
            "structured_error": True,
            "post_error_health": True,
        })
    data = _expect_200(result)
    _content(data)
    finish = (data.get("choices") or [{}])[0].get("finish_reason")
    if value == 1:
        completion = _usage(data)["completion_tokens"]
        require(completion == 1,
                "max_tokens=1 completion usage is invalid")
        require(finish in {"stop", "length"},
                "max_tokens=1 finish_reason is invalid")
        facts = {
            "requested_max_tokens": value,
            "completion_within_limit": True,
            "finish_reason_valid": True,
            "natural_stop": finish == "stop",
            "terminated_by_limit": finish == "length",
        }
    else:
        require(finish == "stop",
                "accepted max_tokens request did not finish by stop")
        facts = {"requested_max_tokens": value, "natural_stop": True}
    return _observation([result], [_normalized_response(data)], facts={
        **facts,
    })


def _exact_echo(
    client: Client,
    config: RunConfig,
    expected: str,
    *,
    system: str | None = None,
) -> Json:
    payload = _base_payload(f"请只输出以下文本，不要添加引号或解释：{expected}", max_tokens=64)
    if system is not None:
        payload["messages"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": "请按系统指令回复。"},
        ]
    result = client.post(payload)
    data = _expect_200(result)
    require(_content(data).strip() == expected, "exact echo differs")
    return _observation([result], [_normalized_response(data)], facts={
        "exact_echo": True,
    })


def _multi_turn(client: Client, config: RunConfig) -> Json:
    payload = _base_payload("unused", max_tokens=64)
    payload["messages"] = [
        {"role": "user", "content": "请记住暗号 ORBIT-731。"},
        {"role": "assistant", "content": "已记住。"},
        {"role": "user", "content": "暗号是什么？只输出暗号。"},
    ]
    result = client.post(payload)
    data = _expect_200(result)
    require(_content(data).strip() == "ORBIT-731", "multi-turn memory failed")
    return _observation([result], [_normalized_response(data)], facts={
        "memory_rule_passed": True,
    })


def _json_object(client: Client, config: RunConfig) -> Json:
    payload = _base_payload('只输出 JSON：{"name":"Alice","age":30}', max_tokens=64)
    payload["response_format"] = {"type": "json_object"}
    result = client.post(payload)
    data = _expect_200(result)
    try:
        parsed = json.loads(_content(data))
    except json.JSONDecodeError as error:
        raise CaseFailure("json_object response is invalid JSON") from error
    require(parsed.get("name") == "Alice" and parsed.get("age") == 30,
            "json_object values differ")
    return _observation([result], [_normalized_response(data)], facts={
        "valid_json": True, "values_match": True,
    })


def _json_schema(client: Client, config: RunConfig) -> Json:
    payload = _base_payload('只输出 Alice 和 30 的 JSON 对象。', max_tokens=64)
    payload["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
                "additionalProperties": False,
            },
        },
    }
    result = client.post(payload)
    data = _expect_200(result)
    try:
        parsed = json.loads(_content(data))
    except json.JSONDecodeError as error:
        raise CaseFailure("json_schema response is invalid JSON") from error
    require(set(parsed) == {"name", "age"}
            and parsed["name"] == "Alice"
            and parsed["age"] == 30,
            "json_schema response violates schema")
    return _observation([result], [_normalized_response(data)], facts={
        "schema_valid": True, "values_match": True,
    })


def _stop_sequence(client: Client, config: RunConfig) -> Json:
    payload = _base_payload("从 1 数到 30，用空格分隔，只输出数字。", max_tokens=128)
    payload["stop"] = ["15"]
    result = client.post(payload)
    data = _expect_200(result)
    content = _content(data)
    require("14" in content, "stop response lacks pre-stop content")
    require("15" not in content and "16" not in content,
            "stop sequence leaked stopped or later content")
    return _observation([result], [_normalized_response(data)], facts={
        "pre_stop_present": True, "post_stop_absent": True,
    })


def _idempotency(client: Client, config: RunConfig) -> Json:
    payload = _base_payload("只输出固定文本 IDEMPOTENT-42。", max_tokens=32)
    payload["seed"] = 42
    first = client.post(payload)
    second = client.post(payload)
    first_data = _expect_200(first)
    second_data = _expect_200(second)
    first_output = _normalized_response(first_data)
    second_output = _normalized_response(second_data)
    require(first_output == second_output, "fixed-seed outputs differ")
    return _observation([first, second], [first_output, second_output], facts={
        "deterministic": True,
    })


def _invalid_payload(client: Client, config: RunConfig, payload: Json) -> Json:
    result = client.post(payload)
    _expect_4xx_and_health(client, config, result)
    return _observation([result], [{"status": result[0]}], facts={
        "rejected_without_5xx": True,
        "structured_error": True,
        "post_error_health": True,
    })


def _truncation(client: Client, config: RunConfig) -> Json:
    require(config.truncation_tokens == 32768,
            "promotion truncation target must remain 32768")
    payload = _base_payload(
        "持续输出字母 A 和空格，直到服务停止。不要解释。",
        max_tokens=config.truncation_tokens,
    )
    payload["min_tokens"] = config.truncation_tokens
    payload["ignore_eos"] = True
    result = client.post(payload, timeout=3600)
    data = _expect_200(result)
    completion = _usage(data)["completion_tokens"]
    require(completion == config.truncation_tokens,
            "exact output truncation token count differs")
    finish = (data.get("choices") or [{}])[0].get("finish_reason")
    require(finish == "length", "exact output truncation finish reason differs")
    return _observation([result], [_normalized_response(data)], facts={
        "exact_completion_tokens": completion,
    })


def _handlers() -> dict[str, Handler]:
    handlers: dict[str, Handler] = {
        "basic_chat": _basic_chat,
        "streaming_usage": _streaming_usage,
        "tool_calling": _auto_tool,
        "reasoning": _reasoning_case,
        "multimodal_input": lambda c, r: _image_case(
            c, r, "multimodal-input", (255, 0, 0), (0, 0, 255)),
        "prefix_cache_hit": _prefix_cache,
        "reasoning_content_split": _reasoning_split,
        "thinking_disabled_top_level": _thinking_disabled_top_level,
        "thinking_true": lambda c, r: _thinking(c, r, True),
        "thinking_false": lambda c, r: _thinking(c, r, False),
        "thinking_default": lambda c, r: _thinking(c, r, None),
        "temperature_0": lambda c, r: _parameter_case(c, r, "temperature", 0.0),
        "temperature_1": lambda c, r: _parameter_case(c, r, "temperature", 1.0),
        "temperature_1_1": lambda c, r: _parameter_case(c, r, "temperature", 1.1),
        "temperature_2": lambda c, r: _parameter_case(c, r, "temperature", 2.0),
        "top_p_0": lambda c, r: _parameter_case(
            c, r, "top_p", 0.0, accept_2xx_or_4xx=True),
        "top_p_0_01": lambda c, r: _parameter_case(c, r, "top_p", 0.01),
        "top_p_0_95": lambda c, r: _parameter_case(c, r, "top_p", 0.95),
        "top_p_1": lambda c, r: _parameter_case(c, r, "top_p", 1.0),
        "top_p_1_1": lambda c, r: _parameter_case(
            c, r, "top_p", 1.1, expect_4xx=True),
        "frequency_penalty_minus_2": lambda c, r: _parameter_case(
            c, r, "frequency_penalty", -2),
        "frequency_penalty_0": lambda c, r: _parameter_case(
            c, r, "frequency_penalty", 0),
        "frequency_penalty_2": lambda c, r: _parameter_case(
            c, r, "frequency_penalty", 2),
        "presence_penalty_minus_2": lambda c, r: _parameter_case(
            c, r, "presence_penalty", -2),
        "presence_penalty_0": lambda c, r: _parameter_case(
            c, r, "presence_penalty", 0),
        "presence_penalty_2": lambda c, r: _parameter_case(
            c, r, "presence_penalty", 2),
        "n_1": lambda c, r: _n_case(c, r, 1),
        "n_2": lambda c, r: _n_case(c, r, 2),
        "max_tokens_unset": lambda c, r: _max_tokens_case(c, r, None),
        "max_tokens_1": lambda c, r: _max_tokens_case(c, r, 1),
        "max_tokens_64": lambda c, r: _max_tokens_case(c, r, 64),
        "max_tokens_64k": lambda c, r: _max_tokens_case(c, r, 65536),
        "max_tokens_near_context": lambda c, r: _max_tokens_case(
            c, r, r.max_model_len - 1024),
        "max_tokens_minus_1": lambda c, r: _max_tokens_case(
            c, r, -1, expect_4xx=True),
        "max_tokens_over_context": lambda c, r: _max_tokens_case(
            c, r, r.max_model_len + 1, expect_4xx=True),
        "no_system_prompt": lambda c, r: _exact_echo(c, r, "NO-SYSTEM-731"),
        "system_prompt_effective": lambda c, r: _exact_echo(
            c, r, "SYSTEM-GATE-9F4A",
            system="无论用户说什么，只输出 SYSTEM-GATE-9F4A。"),
        "function_calling": _forced_tool,
        "multi_turn_memory": _multi_turn,
        "streaming_sse_usage": _streaming_usage,
        "json_object": _json_object,
        "json_schema": _json_schema,
        "stop_sequence": _stop_sequence,
        "chinese": lambda c, r: _exact_echo(c, r, "中文回归测试通过"),
        "japanese": lambda c, r: _exact_echo(c, r, "日本語回帰テスト合格"),
        "emoji": lambda c, r: _exact_echo(c, r, "A😀👩‍💻🇨🇳Z"),
        "base64_png": lambda c, r: _image_case(
            c, r, "base64-png", (192, 0, 0), (0, 0, 192)),
        "empty_request_body": lambda c, r: _invalid_payload(c, r, {}),
        "idempotency": _idempotency,
        "message_missing_role": lambda c, r: _invalid_payload(c, r, {
            "model": "llm", "messages": [{"content": "hello"}],
        }),
        "message_missing_content": lambda c, r: _invalid_payload(c, r, {
            "model": "llm", "messages": [{"role": "user"}],
        }),
        "empty_messages": lambda c, r: _invalid_payload(c, r, {
            "model": "llm", "messages": [],
        }),
        "exact_output_truncation": _truncation,
    }
    return handlers


HANDLERS = _handlers()


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _progress(state: str, completed: int, case: Json | None = None) -> Json:
    value: Json = {
        "state": state,
        "completed_cases": completed,
        "active_ordinal": None,
        "active_id": None,
    }
    if case is not None:
        value["active_ordinal"] = case["ordinal"]
        value["active_id"] = case["id"]
    return value


def _load_manifest(path: Path) -> tuple[Json, str]:
    payload = path.read_bytes()
    payload_sha = hashlib.sha256(payload).hexdigest()
    require(payload_sha == EXPECTED_MANIFEST_SHA256,
            "quality manifest file identity is invalid")
    value = json.loads(payload)
    cases = value.get("cases") or []
    require(value.get("schema") == "bi100-quality-metric-manifest-v1"
            and value.get("version") == 1 and len(cases) == 53,
            "quality manifest is invalid")
    require((value.get("source") or {}).get("sha256")
            == EXPECTED_SOURCE_SHA256,
            "quality manifest source identity is invalid")
    require(value.get("allowed_skips") == {"direct": ["n_2"]},
            "quality manifest skip policy is invalid")
    ids = {case.get("id") for case in cases}
    require(ids == set(HANDLERS), "quality handlers do not match manifest")
    return value, payload_sha


def _selected_cases(manifest: Json, tier: str, requested: list[str]) -> list[Json]:
    cases = manifest["cases"]
    if requested:
        requested_set = set(requested)
        require(len(requested_set) == len(requested), "case ids must be unique")
        known = {case["id"] for case in cases}
        require(requested_set <= known, "unknown quality case requested")
        return [case for case in cases if case["id"] in requested_set]
    rank = TIER_RANK[tier]
    return [case for case in cases if TIER_RANK[case["tier"]] <= rank]


def main() -> int:
    import transformers
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="llm")
    parser.add_argument(
        "--endpoint-mode", choices=("direct", "gateway"), default="direct")
    parser.add_argument("--allow-bare-engine-n2-skip", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tier", choices=tuple(TIER_RANK), default="quick")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--truncation-tokens", type=int, default=32768)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.max_model_len != 262144:
        parser.error("quality gate requires max_model_len=262144")
    if args.model != "llm":
        parser.error("quality gate requires served model name llm")
    if args.truncation_tokens != 32768:
        parser.error("quality gate requires truncation_tokens=32768")
    if args.gpu_count <= 0:
        parser.error("gpu-count must be positive")
    if args.tensor_parallel_size <= 0:
        parser.error("tensor-parallel-size must be positive")
    if not runtime_contract.is_git_revision(args.source_revision):
        parser.error("source-revision must be a fixed Git object id")
    if args.allow_bare_engine_n2_skip and args.endpoint_mode != "direct":
        parser.error("n=2 skip is only valid for direct bare-engine runs")

    manifest, manifest_sha = _load_manifest(args.manifest)
    selected = _selected_cases(manifest, args.tier, args.case)
    expected_runtime = {
        "source_revision": args.source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "gpu_count": args.gpu_count,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "served_model_name": args.model,
    }
    try:
        run_contract, run_contract_sha = runtime_contract.load_runtime_contract(
            args.runtime_contract,
            expected_runtime,
            require_cache_trace=True,
        )
    except runtime_contract.RuntimeContractError as error:
        parser.error(str(error))
    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.is_dir():
        parser.error("tokenizer-path must be a local directory")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True, local_files_only=True)
    tokenizer_metadata = exact_prompt.tokenizer_identity(
        tokenizer_path, tokenizer)
    client = Client(args.base)
    config = RunConfig(args)
    results = []
    report: Json = {
        "schema": RESULT_SCHEMA,
        "version": RESULT_VERSION,
        "qualified": False,
        "quality_run_eligible_for_baseline": False,
        "promotion_authorized": False,
        "label": args.label,
        "run_id_sha256": hashlib.sha256(
            args.run_id.encode("utf-8")).hexdigest(),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": {
            "path_name": args.manifest.name,
            "sha256": manifest_sha,
            "source_sha256": manifest["source"]["sha256"],
            "total_cases": len(manifest["cases"]),
        },
        "runtime": {
            "source_revision": args.source_revision,
            "runtime_identity": args.runtime_identity,
            "runtime_overlay_sha256": run_contract[
                "runtime_overlay_sha256"],
            "service_command_sha256": runtime_contract.sha256_json(
                run_contract["command"]),
            "service_env_sha256": runtime_contract.sha256_json(
                run_contract["environment"]),
            "instance": args.instance,
            "gpu_count": args.gpu_count,
            "tensor_parallel_size": args.tensor_parallel_size,
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path,
            "max_model_len": args.max_model_len,
            "model": args.model,
            "endpoint_mode": args.endpoint_mode,
            "allow_bare_engine_n2_skip": args.allow_bare_engine_n2_skip,
            "cache_trace_v4_attested": run_contract[
                "cache_trace_enabled"],
        },
        "runtime_contract": {
            "sha256": run_contract_sha,
            "contract": run_contract,
        },
        "generator": {
            "runner_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "transformers_version": transformers.__version__,
        },
        "tokenizer": tokenizer_metadata,
        "selection": {
            "tier": args.tier,
            "explicit_cases": args.case,
            "selected_cases": len(selected),
            "allowed_skip_ids": (
                ["n_2"] if args.allow_bare_engine_n2_skip else []),
            "promotion_requires": "extended tier, all 53 cases, baseline comparison",
        },
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_credentials": False,
        },
        "summary": {},
        "group_summary": {},
        "progress": _progress("initialized", 0),
        "cases": results,
    }
    started = time.perf_counter()
    try:
        client.models()
    except CaseFailure as error:
        report["summary"] = {
            "passed": 0,
            "failed": len(selected),
            "total": len(selected),
            "wall_s": time.perf_counter() - started,
            "startup_error": str(error),
        }
        report["progress"] = _progress("startup_failed", 0)
        _atomic_write(args.out, report)
        return 1

    for case in selected:
        case_started = time.perf_counter()
        report["progress"] = _progress("running", len(results), case)
        _atomic_write(args.out, report)
        print(f"[RUN] {case['ordinal']:02d} {case['id']}", flush=True)
        try:
            observation = HANDLERS[case["id"]](client, config)
            skip_reason = observation.pop("_skip_reason", "")
            ok = True
            error_code = ""
            case_status = "skip" if skip_reason else "pass"
        except CaseFailure as error:
            observation = {
                "status_codes": [],
                "finish_reasons": [],
                "prompt_tokens": [],
                "cached_tokens": [],
                "completion_tokens": [],
                "semantic_output_sha256": None,
                "facts": {},
            }
            ok = False
            error_code = str(error)
            skip_reason = ""
            case_status = "fail"
        except Exception as error:
            observation = {
                "status_codes": [],
                "finish_reasons": [],
                "prompt_tokens": [],
                "cached_tokens": [],
                "completion_tokens": [],
                "semantic_output_sha256": None,
                "facts": {},
            }
            ok = False
            error_code = f"unexpected {type(error).__name__}"
            skip_reason = ""
            case_status = "fail"
        results.append({
            **case,
            "ok": ok,
            "status": case_status,
            "skip_reason": skip_reason,
            "elapsed_s": time.perf_counter() - case_started,
            "error_code": error_code,
            "observation": observation,
        })
        report["progress"] = _progress("between_cases", len(results))
        _atomic_write(args.out, report)
        print(f"[{case_status.upper()}] {case['ordinal']:02d} "
              f"{case['id']}", flush=True)
        if not ok and args.fail_fast:
            break

    passed = sum(case["status"] == "pass" for case in results)
    skipped = sum(case["status"] == "skip" for case in results)
    failed = sum(case["status"] == "fail" for case in results)
    groups: dict[str, Json] = {}
    for case in results:
        group = groups.setdefault(case["group"], {
            "passed": 0, "skipped": 0, "failed": 0, "total": 0,
        })
        group["total"] += 1
        status_field = {
            "pass": "passed", "skip": "skipped", "fail": "failed",
        }[case["status"]]
        group[status_field] += 1
    for group in groups.values():
        group["pass_rate"] = (
            group["passed"] + group["skipped"]) / group["total"]
    complete = len(results) == len(selected)
    qualified = complete and failed == 0 and passed + skipped == len(selected)
    report["qualified"] = qualified
    report["quality_run_eligible_for_baseline"] = bool(
        qualified
        and args.tier == "extended"
        and not args.case
        and len(results) == 53
        and args.gpu_count == 4
        and args.tensor_parallel_size == 4
    )
    report["summary"] = {
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "total": len(results),
        "selected_total": len(selected),
        "complete": complete,
        "pass_rate": (
            (passed + skipped) / len(results) if results else 0.0),
        "wall_s": time.perf_counter() - started,
    }
    report["group_summary"] = groups
    report["progress"] = _progress("complete", len(results))
    # A separately qualified CoreX baseline comparison is required.
    report["promotion_authorized"] = False
    _atomic_write(args.out, report)
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
