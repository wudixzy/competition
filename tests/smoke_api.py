#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request
from typing import Any, Callable


Json = dict[str, Any]


def _request_json(method: str,
                  url: str,
                  payload: Json | None = None,
                  *,
                  timeout: float = 180) -> tuple[int, Json]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            data = {"raw": raw}
        return exc.code, data


def post_chat(base: str,
              payload: Json,
              *,
              expect: int = 200,
              timeout: float = 180) -> Json:
    status, data = _request_json(
        "POST", f"{base.rstrip('/')}/v1/chat/completions", payload, timeout=timeout)
    assert status == expect, (status, json.dumps(data, ensure_ascii=False)[:1000])
    return data


def get_models(base: str) -> None:
    status, data = _request_json("GET", f"{base.rstrip('/')}/v1/models", timeout=30)
    assert status == 200, (status, data)
    assert data.get("data"), data


def _message(data: Json) -> Json:
    return data["choices"][0]["message"]


def _assert_content(data: Json) -> str:
    content = _message(data).get("content")
    assert isinstance(content, str) and content.strip(), data
    return content


def _assert_no_reasoning(data: Json) -> None:
    msg = _message(data)
    assert not msg.get("reasoning_content"), data


def _weather_tool() -> list[Json]:
    return [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]


def test_basic_chat(base: str) -> None:
    data = post_chat(base, {
        "model": "llm",
        "messages": [{"role": "user", "content": "用一句话回答：你好"}],
        "max_tokens": 32,
        "thinking": False,
        "temperature": 0,
    })
    _assert_content(data)
    assert data["usage"]["completion_tokens"] > 0, data


def test_thinking_disabled_variants(base: str) -> None:
    for thinking in ({"type": "disabled"}, False, "disabled"):
        data = post_chat(base, {
            "model": "llm",
            "messages": [{"role": "user", "content": "直接回答：1+1=?"}],
            "max_tokens": 32,
            "thinking": thinking,
            "temperature": 0,
        })
        _assert_content(data)
        _assert_no_reasoning(data)


def test_tool_choice_none(base: str) -> None:
    data = post_chat(base, {
        "model": "llm",
        "messages": [{"role": "user", "content": "不要调用工具，直接回答：北京是中国的首都吗？"}],
        "tools": _weather_tool(),
        "tool_choice": "none",
        "max_tokens": 32,
        "thinking": False,
        "temperature": 0,
    })
    _assert_content(data)
    assert not _message(data).get("tool_calls"), data


def test_response_format_json_object(base: str) -> None:
    data = post_chat(base, {
        "model": "llm",
        "messages": [{
            "role": "user",
            "content": '只输出 JSON：{"answer": 4}',
        }],
        "response_format": {"type": "json_object"},
        "max_tokens": 64,
        "thinking": False,
        "temperature": 0,
    })
    parsed = json.loads(_assert_content(data))
    assert isinstance(parsed, dict), data


def test_response_format_json_schema(base: str) -> None:
    data = post_chat(base, {
        "model": "llm",
        "messages": [{
            "role": "user",
            "content": '只输出 JSON：{"answer": 4}',
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "integer"},
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
        "max_tokens": 64,
        "thinking": False,
        "temperature": 0,
    })
    parsed = json.loads(_assert_content(data))
    assert set(parsed) == {"answer"} and isinstance(parsed["answer"], int), data


def test_streaming_sse(base: str) -> None:
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "直接回答：2+3=?"}],
        "max_tokens": 32,
        "thinking": False,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    saw_done = False
    saw_delta = False
    saw_usage = False
    with urllib.request.urlopen(req, timeout=180) as resp:
        assert resp.status == 200, resp.status
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if value == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(value)
            if chunk.get("usage"):
                saw_usage = True
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                saw_delta = True
    assert saw_delta, "stream produced no content delta"
    assert saw_usage, "stream produced no usage chunk"
    assert saw_done, "stream did not end with [DONE]"


def test_bad_request_4xx(base: str) -> None:
    post_chat(base, {"model": "llm", "messages": []}, expect=400)


def test_prefix_cache(base: str) -> None:
    # Long enough to create an exact 8192-token GDN checkpoint before the
    # unaligned final chunk. Cached replay must be token-identical.
    prefix = "请记住以下材料：" + ("信创模盒 BI100 prefix cache 测试材料。" * 620)
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": prefix + "\n问题：材料主题是什么？"}],
        "max_tokens": 32,
        "thinking": False,
        "temperature": 0,
        "seed": 123,
    }
    uncached = post_chat(base, payload, timeout=360)
    cached = post_chat(base, payload, timeout=360)
    details = cached.get("usage", {}).get("prompt_tokens_details") or {}
    assert details.get("cached_tokens", 0) >= 8192, cached
    assert _message(cached) == _message(uncached), (uncached, cached)
    assert cached["choices"][0].get("finish_reason") == (
        uncached["choices"][0].get("finish_reason")), (uncached, cached)
    assert cached["usage"].get("completion_tokens") == (
        uncached["usage"].get("completion_tokens")), (uncached, cached)


def test_stop(base: str) -> None:
    data = post_chat(base, {
        "model": "llm",
        "messages": [{"role": "user", "content": "输出字符串 alpha END beta，不要解释。"}],
        "stop": ["END"],
        "max_tokens": 32,
        "thinking": False,
        "temperature": 0,
    })
    assert "END" not in (_message(data).get("content") or ""), data


def test_tool_call_forced(base: str) -> None:
    data = post_chat(base, {
        "model": "llm",
        "messages": [{"role": "user", "content": "调用 get_weather 查询北京天气。"}],
        "tools": _weather_tool(),
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_weather"},
        },
        "max_tokens": 128,
        "temperature": 0,
    }, timeout=240)
    calls = _message(data).get("tool_calls") or []
    assert calls and calls[0]["function"]["name"] == "get_weather", data
    json.loads(calls[0]["function"].get("arguments") or "{}")


def test_tool_call_forced_streaming(base: str) -> None:
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "调用 get_weather 查询北京天气。"}],
        "tools": _weather_tool(),
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_weather"},
        },
        "max_tokens": 128,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    saw_done = False
    saw_usage = False
    saw_tool_delta = False
    tool_name = None
    arguments = ""
    with urllib.request.urlopen(req, timeout=240) as resp:
        assert resp.status == 200, resp.status
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if value == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(value)
            if chunk.get("usage"):
                saw_usage = True
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            for call in delta.get("tool_calls") or []:
                function = call.get("function") or {}
                if function.get("name"):
                    tool_name = function["name"]
                if function.get("arguments"):
                    arguments += function["arguments"]
                saw_tool_delta = True
    assert saw_tool_delta, "stream produced no tool_call delta"
    assert tool_name == "get_weather", tool_name
    json.loads(arguments or "{}")
    assert saw_usage, "stream produced no usage chunk"
    assert saw_done, "stream did not end with [DONE]"


def test_multilingual_multiturn(base: str) -> None:
    data = post_chat(base, {
        "model": "llm",
        "messages": [
            {"role": "system", "content": "回答要简短。"},
            {"role": "user", "content": "记住这些字符：你好，こんにちは，😀。"},
            {"role": "assistant", "content": "已记住。"},
            {"role": "user", "content": "用一句话确认你看到了这些字符。"},
        ],
        "max_tokens": 64,
        "thinking": False,
        "temperature": 0,
    })
    content = _assert_content(data)
    assert "\ufffd" not in content, data


def test_sampling_boundaries(base: str) -> None:
    data = post_chat(base, {
        "model": "llm",
        "messages": [{"role": "user", "content": "直接回答：A"}],
        "max_tokens": 1,
        "thinking": False,
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    })
    assert data["usage"]["completion_tokens"] <= 1, data

    post_chat(base, {
        "model": "llm",
        "messages": [{"role": "user", "content": "bad temperature"}],
        "max_tokens": 1,
        "temperature": -0.1,
    }, expect=400)


def test_seed_determinism(base: str) -> None:
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "直接回答一个短词：blue"}],
        "max_tokens": 8,
        "thinking": False,
        "temperature": 0,
        "seed": 42,
    }
    first = _assert_content(post_chat(base, payload))
    second = _assert_content(post_chat(base, payload))
    assert first == second, (first, second)


QUICK_TESTS: list[Callable[[str], None]] = [
    test_basic_chat,
    test_thinking_disabled_variants,
    test_tool_choice_none,
    test_response_format_json_object,
    test_response_format_json_schema,
    test_streaming_sse,
    test_bad_request_4xx,
    test_prefix_cache,
]

FULL_ONLY_TESTS: list[Callable[[str], None]] = [
    test_stop,
    test_tool_call_forced,
    test_tool_call_forced_streaming,
    test_multilingual_multiturn,
    test_sampling_boundaries,
    test_seed_determinism,
]


def run_smoke_tests(base: str,
                    tests: list[Callable[[str], None]],
                    *,
                    mode: str,
                    json_out: str = "") -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "mode": mode,
        "base": base,
        "ok": False,
        "tests": results,
    }

    def write_report() -> None:
        if json_out:
            pathlib.Path(json_out).write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    for test in tests:
        t0 = time.perf_counter()
        try:
            test(base)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            results.append({
                "name": test.__name__,
                "ok": False,
                "elapsed_s": elapsed,
                "error": repr(exc),
            })
            write_report()
            print(f"[FAIL] {test.__name__} {elapsed:.2f}s: {exc!r}",
                  flush=True)
            raise
        elapsed = time.perf_counter() - t0
        results.append({
            "name": test.__name__,
            "ok": True,
            "elapsed_s": elapsed,
            "error": "",
        })
        print(f"[PASS] {test.__name__} {elapsed:.2f}s", flush=True)

    report["ok"] = True
    write_report()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    get_models(args.base)
    tests = QUICK_TESTS + (FULL_ONLY_TESTS if args.mode == "full" else [])
    run_smoke_tests(args.base,
                    tests,
                    mode=args.mode,
                    json_out=args.json_out)


if __name__ == "__main__":
    main()
