#!/usr/bin/env python3
"""Run the frozen M1-180 capability, distribution, and optional timing load."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
from typing import Any
import zlib

import exact_chat_prompt
import quality_gate_api as quality


SCHEMA = "bi100-m1-180-arm-observation-v1"
VERSION = 1
STRATA = ("code", "reasoning", "tools", "structured_output",
          "multimodal", "long_context")
SMOKE_PER_STRATUM = 4
FULL_PER_STRATUM = 10
SEED = 20260905
TF_TARGETS = (4096, 16384, 32768, 65536)
PERF_TARGETS = (16384, 32768, 65536)

CODE_CASES = (
    ("print(sum(i*i for i in range(5)))", "30"),
    ("print('bi' + '100')", "bi100"),
    ("print([x for x in range(7) if x % 2][-1])", "5"),
    ("print(len({3, 3, 5, 8}))", "3"),
    ("print('-'.join(reversed(['a','b','c'])))", "c-b-a"),
    ("print(divmod(29, 6))", "(4, 5)"),
    ("print(min({'q': 9, 'k': 4}, key={'q': 9, 'k': 4}.get))", "k"),
    ("print(2 ** 5 + 3)", "35"),
    ("print('abcdef'[1:5:2])", "bd"),
    ("print(all(x < 10 for x in [2, 4, 8]))", "True"),
)
REASONING_CASES = (
    ("A box has 7 rows of 13 bolts. How many bolts?", "91"),
    ("What is 18 percent of 250?", "45"),
    ("A train travels 120 km in 2 hours. What is its speed in km/h?", "60"),
    ("Solve 3x + 5 = 26.", "7"),
    ("What is the next number: 2, 6, 12, 20, 30?", "42"),
    ("A 40 dollar item is discounted by 25 percent. What is the price?", "30"),
    ("Compute the greatest common divisor of 84 and 126.", "42"),
    ("If all Nors are Tivs and no Tiv is a Zed, can a Nor be a Zed?", "NO"),
    ("A rectangle is 9 by 14. What is its area?", "126"),
    ("Convert binary 101101 to decimal.", "45"),
)
TOOL_VALUES = (
    ("osaka", 2), ("beijing", 5), ("lima", 7), ("cairo", 11),
    ("perth", 13), ("seoul", 17), ("accra", 19), ("rome", 23),
    ("quito", 29), ("delhi", 31),
)
STRUCTURED_VALUES = tuple(
    (f"item_{index:02d}", index * 7 + 3) for index in range(10))
COLORS = (
    ("red", (255, 0, 0), ("red", "红")),
    ("blue", (0, 0, 255), ("blue", "蓝")),
    ("green", (0, 180, 0), ("green", "绿")),
    ("yellow", (255, 255, 0), ("yellow", "黄")),
    ("black", (0, 0, 0), ("black", "黑")),
    ("white", (255, 255, 255), ("white", "白")),
    ("purple", (160, 32, 240), ("purple", "紫")),
    ("orange", (255, 128, 0), ("orange", "橙")),
    ("gray", (128, 128, 128), ("gray", "grey", "灰")),
    ("pink", (255, 105, 180), ("pink", "粉")),
)
LONG_TARGETS = (4096, 4096, 4096, 4096, 8192, 8192, 8192, 8192,
                16384, 16384)


class HardEvidenceError(RuntimeError):
    """A request/protocol/finite error that makes the arm invalid."""


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _response(client: quality.Client, payload: dict[str, Any],
              timeout: float = 900) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    status, data = client.post(payload, timeout=timeout)
    elapsed = time.monotonic() - started
    if status != 200 or not isinstance(data, dict):
        raise HardEvidenceError("http_or_json_contract")
    try:
        quality._validate_response_schema(data)
    except quality.CaseFailure as exc:
        raise HardEvidenceError("response_schema_contract") from exc
    if not _finite(data):
        raise HardEvidenceError("nonfinite_response")
    return data, elapsed


def _message(data: dict[str, Any]) -> dict[str, Any]:
    return data["choices"][0]["message"]


def _content(data: dict[str, Any]) -> str:
    value = _message(data).get("content")
    return value if isinstance(value, str) else ""


def _reasoning(data: dict[str, Any]) -> str:
    message = _message(data)
    value = message.get("reasoning_content", message.get("reasoning"))
    return value if isinstance(value, str) else ""


def _payload(prompt: str, max_tokens: int = 96,
             thinking: bool = False) -> dict[str, Any]:
    return {
        "model": "llm",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": SEED,
        "thinking": thinking,
    }


def _solid_png(rgb: tuple[int, int, int]) -> str:
    def chunk(kind: bytes, value: bytes) -> bytes:
        checksum = zlib.crc32(kind + value) & 0xffffffff
        return (struct.pack(">I", len(value)) + kind + value
                + struct.pack(">I", checksum))
    width = height = 128
    scanline = b"\x00" + bytes(rgb) * width
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    image = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
             + chunk(b"IDAT", zlib.compress(scanline * height, 9))
             + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


def _summary(case_id: str, stratum: str, ordinal: int,
             data: dict[str, Any], elapsed: float,
             passed: bool, validator: str) -> dict[str, Any]:
    usage = data["usage"]
    details = usage.get("prompt_tokens_details") or {}
    finish = data["choices"][0].get("finish_reason")
    return {
        "case_id": case_id,
        "stratum": stratum,
        "ordinal": ordinal,
        "stage": "smoke" if ordinal < SMOKE_PER_STRATUM else "extended",
        "pass": passed,
        "validator": validator,
        "http_status": 200,
        "response_contract_complete": True,
        "finish_reason": finish,
        "prompt_tokens": usage["prompt_tokens"],
        "cached_tokens": details.get("cached_tokens", 0),
        "completion_tokens": usage["completion_tokens"],
        "elapsed_s": elapsed,
        "all_values_finite": True,
    }


def _run_code(client: quality.Client, ordinal: int) -> dict[str, Any]:
    snippet, expected = CODE_CASES[ordinal]
    prompt = (f"M1-180 code case {ordinal}. Determine the exact stdout of this "
              f"Python program and output stdout only:\n{snippet}")
    data, elapsed = _response(client, _payload(prompt, 64))
    return _summary(f"code_{ordinal:02d}", "code", ordinal, data, elapsed,
                    _content(data).strip() == expected, "exact_stdout")


def _run_reasoning(client: quality.Client, ordinal: int) -> dict[str, Any]:
    question, expected = REASONING_CASES[ordinal]
    prompt = (f"M1-180 reasoning case {ordinal}. {question} "
              "Solve it independently and end with FINAL=<your answer>.")
    data, elapsed = _response(client, _payload(prompt, 256, True))
    reasoning = _reasoning(data).strip()
    content = _content(data).strip()
    reasoning_protocol_valid = bool(reasoning and content)
    passed = (reasoning_protocol_valid
              and f"FINAL={expected}".casefold()
              in re.sub(r"\s+", "", content).casefold())
    result = _summary(f"reasoning_{ordinal:02d}", "reasoning", ordinal,
                      data, elapsed, passed, "independent_answer")
    result["reasoning_protocol_valid"] = reasoning_protocol_valid
    return result


def _run_tool(client: quality.Client, ordinal: int) -> dict[str, Any]:
    city, days = TOOL_VALUES[ordinal]
    name = f"lookup_{ordinal:02d}"
    payload = _payload(
        f"M1-180 tool case {ordinal}. Call {name} for {city} and {days} days.",
        128)
    payload["tools"] = [{
        "type": "function", "function": {
            "name": name, "description": "Look up a fixed city window.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"},
                               "days": {"type": "integer"}},
                "required": ["city", "days"],
                "additionalProperties": False,
            },
        },
    }]
    payload["tool_choice"] = {"type": "function", "function": {"name": name}}
    data, elapsed = _response(client, payload)
    passed = False
    try:
        calls = quality._normalized_tool_calls(_message(data))
        passed = (len(calls) == 1 and calls[0]["name"] == name
                  and calls[0]["arguments"] == {"city": city, "days": days}
                  and data["choices"][0]["finish_reason"] == "tool_calls")
    except quality.CaseFailure:
        pass
    return _summary(f"tools_{ordinal:02d}", "tools", ordinal, data, elapsed,
                    passed, "forced_tool_exact_arguments")


def _run_structured(client: quality.Client, ordinal: int) -> dict[str, Any]:
    label, value = STRUCTURED_VALUES[ordinal]
    payload = _payload(
        f"M1-180 structured case {ordinal}. Return label {label} and value {value}.",
        64)
    payload["response_format"] = {
        "type": "json_schema", "json_schema": {
            "name": f"m1_180_{ordinal:02d}", "schema": {
                "type": "object",
                "properties": {"label": {"type": "string"},
                               "value": {"type": "integer"}},
                "required": ["label", "value"],
                "additionalProperties": False,
            },
        },
    }
    data, elapsed = _response(client, payload)
    try:
        parsed = json.loads(_content(data))
    except json.JSONDecodeError:
        parsed = None
    passed = parsed == {"label": label, "value": value}
    return _summary(f"structured_output_{ordinal:02d}",
                    "structured_output", ordinal, data, elapsed, passed,
                    "json_schema_exact_values")


def _run_multimodal(client: quality.Client, ordinal: int) -> dict[str, Any]:
    name, rgb, accepted = COLORS[ordinal]
    payload = _payload("unused", 48)
    payload["messages"] = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _solid_png(rgb)}},
        {"type": "text", "text": (
            f"M1-180 image case {ordinal}. State only the dominant color.")},
    ]}]
    data, elapsed = _response(client, payload)
    passed = _matches_color_answer(_content(data), accepted)
    return _summary(f"multimodal_{ordinal:02d}", "multimodal", ordinal,
                    data, elapsed, passed, f"dominant_color_{name}")


def _matches_color_answer(text: str, accepted: tuple[str, ...]) -> bool:
    """Match normalized whole color terms, never arbitrary substrings."""
    normalized = text.casefold().strip()
    english_words = set(re.findall(r"[a-z]+", normalized))
    chinese_aliases = {
        "红": ("红", "红色"), "蓝": ("蓝", "蓝色"),
        "绿": ("绿", "绿色"), "黄": ("黄", "黄色"),
        "黑": ("黑", "黑色"), "白": ("白", "白色"),
        "紫": ("紫", "紫色"), "橙": ("橙", "橙色"),
        "灰": ("灰", "灰色"), "粉": ("粉", "粉色"),
    }
    compact = re.sub(r"[\s\W_]+", "", normalized, flags=re.UNICODE)
    for item in accepted:
        expected = item.casefold()
        if expected.isascii() and expected in english_words:
            return True
        if not expected.isascii() and compact in chinese_aliases.get(
                expected, (expected,)):
            return True
    return False


def _long_prompt(tokenizer: Any, ordinal: int) -> tuple[str, int, str]:
    target = LONG_TARGETS[ordinal]
    marker = f"M180-LONG-{ordinal:02d}-R{ordinal * 17 + 5}"

    def recipe(filler: str) -> tuple[list[dict[str, Any]], None]:
        prompt = (f"M1-180 long-context case {ordinal}. Remember {marker}.\n"
                  + filler + "\nReturn the remembered marker only.")
        return [{"role": "user", "content": prompt}], None

    messages, _, _ = exact_chat_prompt.fit_exact_chat_prompt(
        tokenizer, target, recipe, seed=SEED + ordinal,
        namespace=f"m1-180-long-{ordinal}", thinking=False,
        template_kwargs_mode="direct")
    return messages[0]["content"], target, marker


def _run_long(client: quality.Client, tokenizer: Any,
              ordinal: int) -> dict[str, Any]:
    prompt, target, marker = _long_prompt(tokenizer, ordinal)
    data, elapsed = _response(client, _payload(prompt, 48), timeout=1800)
    passed = _content(data).strip() == marker
    result = _summary(f"long_context_{ordinal:02d}", "long_context",
                      ordinal, data, elapsed, passed, "exact_marker_recall")
    result["target_prompt_tokens"] = target
    if result["prompt_tokens"] != target:
        raise HardEvidenceError("long_context_prompt_token_count")
    return result


def run_cases(client: quality.Client, tokenizer: Any,
              start: int, stop: int) -> list[dict[str, Any]]:
    handlers = {
        "code": lambda n: _run_code(client, n),
        "reasoning": lambda n: _run_reasoning(client, n),
        "tools": lambda n: _run_tool(client, n),
        "structured_output": lambda n: _run_structured(client, n),
        "multimodal": lambda n: _run_multimodal(client, n),
        "long_context": lambda n: _run_long(client, tokenizer, n),
    }
    return [handlers[stratum](ordinal)
            for stratum in STRATA for ordinal in range(start, stop)]


def _case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    capability = report.get("capability") or {}
    return {item["case_id"]: item for item in capability.get("cases") or []}


def smoke_regressions(references: list[dict[str, Any]],
                      candidate: list[dict[str, Any]]) -> list[str]:
    candidate_map = {item["case_id"]: item for item in candidate}
    regressions = []
    for reference in references:
        for case_id, item in _case_map(reference).items():
            if (item.get("stage") == "smoke" and item.get("pass") is True
                    and case_id in candidate_map
                    and candidate_map[case_id].get("pass") is False):
                regressions.append(case_id)
    return sorted(set(regressions))


def _run_child(command: list[str], environment: dict[str, str]) -> None:
    result = subprocess.run(command, check=False, env=environment)
    if result.returncode:
        raise HardEvidenceError(f"child_workload_rc_{result.returncode}")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HardEvidenceError("reference_not_object")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    started = time.monotonic()
    client = quality.Client(args.base)
    client.models("llm")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    if args.arm == "m1_162":
        cases = run_cases(client, tokenizer, 0, SMOKE_PER_STRATUM)
        references = [_load(args.reference_fused_off),
                      _load(args.reference_m1_109)]
        regressions = smoke_regressions(references, cases)
        extended = not regressions
        if extended:
            cases.extend(run_cases(
                client, tokenizer, SMOKE_PER_STRATUM, FULL_PER_STRATUM))
    else:
        cases = run_cases(client, tokenizer, 0, FULL_PER_STRATUM)
        regressions = []
        extended = True

    tf_path = args.out.with_name("teacher_forced_private.json")
    perf_path = args.out.with_name("performance_private.json")
    environment = os.environ.copy()
    key = environment.get("BI100_TEACHER_FORCED_HMAC_KEY", "")
    if len(key) != 64:
        raise HardEvidenceError("teacher_identity_key_missing")
    teacher_forced: dict[str, Any] | None = None
    performance: dict[str, Any] | None = None
    if not regressions:
        _run_child([
            sys.executable, str(Path(__file__).with_name(
                "teacher_forced_topk_api.py")),
            "--base", args.base, "--model-path", str(args.model_path),
            "--attention-runtime-manifest",
            str(args.attention_runtime_manifest),
            "--runtime-identity", args.runtime_identity,
            "--source-revision", args.source_revision,
            "--instance", args.instance,
            "--mode", "candidate" if args.arm == "m1_162" else "control",
            "--targets", ",".join(map(str, TF_TARGETS)),
            "--timeout-s", "3600", "--out", str(tf_path),
        ], environment)
        teacher_forced = _load(tf_path)

    if args.arm == "m1_109" or (args.arm == "m1_162" and not regressions):
        _run_child([
            sys.executable, str(Path(__file__).with_name(
                "attention_operator_tp4_service.py")),
            "--base", args.base, "--model-path", str(args.model_path),
            "--timeout-s", "1800", "--run-id", f"m1-180-{args.arm}",
            "--workload-id", args.workload_id,
            "--selector", "candidate" if args.arm == "m1_162" else "control",
            "--targets", ",".join(map(str, PERF_TARGETS)),
            "--repetitions", "2", "--out", str(perf_path),
        ], environment)
        performance = _load(perf_path)

    report = {
        "schema": SCHEMA, "version": VERSION,
        "arm": args.arm,
        "algorithm_variant": ("fused_off" if args.arm == "fused_off"
                              else f"{args.arm}_fp32_qk" if args.arm == "m1_109"
                              else "m1_162_fp16_qk"),
        "source_revision": args.source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "model_path": str(args.model_path),
        "workload_id": args.workload_id,
        "capability": {
            "strata": list(STRATA),
            "smoke_per_stratum": SMOKE_PER_STRATUM,
            "full_per_stratum": FULL_PER_STRATUM,
            "cases": cases,
            "smoke_completed": sum(item["stage"] == "smoke" for item in cases),
            "extended_triggered": extended,
            "critical_smoke_baseline_only": regressions,
            "complete": len(cases) == len(STRATA) * FULL_PER_STRATUM,
        },
        "teacher_forced": teacher_forced,
        "performance": performance,
        "request_population": {
            "attempted": len(cases)
            + (len(TF_TARGETS) if teacher_forced is not None else 0)
            + (len(PERF_TARGETS) * 2 if performance is not None else 0),
            "completed": len(cases)
            + (len(TF_TARGETS) if teacher_forced is not None else 0)
            + (len(PERF_TARGETS) * 2 if performance is not None else 0),
            "failed": 0,
        },
        "wall_s": time.monotonic() - started,
        "privacy": {
            "prompts_recorded": False, "model_outputs_recorded": False,
            "token_ids_recorded": False, "images_recorded": False,
            "tool_arguments_recorded": False, "credentials_recorded": False,
            "private_teacher_token_keys_nested": teacher_forced is not None,
            "repository_safe": False,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--attention-runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--arm", choices=("fused_off", "m1_109", "m1_162"),
                        required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--reference-fused-off", type=Path)
    parser.add_argument("--reference-m1-109", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.arm == "m1_162" and (
            args.reference_fused_off is None
            or args.reference_m1_109 is None):
        parser.error("candidate requires both capability references")
    try:
        report = run(args)
        args.out.write_text(json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        return 0
    except (HardEvidenceError, OSError, ValueError,
            exact_chat_prompt.PromptConstructionError) as exc:
        print(f"M1-180 workload invalid: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
