import importlib.util
import pathlib
import sys
import types
import unittest
from typing import Any, Dict

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "qwen3_6_scripts" / "protocol.py"
if not PROTOCOL_PATH.exists():
    PROTOCOL_PATH = pathlib.Path(__file__).resolve().parent / "staging_protocol.py"
PROTOCOL_SOURCE = PROTOCOL_PATH.read_text()

for _site in (
        "/usr/local/corex/lib64/python3/dist-packages",
        "/usr/local/corex/lib/python3/dist-packages",
):
    if pathlib.Path(_site).is_dir() and _site not in sys.path:
        sys.path.insert(0, _site)

from pydantic import ValidationError


class _IInfo:
    min = -9223372036854775808
    max = 9223372036854775807


class _Torch(types.ModuleType):
    long = object()

    def iinfo(self, _dtype):
        return _IInfo()


class _Params:

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _GuidedDecodingParams(_Params):

    @classmethod
    def from_optional(cls, **kwargs):
        return cls(**kwargs)


class _SamplingParams(_Params):

    @classmethod
    def from_optional(cls, **kwargs):
        return cls(**kwargs)


class _RequestOutputKind:
    DELTA = "delta"
    FINAL_ONLY = "final_only"


def _install_protocol_stubs():
    torch = _Torch("torch")

    openai_chat = types.ModuleType("openai.types.chat")
    openai_chat.ChatCompletionContentPartParam = Dict[str, Any]

    chat_utils = types.ModuleType("vllm.entrypoints.chat_utils")
    chat_utils.ChatCompletionMessageParam = Dict[str, Any]

    pooling_params = types.ModuleType("vllm.pooling_params")
    pooling_params.PoolingParams = _Params

    sampling_params = types.ModuleType("vllm.sampling_params")
    sampling_params.BeamSearchParams = _Params
    sampling_params.GuidedDecodingParams = _GuidedDecodingParams
    sampling_params.RequestOutputKind = _RequestOutputKind
    sampling_params.SamplingParams = _SamplingParams

    sequence = types.ModuleType("vllm.sequence")
    sequence.Logprob = float

    utils = types.ModuleType("vllm.utils")
    utils.random_uuid = lambda: "unit-test-id"

    modules = {
        "torch": torch,
        "openai": types.ModuleType("openai"),
        "openai.types": types.ModuleType("openai.types"),
        "openai.types.chat": openai_chat,
        "vllm": types.ModuleType("vllm"),
        "vllm.entrypoints": types.ModuleType("vllm.entrypoints"),
        "vllm.entrypoints.chat_utils": chat_utils,
        "vllm.pooling_params": pooling_params,
        "vllm.sampling_params": sampling_params,
        "vllm.sequence": sequence,
        "vllm.utils": utils,
    }
    for name, module in modules.items():
        sys.modules[name] = module


def _load_protocol():
    _install_protocol_stubs()
    module_name = "qwen36_protocol_unit"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, PROTOCOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _base_request(**overrides):
    data = {
        "model": "llm",
        "messages": [{
            "role": "user",
            "content": "hello",
        }],
        "max_tokens": 8,
    }
    data.update(overrides)
    return data


def _tool():
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "lookup value",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    },
                },
            },
        },
    }


class ProtocolUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.protocol = _load_protocol()

    def request(self, **overrides):
        return self.protocol.ChatCompletionRequest.model_validate(
            _base_request(**overrides))

    def assert_validation_error(self, **overrides):
        with self.assertRaises(ValidationError):
            self.request(**overrides)

    def test_thinking_disabled_variants_normalize_to_chat_template_kwargs(self):
        for thinking in (False, "disabled", {
                "type": "disabled"
        }):
            with self.subTest(thinking=thinking):
                request = self.request(
                    thinking=thinking,
                    chat_template_kwargs={"existing": "kept"},
                )
                self.assertEqual(request.chat_template_kwargs["existing"],
                                 "kept")
                self.assertIs(request.chat_template_kwargs["enable_thinking"],
                              False)

    def test_thinking_enabled_variants_are_allowed(self):
        for thinking in (True, "enabled", {
                "type": "enabled"
        }):
            with self.subTest(thinking=thinking):
                request = self.request(thinking=thinking)
                self.assertIs(request.chat_template_kwargs["enable_thinking"],
                              True)

    def test_invalid_thinking_shape_is_rejected(self):
        self.assert_validation_error(thinking={"type": "maybe"})
        self.assert_validation_error(thinking="off")

    def test_tools_default_to_auto_but_explicit_none_is_preserved(self):
        auto = self.request(tools=[_tool()])
        self.assertEqual(auto.tool_choice, "auto")

        none = self.request(tools=[_tool()], tool_choice="none")
        self.assertEqual(none.tool_choice, "none")

    def test_named_tool_requires_matching_tool_and_conflicts_with_guided_json(self):
        named = {
            "type": "function",
            "function": {
                "name": "lookup",
            },
        }
        request = self.request(tools=[_tool()], tool_choice=named)
        self.assertEqual(request.tool_choice.function.name, "lookup")

        self.assert_validation_error(
            tools=[_tool()],
            tool_choice={
                "type": "function",
                "function": {
                    "name": "missing",
                },
            },
        )
        self.assert_validation_error(
            tools=[_tool()],
            tool_choice=named,
            guided_json={"type": "object"},
        )

    def test_guided_decoding_allows_auto_or_none_tool_choice(self):
        for tool_choice in ("none", "auto"):
            with self.subTest(tool_choice=tool_choice):
                request = self.request(
                    tools=[_tool()],
                    tool_choice=tool_choice,
                    guided_json={"type": "object"},
                )
                self.assertEqual(request.tool_choice, tool_choice)

    def test_json_object_uses_generic_schema_regex_backend(self):
        request = self.request(response_format={"type": "json_object"})
        sampling = request.to_sampling_params(default_max_tokens=32)
        guided = sampling.kwargs["guided_decoding"]
        self.assertEqual(guided.kwargs["json"], {"type": "object"})
        self.assertIsNone(guided.kwargs["json_object"])
        self.assertEqual(
            PROTOCOL_SOURCE.count(
                'guided_json_from_schema = {"type": "object"}'),
            2,
        )

    def test_assistant_tool_calls_allow_null_content(self):
        messages = [
            {"role": "user", "content": "look it up"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_lookup_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query":"value"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_lookup_1",
                "content": "result",
            },
        ]
        request = self.request(messages=messages, tools=[_tool()])
        self.assertEqual(request.messages[1]["content"], "")
        self.assertEqual(request.messages[1]["tool_calls"][0]["id"],
                         "call_lookup_1")

    def test_message_without_content_reasoning_or_tool_calls_is_rejected(self):
        self.assert_validation_error(messages=[{
            "role": "assistant",
            "content": None,
        }])


if __name__ == "__main__":
    unittest.main()
