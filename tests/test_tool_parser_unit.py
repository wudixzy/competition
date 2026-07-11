import importlib.util
import json
import pathlib
import re
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER_PATH = ROOT / "qwen3_6_scripts" / "qwen3coder_tool_parser.py"


class _Struct:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _ToolParser:

    def __init__(self, tokenizer):
        self.model_tokenizer = tokenizer
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []

    @property
    def vocab(self):
        return self.model_tokenizer.get_vocab()


class _ToolParserManager:
    tool_parsers = {}

    @classmethod
    def register_module(cls, name=None, force=True, module=None):
        def _register(parser_cls):
            cls.tool_parsers[name or parser_cls.__name__] = parser_cls
            return parser_cls

        return _register(module) if module is not None else _register


class _Logger:

    def debug(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


class _Tokenizer:

    def get_vocab(self):
        return {
            "<tool_call>": 1,
            "</tool_call>": 2,
        }


def _install_vllm_stubs():
    protocol = types.ModuleType("vllm.entrypoints.openai.protocol")
    for name in [
            "ChatCompletionRequest",
            "ChatCompletionToolsParam",
            "DeltaFunctionCall",
            "DeltaMessage",
            "DeltaToolCall",
            "ExtractedToolCallInformation",
            "FunctionCall",
            "ToolCall",
    ]:
        setattr(protocol, name, type(name, (_Struct, ), {}))

    abstract = types.ModuleType(
        "vllm.entrypoints.openai.tool_parsers.abstract_tool_parser")
    abstract.ToolParser = _ToolParser
    abstract.ToolParserManager = _ToolParserManager

    logger = types.ModuleType("vllm.logger")
    logger.init_logger = lambda _name: _Logger()

    tokenizer = types.ModuleType("vllm.transformers_utils.tokenizer")
    tokenizer.AnyTokenizer = object

    modules = {
        "regex": re,
        "vllm": types.ModuleType("vllm"),
        "vllm.entrypoints": types.ModuleType("vllm.entrypoints"),
        "vllm.entrypoints.openai": types.ModuleType("vllm.entrypoints.openai"),
        "vllm.entrypoints.openai.protocol": protocol,
        "vllm.entrypoints.openai.tool_parsers": types.ModuleType(
            "vllm.entrypoints.openai.tool_parsers"),
        "vllm.entrypoints.openai.tool_parsers.abstract_tool_parser": abstract,
        "vllm.logger": logger,
        "vllm.transformers_utils": types.ModuleType(
            "vllm.transformers_utils"),
        "vllm.transformers_utils.tokenizer": tokenizer,
    }
    for name, module in modules.items():
        sys.modules[name] = module


def _load_parser_class():
    _install_vllm_stubs()
    spec = importlib.util.spec_from_file_location(
        "qwen3coder_tool_parser_unit", PARSER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Qwen3CoderToolParser


class Qwen3CoderToolParserUnitTest(unittest.TestCase):

    def test_streaming_argument_name_is_json_escaped(self):
        parser_cls = _load_parser_class()
        parser = parser_cls(_Tokenizer())
        request = _Struct(tools=None)

        text = "<tool_call>"
        parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=text,
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=[1],
            delta_token_ids=[1],
            request=request,
        )

        text += "<function=foo>"
        header = parser.extract_tool_calls_streaming(
            previous_text="<tool_call>",
            current_text=text,
            delta_text="<function=foo>",
            previous_token_ids=[1],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        self.assertEqual(header.tool_calls[0].function.name, "foo")

        opening = parser.extract_tool_calls_streaming(
            previous_text="<tool_call><function=foo>",
            current_text=text,
            delta_text=" ",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        self.assertEqual(opening.tool_calls[0].function.arguments, "{")

        key = 'quote"and\\slash'
        text += f"<parameter={key}>value</parameter>"
        fragment = parser.extract_tool_calls_streaming(
            previous_text="<tool_call><function=foo>",
            current_text=text,
            delta_text=f"<parameter={key}>value</parameter>",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        arguments = fragment.tool_calls[0].function.arguments

        self.assertEqual(json.loads("{" + arguments + "}"), {key: "value"})
        self.assertNotIn(f'"{key}"', arguments)


if __name__ == "__main__":
    unittest.main()
