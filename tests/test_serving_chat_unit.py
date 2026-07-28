import asyncio
import ast
import json
import pathlib
import types
import unittest
from typing import Optional, Union

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVING_CHAT = ROOT / "qwen3_6_scripts" / "serving_chat.py"
SERVING_CHAT_SOURCE = SERVING_CHAT.read_text()
CHAT_UTILS = ROOT / "qwen3_6_scripts" / "chat_utils.py"
CHAT_UTILS_SOURCE = CHAT_UTILS.read_text()
QWEN_MODEL = ROOT / "qwen3_6_scripts" / "qwen3_5.py"
QWEN_MODEL_SOURCE = QWEN_MODEL.read_text()


class _Copyable:

    def model_copy(self, *, deep=False, update=None):
        values = dict(self.__dict__)
        values.update(update or {})
        return type(self)(**values)


class _FakePromptDetails(_Copyable):

    def __init__(self, cached_tokens):
        self.cached_tokens = cached_tokens


class _FakeChoice(_Copyable):

    def __init__(self, index, marker):
        self.index = index
        self.marker = marker


class _FakeUsageInfo:

    def __init__(
        self,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        reasoning_tokens=None,
        prompt_tokens_details=None,
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.reasoning_tokens = reasoning_tokens
        self.prompt_tokens_details = prompt_tokens_details


class _FakeChatCompletionResponse:

    def __init__(
        self,
        model,
        choices,
        usage,
        prompt_logprobs=None,
        id=None,
        created=None,
    ):
        self.id = id
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
        self.prompt_logprobs = prompt_logprobs


def _load_serialize_tool_arguments():
    tree = ast.parse(SERVING_CHAT.read_text(), filename=str(SERVING_CHAT))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_serialize_tool_arguments")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"json": json}
    exec(compile(module, str(SERVING_CHAT), "exec"), namespace)
    return namespace["_serialize_tool_arguments"]


def _load_named_tool_stream_helpers():
    tree = ast.parse(SERVING_CHAT.read_text(), filename=str(SERVING_CHAT))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_named_tool_delta_payload",
            "_consume_named_tool_header_slot",
        }
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Dict": dict, "List": list}
    exec(compile(module, str(SERVING_CHAT), "exec"), namespace)
    return namespace


def _load_named_tool_argument_helpers():
    tree = ast.parse(SERVING_CHAT.read_text(), filename=str(SERVING_CHAT))
    names = {
        "_serialize_tool_arguments",
        "_tool_arguments_are_json_object",
        "_reclassify_named_guided_json",
        "_select_named_tool_arguments",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "json": json,
        "List": list,
        "Optional": Optional,
        "ToolCall": object,
    }
    exec(compile(module, str(SERVING_CHAT), "exec"), namespace)
    return namespace


def _load_sequential_fanout_helpers():
    tree = ast.parse(SERVING_CHAT.read_text(), filename=str(SERVING_CHAT))
    names = {
        "_sequential_greedy_fanout_count",
        "_merge_sequential_chat_responses",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "List": list,
        "ChatCompletionRequest": object,
        "ChatCompletionResponse": _FakeChatCompletionResponse,
        "UsageInfo": _FakeUsageInfo,
    }
    exec(compile(module, str(SERVING_CHAT), "exec"), namespace)
    return namespace


def _load_sequential_fanout_method(merge_responses):
    tree = ast.parse(SERVING_CHAT.read_text(), filename=str(SERVING_CHAT))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "OpenAIServingChat")
    function = next(
        node for node in class_node.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_create_sequential_greedy_fanout")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)

    class FakeErrorResponse:
        pass

    class FakeMetadata:

        def __init__(self, request_id, final_usage_info):
            self.request_id = request_id
            self.final_usage_info = final_usage_info

    logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    namespace = {
        "ChatCompletionRequest": object,
        "Request": object,
        "Optional": Optional,
        "Union": Union,
        "ChatCompletionResponse": _FakeChatCompletionResponse,
        "ErrorResponse": FakeErrorResponse,
        "List": list,
        "random_uuid": lambda: "parent",
        "time": types.SimpleNamespace(time=lambda: 123),
        "logger": logger,
        "RequestResponseMetadata": FakeMetadata,
        "_merge_sequential_chat_responses": merge_responses,
    }
    exec(compile(module, str(SERVING_CHAT), "exec"), namespace)
    return (
        namespace["_create_sequential_greedy_fanout"],
        FakeErrorResponse,
    )


def _load_chat_placeholder_method():
    tree = ast.parse(CHAT_UTILS_SOURCE, filename=str(CHAT_UTILS))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "BaseMultiModalItemTracker")
    function = next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_placeholder_str")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"ModalityStr": str, "Optional": Optional}
    exec(compile(module, str(CHAT_UTILS), "exec"), namespace)
    return namespace["_placeholder_str"]


class ServingChatUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.serialize = staticmethod(_load_serialize_tool_arguments())
        stream_helpers = _load_named_tool_stream_helpers()
        cls.named_delta = staticmethod(
            stream_helpers["_named_tool_delta_payload"])
        cls.consume_named_header = staticmethod(
            stream_helpers["_consume_named_tool_header_slot"])
        helpers = _load_named_tool_argument_helpers()
        cls.select_named_arguments = staticmethod(
            helpers["_select_named_tool_arguments"])
        cls.reclassify_named_json = staticmethod(
            helpers["_reclassify_named_guided_json"])
        fanout = _load_sequential_fanout_helpers()
        cls.fanout_count = staticmethod(
            fanout["_sequential_greedy_fanout_count"])
        cls.merge_fanout = staticmethod(
            fanout["_merge_sequential_chat_responses"])
        method, error_type = _load_sequential_fanout_method(cls.merge_fanout)
        cls.run_fanout = staticmethod(method)
        cls.fanout_error_type = error_type

    def test_tool_arguments_string_is_not_double_json_encoded(self):
        arguments = '{"city": "上海", "unit": "c"}'
        self.assertEqual(self.serialize(arguments), arguments)

    def test_tool_arguments_structured_values_are_json_encoded(self):
        self.assertEqual(json.loads(self.serialize({"city": "上海"})),
                         {"city": "上海"})
        self.assertEqual(json.loads(self.serialize(["a", "b"])), ["a", "b"])

    def test_tool_arguments_none_defaults_to_empty_object(self):
        self.assertEqual(self.serialize(None), "{}")

    def test_named_tool_calls_use_tool_finish_reason(self):
        self.assertIn("named_tool_called = False", SERVING_CHAT_SOURCE)
        self.assertIn("named_tool_called = True", SERVING_CHAT_SOURCE)
        self.assertIn(
            "auto_tools_called or named_tool_called", SERVING_CHAT_SOURCE)
        self.assertIn(
            "auto_tools_called or tool_choice_function_name",
            SERVING_CHAT_SOURCE,
        )

    def test_named_stream_tool_identity_is_stable_and_sent_once(self):
        first = self.named_delta(
            "terminal", '{"command":', 0, "call-stable", True)
        following = self.named_delta(
            "terminal", '"pwd"}', 0, "call-stable", False)

        self.assertEqual(first, {
            "id": "call-stable",
            "type": "function",
            "index": 0,
            "function": {
                "name": "terminal",
                "arguments": '{"command":',
            },
        })
        self.assertEqual(following, {
            "index": 0,
            "function": {"arguments": '"pwd"}'},
        })
        self.assertIn("named_tool_call_ids = (", SERVING_CHAT_SOURCE)
        self.assertIn(
            "named_tool_header_sent = [False] * num_choices",
            SERVING_CHAT_SOURCE,
        )
        self.assertNotIn("previous_num_tokens[i] == 0", SERVING_CHAT_SOURCE)

    def test_named_stream_header_is_consumed_even_for_zero_token_delta(self):
        header_sent = [False, False]
        self.assertTrue(self.consume_named_header(header_sent, 0))
        self.assertFalse(self.consume_named_header(header_sent, 0))
        self.assertTrue(self.consume_named_header(header_sent, 1))
        self.assertEqual(header_sent, [True, True])

    def test_named_nonstream_keeps_valid_raw_json_exactly(self):
        raw = '{ "key" : "TOOLS-731" }'
        parsed = [types.SimpleNamespace(function=types.SimpleNamespace(
            name="lookup_quality_marker", arguments='{"key":"other"}'))]
        self.assertEqual(
            self.select_named_arguments(
                raw, "lookup_quality_marker", parsed),
            raw,
        )

    def test_named_guided_json_is_not_misclassified_as_reasoning(self):
        raw = '{"key":"AGENT-235K-731","ordinal":235000}'
        reasoning, output = self.reclassify_named_json(raw, "")
        self.assertIsNone(reasoning)
        self.assertEqual(output, raw)

    def test_unterminated_non_json_reasoning_is_not_reclassified(self):
        raw = "unfinished private reasoning"
        reasoning, output = self.reclassify_named_json(raw, "")
        self.assertEqual(reasoning, raw)
        self.assertEqual(output, "")

    def test_named_nonstream_recovers_unique_same_name_parser_call(self):
        parsed = [types.SimpleNamespace(function=types.SimpleNamespace(
            name="report_agent_marker",
            arguments={"key": "AGENT-235K-731", "ordinal": 235000},
        ))]
        selected = self.select_named_arguments(
            "<tool_call>not-json</tool_call>",
            "report_agent_marker",
            parsed,
        )
        self.assertEqual(json.loads(selected), {
            "key": "AGENT-235K-731",
            "ordinal": 235000,
        })

    def test_named_nonstream_rejects_ambiguous_or_wrong_parser_call(self):
        raw = "<tool_call>not-json</tool_call>"
        wrong = [types.SimpleNamespace(function=types.SimpleNamespace(
            name="other", arguments='{"key":"x"}'))]
        ambiguous = wrong + wrong
        invalid = [types.SimpleNamespace(function=types.SimpleNamespace(
            name="report_agent_marker", arguments="not-json"))]
        for parsed in (wrong, ambiguous, invalid, None):
            with self.subTest(parsed=parsed):
                self.assertEqual(
                    self.select_named_arguments(
                        raw, "report_agent_marker", parsed),
                    raw,
                )

    def test_named_nonstream_invokes_parser_only_for_invalid_json(self):
        branch = SERVING_CHAT_SOURCE.index(
            "parsed_named_tool_calls: Optional[List[ToolCall]]")
        guard = SERVING_CHAT_SOURCE.index(
            "not _tool_arguments_are_json_object(output_text)", branch)
        parser_call = SERVING_CHAT_SOURCE.index(
            ".extract_tool_calls(", guard)
        selector = SERVING_CHAT_SOURCE.index(
            "named_arguments = _select_named_tool_arguments", parser_call)
        self.assertLess(branch, guard)
        self.assertLess(guard, parser_call)
        self.assertLess(parser_call, selector)

    def test_empty_messages_are_rejected_before_async_work(self):
        guard = 'if not request.messages:'
        guard_pos = SERVING_CHAT_SOURCE.index(guard)
        self.assertIn("messages must contain at least one message",
                      SERVING_CHAT_SOURCE)
        for later_operation in [
                "await self._check_model(request)",
                "await self.engine_client.get_tokenizer(lora_request)",
                "parse_chat_messages_futures(",
        ]:
            self.assertLess(
                guard_pos, SERVING_CHAT_SOURCE.index(later_operation))

    def test_sequential_fanout_is_limited_to_exact_greedy_n2_shape(self):
        values = {
            "n": 2,
            "temperature": 0,
            "stream": False,
            "use_beam_search": False,
            "best_of": None,
            "prompt_logprobs": None,
        }
        request = types.SimpleNamespace(**values)
        self.assertEqual(self.fanout_count(request, 1), 2)
        self.assertEqual(self.fanout_count(request, 2), 0)

        rejected = {
            "n": 3,
            "temperature": 0.7,
            "stream": True,
            "use_beam_search": True,
            "best_of": 2,
            "prompt_logprobs": 1,
        }
        for field, value in rejected.items():
            with self.subTest(field=field):
                variant = dict(values)
                variant[field] = value
                self.assertEqual(
                    self.fanout_count(types.SimpleNamespace(**variant), 1),
                    0,
                )

    def test_sequential_fanout_merges_choices_and_usage_once(self):
        first_details = _FakePromptDetails(cached_tokens=0)
        responses = [
            _FakeChatCompletionResponse(
                model="llm",
                choices=[_FakeChoice(index=0, marker="first")],
                usage=_FakeUsageInfo(
                    prompt_tokens=11,
                    completion_tokens=2,
                    total_tokens=13,
                    reasoning_tokens=1,
                    prompt_tokens_details=first_details,
                ),
                prompt_logprobs=["prompt"],
            ),
            _FakeChatCompletionResponse(
                model="llm",
                choices=[_FakeChoice(index=0, marker="second")],
                usage=_FakeUsageInfo(
                    prompt_tokens=11,
                    completion_tokens=3,
                    total_tokens=14,
                    reasoning_tokens=None,
                    prompt_tokens_details=_FakePromptDetails(
                        cached_tokens=11),
                ),
                prompt_logprobs=["prompt"],
            ),
        ]
        merged = self.merge_fanout(responses, "chat-parent", 123)

        self.assertEqual(merged.id, "chat-parent")
        self.assertEqual(merged.created, 123)
        self.assertEqual(
            [(choice.index, choice.marker) for choice in merged.choices],
            [(0, "first"), (1, "second")],
        )
        self.assertEqual(merged.usage.prompt_tokens, 11)
        self.assertEqual(merged.usage.completion_tokens, 5)
        self.assertEqual(merged.usage.total_tokens, 16)
        self.assertEqual(merged.usage.reasoning_tokens, 1)
        self.assertEqual(
            merged.usage.prompt_tokens_details.cached_tokens, 0)
        self.assertIsNot(
            merged.usage.prompt_tokens_details, first_details)
        self.assertEqual([choice.index for choice in responses[0].choices], [0])

    def test_sequential_fanout_merge_fails_closed_on_contract_drift(self):
        def response(*, model="llm", prompt=11, completion=2, choices=1):
            return _FakeChatCompletionResponse(
                model=model,
                choices=[
                    _FakeChoice(index=0, marker=str(index))
                    for index in range(choices)
                ],
                usage=_FakeUsageInfo(
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=prompt + (completion or 0),
                ),
            )

        invalid_pairs = [
            [response()],
            [response(), response(model="other")],
            [response(), response(prompt=12)],
            [response(), response(completion=None)],
            [response(), response(choices=2)],
        ]
        for pair in invalid_pairs:
            with self.subTest(pair_size=len(pair)):
                with self.assertRaises(ValueError):
                    self.merge_fanout(pair, "chat-parent", 123)

    def test_sequential_fanout_guard_precedes_template_and_is_observable(self):
        multi_choice_guard = SERVING_CHAT_SOURCE.index(
            "if request.n is not None and request.n > 1:")
        scheduler_lookup = SERVING_CHAT_SOURCE.index(
            "scheduler_config = await self.engine_client."
            "get_scheduler_config()")
        guard = SERVING_CHAT_SOURCE.index(
            "fanout_count = _sequential_greedy_fanout_count")
        template = SERVING_CHAT_SOURCE.index("parse_chat_messages_futures(")
        self.assertLess(multi_choice_guard, scheduler_lookup)
        self.assertLess(scheduler_lookup, guard)
        self.assertLess(guard, template)
        self.assertIn(
            "return await self._create_sequential_greedy_fanout(",
            SERVING_CHAT_SOURCE,
        )
        self.assertIn('update={"n": 1}', SERVING_CHAT_SOURCE)
        self.assertIn("[BI100 N_FANOUT]", SERVING_CHAT_SOURCE)

    def test_sequential_fanout_runs_two_n1_children_and_restores_metadata(self):
        class FakeRequest:

            def __init__(self, n):
                self.n = n

            def model_copy(self, *, deep=False, update=None):
                return FakeRequest((update or {}).get("n", self.n))

        def response(marker, completion):
            return _FakeChatCompletionResponse(
                model="llm",
                choices=[_FakeChoice(index=0, marker=marker)],
                usage=_FakeUsageInfo(
                    prompt_tokens=11,
                    completion_tokens=completion,
                    total_tokens=11 + completion,
                ),
            )

        class FakeServing:

            def __init__(self, responses):
                self.responses = list(responses)
                self.child_ns = []

            async def create_chat_completion(self, request, raw_request):
                self.child_ns.append(request.n)
                return self.responses.pop(0)

            def create_error_response(self, message):
                raise AssertionError(message)

        raw_request = types.SimpleNamespace(state=types.SimpleNamespace())
        serving = FakeServing([response("first", 2), response("second", 3)])
        merged = asyncio.run(self.run_fanout(
            serving, FakeRequest(2), raw_request, 2))
        self.assertEqual(serving.child_ns, [1, 1])
        self.assertEqual([choice.index for choice in merged.choices], [0, 1])
        self.assertEqual(merged.usage.completion_tokens, 5)
        self.assertEqual(
            raw_request.state.request_metadata.request_id, "chat-parent")
        self.assertIs(
            raw_request.state.request_metadata.final_usage_info, merged.usage)

        error = self.fanout_error_type()
        serving = FakeServing([error, response("unused", 1)])
        returned = asyncio.run(self.run_fanout(
            serving, FakeRequest(2), raw_request, 2))
        self.assertIs(returned, error)
        self.assertEqual(serving.child_ns, [1])

    def test_qwen36_image_placeholder_uses_native_vision_tokens(self):
        placeholder_str = _load_chat_placeholder_method()
        tracker = types.SimpleNamespace(
            _model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(model_type="qwen3_5_moe")),
            _tokenizer=None,
        )
        self.assertEqual(
            placeholder_str(tracker, "image", 1),
            "<|vision_start|><|image_pad|><|vision_end|>",
        )

    def test_qwen36_cached_image_tokens_use_visual_suffix(self):
        self.assertIn("if num_placeholders:", QWEN_MODEL_SOURCE)
        self.assertIn("image_embeds[-num_placeholders:]", QWEN_MODEL_SOURCE)
        self.assertNotIn(
            "image token count ({num_placeholders}) does not match",
            QWEN_MODEL_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
