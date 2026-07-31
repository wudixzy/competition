import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tests" / "summarize_api_4xx_log.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_api_4xx_log", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SHAPE = (
    "messages=3 systems=1 system_part_msgs=1 system_text_parts=2 "
    "system_other_parts=0 tools=2 tool_msgs=1 assistant_tool_msgs=1 "
    "strict_false=2 strict_true=0 choice=auto images=1 image_data=1 "
    "image_remote=0 image_other=0 stream=1 n=1"
)


class SummarizeApi4xxLogTest(unittest.TestCase):

    def summarize(self, text):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "server.log"
            path.write_text(text, encoding="utf-8")
            return MODULE.summarize(path)

    def test_complete_attribution_groups_only_bounded_fields(self):
        report, qualified = self.summarize(
            "WARNING [BI100 4XX] endpoint=request_validation code=400 "
            f"reason=request_validation_tool_strict {SHAPE} errors=1 "
            "validation_field=tools "
            "validation_type=value_error.extra\n"
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n"
        )
        self.assertTrue(qualified)
        self.assertTrue(report["complete"])
        self.assertEqual(
            report["by_reason"], {"request_validation_tool_strict": 1})
        self.assertEqual(report["by_validation_field"], {"tools": 1})
        self.assertEqual(
            report["by_validation_type"], {"value_error.extra": 1})
        self.assertEqual(report["request_shapes"], [{
            "messages": 3,
            "systems": 1,
            "system_part_msgs": 1,
            "system_text_parts": 2,
            "system_other_parts": 0,
            "tools": 2,
            "tool_msgs": 1,
            "assistant_tool_msgs": 1,
            "strict_false": 2,
            "strict_true": 0,
            "choice": "auto",
            "images": 1,
            "image_data": 1,
            "image_remote": 0,
            "image_other": 0,
            "stream": 1,
            "n": 1,
            "count": 1,
        }])
        self.assertNotIn("prompt", repr(report).lower())
        self.assertNotIn("schema contents", repr(report).lower())

    def test_missing_or_malformed_marker_fails_closed(self):
        report, qualified = self.summarize(
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n")
        self.assertFalse(qualified)
        self.assertFalse(report["complete"])
        self.assertEqual(report["attribution_delta"], 1)

        report, qualified = self.summarize(
            "WARNING [BI100 4XX] endpoint=request_validation code=400 "
            f"reason=unknown {SHAPE} errors=1\n")
        self.assertFalse(qualified)
        self.assertEqual(report["malformed_marker_count"], 1)

    def test_inconsistent_image_counts_fail_closed(self):
        malformed = SHAPE.replace("images=1", "images=2")
        report, qualified = self.summarize(
            "WARNING [BI100 4XX] endpoint=chat code=400 "
            f"reason=image_count_limit {malformed}\n"
        )
        self.assertFalse(qualified)
        self.assertEqual(report["malformed_marker_count"], 1)

    def test_unclassified_chat_error_blocks_qualification(self):
        report, qualified = self.summarize(
            "WARNING [BI100 4XX] endpoint=chat code=400 "
            f"reason=unclassified_chat_error {SHAPE}\n"
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n")
        self.assertFalse(qualified)
        self.assertTrue(report["complete"])
        self.assertFalse(report["classified"])

    def test_known_sampling_and_context_errors_are_classified(self):
        lines = []
        for reason in (
            "invalid_top_p",
            "invalid_max_tokens",
            "context_length_exceeded",
        ):
            lines.extend([
                "WARNING [BI100 4XX] endpoint=chat code=400 "
                f"reason={reason} {SHAPE}\n",
                'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
                "400 Bad Request\n",
            ])
        report, qualified = self.summarize("".join(lines))
        self.assertTrue(qualified, report)
        self.assertTrue(report["classified"])
        self.assertEqual(report["by_reason"], {
            "context_length_exceeded": 1,
            "invalid_max_tokens": 1,
            "invalid_top_p": 1,
        })

    def test_request_stage_detail_is_bounded_and_summarized(self):
        report, qualified = self.summarize(
            "WARNING [BI100 4XX DETAIL] endpoint=chat "
            "stage=multimodal_load exception_type=TimeoutError\n"
            "WARNING [BI100 4XX] endpoint=chat code=400 "
            f"reason=multimodal_load_failed {SHAPE}\n"
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n"
        )
        self.assertTrue(qualified, report)
        self.assertEqual(
            report["by_failure_stage"], {"multimodal_load": 1})
        self.assertEqual(
            report["by_exception_type"], {"TimeoutError": 1})
        self.assertEqual(report["malformed_detail_marker_count"], 0)

    def test_malformed_request_stage_detail_fails_closed(self):
        report, qualified = self.summarize(
            "WARNING [BI100 4XX DETAIL] endpoint=chat "
            "stage=private/stage exception_type=private/value\n"
        )
        self.assertFalse(qualified)
        self.assertEqual(report["malformed_detail_marker_count"], 1)

    def test_zero_4xx_is_complete(self):
        report, qualified = self.summarize(
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "200 OK\n")
        self.assertTrue(qualified)
        self.assertEqual(report["chat_4xx_access_count"], 0)

    def test_legacy_v2_shape_is_reconciled_without_inventing_fields(self):
        legacy = (
            "messages=2 systems=0 tools=1 tool_msgs=0 "
            "assistant_tool_msgs=0 strict_false=0 strict_true=1 "
            "choice=none image=0 stream=0 n=1"
        )
        report, qualified = self.summarize(
            "WARNING [BI100 4XX] endpoint=request_validation code=400 "
            "reason=request_validation_tool_strict "
            f"{legacy} errors=1\n"
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n"
        )
        self.assertTrue(qualified, report)
        self.assertEqual(
            report["by_validation_field"], {"unknown": 1})
        self.assertEqual(
            report["by_validation_type"], {"unknown": 1})
        self.assertEqual(report["request_shapes"], [{
            "messages": 2,
            "systems": 0,
            "tools": 1,
            "tool_msgs": 0,
            "assistant_tool_msgs": 0,
            "strict_false": 0,
            "strict_true": 1,
            "image": 0,
            "stream": 0,
            "choice": "none",
            "n": 1,
            "shape_version": 2,
            "count": 1,
        }])

    def test_root_unknown_and_multiple_validation_dimensions_are_counted(self):
        report, qualified = self.summarize(
            "WARNING [BI100 4XX] endpoint=request_validation code=400 "
            "reason=request_validation_unknown errors=1 "
            "validation_field=root validation_type=value_error\n"
            "WARNING [BI100 4XX] endpoint=request_validation code=400 "
            "reason=request_validation_unknown errors=0 "
            "validation_field=unknown validation_type=unknown\n"
            "WARNING [BI100 4XX] endpoint=request_validation code=400 "
            "reason=request_validation_other errors=2 "
            "validation_field=multiple validation_type=multiple\n"
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n"
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n"
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n"
        )
        self.assertTrue(qualified, report)
        self.assertEqual(report["by_validation_field"], {
            "multiple": 1,
            "root": 1,
            "unknown": 1,
        })
        self.assertEqual(report["by_validation_type"], {
            "multiple": 1,
            "unknown": 1,
            "value_error": 1,
        })

    def test_malformed_validation_identifier_fails_closed(self):
        report, qualified = self.summarize(
            "WARNING [BI100 4XX] endpoint=request_validation code=400 "
            "reason=request_validation_unknown errors=1 "
            "validation_field=private/value "
            "validation_type=value_error\n"
        )
        self.assertFalse(qualified)
        self.assertEqual(report["malformed_marker_count"], 1)


if __name__ == "__main__":
    unittest.main()
