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
    "messages=3 systems=1 tools=2 tool_msgs=1 assistant_tool_msgs=1 "
    "strict_false=2 strict_true=0 choice=auto image=0 stream=1 n=1"
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
            f"reason=request_validation_tool_strict {SHAPE} errors=1\n"
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n"
        )
        self.assertTrue(qualified)
        self.assertTrue(report["complete"])
        self.assertEqual(
            report["by_reason"], {"request_validation_tool_strict": 1})
        self.assertEqual(report["request_shapes"], [{
            "messages": 3,
            "systems": 1,
            "tools": 2,
            "tool_msgs": 1,
            "assistant_tool_msgs": 1,
            "strict_false": 2,
            "strict_true": 0,
            "choice": "auto",
            "image": 0,
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

    def test_unclassified_chat_error_blocks_qualification(self):
        report, qualified = self.summarize(
            "WARNING [BI100 4XX] endpoint=chat code=400 "
            f"reason=unclassified_chat_error {SHAPE}\n"
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n")
        self.assertFalse(qualified)
        self.assertTrue(report["complete"])
        self.assertFalse(report["classified"])

    def test_zero_4xx_is_complete(self):
        report, qualified = self.summarize(
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "200 OK\n")
        self.assertTrue(qualified)
        self.assertEqual(report["chat_4xx_access_count"], 0)


if __name__ == "__main__":
    unittest.main()
