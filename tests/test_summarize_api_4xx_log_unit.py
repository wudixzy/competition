import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tests" / "summarize_api_4xx_log.py"
SPEC = importlib.util.spec_from_file_location("summarize_api_4xx_log",
                                               MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SummarizeApi4xxLogTest(unittest.TestCase):

    def summarize(self, text):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "server.log"
            path.write_text(text, encoding="utf-8")
            return MODULE.summarize(path)

    def test_complete_attribution_groups_only_fixed_fields(self):
        report, complete = self.summarize(
            "WARNING [BI100 4XX] endpoint=chat code=400 "
            "reason=n_exceeds_max_num_seqs messages=1 systems=0 tools=0 "
            "image=0 stream=1 n=2\n"
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n"
            "WARNING [BI100 4XX] endpoint=request_validation code=400 "
            "reason=request_validation_messages messages=0 systems=0 "
            "tools=0 image=0 stream=0 n=unset errors=1\n"
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n"
        )
        self.assertTrue(complete)
        self.assertTrue(report["complete"])
        self.assertEqual(report["chat_4xx_access_count"], 2)
        self.assertEqual(report["attributed_count"], 2)
        self.assertEqual(report["attribution_delta"], 0)
        self.assertEqual(report["by_reason"], {
            "n_exceeds_max_num_seqs": 1,
            "request_validation_messages": 1,
        })
        self.assertEqual(report["request_shapes"][0]["n"], None)
        self.assertEqual(report["request_shapes"][1]["n"], 2)
        serialized = str(report)
        self.assertNotIn("Bad Request", serialized)
        self.assertNotIn("127.0.0.1", serialized)

    def test_access_4xx_without_marker_fails_closed(self):
        report, complete = self.summarize(
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n"
        )
        self.assertFalse(complete)
        self.assertEqual(report["attribution_delta"], 1)

    def test_marker_without_access_4xx_fails_closed(self):
        report, complete = self.summarize(
            "WARNING [BI100 4XX] endpoint=chat code=400 "
            "reason=empty_messages messages=0 systems=0 tools=0 "
            "image=0 stream=0 n=unset\n"
        )
        self.assertFalse(complete)
        self.assertEqual(report["attribution_delta"], -1)

    def test_mismatched_status_code_fails_closed(self):
        report, complete = self.summarize(
            "WARNING [BI100 4XX] endpoint=chat code=422 "
            "reason=empty_messages messages=0 systems=0 tools=0 "
            "image=0 stream=0 n=unset\n"
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n"
        )
        self.assertFalse(complete)
        self.assertEqual(report["by_access_code"], {"400": 1})
        self.assertEqual(report["by_attributed_code"], {"422": 1})

    def test_unknown_reason_is_malformed_without_leaking_value(self):
        report, complete = self.summarize(
            "WARNING [BI100 4XX] endpoint=chat code=400 "
            "reason=private_prompt_failed messages=1 systems=0 tools=0 "
            "image=0 stream=0 n=1\n"
            'INFO: 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" '
            "400 Bad Request\n"
        )
        self.assertFalse(complete)
        self.assertEqual(report["malformed_marker_count"], 1)
        self.assertNotIn("private_prompt_failed", str(report))

    def test_non_chat_4xx_is_out_of_scope(self):
        report, complete = self.summarize(
            'INFO: 127.0.0.1 "POST /v1/completions HTTP/1.1" '
            "400 Bad Request\n"
        )
        self.assertTrue(complete)
        self.assertEqual(report["chat_4xx_access_count"], 0)


if __name__ == "__main__":
    unittest.main()
