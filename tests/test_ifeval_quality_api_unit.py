from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/ifeval_quality_api.py"
SPEC = importlib.util.spec_from_file_location("ifeval_quality_api", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "tests"))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


class IFEvalQualityAPITest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.manifest_sha, cls.rows = MODULE.load_manifest(
            MODULE.DEFAULT_MANIFEST)

    def test_manifest_and_request_semantics_are_exact(self):
        self.assertEqual(
            self.manifest_sha, MODULE.EXPECTED_MANIFEST_SHA256)
        self.assertEqual(len(self.rows), 64)
        payload = MODULE.request_payload(
            self.rows[0]["prompt"], "llm", self.manifest)
        self.assertEqual(payload, {
            "model": "llm",
            "messages": [{
                "role": "user", "content": self.rows[0]["prompt"]}],
            "max_tokens": 4096,
            "temperature": 0,
            "seed": 20260725,
            "stream": False,
        })
        self.assertNotIn("thinking", payload)
        self.assertNotIn("chat_template", payload)

    def test_response_normalization_preserves_quality_fields(self):
        normalized = MODULE.normalize_response({
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": "answer",
                    "reasoning_content": "reasoning",
                },
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }, 1.5)
        self.assertEqual(normalized["content"], "answer")
        self.assertEqual(normalized["reasoning_content"], "reasoning")
        self.assertEqual(normalized["usage"]["cached_tokens"], 0)

    def test_summary_reports_prompt_instruction_id_and_family_counts(self):
        summary = MODULE.summarize([
            {
                "key": 1,
                "instruction_id_list": ["keywords:existence", "stop:end"],
                "strict": [True, False],
                "loose": [True, True],
            },
            {
                "key": 2,
                "instruction_id_list": ["keywords:existence"],
                "strict": [True],
                "loose": [True],
            },
        ])
        self.assertEqual(summary["prompt_total"], 2)
        self.assertEqual(summary["strict_prompt_passed"], 1)
        self.assertEqual(summary["loose_prompt_passed"], 2)
        self.assertEqual(summary["instruction_total"], 3)
        self.assertEqual(
            summary["by_instruction_id"]["keywords:existence"],
            {"total": 2, "strict_passed": 2, "loose_passed": 2},
        )

    def test_checkpoint_is_private_and_bound_to_run(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            path = Path(temporary) / "checkpoint.json"
            MODULE.write_checkpoint(path, "a" * 64, {
                1: {"content": "temporary raw output"}})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            value = MODULE.parse_checkpoint(path, "a" * 64)
            self.assertEqual(value[1]["content"], "temporary raw output")
            with self.assertRaisesRegex(ValueError, "identity"):
                MODULE.parse_checkpoint(path, "b" * 64)


if __name__ == "__main__":
    unittest.main()
