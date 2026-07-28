import hashlib
import json
import pathlib
import sys
import unittest
from unittest.mock import patch
from urllib.error import URLError

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import bench_m1_104_admission64_policy_matrix as m104


def record(target, pair, phase, cached=0, prompt=None, output="same"):
    return {"target_tokens": target, "pair": pair, "phase": phase,
            "cached_tokens": cached, "prompt_tokens": prompt or target,
            "ok": True, "output_sha256": output}


class M1_104UnitTest(unittest.TestCase):
    def test_tools_are_fixed_and_complete(self):
        tools = m104.make_tools()
        self.assertEqual(len(tools), 29)
        self.assertEqual(tools[0]["function"]["name"], "read_file_0")
        self.assertFalse(tools[-1]["function"]["parameters"]["additionalProperties"])

    def test_normalized_output_is_stable_and_separated(self):
        body = {"choices": [{"message": {"content": "a", "reasoning_content": "b",
                                           "tool_calls": [{"id": "x", "function": {"name": "f"}}]}}]}
        result = m104.normalized_output(body)
        self.assertEqual(result["content_sha256"], hashlib.sha256(b"a").hexdigest())
        self.assertEqual(result["reasoning_sha256"], hashlib.sha256(b"b").hexdigest())
        self.assertEqual(len(result["sha256"]), 64)
        self.assertEqual(result, m104.normalized_output(json.loads(json.dumps(body))))

    def test_validate_accepts_complete_cold_warm_matrix(self):
        rows = []
        for target in m104.SHAPES:
            for pair in range(m104.PAIR_COUNT):
                rows += [record(target, pair, "cold"), record(target, pair, "warm", cached=target - 1)]
        m104.validate_report({"requests": rows, "service_healthy": True})

    def test_validate_rejects_missing_request(self):
        rows = [record(target, pair, phase) for target in m104.SHAPES for pair in range(m104.PAIR_COUNT)
                for phase in ("cold", "warm")]
        with self.assertRaisesRegex(ValueError, "expected 18"):
            m104.validate_report({"requests": rows[:-1], "service_healthy": True})

    def test_validate_rejects_nonzero_cold_cache(self):
        rows = [record(target, pair, phase, cached=1 if phase == "cold" else 2)
                for target in m104.SHAPES for pair in range(m104.PAIR_COUNT)
                for phase in ("cold", "warm")]
        with self.assertRaisesRegex(ValueError, "cold cache"):
            m104.validate_report({"requests": rows, "service_healthy": True})

    def test_validate_rejects_hash_or_token_mismatch(self):
        rows = [record(target, pair, phase, output=("cold" if phase == "cold" else "warm"))
                for target in m104.SHAPES for pair in range(m104.PAIR_COUNT)
                for phase in ("cold", "warm")]
        with self.assertRaisesRegex(ValueError, "output mismatch"):
            m104.validate_report({"requests": rows, "service_healthy": True})
        rows[0]["output_sha256"] = rows[1]["output_sha256"] = "same"
        rows[0]["prompt_tokens"] = m104.SHAPES[0] + m104.TOKEN_ERROR_LIMIT
        with self.assertRaisesRegex(ValueError, "target mismatch"):
            m104.validate_report({"requests": rows, "service_healthy": True})

    def test_validate_rejects_unhealthy_service(self):
        rows = [record(target, pair, phase) for target in m104.SHAPES for pair in range(m104.PAIR_COUNT)
                for phase in ("cold", "warm")]
        with self.assertRaisesRegex(ValueError, "health"):
            m104.validate_report({"requests": rows, "service_healthy": False})

    def test_health_check_handles_unreachable_service(self):
        with patch("urllib.request.urlopen", side_effect=URLError("offline")):
            self.assertFalse(m104.service_health("http://unit.test", 1))
