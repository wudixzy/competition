import importlib.util
import json
import math
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/compare_cpu_kv_pressure_ab.py"
SPEC = importlib.util.spec_from_file_location("pressure_ab", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _request(name, expected, cached, elapsed, sha="a" * 64):
    return {
        "name": name,
        "status": "ok",
        "expected_prompt_tokens": expected,
        "summary": {
            "prompt_tokens": expected,
            "cached_tokens": cached,
            "completion_tokens": 8,
            "finish_reason": "stop",
            "message_sha256": sha,
            "elapsed_s": elapsed,
        },
    }


def _report(mode, after, run_id="same"):
    requests = [
        _request("target_cold", 64, 0, 2.0),
        _request("target_immediate_warm", 64, 62, 1.0),
        _request("pressure_cold_0000", 32, 0, 1.5),
        _request("target_after_pressure", 64, after, 4.0),
        _request("target_refreshed", 64, 62, 1.0),
    ]
    return {
        "schema": MODULE.INPUT_SCHEMA,
        "version": MODULE.VERSION,
        "params": {
            "run_id": run_id,
            "mode": mode,
            "json_out": "/tmp/report.json",
            "max_control_cached": 16,
            "min_candidate_cached": 48,
            "target_prompt_tokens": 64,
            "pressure_prompt_tokens": 32,
            "pressure_count": 1,
            "max_tokens": 8,
            "block_size": 16,
            "timeout_s": 900.0,
        },
        "requests": requests,
        "validation": {"qualified": True, "reasons": []},
        "qualified": True,
    }


class CompareCpuKvPressureAbTest(unittest.TestCase):

    def test_passes_and_emits_summary_only(self):
        result = MODULE.compare(_report("control", 8), _report("candidate", 48))
        self.assertTrue(result["qualified"])
        self.assertEqual(result["after_pressure"]["cached_tokens_delta"], 40)
        self.assertEqual(result["after_pressure"]["elapsed_delta_s"], 0.0)
        self.assertEqual(result["after_pressure"]["elapsed_ratio"], 1.0)
        self.assertEqual(len(result["requests"]), 5)
        self.assertNotIn("messages", repr(result))

    def test_run_id_mismatch_fails(self):
        result = MODULE.compare(_report("control", 8, "one"),
                                _report("candidate", 48, "two"))
        self.assertFalse(result["qualified"])
        self.assertTrue(any("params differ" in reason for reason in result["reasons"]))

    def test_hash_mismatch_fails(self):
        candidate = _report("candidate", 48)
        candidate["requests"][1]["summary"]["message_sha256"] = "b" * 64
        result = MODULE.compare(_report("control", 8), candidate)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("message_sha256 differs" in reason
                            for reason in result["reasons"]))

    def test_thresholds_fail_closed(self):
        result = MODULE.compare(_report("control", 20), _report("candidate", 47))
        self.assertFalse(result["qualified"])
        self.assertTrue(any("control target_after_pressure" in reason
                            for reason in result["reasons"]))
        self.assertTrue(any("candidate target_after_pressure" in reason
                            for reason in result["reasons"]))

    def test_nonfinite_elapsed_and_error_request_fail(self):
        control = _report("control", 8)
        control["requests"][0]["summary"]["elapsed_s"] = math.inf
        candidate = _report("candidate", 48)
        candidate["requests"][2]["status"] = "error"
        result = MODULE.compare(control, candidate)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("elapsed_s is not positive finite" in reason
                            for reason in result["reasons"]))
        self.assertTrue(any("is not ok" in reason for reason in result["reasons"]))

    def test_cli_writes_atomically_and_returns_failure(self):
        control = _report("control", 8)
        candidate = _report("candidate", 48)
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            control_path = directory / "control.json"
            candidate_path = directory / "candidate.json"
            out = directory / "out.json"
            control_path.write_text(json.dumps(control), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            self.assertEqual(MODULE.main(["--control", str(control_path),
                                          "--candidate", str(candidate_path),
                                          "--out", str(out)]), 0)
            self.assertTrue(json.loads(out.read_text())["qualified"])


if __name__ == "__main__":
    unittest.main()
