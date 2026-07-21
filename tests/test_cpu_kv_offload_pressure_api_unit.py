import argparse
import math
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import cpu_kv_offload_pressure_api as pressure_api


class PressureApiArgumentTest(unittest.TestCase):

    def _base_args(self, target: int, pressure: int) -> list[str]:
        json_out = tempfile.NamedTemporaryFile(delete=False).name
        return [
            "--target-prompt-tokens",
            str(target),
            "--pressure-prompt-tokens",
            str(pressure),
            "--pressure-count",
            "1",
            "--json-out",
            json_out,
        ]

    def test_default_min_candidate_cached_is_target_minus_32(self):
        args = pressure_api.parse_args(self._base_args(70000, 64))
        self.assertEqual(args.min_candidate_cached, 70000 - 32)

    def test_pressure_count_must_be_positive(self):
        with self.assertRaises(SystemExit):
            pressure_api.parse_args([
                "--target-prompt-tokens",
                "64",
                "--pressure-prompt-tokens",
                "64",
                "--pressure-count",
                "0",
                "--json-out",
                tempfile.NamedTemporaryFile(delete=False).name,
            ])

    def test_prompt_tokens_plus_max_must_fit_window(self):
        with self.assertRaises(SystemExit):
            pressure_api.parse_args([
                "--target-prompt-tokens",
                "262140",
                "--pressure-prompt-tokens",
                "64",
                "--max-tokens",
                "8",
                "--pressure-count",
                "1",
                "--json-out",
                tempfile.NamedTemporaryFile(delete=False).name,
            ])

    def test_thresholds_and_timeout_must_be_bounded(self):
        for extra in (
            ["--min-candidate-cached", "65"],
            ["--max-control-cached", "65"],
            ["--timeout-s", "0"],
            ["--timeout-s", "nan"],
        ):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                pressure_api.parse_args(self._base_args(64, 64) + extra)


class PressureApiValidationTest(unittest.TestCase):

    def _args(self, mode: str, target: int = 64) -> argparse.Namespace:
        return pressure_api.parse_args([
            "--target-prompt-tokens",
            str(target),
            "--pressure-prompt-tokens",
            "64",
            "--pressure-count",
            "1",
            "--mode",
            mode,
            "--max-tokens",
            "1",
            "--json-out",
            tempfile.NamedTemporaryFile(delete=False).name,
        ])

    def _request(self, name: str, expected: int, summary: dict[str, object] | None,
                 status: str = "ok") -> dict[str, object]:
        request: dict[str, object] = {
            "name": name,
            "status": status,
            "expected_prompt_tokens": expected,
        }
        if summary is not None:
            request["summary"] = summary
        if status != "ok":
            request["error"] = "synthetic"
        return request

    def _summary(self, prompt_tokens: int, cached_tokens: int, completion_tokens: int,
                 finish_reason: str, sha: str = "same", elapsed: float = 1.0
                 ) -> dict[str, object]:
        return {
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "message_sha256": sha,
            "elapsed_s": elapsed,
        }

    def test_target_responses_must_be_consistent(self):
        args = self._args("control")
        requests = [
            self._request("target_cold", 64,
                          self._summary(64, 0, 8, "stop")),
            self._request("target_immediate_warm", 64,
                          self._summary(64, 62, 8, "stop")),
            self._request("pressure_cold_0000", 64,
                          self._summary(64, 0, 8, "stop")),
            self._request("target_after_pressure", 64,
                          self._summary(64, 16, 8, "stop")),
            self._request("target_refreshed", 64,
                          self._summary(64, 62, 8, "stop")),
        ]
        validation = pressure_api.evaluate_validation(args, requests)
        self.assertTrue(validation["qualified"])

        requests[4]["summary"]["message_sha256"] = "different"
        validation = pressure_api.evaluate_validation(args, requests)
        self.assertFalse(validation["qualified"])
        self.assertTrue(any("message" in reason for reason in validation["reasons"]))

        requests[0]["summary"]["cached_tokens"] = 1
        validation = pressure_api.evaluate_validation(args, requests)
        self.assertFalse(validation["qualified"])
        self.assertTrue(any("zero cached" in reason
                            for reason in validation["reasons"]))

    def test_control_and_candidate_thresholds(self):
        control_args = self._args("control")
        control_requests = [
            self._request("target_cold", 64,
                          self._summary(64, 0, 8, "stop")),
            self._request("target_immediate_warm", 64,
                          self._summary(64, 62, 8, "stop")),
            self._request("pressure_cold_0000", 64,
                          self._summary(64, 0, 8, "stop")),
            self._request("target_after_pressure", 64,
                          self._summary(64, 20, 8, "stop")),
            self._request("target_refreshed", 64,
                          self._summary(64, 62, 8, "stop")),
        ]
        control_validation = pressure_api.evaluate_validation(
            control_args, control_requests)
        self.assertFalse(control_validation["qualified"])
        self.assertTrue(any("control threshold" in reason
                            for reason in control_validation["reasons"]))

        control_args.max_control_cached = 24
        control_validation = pressure_api.evaluate_validation(
            control_args, control_requests)
        self.assertTrue(control_validation["qualified"])

        candidate_args = self._args("candidate", target=128)
        candidate_args.min_candidate_cached = 96
        candidate_args.max_control_cached = 16
        candidate_requests = [
            self._request("target_cold", 128,
                          self._summary(128, 0, 8, "stop")),
            self._request("target_immediate_warm", 128,
                          self._summary(128, 126, 8, "stop")),
            self._request("pressure_cold_0000", 64,
                          self._summary(64, 0, 8, "stop")),
            self._request("target_after_pressure", 128,
                          self._summary(128, 80, 8, "stop")),
            self._request("target_refreshed", 128,
                          self._summary(128, 126, 8, "stop")),
        ]
        fail_validation = pressure_api.evaluate_validation(
            candidate_args, candidate_requests)
        self.assertFalse(fail_validation["qualified"])
        self.assertTrue(any("below" in reason
                            for reason in fail_validation["reasons"]))

        candidate_requests[3]["summary"]["cached_tokens"] = 96
        pass_validation = pressure_api.evaluate_validation(
            candidate_args, candidate_requests)
        self.assertTrue(pass_validation["qualified"])

    def test_error_and_non_finite_or_imprecise_summaries_fail(self):
        args = self._args("control")
        valid_summary = self._summary(64, 32, 8, "stop")
        cold_summary = self._summary(64, 0, 8, "stop")
        non_finite = self._summary(64, 32, 8, "stop", elapsed=math.inf)

        requests = [
            self._request("target_cold", 64, cold_summary),
            self._request("target_immediate_warm", 64, valid_summary),
            self._request("pressure_cold_0000", 64, valid_summary),
            self._request("target_after_pressure", 64,
                          self._summary(64, 8, 8, "stop")),
            self._request("target_refreshed", 64, valid_summary),
        ]
        self.assertTrue(pressure_api.evaluate_validation(args, requests)["qualified"])

        requests[1]["summary"] = non_finite
        self.assertFalse(
            pressure_api.evaluate_validation(args, requests)["qualified"])

        # imprecise prompt token count for target should also fail
        requests = [
            self._request("target_cold", 64,
                          self._summary(63, 0, 8, "stop")),
            self._request("target_immediate_warm", 64, valid_summary),
            self._request("pressure_cold_0000", 64, valid_summary),
            self._request("target_after_pressure", 64, valid_summary),
            self._request("target_refreshed", 64, valid_summary),
        ]
        bad_validation = pressure_api.evaluate_validation(args, requests)
        self.assertFalse(bad_validation["qualified"])
        self.assertTrue(any("mismatch" in reason for reason in bad_validation["reasons"]))

        requests = [
            self._request("target_cold", 64,
                          self._summary(64, 0, 0, "length")),
            self._request("target_immediate_warm", 64,
                          self._summary(64, 62, 0, "length")),
            self._request("pressure_cold_0000", 64, valid_summary),
            self._request("target_after_pressure", 64,
                          self._summary(64, 8, 0, "length")),
            self._request("target_refreshed", 64,
                          self._summary(64, 62, 0, "length")),
        ]
        no_completion = pressure_api.evaluate_validation(args, requests)
        self.assertFalse(no_completion["qualified"])
        self.assertTrue(any("no completion" in reason
                            for reason in no_completion["reasons"]))


if __name__ == "__main__":
    unittest.main()
