from __future__ import annotations

import copy
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "compare_m1_104_admission64_paired_ab.py"
SPEC = importlib.util.spec_from_file_location("m1_104", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def h(value: int) -> str:
    return f"{value:064x}"


def measurement(mode: str, hit: float, weighted: float, output: float = 21.0, ttft: float = 4.0, run: str = "r") -> dict:
    requests = []
    for i in range(18):
        base = {"request_id": f"q{i:02d}", "target_prompt_tokens": 4096 + i, "salt_sha256": h(1000 + i), "output_sha256": h(2000 + i), "first_token_sha256": h(3000 + i), "finish_reason": "stop", "completion_tokens": 8}
        requests.append({**base, "cold": {**base, "ttft_s": 2.0, "output_sha256": h(4000 + i)}, "warm": {**base, "ttft_s": 1.0, "output_sha256": h(4000 + i)}})
    return {"schema": MODULE.MEASUREMENT_SCHEMA, "mode": mode, "run_id": run, "request_count": 18, "request_manifest_sha256": h(777), "target_order": [f"q{i:02d}" for i in range(18)], "qualified_measurement": True, "reasons": [], "aggregate": {"effective_hit_rate": hit, "success_rate": 100.0, "cold_cached_tokens": 0, "output_tps_p10": output, "ttft_p90_s": ttft, "weighted": weighted}, "requests": requests}


def pairs(candidate_hit=54.0, candidate_weighted=104.0, output=21.0, ttft=4.0):
    controls = [measurement("control", 50.0, 100.0, run=f"r{i}") for i in range(3)]
    candidates = [measurement("candidate", candidate_hit, candidate_weighted, output, ttft, run=f"r{i}") for i in range(3)]
    return controls, candidates


class ComparatorTest(unittest.TestCase):
    def test_valid_three_pair_policy_v2(self):
        controls, candidates = pairs()
        result = MODULE.compare(controls, candidates)
        self.assertTrue(result["qualified"], result["reasons"])
        self.assertTrue(result["decision"]["m1_85_full_quality_authorized"])
        self.assertFalse(result["decision"]["default_or_main_authorized"])

    def test_old_overstrict_case_is_requalified_by_or_rule(self):
        controls, candidates = pairs(candidate_hit=52.5, candidate_weighted=104.0)
        result = MODULE.compare(controls, candidates)
        self.assertTrue(result["qualified"], result["reasons"])

    def test_absolute_hit_gate_rejects_low_candidate(self):
        controls, candidates = pairs(candidate_hit=49.9)
        result = MODULE.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("effective hit" in x for x in result["reasons"]))

    def test_quality_and_single_outlier_reject(self):
        controls, candidates = pairs()
        candidates[0]["aggregate"]["output_tps_p10"] = 19.0
        result = MODULE.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("Output TPS" in x for x in result["reasons"]))

    def test_order_salt_and_hash_must_match(self):
        controls, candidates = pairs()
        candidates[0]["requests"][1]["salt_sha256"] = h(9999)
        result = MODULE.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("identity" in x or "salt" in x for x in result["reasons"]))

    def test_failure_is_complete_and_does_not_mutate_inputs(self):
        controls, candidates = pairs()
        candidates[1]["aggregate"]["success_rate"] = 99.0
        original = copy.deepcopy((controls, candidates))
        result = MODULE.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertIn("pairs", result)
        self.assertIn("decision", result)
        self.assertEqual((controls, candidates), original)

    def test_exactly_three_pairs_required(self):
        controls, candidates = pairs()
        result = MODULE.compare(controls[:2], candidates[:2])
        self.assertFalse(result["qualified"])
        self.assertIn("exactly three", result["reasons"][0])


if __name__ == "__main__":
    unittest.main()
