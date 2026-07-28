from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tests import bench_m1_104_admission64_policy_matrix as measurement
from tests import compare_m1_104_admission64_paired_ab as module


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def report(
    mode: str,
    *,
    ab_pair: int = 1,
    cold_fraction: float,
    warm_fraction: float,
    output_tps: float = 21.0,
    ttft: float = 4.0,
) -> dict:
    policy = "fine32" if mode == "control" else "admission64"
    decode_window = 64.0 / output_tps
    requests = []
    contracts = []
    for target in measurement.SHAPES:
        for pair in measurement.PAIRS:
            for phase in measurement.PHASES:
                request_id = f"{target}_pair{pair}_{phase}"
                output_id = f"{target}_pair{pair}"
                if phase == "cold":
                    cached = (
                        0 if not requests else int(target * cold_fraction))
                else:
                    cached = int(target * warm_fraction)
                row = {
                    "request_id": request_id,
                    "target_prompt_tokens": target,
                    "pair": pair,
                    "phase": phase,
                    "salt_sha256": digest(f"{target}:{pair}"),
                    "rendered_tokens_local": target,
                    "seed": measurement.SEED,
                    "ok": True,
                    "http_status": 200,
                    "done_seen": True,
                    "terminal_choice_seen": True,
                    "usage_seen": True,
                    "data_event_count": 4,
                    "malformed_sse_count": 0,
                    "health_after": True,
                    "prompt_tokens": target,
                    "cached_tokens": cached,
                    "completion_tokens": 64,
                    "finish_reason": "length",
                    "ttft_s": ttft,
                    "latency_s": ttft + decode_window,
                    "decode_window_s": decode_window,
                    "output_tps": output_tps,
                    "first_token_sha256": digest(f"first:{output_id}"),
                    "output_sha256": digest(f"output:{output_id}"),
                    "content_sha256": digest(f"content:{output_id}"),
                    "reasoning_sha256": digest(
                        f"reasoning:{output_id}"),
                    "tool_calls_sha256": measurement._sha256_json([]),
                    "tool_call_delta_count": 0,
                }
                requests.append(row)
                contracts.append(measurement.request_contract(row))
    return {
        "schema": measurement.SCHEMA,
        "version": measurement.VERSION,
        "mode": mode,
        "policy": policy,
        "ab_pair": ab_pair,
        "request_count": measurement.REQUEST_COUNT,
        "request_manifest_sha256": measurement._sha256_json(contracts),
        "target_order": [row["request_id"] for row in requests],
        "fixed": {
            "shapes": list(measurement.SHAPES),
            "pairs": list(measurement.PAIRS),
            "phases": list(measurement.PHASES),
            "seed": measurement.SEED,
            "tool_count": measurement.TOOL_COUNT,
            "max_tokens": measurement.MAX_TOKENS,
            "temperature": 0,
            "thinking": False,
            "tool_choice": "none",
            "stream_usage": True,
            "salt_namespace_sha256": digest("m1-104-fixed"),
            "corpus": [{"name": "fixed.txt", "sha256": digest("corpus")}],
        },
        "aggregate": measurement.aggregate(requests),
        "qualified_measurement": True,
        "reasons": [],
        "requests": requests,
        "privacy": {
            "contains_raw_prompt": False,
            "contains_raw_output": False,
            "contains_tools": False,
            "contains_credentials": False,
        },
    }


def pairs(
    *,
    control_cold: float = 0.10,
    control_warm: float = 0.80,
    candidate_cold: float = 0.40,
    candidate_warm: float = 0.95,
    candidate_output: float = 21.0,
    candidate_ttft: float = 4.0,
) -> tuple[list[dict], list[dict]]:
    controls = [
        report(
            "control",
            ab_pair=ab_pair,
            cold_fraction=control_cold,
            warm_fraction=control_warm,
        )
        for ab_pair in range(1, 4)
    ]
    candidates = [
        report(
            "candidate",
            ab_pair=ab_pair,
            cold_fraction=candidate_cold,
            warm_fraction=candidate_warm,
            output_tps=candidate_output,
            ttft=candidate_ttft,
        )
        for ab_pair in range(1, 4)
    ]
    return controls, candidates


class M1104Admission64PairedAbUnitTest(unittest.TestCase):

    def test_valid_three_pair_policy_v2_qualifies(self):
        controls, candidates = pairs()
        result = module.compare(controls, candidates)
        self.assertTrue(result["qualified"], result["reasons"])
        self.assertTrue(
            result["decision"]["m1_85_full_quality_authorized"])
        self.assertFalse(
            result["decision"]["default_policy_change_authorized"])

    def test_weighted_or_path_qualifies_without_hit_reduction(self):
        controls, candidates = pairs(
            control_cold=0.25,
            control_warm=0.95,
            candidate_cold=0.25,
            candidate_warm=0.95,
            candidate_ttft=3.8,
        )
        result = module.compare(controls, candidates)
        self.assertTrue(result["qualified"], result["reasons"])
        self.assertTrue(result["summary"]["median_benefit_paths"][
            "weighted_gain_at_least_3pct_without_hit_reduction"])

    def test_absolute_hit_and_output_floors_are_hard(self):
        controls, candidates = pairs(
            candidate_cold=0.0,
            candidate_warm=0.90,
            candidate_output=19.99,
        )
        result = module.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("below 50%" in reason
                            for reason in result["reasons"]))
        self.assertTrue(any("below 20" in reason
                            for reason in result["reasons"]))

    def test_single_and_median_regressions_reject(self):
        controls, candidates = pairs(
            candidate_output=19.9,
            candidate_ttft=4.21,
        )
        result = module.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("Output TPS regression" in reason
                            for reason in result["reasons"]))
        self.assertTrue(any("TTFT P90 regression" in reason
                            for reason in result["reasons"]))

    def test_two_of_three_pairs_must_pass_benefit_path(self):
        controls, candidates = pairs(
            control_cold=0.25,
            control_warm=0.80,
        )
        candidates[1] = report(
            "candidate", ab_pair=2,
            cold_fraction=0.25, warm_fraction=0.80)
        candidates[2] = report(
            "candidate", ab_pair=3,
            cold_fraction=0.25, warm_fraction=0.80)
        result = module.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "fewer than two pairs pass a policy-v2 benefit path",
            result["reasons"],
        )

    def test_request_salt_or_output_change_rejects(self):
        controls, candidates = pairs()
        for index in (0, 1):
            candidates[0]["requests"][index]["salt_sha256"] = digest(
                "different")
        candidates[0]["request_manifest_sha256"] = measurement._sha256_json([
            measurement.request_contract(row)
            for row in candidates[0]["requests"]
        ])
        for index in (0, 1):
            candidates[1]["requests"][index]["output_sha256"] = digest(
                "different")
        result = module.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "manifest" in reason or "workload" in reason
            for reason in result["reasons"]))
        self.assertTrue(any("output differs" in reason
                            for reason in result["reasons"]))

    def test_unbound_aggregate_is_rejected(self):
        controls, candidates = pairs()
        candidates[0]["aggregate"]["weighted"] += 1.0
        result = module.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("aggregate.weighted is not bound" in reason
                            for reason in result["reasons"]))

    def test_invalid_measurement_is_complete_negative_evidence(self):
        controls, candidates = pairs()
        candidates[0]["qualified_measurement"] = False
        candidates[0]["reasons"] = ["failed"]
        original = copy.deepcopy((controls, candidates))
        result = module.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertEqual((controls, candidates), original)
        self.assertIn("pairs", result)
        self.assertIn("decision", result)

    def test_exactly_three_pairs_required(self):
        controls, candidates = pairs()
        result = module.compare(controls[:2], candidates[:2])
        self.assertFalse(result["qualified"])
        self.assertIn(
            "exactly three control/candidate pairs are required",
            result["reasons"],
        )

    def test_outer_ab_pair_identity_is_bound(self):
        controls, candidates = pairs()
        candidates[1]["ab_pair"] = 1
        result = module.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("measurement contract differs" in reason
                            for reason in result["reasons"]))

    def test_cli_rejects_reused_report_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "report.json"
            output = root / "comparison.json"
            source.write_text(
                json.dumps(report(
                    "control",
                    ab_pair=1,
                    cold_fraction=0.1,
                    warm_fraction=0.8,
                )) + "\n",
                encoding="utf-8",
            )
            args = []
            for mode in ("control", "candidate"):
                for _ in range(3):
                    args.extend((f"--{mode}", str(source)))
            args.extend(("--out", str(output)))
            with redirect_stdout(io.StringIO()):
                rc = module.main(args)
            self.assertEqual(rc, 1)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(value["qualified"])
            self.assertTrue(any("paths must be unique" in reason
                                for reason in value["reasons"]))


if __name__ == "__main__":
    unittest.main()
