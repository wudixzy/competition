from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
REPLAY_PATH = ROOT / "scripts" / "replay_selected_dataset.py"
REPLAY_SPEC = importlib.util.spec_from_file_location(
    "selected_replay_for_qualification", REPLAY_PATH)
REPLAY = importlib.util.module_from_spec(REPLAY_SPEC)
assert REPLAY_SPEC.loader is not None
sys.modules[REPLAY_SPEC.name] = REPLAY
REPLAY_SPEC.loader.exec_module(REPLAY)

TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
import qualify_selected_dataset_replay as qualifier


def _source() -> dict:
    turns = []
    for conversation, count in enumerate(qualifier.EXPECTED_CONVERSATION_TURNS):
        for turn in range(count):
            result = REPLAY.StreamResult(
                ok=True,
                content=f"private-{conversation}-{turn}",
                latency_s=2.0,
                ttft_s=1.0,
                prompt_tokens=100 + turn * 20,
                cached_tokens=turn * 16,
                completion_tokens=16,
                finish_reason="length",
            )
            turns.append(REPLAY.redacted_turn(
                conversation, turn, 2 + 2 * turn, result))
    return REPLAY.summarize(
        "unit-selected",
        pathlib.Path("chat_dataset_v0.json"),
        qualifier.EXPECTED_DATASET_SHA256,
        4,
        13,
        turns,
        26.1,
    )


class SelectedDatasetQualificationTest(unittest.TestCase):

    def test_fixed_redacted_replay_qualifies(self):
        source = _source()
        source["turns"][0]["unknown_raw_field"] = "must not survive"
        report = qualifier.qualify(source)
        self.assertTrue(report["qualified"], report)
        self.assertNotIn("unknown_raw_field", report["turns"][0])
        self.assertEqual(
            report["scope"],
            "selected-13-turn-supplemental-not-official-score",
        )

    def test_dataset_drift_fails_closed(self):
        source = _source()
        source["dataset"]["sha256"] = "b" * 64
        report = qualifier.qualify(source)
        self.assertFalse(report["qualified"], report)
        self.assertIn(
            "dataset contract differs from the frozen selection",
            report["reasons"],
        )

    def test_failed_turn_fails_closed(self):
        source = _source()
        source["turns"][5]["ok"] = False
        report = qualifier.qualify(source)
        self.assertFalse(report["qualified"], report)
        self.assertIn("turn 5 request failed", report["reasons"])

    def test_metric_tampering_fails_closed(self):
        source = _source()
        source["aggregate"]["weighted_score_proxy"] += 1
        report = qualifier.qualify(source)
        self.assertFalse(report["qualified"], report)
        self.assertIn(
            "aggregate weighted_score_proxy is inconsistent",
            report["reasons"],
        )

    def test_nonfinite_metric_fails_closed(self):
        source = copy.deepcopy(_source())
        source["turns"][0]["ttft_s"] = float("nan")
        report = qualifier.qualify(source)
        self.assertFalse(report["qualified"], report)
        self.assertIn("turn 0 TTFT is invalid", report["reasons"])


if __name__ == "__main__":
    unittest.main()
