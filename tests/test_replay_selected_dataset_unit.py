import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/replay_selected_dataset.py"
SPEC = importlib.util.spec_from_file_location("selected_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SelectedDatasetReplayTest(unittest.TestCase):

    def test_redacted_report_and_weighted_proxy(self):
        first = MODULE.StreamResult(
            ok=True,
            content="private model output one",
            latency_s=3.0,
            ttft_s=1.0,
            prompt_tokens=100,
            cached_tokens=0,
            completion_tokens=40,
            finish_reason="stop",
        )
        second = MODULE.StreamResult(
            ok=True,
            content="private model output two",
            latency_s=2.0,
            ttft_s=0.5,
            prompt_tokens=160,
            cached_tokens=96,
            completion_tokens=30,
            finish_reason="stop",
        )
        turns = [
            MODULE.redacted_turn(0, 0, 2, first),
            MODULE.redacted_turn(0, 1, 4, second),
        ]
        report = MODULE.summarize(
            "unit",
            pathlib.Path("secret-dataset.json"),
            "a" * 64,
            1,
            2,
            turns,
            5.0,
        )
        encoded = json.dumps(report)
        self.assertNotIn("private model output", encoded)
        self.assertFalse(report["privacy"]["contains_raw_messages"])
        self.assertFalse(report["privacy"]["contains_raw_model_output"])
        self.assertTrue(report["validation"]["complete_replay"])
        self.assertTrue(report["validation"]["all_successful"])
        self.assertAlmostEqual(
            report["aggregate"]["cache_hit_rate"], 96 / 260)
        self.assertGreater(report["aggregate"]["weighted_score_proxy"], 0)

    def test_failure_stays_redacted_and_marks_replay_incomplete(self):
        failure = MODULE.StreamResult(
            ok=False,
            content="",
            latency_s=0.1,
            ttft_s=None,
            prompt_tokens=0,
            cached_tokens=0,
            completion_tokens=0,
            finish_reason=None,
            error_kind="HTTPError:400",
        )
        turns = [MODULE.redacted_turn(0, 0, 1, failure)]
        report = MODULE.summarize(
            "unit",
            pathlib.Path("dataset.json"),
            "b" * 64,
            2,
            3,
            turns,
            0.1,
        )
        self.assertFalse(report["validation"]["complete_replay"])
        self.assertFalse(report["validation"]["all_successful"])
        self.assertEqual(report["validation"]["success_rate"], 0.0)
        self.assertEqual(report["turns"][0]["error_kind"], "HTTPError:400")

    def test_dataset_shape_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "dataset.json"
            path.write_text(json.dumps([{
                "system_prompt": "system",
                "user_questions": ["one", "two"],
            }]))
            data = MODULE.load_dataset(path)
            self.assertEqual(len(data), 1)

            path.write_text(json.dumps({"user_questions": []}))
            with self.assertRaisesRegex(ValueError, "root must be a list"):
                MODULE.load_dataset(path)


if __name__ == "__main__":
    unittest.main()
