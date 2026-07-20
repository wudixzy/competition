import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/summarize_dataset_shaped_matrix.py"
SPEC = importlib.util.spec_from_file_location("matrix_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DatasetMatrixSummaryTest(unittest.TestCase):

    def test_complete_matrix_and_weighted_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            requests = root / "requests"
            requests.mkdir()
            for target in (4096, 7800, 16000):
                for pair in (1, 2, 3):
                    for phase in ("cold", "warm"):
                        cached = target - 16 if phase == "warm" else 0
                        payload = {
                            "timing": {
                                "ok": True,
                                "prompt_tokens": target,
                                "cached_tokens": cached,
                                "completion_tokens": 64,
                                "ttft_s": 2.0 if phase == "cold" else 1.0,
                                "latency_s": 4.0,
                                "output_tps_decode": 21.0,
                            }
                        }
                        path = requests / f"{target}_pair{pair}_{phase}.json"
                        path.write_text(json.dumps(payload))
            report = MODULE.summarize(root)
            self.assertTrue(report["validation"]["complete_matrix"])
            self.assertEqual(report["validation"]["success_rate"], 1.0)
            self.assertEqual(report["aggregate"]["output_tps_p10"], 21.0)
            aggregate = report["aggregate"]
            expected = (
                21.0 * 16.796
                + aggregate["input_tps_aggregate"] * 2.799
                + aggregate["cache_tps_aggregate"] * 0.56)
            self.assertAlmostEqual(aggregate["weighted_score"], expected)

    def test_missing_request_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "requests").mkdir()
            self.assertFalse(
                MODULE.summarize(root)["validation"]["complete_matrix"])


if __name__ == "__main__":
    unittest.main()
