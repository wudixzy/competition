from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/compare_bi100_preflights.py"
SPEC = importlib.util.spec_from_file_location("compare_preflights", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _preflight(*, free_delta: int = 0, checksum_delta: float = 0.0) -> dict:
    size = 1024
    total = 64 * 1024 ** 3
    return {
        "schema": MODULE.SOURCE_SCHEMA,
        "version": MODULE.VERSION,
        "gpus": [0, 1, 2, 3],
        "matmul_size": size,
        "timeout_s": 25.0,
        "ok": True,
        "results": [{
            "gpu": gpu,
            "device_name": "BI-V100",
            "device_capability": [7, 0],
            "free": total - (gpu + 1) * 1024 ** 2 + free_delta,
            "total": total,
            "checksum": float(size ** 3) + checksum_delta,
            "stage": "done",
            "ok": True,
            "returncode": 0,
        } for gpu in range(4)],
    }


class CompareBi100PreflightsTest(unittest.TestCase):

    def test_free_memory_may_change_between_service_lifetimes(self):
        report = MODULE.compare([
            ("before_legacy", _preflight()),
            ("after_legacy", _preflight(free_delta=-64 * 1024 ** 2)),
            ("after_candidate", _preflight(free_delta=-32 * 1024 ** 2)),
        ])
        self.assertTrue(report["qualified"], report)

    def test_topology_change_fails_closed(self):
        changed = _preflight()
        changed["results"][2]["total"] -= 1024 ** 3
        report = MODULE.compare([
            ("before_legacy", _preflight()),
            ("after_legacy", changed),
        ])
        self.assertFalse(report["qualified"])
        self.assertTrue(any("differs from the first" in reason
                            for reason in report["reasons"]))

    def test_bad_checksum_fails_closed(self):
        report = MODULE.compare([
            ("before_legacy", _preflight()),
            ("after_legacy", _preflight(checksum_delta=1.0)),
        ])
        self.assertFalse(report["qualified"])
        self.assertTrue(any("checksum" in reason
                            for reason in report["reasons"]))

    def test_duplicate_labels_fail_closed(self):
        report = MODULE.compare([
            ("same", _preflight()),
            ("same", _preflight()),
        ])
        self.assertFalse(report["qualified"])
        self.assertIn(
            "preflight stage labels must be unique", report["reasons"])


if __name__ == "__main__":
    unittest.main()
