from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/gdn_action_broadcast_gate.py"
SPEC = importlib.util.spec_from_file_location("gdn_action_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _Input:

    def __init__(self, gdn_restore_key=None, gdn_capture_points=None,
                 gdn_evict_keys=None, **kwargs):
        self.gdn_restore_key = gdn_restore_key
        self.gdn_capture_points = gdn_capture_points
        self.gdn_evict_keys = gdn_evict_keys

    def as_broadcastable_tensor_dict(self):
        return {
            "gdn_restore_key": self.gdn_restore_key,
            "gdn_capture_points": self.gdn_capture_points,
            "gdn_evict_keys": self.gdn_evict_keys,
        }

    @classmethod
    def from_broadcasted_tensor_dict(cls, payload):
        return cls(**payload)


class _DroppingInput(_Input):

    def as_broadcastable_tensor_dict(self):
        payload = super().as_broadcastable_tensor_dict()
        payload.pop("gdn_restore_key")
        return payload


def _model_source(directory: str, *, fail_fast: bool = True) -> Path:
    path = Path(directory) / "qwen3_5.py"
    text = "\n".join((
        "saved_state = self._gdn_prefix_cache.get(restore_key)",
        "if saved_state is None:",
        "    raise RuntimeError('scheduler requested a missing GDN prefix state')",
    )) if fail_fast else "saved_state = None\n"
    path.write_text(text, encoding="utf-8")
    return path


class GdnActionBroadcastGateUnitTest(unittest.TestCase):

    def test_four_rank_reconstruction_and_fail_fast_qualify(self):
        with tempfile.TemporaryDirectory() as directory:
            report = MODULE.build_report(
                _Input, _Input, _model_source(directory))
        self.assertTrue(report["qualified"])
        self.assertEqual(len(report["cases"]), 2)
        self.assertTrue(all(
            case["rank_reconstruction_count"] == 4
            for case in report["cases"]))

    def test_missing_action_field_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            report = MODULE.build_report(
                _DroppingInput, _Input, _model_source(directory))
        self.assertFalse(report["qualified"])
        self.assertTrue(any(
            "fields_present" in reason for reason in report["reasons"]))

    def test_missing_restore_guard_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            report = MODULE.build_report(
                _Input, _Input, _model_source(directory, fail_fast=False))
        self.assertFalse(report["qualified"])
        self.assertIn(
            "model source lacks missing-restore fail-fast", report["reasons"])


if __name__ == "__main__":
    unittest.main()
