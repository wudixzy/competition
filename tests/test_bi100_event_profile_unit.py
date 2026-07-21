from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "qwen3_6_scripts" / "bi100_profile.py"


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, template, *args):
        self.messages.append(template % args)


class _Event:
    def __init__(self, enable_timing=False):
        self.recorded = False

    def record(self):
        self.recorded = True

    def elapsed_time(self, other):
        assert self.recorded and other.recorded
        return 2.5


def _load_profile(environment):
    logger = _Logger()
    torch_module = types.SimpleNamespace(cuda=types.SimpleNamespace(
        Event=_Event, synchronize=lambda: None))
    vllm_module = types.ModuleType("vllm")
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: logger
    spec = importlib.util.spec_from_file_location(
        f"bi100_profile_under_test_{id(environment)}", PROFILE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    modules = {
        "torch": torch_module,
        "vllm": vllm_module,
        "vllm.logger": logger_module,
    }
    with patch.dict(os.environ, environment, clear=False), patch.dict(
            sys.modules, modules):
        spec.loader.exec_module(module)
    return module, logger, modules


class Bi100EventProfileTest(unittest.TestCase):
    def test_event_mode_filters_counts_and_aggregates(self):
        module, logger, modules = _load_profile({
            "BI100_PROFILE": "1",
            "BI100_PROFILE_MODE": "event",
            "BI100_PROFILE_FILTER": "model.*,paged_attn.*",
        })
        with patch.dict(sys.modules, modules), patch(
                "time.monotonic_ns", side_effect=(1_000_000, 5_000_000)):
            with module.bi100_timer("model.forward"):
                module.bi100_profile_count(
                    "paged_attn.prefix_dispatch", path="pytorch",
                    query_len=16, context_len=65520)
                module.bi100_profile_count(
                    "paged_attn.prefix_dispatch", path="pytorch",
                    query_len=16, context_len=65520)
            with module.bi100_timer("moe.routed"):
                pass
            payload = module.bi100_profile_flush(
                tp_rank=2,
                phase="prefill", prefill_tokens=16, decode_tokens=0,
                context_len=65520)

        self.assertEqual(payload["schema"], "bi100-profile-event-v1")
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["tp_rank"], 2)
        self.assertEqual(payload["event_count"], 1)
        self.assertEqual(payload["model_forward_event_count"], 1)
        self.assertEqual(payload["regions"]["model.forward"]["total_ms"], 2.5)
        self.assertEqual(payload["counters"][0]["count"], 2)
        self.assertEqual(payload["host_model_start_to_flush_ms"], 4.0)
        self.assertEqual(len(logger.messages), 1)
        encoded = logger.messages[0].split("] ", 1)[1]
        self.assertEqual(json.loads(encoded), payload)

    def test_counter_rejects_non_scalar_metadata(self):
        module, _, _ = _load_profile({
            "BI100_PROFILE": "1",
            "BI100_PROFILE_MODE": "event",
        })
        with self.assertRaises(TypeError):
            module.bi100_profile_count("path", token_ids=[1, 2, 3])

    def test_flush_rejects_unbounded_metadata_and_invalid_rank(self):
        module, _, modules = _load_profile({
            "BI100_PROFILE": "1",
            "BI100_PROFILE_MODE": "event",
        })
        with patch.dict(sys.modules, modules):
            with module.bi100_timer("model.forward"):
                pass
            with self.assertRaises(TypeError):
                module.bi100_profile_flush(
                    tp_rank=0, raw_tokens=[1, 2], phase="prefill")
            with self.assertRaises(TypeError):
                module.bi100_profile_flush(tp_rank=True, phase="prefill")

    def test_invalid_enabled_mode_fails_at_import(self):
        with self.assertRaises(RuntimeError):
            _load_profile({
                "BI100_PROFILE": "1",
                "BI100_PROFILE_MODE": "invalid",
            })

    def test_invalid_boolean_or_filter_fails_at_import(self):
        with self.assertRaises(RuntimeError):
            _load_profile({
                "BI100_PROFILE": "yes",
            })
        with self.assertRaises(RuntimeError):
            _load_profile({
                "BI100_PROFILE": "1",
                "BI100_PROFILE_MODE": "event",
                "BI100_PROFILE_FILTER": "model.*,raw token*",
            })

    def test_failed_forward_cannot_contaminate_the_next_transaction(self):
        module, logger, modules = _load_profile({
            "BI100_PROFILE": "1",
            "BI100_PROFILE_MODE": "event",
        })

        @module.bi100_profile_transaction
        def run_forward(fail):
            with module.bi100_timer("model.forward"):
                module.bi100_profile_count("model.path", path="test")
                if fail:
                    raise ValueError("expected failure")
            return module.bi100_profile_flush(
                tp_rank=0,
                phase="prefill",
                prefill_tokens=16,
                decode_tokens=0,
                context_len=0,
            )

        with patch.dict(sys.modules, modules), self.assertRaises(ValueError):
            run_forward(True)
        with patch.dict(sys.modules, modules):
            payload = run_forward(False)

        self.assertEqual(payload["event_count"], 1)
        self.assertEqual(payload["model_forward_event_count"], 1)
        self.assertEqual(payload["counters"][0]["count"], 1)
        self.assertEqual(len(logger.messages), 1)

    def test_successful_transaction_without_flush_fails_closed(self):
        module, _, modules = _load_profile({
            "BI100_PROFILE": "1",
            "BI100_PROFILE_MODE": "event",
        })

        @module.bi100_profile_transaction
        def run_forward():
            with module.bi100_timer("model.forward"):
                pass

        with patch.dict(sys.modules, modules), self.assertRaisesRegex(
                RuntimeError, "without a flush"):
            run_forward()


if __name__ == "__main__":
    unittest.main()
