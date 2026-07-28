#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import qwen36_diagnostic_api as api  # noqa: E402


class Qwen36DiagnosticApiUnitTest(unittest.TestCase):
    def test_response_summary_is_privacy_safe(self) -> None:
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "private generated text",
                    "reasoning_content": "private reasoning",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 8},
            },
        }
        summary = api._response_summary(response)
        encoded = json.dumps(summary)
        self.assertNotIn("private generated text", encoded)
        self.assertNotIn("private reasoning", encoded)
        self.assertEqual(summary["cached_tokens"], 8)
        self.assertEqual(summary["completion_tokens"], 3)
        self.assertEqual(len(summary["message_sha256"]), 64)

    def test_png_fixture_is_embedded_and_valid(self) -> None:
        value = api._solid_png_data_url((255, 0, 0))
        prefix = "data:image/png;base64,"
        self.assertTrue(value.startswith(prefix))
        decoded = base64.b64decode(value[len(prefix):])
        self.assertTrue(decoded.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(decoded.endswith(b"IEND\xaeB`\x82"))

    def test_non_200_response_uses_digest_in_error(self) -> None:
        with mock.patch.object(
                api, "_request_json", return_value=(500, {"secret": "body"})):
            with self.assertRaisesRegex(
                    AssertionError, r"response_sha256=[0-9a-f]{64}"):
                api._post_chat(
                    "http://127.0.0.1:8000",
                    {"model": "llm"},
                    timeout_s=1,
                )

    def test_response_requires_generated_tokens(self) -> None:
        response = {
            "choices": [{
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0},
        }
        with self.assertRaisesRegex(
                AssertionError, "generated no completion tokens"):
            api._response_summary(response)


class Qwen36DiagnosticHarnessStaticTest(unittest.TestCase):
    def test_harness_is_diagnostic_only_and_keeps_capacity(self) -> None:
        harness = (
            ROOT / "scripts" / "run_qwen36_diagnostic_gate.sh"
        ).read_text(encoding="utf-8")
        for marker in (
            "umask 077",
            'if [[ "$RUN_ROOT" != /tmp/* ]]',
            "--max-model-len 262144",
            "verify_qwen36_diagnostic_checkpoint.py",
            "verify_bare_host_runtime_identity.py",
            "runtime_overlay_identity.json",
            "runtime_tree_sha256",
            "BI100_DIAGNOSTIC_LAYER_TRACE=1",
            "BI100_HYBRID_KV_ACCOUNTING=full_attention",
            "prefix_boundary_api.py",
            "qwen36_diagnostic_api.py",
            "qwen36_quality_contract_diagnostic.py",
            "quality_contract_gate.json",
            '"quality_contract": read_rc("quality_contract_gate.rc")',
            '"n_cross_case_contract": quality_contract.get(',
            "qwen36_compat_http_gate.py",
            "--multiple-system-parts-expected-status 200",
            "--image-limit 1",
            "compat_http_gate.json",
            '"compat_http": read_rc("compat_http_gate.rc")',
            "qwen36_tool_http_gate.py",
            "--strict-false-expected-status 200",
            "--object-history-expected-status 200",
            "tool_http_gate.json",
            '"tool_http": read_rc("tool_http_gate.rc")',
            '"streaming_contract_qualified": (',
            '"streaming_equivalence_qualified": (',
            "bi100_stop_process_group",
            '"$ACTIVE_PGID" "$ACTIVE_PID" 60 20 \\\n'
            '            "$ACTIVE_STARTTIME" "$ACTIVE_SESSION_TOKEN"',
            "ACTIVE_SESSION_TOKEN",
            "--kill-after=90s 240s",
            "service_postflight_gate.py",
            "compare_bi100_preflights.py",
            '--expected-gpus "$GPU_LIST"',
            "--max-free-memory-drop-bytes 1073741824",
            "preflight_comparison.json",
            '"preflight_comparison": read_rc(',
            "service_contract.json",
            "qwen36-diagnostic-service-contract-v1",
            "exec_bi100_session.py",
            "process_group_identity.json",
            '"process_group": read_rc("process_group.rc")',
            "wait_http_health.py",
            '--starttime-ticks "$ACTIVE_STARTTIME"',
            '--out "$RUN_ROOT/startup.json"',
            "active_pid_is_same",
            "scan_timeout_rcs",
            "perform_postflight",
            "qwen36-diagnostic-cleanup-v1",
            "cleanup_status.json",
            "trap 'exit 143' TERM",
            "trap '' TERM INT",
            "production_promotion_authorized",
            "unset CUDA_VISIBLE_DEVICES",
            "all(value == 0 for value in gates.values())",
            '"full_model_evaluated": False',
            '"performance_evaluated": False',
        ):
            self.assertIn(marker, harness)
        self.assertNotIn("computility-run.yaml", harness)
        self.assertNotIn("git push", harness)

    def test_gpu_preflight_wrappers_cover_child_cleanup_window(self) -> None:
        harness = (
            ROOT / "scripts" / "run_qwen36_diagnostic_gate.sh"
        ).read_text(encoding="utf-8")
        marker = 'python3 "$ROOT/tests/bi100_preflight.py"'
        positions = []
        offset = 0
        while True:
            position = harness.find(marker, offset)
            if position < 0:
                break
            positions.append(position)
            offset = position + len(marker)
        self.assertEqual(len(positions), 2)
        for position in positions:
            self.assertIn(
                "--kill-after=90s 240s",
                harness[max(0, position - 180):position],
            )

    def test_harness_rejects_duplicate_gpus_and_unsafe_instance_label(
            self) -> None:
        script = ROOT / "scripts" / "run_qwen36_diagnostic_gate.sh"
        duplicate = subprocess.run(
            [
                "bash", str(script), "/missing-model", "2", "0,0",
                "single-card", "/tmp/unused-diagnostic-run",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("unique physical indices", duplicate.stderr)

        unsafe = subprocess.run(
            [
                "bash", str(script), "/missing-model", "1", "0",
                "contains/a/slash", "/tmp/unused-diagnostic-run",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(unsafe.returncode, 2)
        self.assertIn("short non-sensitive label", unsafe.stderr)

    def test_harness_rejects_non_tmp_run_root_before_model_access(self) -> None:
        script = ROOT / "scripts" / "run_qwen36_diagnostic_gate.sh"
        result = subprocess.run(
            [
                "bash", str(script), "/missing-model", "1", "0",
                "single-card", "/var/tmp/unused-diagnostic-run",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 3)
        self.assertIn("private /tmp path", result.stderr)

    def test_postflight_orders_cleanup_before_gpu_comparison(self) -> None:
        harness = (
            ROOT / "scripts" / "run_qwen36_diagnostic_gate.sh"
        ).read_text(encoding="utf-8")
        start = harness.index("perform_postflight()")
        end = harness.index("\ncleanup()", start)
        body = harness[start:end]
        cleanup = body.index("stop_service\n")
        process_scan = body.index("run_service_postflight\n")
        gpu_probe = body.index("run_gpu_preflight_after\n")
        comparison = body.index("run_preflight_comparison\n")
        fatal_scan = body.index("scan_fatal_logs\n")
        self.assertLess(cleanup, process_scan)
        self.assertLess(process_scan, gpu_probe)
        self.assertLess(gpu_probe, comparison)
        self.assertLess(comparison, fatal_scan)

    def test_log_and_timeout_scans_are_recursive(self) -> None:
        harness = (
            ROOT / "scripts" / "run_qwen36_diagnostic_gate.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'find "$RUN_ROOT" -type f \\\n'
            '        \\( -name \'*.log\' -o -name \'*.stdout\' '
            '-o -name \'*.stderr\' \\)',
            harness,
        )
        self.assertIn(
            'find "$RUN_ROOT" -type f -name \'*.rc\' -print0',
            harness,
        )

    def test_layer_trace_is_opt_in_and_post_moe(self) -> None:
        model = (
            ROOT / "qwen3_6_scripts" / "qwen3_5.py"
        ).read_text(encoding="utf-8")
        env_index = model.index('os.getenv("BI100_DIAGNOSTIC_LAYER_TRACE")')
        moe_index = model.index('with bi100_timer("layer.moe")', env_index)
        completed_index = model.index("stage=completed", moe_index)
        self.assertLess(moe_index, completed_index)
        yaml = (ROOT / "computility-run.yaml").read_text(encoding="utf-8")
        self.assertNotIn("BI100_DIAGNOSTIC_LAYER_TRACE", yaml)

    def test_submission_preflight_forbids_trace_leak(self) -> None:
        preflight = (
            ROOT / "tests" / "submission_preflight.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"BI100_DIAGNOSTIC_LAYER_TRACE"', preflight)


if __name__ == "__main__":
    unittest.main()
