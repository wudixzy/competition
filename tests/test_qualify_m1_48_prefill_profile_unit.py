from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.qualify_m1_48_prefill_profile import PROFILE_FILTER, qualify
from tests.summarize_prefill_path_profile import summarize
from tests.test_prefill_path_profile_summary_unit import _payload


DIGEST = "a" * 64


def _inputs():
    m1_49 = {
        "schema": "bi100-m1-49-long-context-qualification-v1",
        "version": 1,
        "qualified": True,
        "reasons": [],
        "scope": "hybrid-kv-capacity-correctness-not-prefill-speed",
        "candidate_startup": {
            "mode": "full_attention",
            "attention_layers": 10,
        },
    }
    runtime_install = {
        "schema": "bi100-bare-host-runtime-install-v2",
        "version": 2,
        "qualified": True,
        "system_site_packages_modified": False,
        "startup_profile_guard_patch": True,
        "worker_sha256": DIGEST,
        "files": {
            name: {
                "same": True,
                "source_sha256": DIGEST,
                "installed_sha256": DIGEST,
            }
            for name in (
                "vllm_model",
                "bi100_profile",
                "paged_attention",
                "xformers_backend",
            )
        },
    }
    runtime_identity = {
        "schema": "bi100-m1-48-runtime-identity-v1",
        "version": 1,
        "qualified": True,
        "reasons": [],
        "source_revision": "b" * 40,
        "startup_profile_guard_patch": True,
        "install_worker_sha256": DIGEST,
        "runtime_worker_sha256": DIGEST,
        "files": {
            name: {
                "current_source_sha256": DIGEST,
                "install_source_sha256": DIGEST,
                "installed_sha256": DIGEST,
                "runtime_installed_sha256": DIGEST,
                "same": True,
            }
            for name in (
                "vllm_model",
                "bi100_profile",
                "paged_attention",
                "xformers_backend",
            )
        },
    }
    startup = {
        "schema": "bi100-hybrid-kv-startup-v1",
        "version": 1,
        "qualified": True,
        "reasons": [],
        "mode": "full_attention",
        "config_mode": "full_attention",
        "expected_attention_layers": 10,
        "observed_attention_layers": 10,
        "observed_layer_count": 40,
        "full_attention_ordinals": list(range(10)),
        "num_key_value_heads": 4,
        "rank_kv_heads": 1,
        "head_dim": 256,
        "tensor_parallel_size": 4,
        "dtype": "float16",
        "dtype_bytes": 2,
        "expected_kv_bytes_per_block": 163840,
        "max_model_len_required": 262144,
        "block_size": 16,
        "required_gpu_blocks": 16384,
        "observed_max_seq_len": 262144,
        "runtime_contract": {"accounting": "full_attention"},
        "runtime_contract_invariant_sha256": DIGEST,
    }
    preflight = {
        "schema": "bi100-gpu-preflight-comparison-v1",
        "version": 1,
        "qualified": True,
        "reasons": [],
        "max_free_memory_drop_bytes": 1_073_741_824,
        "stages": [
            {
                "label": label,
                "qualified": True,
                "results": [{"gpu": gpu, "ok": True} for gpu in range(4)],
            }
            for label in ("before_control", "after_control", "after_profile")
        ],
    }
    protocol = {
        "stream": True,
        "max_tokens": 1,
        "min_tokens": 1,
        "temperature": 0,
        "seed": 20260722,
        "thinking": False,
        "target_prompt_tokens": 235000,
        "max_model_len": 262144,
    }
    request = {
        "prompt_tokens": 235000,
        "cached_tokens": 0,
        "completion_tokens": 1,
        "ttft_s": 20.0,
        "elapsed_s": 20.1,
        "output_sha256": DIGEST,
    }
    service = {
        "schema": "bi100-m1-48-prefill-service-v1",
        "version": 1,
        "run_id": "m148-unit",
        "protocol": protocol,
        "request": request,
        "qualified_measurement": True,
        "reasons": [],
    }
    control_service = {**service, "mode": "control"}
    profile_service = copy.deepcopy(service)
    profile_service["mode"] = "profile"
    profile_service["request"]["ttft_s"] = 21.0
    summary = {
        "schema": "bi100-m1-48-prefill-path-profile-v2",
        "version": 2,
        "qualified_profile": True,
        "reasons": [],
        "request": {
            "group_index": 0,
            "prefill_tokens": 235000,
            "expected_chunk_size": 8192,
            "block_size": 16,
            "num_attention_heads": 16,
            "query_heads_per_rank": 4,
            "forward_count": 29,
            "tp_ranks": [0, 1, 2, 3],
            "profile_overhead_limit_fraction": 0.15,
            "profile_overhead_fraction": 0.05,
            "control_ttft_s": 20.0,
            "profile_ttft_s": 21.0,
            "model_rank_spread_fraction": 0.01,
            "max_forward_model_rank_spread_fraction": 0.02,
            "control_output_sha256": DIGEST,
        },
        "full_attention": {
            "inclusive_ms_per_rank_mean": 10000.0,
            "paged_segment_ms_per_rank_mean": 8000.0,
            "attention_unattributed_ms_per_rank_mean": 500.0,
            "paged_unattributed_ms_per_rank_mean": 100.0,
        },
    }
    common_log = (
        "[BI100] runtime overlay active: /runtime\n"
        "[BI100] GDN cache; policy=admission64 restore=direct\n"
        "[BI100] M1-49 runtime contract; accounting=full_attention\n"
    )
    control_log = common_log + (
        "[BI100] M1-48 profile contract; enabled=0 mode=event "
        f"include_startup=0 filter={PROFILE_FILTER}\n"
    )
    profile_log = common_log + (
        "[BI100] M1-48 profile contract; enabled=1 mode=event "
        f"include_startup=0 filter={PROFILE_FILTER}\n"
        "[BI100_PROFILE_EVENT] {}\n"
    )
    source_sha256 = {
        name: DIGEST for name in (
            "m1_49",
            "runtime_install",
            "runtime_identity",
            "preflight",
            "control_startup",
            "profile_startup",
            "control_service",
            "profile_service",
            "profile_summary",
            "prequalification_cleanup",
            "source_revision",
            "control_log",
            "profile_log",
        )
    }
    values = {
        "m1_49": m1_49,
        "runtime_install": runtime_install,
        "runtime_identity": runtime_identity,
        "preflight": preflight,
        "control_startup": copy.deepcopy(startup),
        "profile_startup": copy.deepcopy(startup),
        "control_service": control_service,
        "profile_service": profile_service,
        "profile_summary": summary,
        "prequalification_cleanup": "0\n",
        "source_revision": "b" * 40 + "\n",
        "control_log": control_log,
        "profile_log": profile_log,
        "source_sha256": source_sha256,
    }
    values["recomputed_profile_summary"] = copy.deepcopy(summary)
    return values


class QualifyM148PrefillProfileTest(unittest.TestCase):
    def test_fixed_evidence_qualifies_without_authorizing_promotion(self):
        report = qualify(**_inputs())
        self.assertTrue(report["qualified"], report)
        self.assertFalse(report["promotion_authorized"])
        self.assertEqual(
            report["scope"], "post-m1-49-diagnostic-path-ranking-only")

    def test_missing_m1_49_prerequisite_fails(self):
        inputs = _inputs()
        inputs["m1_49"]["qualified"] = False
        report = qualify(**inputs)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "M1-49 long-context prerequisite is not qualified",
            report["reasons"],
        )

    def test_control_profile_event_or_fatal_log_fails(self):
        inputs = _inputs()
        inputs["control_log"] += "[BI100_PROFILE_EVENT] {}\n"
        inputs["profile_log"] += "Gloo connectFullMesh failed\n"
        report = qualify(**inputs)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "control log unexpectedly contains profile events",
            report["reasons"],
        )
        self.assertIn("profile log contains a fatal signature", report["reasons"])

    def test_summary_tampering_fails_closed(self):
        mutations = {
            "ttft": lambda value: value["profile_summary"]["request"].__setitem__(
                "control_ttft_s", 999.0),
            "aggregate_spread": lambda value: value["profile_summary"][
                "request"].__setitem__("model_rank_spread_fraction", 99.0),
            "forward_spread": lambda value: value["profile_summary"][
                "request"].__setitem__(
                    "max_forward_model_rank_spread_fraction", 99.0),
            "output_digest": lambda value: value["profile_summary"][
                "request"].__setitem__("control_output_sha256", "c" * 64),
            "full_attention": lambda value: value["profile_summary"].pop(
                "full_attention"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                inputs = _inputs()
                mutate(inputs)
                report = qualify(**inputs)
                self.assertFalse(report["qualified"], report)

    def test_failed_prequalification_cleanup_fails(self):
        inputs = _inputs()
        inputs["prequalification_cleanup"] = "1\n"

        report = qualify(**inputs)

        self.assertFalse(report["qualified"])
        self.assertIn(
            "M1-48 pre-qualification cleanup did not pass",
            report["reasons"],
        )

    def test_cli_recomputes_summary_from_log_and_service_sources(self):
        inputs = _inputs()
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            evidence = {}
            for name in (
                    "m1_49", "runtime_install", "runtime_identity",
                    "preflight", "control_startup", "profile_startup",
                    "control_service", "profile_service"):
                path = directory / f"{name}.json"
                path.write_text(
                    json.dumps(inputs[name]), encoding="utf-8")
                evidence[name] = path

            control_log = directory / "control.log"
            control_log.write_text(inputs["control_log"], encoding="utf-8")
            profile_log = directory / "profile.log"
            lines = [
                inputs["profile_log"].replace(
                    "[BI100_PROFILE_EVENT] {}\n", "").rstrip()
            ]
            context = 0
            index = 0
            while context < 235000:
                tokens = min(8192, 235000 - context)
                base = _payload(
                    index,
                    tokens,
                    context,
                    True,
                    capture_points=(1 if context + tokens == 235000 else 0),
                )
                for rank, pid in enumerate((101, 102, 103, 104)):
                    event = copy.deepcopy(base)
                    event["tp_rank"] = rank
                    lines.append(
                        f"(VllmWorkerProcess pid={pid}) "
                        "[BI100_PROFILE_EVENT] "
                        + json.dumps(event, separators=(",", ":")))
                context += tokens
                index += 1
            profile_log.write_text(
                "\n".join(lines) + "\n", encoding="utf-8")

            summary_path = directory / "profile_summary.json"
            summary = summarize(
                profile_log,
                expected_prefill_tokens=235000,
                expected_processes=4,
                profile_service=evidence["profile_service"],
                control_service=evidence["control_service"],
                expected_chunk_size=8192,
                block_size=16,
            )
            self.assertTrue(summary["qualified_profile"], summary)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            revision = directory / "source_revision.txt"
            revision.write_text("b" * 40 + "\n", encoding="utf-8")
            cleanup = directory / "cleanup.rc"
            cleanup.write_text("0\n", encoding="utf-8")
            output = directory / "qualification.json"

            command = [
                sys.executable,
                str(Path(__file__).with_name(
                    "qualify_m1_48_prefill_profile.py")),
            ]
            for flag, name in (
                    ("--m1-49", "m1_49"),
                    ("--runtime-install", "runtime_install"),
                    ("--runtime-identity", "runtime_identity"),
                    ("--preflight", "preflight"),
                    ("--control-startup", "control_startup"),
                    ("--profile-startup", "profile_startup"),
                    ("--control-service", "control_service"),
                    ("--profile-service", "profile_service")):
                command.extend((flag, str(evidence[name])))
            command.extend((
                "--profile-summary", str(summary_path),
                "--prequalification-cleanup", str(cleanup),
                "--source-revision", str(revision),
                "--control-log", str(control_log),
                "--profile-log", str(profile_log),
                "--out", str(output),
            ))
            result = subprocess.run(
                command, text=True, capture_output=True, check=False)

            report = json.loads(output.read_text(encoding="utf-8"))
            summary["request"]["control_ttft_s"] = 999.0
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            tampered_result = subprocess.run(
                command, text=True, capture_output=True, check=False)
            tampered_report = json.loads(
                output.read_text(encoding="utf-8"))
        self.assertEqual(
            result.returncode, 0,
            f"stderr={result.stderr}\nstdout={result.stdout}\nreport={report}")
        self.assertTrue(report["qualified"], report)
        self.assertEqual(tampered_result.returncode, 1)
        self.assertIn(
            "M1-48 profile summary differs from source recomputation",
            tampered_report["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
