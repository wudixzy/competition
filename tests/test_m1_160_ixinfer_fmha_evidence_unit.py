from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs"
    / "experiments"
    / "evidence"
    / "M1_160_IXINFER_FMHA_CAPABILITY_20260730"
)
SOURCE_REVISION = "9a2a87b88f8d450d587b82c073a055fb5742eccc"
EXTENSION_SHA256 = (
    "3c83ae9c0bb35096bd41c9bc9be4481710ce234335bcdc3b0ff80589d4b39b5a"
)
EXPECTED_CASES = (
    "bshd_d128_mha",
    "bshd_d256_mha",
    "bhsd_d128_mha",
    "bshd_d256_gqa_causal",
)


class M1160IxinferFmhaEvidenceTest(unittest.TestCase):
    def test_manifest_authenticates_recursive_evidence(self):
        rows = [
            line.split("  ", 1)
            for line in (EVIDENCE / "SHA256SUMS")
            .read_text(encoding="ascii")
            .splitlines()
        ]
        expected = {
            f"./{path.relative_to(EVIDENCE).as_posix()}"
            for path in EVIDENCE.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        }
        self.assertEqual({name for _, name in rows}, expected)
        for digest, name in rows:
            path = EVIDENCE / name.removeprefix("./")
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), digest
            )

    def test_all_reasonable_header_contract_cells_fail_bad_param(self):
        matrix = json.loads(
            (EVIDENCE / "matrix.json").read_text(encoding="ascii")
        )
        self.assertEqual(matrix["source_revision"], SOURCE_REVISION)
        self.assertEqual(matrix["extension_sha256"], EXTENSION_SHA256)
        self.assertEqual(
            tuple(row["name"] for row in matrix["cases"]),
            EXPECTED_CASES,
        )
        self.assertTrue(
            matrix["conclusion"]["all_dispatches_rejected_bad_param"]
        )
        self.assertFalse(
            matrix["conclusion"]["documented_contract_usable"]
        )
        self.assertFalse(
            matrix["conclusion"]["continue_ixinfer_parameter_guessing"]
        )
        for row in matrix["cases"]:
            self.assertEqual(
                row["cuinfer_statuses"], ["CUINFER_STATUS_BAD_PARAM"]
            )
            self.assertNotEqual(row["returncode"], 0)
            self.assertFalse(row["timed_out"])
            self.assertFalse(row["result_written"])

    def test_failure_is_clean_and_does_not_authorize_continuation(self):
        matrix = json.loads(
            (EVIDENCE / "matrix.json").read_text(encoding="ascii")
        )
        postflight = json.loads(
            (EVIDENCE / "postflight_after.json").read_text(
                encoding="ascii"
            )
        )
        preflight = json.loads(
            (EVIDENCE / "preflight_after.json").read_text(encoding="ascii")
        )
        self.assertTrue(matrix["fatal_scan"]["qualified"])
        self.assertFalse(any(
            matrix["fatal_scan"]["category_counts"].values()
        ))
        self.assertEqual(
            matrix["authorization"],
            {
                "main_or_yaml_change_authorized": False,
                "runtime_overlay_authorized": False,
                "tp4_service_authorized": False,
            },
        )
        self.assertTrue(postflight["qualified"])
        self.assertEqual(postflight["gpu_processes"], [])
        self.assertTrue(preflight["ok"])
        self.assertTrue(preflight["cleanup_reaped"])
        self.assertEqual(preflight["gpus"], [1, 2, 3])

    def test_identity_matches_source_artifacts(self):
        identity = json.loads(
            (EVIDENCE / "identity.json").read_text(encoding="ascii")
        )
        self.assertEqual(identity["source_revision"], SOURCE_REVISION)
        self.assertEqual(identity["extension_sha256"], EXTENSION_SHA256)
        paths = {
            "probe_source_sha256":
                ROOT / "tests" / "corex_ixinfer_fmha_probe_ext.cu",
            "build_script_sha256":
                ROOT / "tests" / "build_corex_ixinfer_fmha_probe.sh",
            "probe_runner_sha256":
                ROOT / "tests" / "run_corex_ixinfer_fmha_probe.py",
            "matrix_runner_sha256":
                ROOT
                / "tests"
                / "run_m1_160_ixinfer_fmha_capability_matrix.py",
        }
        for field, path in paths.items():
            self.assertEqual(
                identity[field],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
