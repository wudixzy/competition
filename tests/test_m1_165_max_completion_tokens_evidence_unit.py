import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_165_MAX_COMPLETION_TOKENS_COMPAT_20260730"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M1165MaxCompletionTokensEvidenceUnitTest(unittest.TestCase):

    def test_runtime_probe_qualifies_every_case(self):
        report = json.loads(
            (EVIDENCE / "runtime_probe.json").read_text(encoding="utf-8"))
        self.assertTrue(report["qualified"], report["reasons"])
        self.assertTrue(report["model_has_max_completion_tokens"])
        self.assertEqual(report["case_count"], 10)
        self.assertEqual(report["http_500_count"], 0)
        self.assertTrue(all(case["matched"] for case in report["cases"]))

        cases = {case["name"]: case for case in report["cases"]}
        self.assertEqual(
            cases["both_new_field_precedes"]["sampling_max_tokens"], 7)
        self.assertEqual(cases["legacy_only"]["sampling_max_tokens"], 37)
        for name in (
            "invalid_completion_type",
            "invalid_completion_boundary",
            "unrelated_unknown_field",
        ):
            self.assertEqual(cases[name]["status_code"], 400)

    def test_runtime_overlay_identity_is_attested(self):
        probe = json.loads(
            (EVIDENCE / "runtime_probe.json").read_text(encoding="utf-8"))
        summary = json.loads(
            (EVIDENCE / "verification_summary.json").read_text(
                encoding="utf-8"))
        self.assertTrue(summary["qualified"])
        self.assertEqual(summary["runtime_source"]["install_exit_code"], 0)
        self.assertEqual(summary["probe"]["final_exit_code"], 0)
        self.assertEqual(summary["probe"]["matched_case_count"], 10)
        self.assertFalse(summary["log_review"]["request_values_or_content_logged"])
        self.assertEqual(
            _sha256(EVIDENCE / "runtime_probe.json"),
            summary["probe"]["json_sha256"],
        )
        self.assertEqual(
            _sha256(ROOT / "tests" / "probe_max_completion_tokens_runtime.py"),
            summary["probe"]["script_sha256"],
        )
        for name, runtime_file in probe["runtime_files"].items():
            self.assertEqual(
                runtime_file["sha256"],
                summary["runtime_overlay_identity"][f"{name}.py"]["sha256"],
            )
            self.assertTrue(
                summary["runtime_overlay_identity"][f"{name}.py"][
                    "source_equals_installed"
                ])

    def test_field_audit_is_strict_and_complete(self):
        audit = json.loads(
            (EVIDENCE / "field_audit.json").read_text(encoding="utf-8"))
        self.assertTrue(audit["qualified"], audit["reasons"])
        self.assertEqual(audit["strict_extra_policy"], "forbid")
        self.assertIn(
            "max_completion_tokens", audit["lossless_alias_fields"])
        self.assertIn(
            "reasoning_effort", audit["classified_openai_only_fields"])
        self.assertIn(
            "service_tier", audit["classified_openai_only_fields"])
        self.assertEqual(
            _sha256(ROOT / audit["contract_path"]),
            audit["contract_sha256"],
        )
        self.assertEqual(
            _sha256(ROOT / audit["protocol_path"]),
            audit["protocol_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
