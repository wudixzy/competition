import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_167_CHAT_FIELD_INTERACTIONS_20260731"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M1167ChatFieldInteractionsEvidenceUnitTest(unittest.TestCase):

    def test_evidence_is_qualified_and_bound_to_runtime_source(self):
        audit = json.loads(
            (EVIDENCE / "field_audit.json").read_text(encoding="utf-8"))
        runtime = json.loads(
            (EVIDENCE / "runtime_probe.json").read_text(encoding="utf-8"))

        self.assertTrue(audit["qualified"], audit["reasons"])
        self.assertTrue(runtime["qualified"], runtime["reasons"])
        self.assertEqual(runtime["case_count"], 18)
        self.assertEqual(runtime["http_500_count"], 0)
        self.assertTrue(runtime["required_fields_present"])
        self.assertTrue(all(row["matched"] for row in runtime["cases"]))
        self.assertEqual(
            audit["protocol_sha256"],
            runtime["runtime_files"]["protocol"]["sha256"],
        )
        self.assertEqual(
            audit["protocol_sha256"],
            _sha256(ROOT / audit["protocol_path"]),
        )

    def test_manifest_covers_every_evidence_file(self):
        lines = (EVIDENCE / "SHA256SUMS").read_text(
            encoding="ascii").splitlines()
        observed = {}
        for line in lines:
            digest, name = line.split("  ", 1)
            observed[name] = digest
        expected = {
            path.name: _sha256(path)
            for path in EVIDENCE.glob("*.json")
        }
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
