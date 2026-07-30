from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_153_RUNTIME_AND_PROMPT_PREFLIGHT_20260730"
)
SOURCE_REVISION = "a0c70d480f00ea43728f2d1d9e8063f7da93ee6b"
RUNTIME_TREE_SHA256 = (
    "3a89b9eec39792ac5fb4577ab6b148b1319f5ae1a40b4ca76b869ff39cd57631"
)


class M1153RuntimePromptEvidenceUnitTest(unittest.TestCase):

    def test_manifest_authenticates_every_evidence_file(self):
        rows = [
            line.split("  ", 1)
            for line in (EVIDENCE / "SHA256SUMS").read_text(
                encoding="ascii").splitlines()
        ]
        expected = {
            path.name
            for path in EVIDENCE.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        }
        self.assertEqual({name for _, name in rows}, expected)
        for digest, name in rows:
            self.assertEqual(
                hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest(),
                digest,
            )

    def test_exact_runtime_and_loader_identity_qualify(self):
        ensured = json.loads(
            (EVIDENCE / "runtime_ensure.json").read_text(encoding="ascii"))
        verified = json.loads(
            (EVIDENCE / "runtime_verification.json").read_text(
                encoding="ascii"))

        self.assertTrue(ensured["qualified"])
        self.assertTrue(ensured["source_tree_clean"])
        self.assertFalse(ensured["cache_hit"])
        self.assertEqual(ensured["source_revision"], SOURCE_REVISION)
        self.assertEqual(
            ensured["runtime_tree_sha256"], RUNTIME_TREE_SHA256)

        self.assertTrue(verified["qualified"])
        self.assertEqual(verified["reasons"], [])
        self.assertEqual(verified["source_revision"], SOURCE_REVISION)
        self.assertEqual(
            verified["runtime_tree_sha256"], RUNTIME_TREE_SHA256)
        self.assertEqual(
            verified["files"]["bi100_external_extension"],
            {"generated": False, "same": True},
        )
        self.assertTrue(all(
            row["same"] for row in verified["files"].values()))
        self.assertTrue(all(verified["fixed_source_identity"].values()))

    def test_p90_prompt_lengths_and_partial_boundaries_qualify(self):
        report = json.loads(
            (EVIDENCE / "prompt_construction.json").read_text(
                encoding="ascii"))
        self.assertTrue(report["qualified"])
        self.assertEqual(report["reasons"], [])
        self.assertEqual(
            [row["actual_prompt_tokens"] for row in report["cold"]],
            [8192, 16384, 24576, 32768, 49152, 65536],
        )
        self.assertEqual(
            [row["target_prompt_tokens"] for row in report["partial"]],
            [16384, 32768, 49152, 65536],
        )
        self.assertEqual(
            [row["cached_prefix_tokens"] for row in report["partial"]],
            [8192, 24576, 40960, 57344],
        )
        self.assertEqual(
            [row["residual_prefill_tokens"] for row in report["partial"]],
            [8192, 8192, 8192, 8192],
        )
        self.assertEqual(
            report["privacy"],
            {
                "credentials_recorded": False,
                "prompts_recorded": False,
                "token_ids_recorded": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
