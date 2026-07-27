from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_69_IMAGE_COMPAT"
)
EXPECTED_SHA256 = {
    "baseline-multimodal-v3.json":
        "da2c9aeef44784d86d98326729146cc336d48dca5afd5edf4816c26fd51b9999",
    "baseline-request-v3.json":
        "da2b22d547ea8411aa292ca4f0feecda95dc118da033e1fcfcf6960aef2e127b",
    "baseline-template-v3.json":
        "55edfbae752f51b933ed94ddbc1c1d61d214c40ddddfbec17b0057582887aea0",
    "candidate-multimodal-v3.json":
        "da2c9aeef44784d86d98326729146cc336d48dca5afd5edf4816c26fd51b9999",
    "candidate-request-v3.json":
        "da2b22d547ea8411aa292ca4f0feecda95dc118da033e1fcfcf6960aef2e127b",
    "candidate-template-v3.json":
        "e2660b736e6e87a436da990c111145807c72fe4cc5b31310b1447bee154efae8",
    "error-scan-clean.json":
        "3b6963b911261d6198081f77d4fa2e75d1438e9d68606e5e66978c7f7f41c52a",
    "four-card-postflight-20260728.json":
        "b306de3d900b9b8d66cbc66e3e9994798180485b0de53131e4caed4030e309fc",
    "four-card-preflight-20260728.json":
        "2f007f9ab23bdf78a2b1345edcd6bd891aa4c2036fc5d1dca983a20e3bb962a5",
    "gpu1-preflight-final-pure.json":
        "05794a4e890c93ded859cfe68a320641f72da4884c1ca1b49d4f3f7959e324d7",
    "gpu1-preflight-initial-pure.json":
        "05794a4e890c93ded859cfe68a320641f72da4884c1ca1b49d4f3f7959e324d7",
    "input-diff-dd3f1cc.txt":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "install.json":
        "4adfae70ebfc6b8ce46d33c05251d4730413dac4a7ec4e759821b553fccdb5aa",
    "multimodal-preprocessing-dd3f1cc.json":
        "36e1d39ac1568102e1dede1a09f543686fa2b00bd5468e469b14e80f8cc3b3e2",
    "run-status-v1.json":
        "e43f7b0d7524c6db7b3ad5881188322587a35f8c655be42f5bddf651e89eb209",
    "service-postflight-final.json":
        "4a2af2504bb011dfba638a5d65f4e58c07a43c0250771fc23fe779351468f85d",
    "worktree-verification.txt":
        "e8b9515cc7d030470c39a6f11148a939aab0e8ef5ed32bec2a230b86dda34a7d",
}


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


class M169ImageCompatEvidenceTest(unittest.TestCase):

    def test_evidence_hashes_and_manifest_are_bound(self):
        manifest = {}
        for line in (EVIDENCE / "SHA256SUMS").read_text(
                encoding="utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            manifest[name] = digest
        self.assertEqual(manifest, EXPECTED_SHA256)
        self.assertEqual(
            {path.name for path in EVIDENCE.iterdir()},
            set(EXPECTED_SHA256) | {"SHA256SUMS"},
        )
        for name, expected in EXPECTED_SHA256.items():
            actual = hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)

    def test_exact_runtime_and_expected_exit_codes_qualify(self):
        install = load("install.json")
        status = load("run-status-v1.json")
        self.assertTrue(install["qualified"])
        self.assertEqual(
            install["source_revision"],
            "37001edff643d98bf41bf4a52e0a145329003315",
        )
        self.assertEqual(
            install["runtime_tree_sha256"],
            "9cb9bf5b21260826372d8f9496a23bc501cb4052ce7c701dbc401be0952f6549",
        )
        self.assertTrue(status["qualified"])
        self.assertEqual(status["rcs"], status["expected_rcs"])
        self.assertTrue(status["runtime_inputs_diff_empty"])
        self.assertEqual(
            status["probe_revision"],
            "dd3f1cc413f32848b4b2996bd2e006823465c700",
        )
        self.assertEqual(
            status["runtime_tree_sha256"], install["runtime_tree_sha256"])
        self.assertEqual((EVIDENCE / "input-diff-dd3f1cc.txt").read_bytes(),
                         b"")

    def test_system_text_parts_fix_is_isolated_at_tokenizer_boundary(self):
        candidate_request = load("candidate-request-v3.json")
        baseline_request = load("baseline-request-v3.json")
        candidate = load("candidate-template-v3.json")
        baseline = load("baseline-template-v3.json")

        for report in (candidate_request, baseline_request):
            self.assertTrue(report["qualified"])
            self.assertEqual(report["matched_count"], 12)
            self.assertEqual(report["mismatch_count"], 0)
        self.assertTrue(candidate["qualified"])
        self.assertTrue(all(candidate["checks"].values()))
        self.assertFalse(baseline["qualified"])
        failed = {
            name for name, passed in baseline["checks"].items()
            if not passed
        }
        self.assertEqual(failed, {"system_text_parts_token_exact"})

        fixed = candidate["pairs"]["system_text_parts"]
        old = baseline["pairs"]["system_text_parts"]
        self.assertEqual(fixed["parts_token_count"], 32)
        self.assertEqual(fixed["parts_token_count"],
                         fixed["normalized_token_count"])
        self.assertEqual(fixed["parts_sha256"], fixed["normalized_sha256"])
        self.assertIsNone(fixed["parts_error_type"])
        self.assertEqual(old["parts_error_type"], "TemplateError")
        self.assertIsNone(old["parts_sha256"])
        self.assertIsNone(old["parts_token_count"])

    def test_multimodal_limit_and_preprocessing_contracts_qualify(self):
        candidate = load("candidate-multimodal-v3.json")
        baseline = load("baseline-multimodal-v3.json")
        preprocessing = load("multimodal-preprocessing-dd3f1cc.json")

        for report in (candidate, baseline):
            self.assertTrue(report["qualified"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["model"]["max_model_len"], 262144)
            self.assertEqual(report["default"]["effective"]["image"], 1)
            self.assertEqual(
                report["explicit_image_two"]["effective"]["image"], 2)
            self.assertEqual(report["default"]["max_multimodal_tokens"], 1280)
            self.assertEqual(
                report["explicit_image_two"]["max_multimodal_tokens"], 2560)

        self.assertTrue(preprocessing["qualified"])
        self.assertTrue(all(preprocessing["checks"].values()))
        self.assertEqual(preprocessing["model"]["max_model_len"], 262144)
        one = preprocessing["one_image"]
        two = preprocessing["two_images"]
        repeated = preprocessing["two_images_repeated"]
        reversed_images = preprocessing["two_images_reversed"]
        self.assertEqual(one["processed_image_tokens"], 64)
        self.assertEqual(two["processed_image_tokens"], 128)
        self.assertEqual(one["mapped_shapes"]["pixel_values"], [[256, 1536]])
        self.assertEqual(two["mapped_shapes"]["pixel_values"], [[512, 1536]])
        self.assertTrue(all(one["mapped_finite"].values()))
        self.assertTrue(all(two["mapped_finite"].values()))
        self.assertEqual(two["processed_token_sha256"],
                         repeated["processed_token_sha256"])
        self.assertEqual(two["mapped_sha256"], repeated["mapped_sha256"])
        self.assertNotEqual(two["processed_token_sha256"],
                            reversed_images["processed_token_sha256"])
        self.assertNotEqual(two["mapped_sha256"]["pixel_values"],
                            reversed_images["mapped_sha256"]["pixel_values"])

    def test_gpu_health_and_all_postflights_are_explicit(self):
        four_pre = load("four-card-preflight-20260728.json")
        four_post = load("four-card-postflight-20260728.json")
        gpu1_pre = load("gpu1-preflight-initial-pure.json")
        gpu1_post = load("gpu1-preflight-final-pure.json")
        service_post = load("service-postflight-final.json")
        error_scan = load("error-scan-clean.json")

        self.assertFalse(four_pre["ok"])
        results = {result["gpu"]: result for result in four_pre["results"]}
        self.assertTrue(results[1]["ok"])
        for gpu in (0, 2, 3):
            self.assertFalse(results[gpu]["ok"])
            self.assertEqual(results[gpu]["stage"], "timeout")
            self.assertEqual(results[gpu]["last_progress_stage"],
                             "mem_get_info")
            self.assertEqual(results[gpu]["termination"], "sigterm")
            self.assertTrue(results[gpu]["cleanup_reaped"])

        self.assertEqual(gpu1_pre, gpu1_post)
        self.assertTrue(gpu1_pre["ok"])
        for postflight in (four_post, service_post):
            self.assertTrue(postflight["qualified"])
            self.assertEqual(postflight["settling"]["final_clean_streak"], 3)
            self.assertEqual(postflight["api_server_pids"], [])
            self.assertEqual(postflight["worker_pids"], [])
            self.assertEqual(postflight["gpu_processes"], [])
        self.assertTrue(error_scan["fatal_scan_clean"])
        self.assertEqual(error_scan["hits"], [])


if __name__ == "__main__":
    unittest.main()
