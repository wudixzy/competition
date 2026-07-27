from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_70_DIAGNOSTIC_HTTP_V3_20260728"
)
MANIFEST_SHA256 = (
    "b93943c39b87178b7593895e0ed00c85234d3f1f52b7f84d02f45f9283b613dc"
)
ARMS = ("baseline_default", "candidate_default", "candidate_image2")


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def cases(arm: str) -> dict[str, dict]:
    report = load(f"{arm}/probe.json")
    return {row["name"]: row for row in report["cases"]}


class M170DiagnosticHttpV3EvidenceTest(unittest.TestCase):

    def test_manifest_binds_every_evidence_file(self):
        manifest_path = EVIDENCE / "SHA256SUMS"
        self.assertEqual(
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            MANIFEST_SHA256,
        )
        manifest = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split(maxsplit=1)
            manifest[relative.removeprefix("./")] = digest
        actual_files = {
            path.relative_to(EVIDENCE).as_posix()
            for path in EVIDENCE.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        }
        self.assertEqual(set(manifest), actual_files)
        for relative, expected in manifest.items():
            actual = hashlib.sha256(
                (EVIDENCE / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_runner_and_comparison_are_qualified_without_promotion(self):
        runner = load("runner_status.json")
        comparison = load("comparison.json")
        self.assertEqual(
            runner["source_revision"],
            "dfba141669518f554ed72f9372526b7de6bdb0b2",
        )
        self.assertEqual(runner["instance"], "ssh-73ca29ba")
        self.assertEqual(runner["physical_gpu"], 1)
        self.assertEqual(runner["returncode"], 0)
        self.assertTrue(runner["qualified"])
        self.assertTrue(all(value == 0 for value in runner["gates"].values()))
        self.assertFalse(runner["full_model_evaluated"])
        self.assertFalse(runner["semantic_quality_evaluated"])
        self.assertFalse(runner["production_promotion_authorized"])

        self.assertTrue(comparison["qualified"])
        self.assertEqual(comparison["reasons"], [])
        self.assertTrue(all(comparison["checks"].values()))
        self.assertFalse(comparison["default_image_limit_change_authorized"])
        self.assertFalse(comparison["production_promotion_authorized"])
        self.assertTrue(all(
            value is False for value in comparison["privacy"].values()))

        ports = {
            "baseline_default": 8018,
            "candidate_default": 8019,
            "candidate_image2": 8020,
        }
        for arm in ARMS:
            status = load(f"{arm}/status.json")
            self.assertTrue(status["qualified"], arm)
            self.assertEqual(status["port"], ports[arm])
            self.assertTrue(all(
                value == 0 for value in status["gates"].values()))

    def test_system_and_one_image_generation_are_exact(self):
        mapped = {arm: cases(arm) for arm in ARMS}
        baseline_multi = mapped["baseline_default"][
            "multiple_system_text_parts"]["evidence"]
        self.assertEqual(baseline_multi["http_status"], 400)

        canonical_shas = set()
        image_shas = set()
        for arm in ARMS:
            self.assertTrue(all(
                row["ok"] for row in mapped[arm].values()), arm)
            model = mapped[arm]["models_262144_contract"]["evidence"]
            self.assertEqual(model["max_model_len"], 262144)
            single = mapped[arm]["single_system_text_parts"]["evidence"]
            self.assertEqual(single["http_status"], 200)
            self.assertTrue(single["canonical_generation_exact"])
            canonical_shas.add(
                mapped[arm]["canonical_system_string"]["evidence"][
                    "message_sha256"])
            image_shas.add(
                mapped[arm]["one_image"]["evidence"]["message_sha256"])
        self.assertEqual(len(canonical_shas), 1)
        self.assertEqual(len(image_shas), 1)

        for arm in ("candidate_default", "candidate_image2"):
            multi = mapped[arm][
                "multiple_system_text_parts"]["evidence"]
            self.assertEqual(multi["http_status"], 200)
            self.assertTrue(multi["canonical_generation_exact"])

        image2 = mapped["candidate_image2"][
            "image_at_limit_replay"]["evidence"]
        self.assertEqual(image2["image_count"], 2)
        self.assertTrue(image2["exact_generation_match"])
        self.assertEqual(image2["first"]["cached_tokens"], 0)
        self.assertEqual(image2["replay"]["cached_tokens"], 448)
        self.assertEqual(
            image2["first"]["message_sha256"],
            image2["replay"]["message_sha256"],
        )

    def test_image_limit_4xx_is_complete_and_private(self):
        expected_images = {
            "candidate_default": 2,
            "candidate_image2": 3,
        }
        for arm, image_count in expected_images.items():
            report = load(f"{arm}/attribution.json")
            self.assertTrue(report["qualified"], arm)
            self.assertTrue(report["classified"], arm)
            self.assertTrue(report["complete"], arm)
            self.assertEqual(report["attribution_delta"], 0)
            self.assertEqual(report["malformed_marker_count"], 0)
            self.assertEqual(report["chat_4xx_access_count"], 1)
            self.assertEqual(
                report["by_reason"], {"image_count_limit": 1})
            self.assertEqual(len(report["request_shapes"]), 1)
            shape = report["request_shapes"][0]
            self.assertEqual(shape["images"], image_count)
            self.assertEqual(shape["image_data"], image_count)
            self.assertEqual(shape["image_remote"], 0)
            self.assertEqual(shape["image_other"], 0)
            self.assertTrue(all(
                value is False for value in report["privacy"].values()))

    def test_checkpoint_runtime_cleanup_and_gpu_state_are_bound(self):
        checkpoint = load("checkpoint_verify.json")
        runtime = load("runtime_pair.json")
        self.assertTrue(checkpoint["qualified"])
        self.assertTrue(checkpoint["full_hash_checked"])
        self.assertTrue(checkpoint["tensor_contract_preserved"])
        self.assertEqual(checkpoint["layer_count"], 4)
        self.assertEqual(checkpoint["visual_weight_count"], 333)
        self.assertEqual(checkpoint["weight_count"], 424)
        self.assertTrue(runtime["qualified"])
        self.assertEqual(
            runtime["observed_runtime_file_delta"],
            ["api_server", "protocol"],
        )

        for path in (
                "fatal_scan.txt",
                "timeout_scan.txt",
                "baseline_default/fatal_scan.txt",
                "candidate_default/fatal_scan.txt",
                "candidate_image2/fatal_scan.txt"):
            self.assertEqual((EVIDENCE / path).read_bytes(), b"", path)

        for arm in ARMS:
            postflight = load(f"{arm}/service_postflight.json")
            self.assertTrue(postflight["qualified"], arm)
            self.assertEqual(postflight["api_server_pids"], [])
            self.assertEqual(postflight["worker_pids"], [])
            self.assertEqual(postflight["gpu_processes"], [])
            self.assertEqual(
                postflight["settling"]["final_clean_streak"], 3)

        final_postflight = load("final_postflight.json")
        self.assertTrue(final_postflight["qualified"])
        self.assertEqual(final_postflight["api_server_pids"], [])
        self.assertEqual(final_postflight["worker_pids"], [])
        self.assertEqual(final_postflight["gpu_processes"], [])
        self.assertEqual(
            final_postflight["settling"]["final_clean_streak"], 3)

        initial = load("initial_preflight.json")["results"][0]
        final = load("final_preflight.json")["results"][0]
        self.assertEqual(initial["free"], 34057748480)
        self.assertEqual(final["free"], initial["free"])
        self.assertEqual(final["checksum"], initial["checksum"])
        comparison = load("final_preflight_comparison.json")
        self.assertTrue(comparison["qualified"])
        self.assertEqual(
            comparison["stages"][1][
                "free_memory_drop_from_first_bytes"],
            {"1": 0},
        )


if __name__ == "__main__":
    unittest.main()
