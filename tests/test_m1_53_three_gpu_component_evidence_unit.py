from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/experiments/evidence/"
    "M1_53_THREE_GPU_COMPONENT_GATES_20260725.json"
)


class M153ThreeGpuComponentEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_source_and_three_gpu_scope_are_explicit(self):
        self.assertEqual(
            self.value["source"]["revision"],
            "fbb37b2ffe42e144f55c584268c9e87adec39a9b",
        )
        scope = self.value["scope"]
        self.assertEqual(scope["physical_gpus_used"], [1, 2, 3])
        self.assertFalse(scope["tp3_model_service_valid"])
        self.assertFalse(scope["model_service_started"])
        self.assertFalse(scope["model_weights_loaded"])
        self.assertFalse(scope["tp4_service_or_performance_conclusion_allowed"])

    def test_runtime_overlay_is_reproducible_and_metadata_only(self):
        runtime = self.value["runtime_reproducibility"]
        self.assertTrue(runtime["qualified"])
        self.assertEqual(len(set(runtime["runtime_tree_sha256"])), 1)
        self.assertEqual(runtime["site_packages_diff_rc"], 0)
        self.assertEqual(runtime["site_packages_differing_files"], 0)
        self.assertFalse(runtime["system_site_packages_modified"])
        parent = runtime["parent_runtime"]
        self.assertEqual(
            parent["implementation_diff_rc_excluding_dist_info_provenance"], 0)
        self.assertEqual(parent["implementation_differing_files"], 0)

    def test_all_tp4_rank_local_qgkv_results_are_exact(self):
        loader = self.value["tp4_rank_local_qgkv_loader"]
        self.assertTrue(loader["qualified"])
        self.assertEqual(
            {row["tp_rank"] for row in loader["assignments"]},
            {0, 1, 2, 3},
        )
        self.assertTrue(loader["all_loaded"])
        self.assertTrue(loader["all_weight_segments_exact"])
        self.assertTrue(loader["all_output_segments_exact"])
        self.assertEqual(loader["max_abs"], 0.0)

        benchmark = self.value["packed_qgkv_microbenchmark"]
        self.assertFalse(benchmark["parameter_scan_performed"])
        self.assertEqual(len(benchmark["results"]), 6)
        for row in benchmark["results"]:
            self.assertTrue(row["exact"])
            self.assertEqual(row["max_abs"], 0.0)
            self.assertGreater(row["speedup"], 1.0)

    def test_communication_and_postflight_are_exact(self):
        communication = self.value["three_rank_communication"]
        self.assertTrue(communication["qualified"])
        for gate in ("nccl_preflight", "vllm_group_preflight"):
            self.assertTrue(communication[gate]["ok"])
            self.assertEqual(communication[gate]["rank_values"],
                             [6.0, 6.0, 6.0])
            self.assertEqual(communication[gate]["timed_out_ranks"], [])
        ixformer = communication["ixformer_allreduce"]
        self.assertTrue(ixformer["ok"])
        self.assertFalse(ixformer["ipc_initiated"])
        self.assertEqual(ixformer["parity_max_abs"], 0.0)

        health = self.value["gpu_health_after_all_gates"]
        self.assertTrue(health["ok"])
        self.assertEqual(
            health["free_bytes"], health["total_bytes"])
        self.assertEqual(health["residual_component_or_service_processes"], 0)

    def test_quality_and_promotion_remain_closed(self):
        quality = self.value["quality_status"]
        self.assertFalse(quality["model_output_regression_run"])
        self.assertFalse(quality["model_capability_non_regression_established"])
        decision = self.value["decision"]
        self.assertTrue(decision["tp4_service_ab_required"])
        self.assertTrue(decision["tp4_quality_gates_required"])
        self.assertFalse(decision["tp4_performance_claim_established"])
        self.assertFalse(decision["model_quality_non_regression_established"])
        self.assertFalse(decision["overall_candidate_promotion_authorized"])
        self.assertFalse(decision["main_or_yaml_change_authorized"])
        for digest in self.value["artifact_sha256"].values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)
        for contains_sensitive in self.value["privacy"].values():
            self.assertFalse(contains_sensitive)


if __name__ == "__main__":
    unittest.main()
