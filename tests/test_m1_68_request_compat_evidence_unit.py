from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_68_REQUEST_COMPAT"
)
EXPECTED_SHA256 = {
    "gpu-postflight-20260727.json":
        "b8f3b142ac6bccbb84aef7f3f919749730c6c7f8735f86431eb99950de4c2ebc",
    "gpu-preflight-20260727.json":
        "2f007f9ab23bdf78a2b1345edcd6bd891aa4c2036fc5d1dca983a20e3bb962a5",
    "m1-68-request-compat-6c7c9db.json":
        "80389cf778cb0582763f71337b4129356b462340ae1d477d310ce92dc0974952",
    "m1-68-request-compat-baseline-c78d55d.json":
        "334d89c17b4ffadb41b777104d48fc91eaa6995b8d1bfcc861c46f2cdc41be91",
    "m1-68-request-compat-cdb1bc4.json":
        "c9c2256ad4d971a2086ed6658cbdc02ec1c3b8dd1b8bfd75c4e98da90ca52bff",
    "m1-68-template-compat-baseline-c78d55d.json":
        "0ae70cf7980b6c8ab1b528b422b7efe91f2f8766a59dadb052aa80eab91e7867",
    "m1-68-template-compat-cdb1bc4.json":
        "af772dc3596e1f7696c5101006df42417f06808db93c063e4cb628f370d9785d",
    "runtime-install-cdb1bc4.json":
        "90f5b939b114370f8ccfb03b57f532335b3182cb58252b2e25e5fe61163c4a98",
}


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


class M168RequestCompatEvidenceTest(unittest.TestCase):

    def test_evidence_hashes_are_bound(self):
        for name, expected in EXPECTED_SHA256.items():
            actual = hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)

    def test_exact_runtime_and_candidate_matrix_qualify(self):
        runtime = load("runtime-install-cdb1bc4.json")
        request = load("m1-68-request-compat-cdb1bc4.json")
        template = load("m1-68-template-compat-cdb1bc4.json")
        self.assertTrue(runtime["qualified"])
        self.assertEqual(
            runtime["source_revision"],
            "cdb1bc41f728a5610a3632ad7923d73a90748919",
        )
        self.assertEqual(
            runtime["runtime_tree_sha256"],
            "5196f4030ddc23a716f1f8d89f2c9967aabf3d2c90f361244b8062e37d563d8c",
        )
        for name in ("protocol", "chat_utils", "api_server"):
            self.assertTrue(runtime["files"][name]["same"], name)
        self.assertTrue(request["qualified"])
        self.assertEqual(request["matched_count"], 11)
        self.assertEqual(request["mismatch_count"], 0)
        self.assertTrue(template["qualified"])
        self.assertTrue(all(template["checks"].values()))

    def test_tokenizer_pairs_are_exact_and_fail_closed_guards_hold(self):
        report = load("m1-68-template-compat-cdb1bc4.json")
        history = report["pairs"]["object_history"]
        strict = report["pairs"]["strict_false"]
        self.assertEqual(
            history["string_token_count"], history["object_token_count"])
        self.assertEqual(
            history["string_sha256"], history["object_sha256"])
        self.assertEqual(
            strict["omitted_token_count"], strict["false_token_count"])
        self.assertEqual(
            strict["omitted_sha256"], strict["false_sha256"])
        self.assertTrue(report["checks"]["strict_true_rejected"])
        self.assertTrue(report["checks"]["tool_choice_required_rejected"])

    def test_negative_intermediate_and_baseline_remain_negative(self):
        intermediate = load("m1-68-request-compat-6c7c9db.json")
        baseline_request = load(
            "m1-68-request-compat-baseline-c78d55d.json")
        baseline_template = load(
            "m1-68-template-compat-baseline-c78d55d.json")
        self.assertFalse(intermediate["qualified"])
        self.assertEqual(
            set(intermediate["mismatches"]),
            {
                "assistant_tool_arguments_object",
                "assistant_tool_arguments_invalid_json",
            },
        )
        self.assertFalse(baseline_request["qualified"])
        self.assertFalse(baseline_template["qualified"])

    def test_failed_gpu_preflight_has_clean_postflight(self):
        preflight = load("gpu-preflight-20260727.json")
        postflight = load("gpu-postflight-20260727.json")
        self.assertFalse(preflight["ok"])
        results = {result["gpu"]: result for result in preflight["results"]}
        self.assertTrue(results[1]["ok"])
        for gpu in (0, 2, 3):
            self.assertFalse(results[gpu]["ok"])
            self.assertEqual(results[gpu]["stage"], "timeout")
            self.assertEqual(results[gpu]["last_progress_stage"],
                             "mem_get_info")
            self.assertEqual(results[gpu]["termination"], "sigterm")
            self.assertTrue(results[gpu]["cleanup_reaped"])
        self.assertTrue(postflight["qualified"])
        self.assertEqual(postflight["settling"]["final_clean_streak"], 3)
        self.assertEqual(postflight["api_server_pids"], [])
        self.assertEqual(postflight["worker_pids"], [])
        self.assertEqual(postflight["gpu_processes"], [])


if __name__ == "__main__":
    unittest.main()
