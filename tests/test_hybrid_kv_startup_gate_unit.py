from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/hybrid_kv_startup_gate.py"
SPEC = importlib.util.spec_from_file_location("hybrid_startup", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
FULL_ATTENTION_ORDINALS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]


def _log(
    gpu_blocks: int = 67_512,
    cpu_blocks: int = 26_212,
    mode: str = "full_attention",
    configured_layers: int = 10,
    model_path: str = "/model",
    cache_trace: str = "0",
    cpu_kv_offload: str = "0",
    block_major_cpu_kv: str = "0",
    kernel_profile: str = "submission",
) -> str:
    service_contract = dict(MODULE.FIXED_SERVICE_CONTRACT)
    service_contract.update(MODULE.KERNEL_PROFILES[kernel_profile])
    service_contract.update({
        "accounting": mode,
        "cache_trace": cache_trace,
        "cpu_kv_offload": cpu_kv_offload,
        "block_major_cpu_kv": block_major_cpu_kv,
        "model_path": model_path,
        "runtime_site_packages": "system",
    })
    contract_line = " ".join(
        f"{name}={value}" for name, value in service_contract.items())
    accounting = "".join(
        f"[BI100] Qwen hybrid KV accounting; tp_rank={rank} "
        f"env_mode={mode} config_mode={mode} "
        f"configured_kv_layers={configured_layers} "
        "full_attention_layers=10 "
        "full_attention_ordinals=3,7,11,15,19,23,27,31,35,39\n"
        for rank in range(4)
    )
    block_major = ""
    if block_major_cpu_kv == "1":
        block_major = "".join(
            "[BI100 BLOCK KV] capacity reserve "
            "blocks=1024 bytes=167772160 "
            "profiled_blocks=68536 usable_blocks=67512\n"
            f"[BI100 BLOCK KV] enabled device=cuda:{rank} "
            "gpu_blocks=67512 cpu_blocks=26212 layers=10 "
            "block_bytes=163840 staging_blocks=512 staging_buffers=2\n"
            for rank in range(4)
        )
    return (
        f"[BI100] M1-49 runtime contract; {contract_line}\n"
        "Initializing engine dtype=torch.float16 max_seq_len=262144 "
        "block_size=16 swap_space=4\n"
        f"# GPU blocks: {gpu_blocks}, # CPU blocks: {cpu_blocks}\n"
        + accounting
        + block_major
    )


def _contract(mode: str = "full_attention") -> dict:
    return {
        "model_config_sha256": "a" * 64,
        "service": {"accounting": mode},
        "engine": {
            "max_seq_len": 262_144,
            "block_size": 16,
            "swap_space_gib": 4.0,
            "dtype": "float16",
        },
    }


class HybridKvStartupGateTest(unittest.TestCase):

    def test_bare_host_installer_stages_docker_overrides(self):
        source = (ROOT / "scripts/install_bi100_bare_host_runtime.sh").read_text(
            encoding="utf-8")
        self.assertIn("vendor_overrides/vllm/core/block", source)
        self.assertIn("bash ./patch_ops.sh", source)
        self.assertIn('cd /tmp', source)
        self.assertIn('versions["transformers"] == "4.55.3"', source)
        self.assertIn("get_num_attention_layers", source)
        self.assertIn("SYSTEM_PYTHONPATH", source)
        self.assertIn("system_site_packages_modified", source)
        self.assertIn("expected_runtime", source)
        self.assertIn("block_manager_base_sha256", source)
        self.assertIn('"cache_trace_outputs"', source)
        self.assertIn('"$EXPECTED_ROOT/vllm/outputs.py"', source)
        self.assertIn('"source_revision": source_revision', source)
        self.assertIn('"cache_trace_patcher_sha256"', source)
        self.assertIn('"offline_metadata_normalizer_sha256"', source)
        self.assertIn("normalize_offline_distribution.py", source)
        self.assertIn("refuses a dirty source tree", source)
        self.assertIn('mv "$RUNTIME_STAGE" "$RUNTIME_ROOT"', source)
        self.assertIn("resolved outside staged overlay", source)

    def test_model_emits_runtime_accounting_contract(self):
        source = (ROOT / "qwen3_6_scripts/qwen3_5.py").read_text(
            encoding="utf-8")
        self.assertIn(
            "[BI100] Qwen hybrid KV accounting; tp_rank=%d ",
            source)
        self.assertIn("full_attention_ordinals=%s", source)

    def test_candidate_contract_qualifies(self):
        layers = ["linear_attention"] * 40
        for index in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39):
            layers[index] = "attention"
        report = MODULE.evaluate(
            _log(),
            mode="full_attention",
            config_mode="full_attention",
            layers_block_type=layers,
            full_attention_ordinals=FULL_ATTENTION_ORDINALS,
            num_key_value_heads=2,
            head_dim=256,
            max_model_len=262_144,
            block_size=16,
            tensor_parallel_size=4,
            runtime_contract=_contract(),
        )
        self.assertTrue(report["qualified"], report)
        self.assertEqual(report["observed_attention_layers"], 10)
        self.assertEqual(report["observed_gpu_tokens"], 1_080_192)
        self.assertEqual(report["expected_kv_bytes_per_block"], 163_840)

    def test_runtime_contract_is_parsed_from_the_service_log(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = pathlib.Path(directory)
            (model_path / "config.json").write_text(
                '{"model_type":"qwen3_5_moe"}\n', encoding="utf-8")
            contract, reasons = MODULE._runtime_contract(
                _log(model_path=str(model_path)),
                model_path,
                mode="full_attention",
                max_model_len=262_144,
                block_size=16,
                tensor_parallel_size=4,
            )
        self.assertEqual(reasons, [])
        self.assertEqual(
            contract["service"]["accounting"], "full_attention")
        self.assertEqual(contract["engine"]["block_size"], 16)

    def test_diagnostic_trace_contract_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = pathlib.Path(directory)
            (model_path / "config.json").write_text(
                '{}\n', encoding="utf-8")
            contract, reasons = MODULE._runtime_contract(
                _log(model_path=str(model_path), cache_trace="1"),
                model_path,
                mode="full_attention",
                max_model_len=262_144,
                block_size=16,
                tensor_parallel_size=4,
                expected_cache_trace="1",
            )
            _, mismatch_reasons = MODULE._runtime_contract(
                _log(model_path=str(model_path), cache_trace="1"),
                model_path,
                mode="full_attention",
                max_model_len=262_144,
                block_size=16,
                tensor_parallel_size=4,
            )
        self.assertEqual(reasons, [])
        self.assertEqual(contract["service"]["cache_trace"], "1")
        self.assertTrue(any(
            "cache_trace" in reason for reason in mismatch_reasons))

    def test_strict_reference_kernel_profile_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = pathlib.Path(directory)
            (model_path / "config.json").write_text(
                '{}\n', encoding="utf-8")
            contract, reasons = MODULE._runtime_contract(
                _log(
                    model_path=str(model_path),
                    kernel_profile="strict-reference",
                ),
                model_path,
                mode="full_attention",
                max_model_len=262_144,
                block_size=16,
                tensor_parallel_size=4,
                expected_kernel_profile="strict-reference",
            )
            _, submission_reasons = MODULE._runtime_contract(
                _log(
                    model_path=str(model_path),
                    kernel_profile="strict-reference",
                ),
                model_path,
                mode="full_attention",
                max_model_len=262_144,
                block_size=16,
                tensor_parallel_size=4,
            )
        self.assertEqual(reasons, [])
        self.assertEqual(contract["service"]["moe_direct"], "0")
        self.assertEqual(contract["service"]["gdn_packed"], "0")
        self.assertTrue(any(
            "moe_direct" in reason for reason in submission_reasons))
        self.assertTrue(any(
            "gdn_packed" in reason for reason in submission_reasons))

    def test_block_major_candidate_contract_and_rank_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = pathlib.Path(directory)
            (model_path / "config.json").write_text(
                '{}\n', encoding="utf-8")
            log = _log(
                model_path=str(model_path),
                cpu_kv_offload="1",
                block_major_cpu_kv="1",
            )
            contract, reasons = MODULE._runtime_contract(
                log,
                model_path,
                mode="full_attention",
                max_model_len=262_144,
                block_size=16,
                tensor_parallel_size=4,
                expected_cpu_kv_offload="1",
                expected_block_major_cpu_kv="1",
            )
        layers = ["linear_attention"] * 40
        for index in FULL_ATTENTION_ORDINALS:
            layers[index] = "attention"
        report = MODULE.evaluate(
            log,
            mode="full_attention",
            config_mode="full_attention",
            layers_block_type=layers,
            full_attention_ordinals=FULL_ATTENTION_ORDINALS,
            num_key_value_heads=2,
            head_dim=256,
            max_model_len=262_144,
            block_size=16,
            tensor_parallel_size=4,
            runtime_contract=contract,
            contract_reasons=reasons,
            block_major_cpu_kv=True,
        )
        self.assertTrue(report["qualified"], report)
        self.assertEqual(len(report["block_major_capacity_reports"]), 4)
        self.assertEqual(len(report["block_major_cache_reports"]), 4)

    def test_invalid_expected_trace_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = pathlib.Path(directory)
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                MODULE._runtime_contract(
                    "", model_path,
                    mode="full_attention",
                    max_model_len=262_144,
                    block_size=16,
                    tensor_parallel_size=4,
                    expected_cache_trace="yes",
                )

    def test_duplicate_service_contract_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = pathlib.Path(directory)
            (model_path / "config.json").write_text("{}\n", encoding="utf-8")
            log = _log(model_path=str(model_path))
            _, reasons = MODULE._runtime_contract(
                log + log.splitlines()[0] + "\n",
                model_path,
                mode="full_attention",
                max_model_len=262_144,
                block_size=16,
                tensor_parallel_size=4,
            )
        self.assertTrue(any("exactly one" in reason for reason in reasons))

    def test_legacy_contract_qualifies(self):
        report = MODULE.evaluate(
            _log(16_878, 6_553, "legacy40", 40),
            mode="legacy40",
            config_mode="legacy40",
            layers_block_type=["attention"] * 40,
            full_attention_ordinals=FULL_ATTENTION_ORDINALS,
            num_key_value_heads=2,
            head_dim=256,
            max_model_len=262_144,
            block_size=16,
            tensor_parallel_size=4,
            runtime_contract=_contract("legacy40"),
        )
        self.assertTrue(report["qualified"], report)
        self.assertEqual(report["expected_kv_bytes_per_block"], 655_360)

    def test_wrong_mode_or_layer_count_fails(self):
        report = MODULE.evaluate(
            _log(),
            mode="full_attention",
            config_mode="legacy40",
            layers_block_type=["attention"] * 40,
            full_attention_ordinals=FULL_ATTENTION_ORDINALS,
            num_key_value_heads=2,
            head_dim=256,
            max_model_len=262_144,
            block_size=16,
            tensor_parallel_size=4,
            runtime_contract=_contract(),
        )
        self.assertFalse(report["qualified"])
        self.assertTrue(any("serialized config mode" in reason
                            for reason in report["reasons"]))
        self.assertTrue(any("attention layer count" in reason
                            for reason in report["reasons"]))

    def test_missing_or_insufficient_capacity_fails(self):
        report = MODULE.evaluate(
            _log(100, 0, "legacy40", 40),
            mode="legacy40",
            config_mode="legacy40",
            layers_block_type=["attention"] * 40,
            full_attention_ordinals=FULL_ATTENTION_ORDINALS,
            num_key_value_heads=2,
            head_dim=256,
            max_model_len=262_144,
            block_size=16,
            tensor_parallel_size=4,
            runtime_contract=_contract(),
        )
        self.assertFalse(report["qualified"])
        self.assertTrue(any("below required" in reason
                            for reason in report["reasons"]))
        self.assertTrue(any("CPU block count" in reason
                            for reason in report["reasons"]))

    def test_missing_runtime_rank_report_fails(self):
        report = MODULE.evaluate(
            _log().replace(
                "[BI100] Qwen hybrid KV accounting;", "ignored;", 2),
            mode="full_attention",
            config_mode="full_attention",
            layers_block_type=(
                ["linear_attention"] * 30 + ["attention"] * 10),
            full_attention_ordinals=FULL_ATTENTION_ORDINALS,
            num_key_value_heads=2,
            head_dim=256,
            max_model_len=262_144,
            block_size=16,
            tensor_parallel_size=4,
            runtime_contract=_contract(),
        )
        self.assertFalse(report["qualified"])
        self.assertTrue(any("per TP rank" in reason
                            for reason in report["reasons"]))

    def test_duplicate_runtime_rank_report_fails(self):
        report = MODULE.evaluate(
            _log().replace("tp_rank=3", "tp_rank=2"),
            mode="full_attention",
            config_mode="full_attention",
            layers_block_type=(
                ["linear_attention"] * 30 + ["attention"] * 10),
            full_attention_ordinals=FULL_ATTENTION_ORDINALS,
            num_key_value_heads=2,
            head_dim=256,
            max_model_len=262_144,
            block_size=16,
            tensor_parallel_size=4,
            runtime_contract=_contract(),
        )
        self.assertFalse(report["qualified"])
        self.assertTrue(any("TP ranks" in reason
                            for reason in report["reasons"]))


if __name__ == "__main__":
    unittest.main()
