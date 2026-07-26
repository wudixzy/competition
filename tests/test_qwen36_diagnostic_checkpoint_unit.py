#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_qwen36_diagnostic_checkpoint as builder  # noqa: E402
import verify_qwen36_diagnostic_checkpoint as verifier  # noqa: E402


def _write_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    ordered = sorted(tensors.items())
    for name, payload in ordered:
        header[name] = {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for _, payload in ordered:
            stream.write(payload)


def _source_config() -> dict[str, object]:
    layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ] * 2
    return {
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            **builder.TARGET_TEXT_CONTRACT,
            "num_hidden_layers": len(layer_types),
            "layer_types": layer_types,
            "vocab_size": 248320,
        },
        "vision_config": {
            "out_hidden_size": 2048,
        },
    }


def _layer_tensor_names(layer: int, layer_type: str) -> list[str]:
    prefix = f"model.language_model.layers.{layer}."
    names = [prefix + suffix for suffix in builder.COMMON_REQUIRED_SUFFIXES]
    if layer_type == "linear_attention":
        names.extend(prefix + suffix
                     for suffix in builder.LINEAR_REQUIRED_SUFFIXES)
    else:
        names.extend(prefix + suffix
                     for suffix in builder.FULL_REQUIRED_SUFFIXES)
    return sorted(names)


def _create_source(root: Path) -> Path:
    root.mkdir()
    config = _source_config()
    (root / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        '{"version":"diagnostic-test"}\n', encoding="utf-8")
    (root / "chat_template.jinja").write_text(
        "{{ messages }}\n", encoding="utf-8")
    (root / ".config.json.swp").write_bytes(b"must-not-copy")
    (root / "config.json~").write_bytes(b"must-not-copy")

    shards: dict[str, dict[str, bytes]] = {
        "model-00001-of-00002.safetensors": {},
        "model-00002-of-00002.safetensors": {},
    }
    weight_map: dict[str, str] = {}

    global_names = sorted(builder.GLOBAL_REQUIRED_WEIGHTS | {
        "model.visual.blocks.0.attn.proj.weight",
        "mtp.fc.weight",
    })
    for index, name in enumerate(global_names):
        shard = "model-00001-of-00002.safetensors"
        payload = bytes([index + 1]) * (index + 3)
        shards[shard][name] = payload
        weight_map[name] = shard

    layer_types = config["text_config"]["layer_types"]  # type: ignore[index]
    for layer, layer_type in enumerate(layer_types):
        shard = (
            "model-00001-of-00002.safetensors"
            if layer < 4 else "model-00002-of-00002.safetensors"
        )
        for index, name in enumerate(_layer_tensor_names(layer, layer_type)):
            payload = bytes([(layer + index + 1) % 251]) * (index + 1)
            shards[shard][name] = payload
            weight_map[name] = shard

    for shard, tensors in shards.items():
        _write_safetensors(root / shard, tensors)
    total_size = sum(
        len(payload) for tensors in shards.values() for payload in tensors.values()
    )
    (root / builder.INDEX_NAME).write_text(
        json.dumps({
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


class Qwen36DiagnosticCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = _create_source(self.root / "source")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_build_preserves_exact_contract_and_filters_depth(self) -> None:
        output = self.root / "diagnostic"
        manifest = builder.build_checkpoint(self.source, output, 1)

        config = json.loads((output / "config.json").read_text())
        self.assertEqual(config["text_config"]["num_hidden_layers"], 4)
        self.assertEqual(
            config["text_config"]["layer_types"],
            ["linear_attention"] * 3 + ["full_attention"],
        )
        index = json.loads((output / builder.INDEX_NAME).read_text())
        names = set(index["weight_map"])
        self.assertTrue(builder.GLOBAL_REQUIRED_WEIGHTS.issubset(names))
        self.assertTrue(any(name.startswith("model.visual.") for name in names))
        self.assertTrue(any(name.startswith("mtp.") for name in names))
        self.assertFalse(any(
            builder._layer_index(name) is not None
            and builder._layer_index(name) >= 4
            for name in names
        ))
        self.assertTrue((output / "tokenizer.json").is_file())
        self.assertFalse((output / ".config.json.swp").exists())
        self.assertFalse((output / "config.json~").exists())
        self.assertEqual(
            manifest["diagnostic"]["tensor_payload_transform"],
            "none-byte-for-byte-copy",
        )

        report = verifier.verify_checkpoint(
            self.source,
            output,
            full_hash=True,
            compare_source_bytes=True,
        )
        self.assertTrue(report["qualified"])
        self.assertTrue(report["tensor_contract_preserved"])
        self.assertFalse(report["production_promotion_authorized"])
        self.assertEqual(report["layer_count"], 4)

    def test_dry_plan_reports_one_complete_cycle(self) -> None:
        plan = builder.describe_plan(self.source, 1)
        self.assertEqual(plan["retained_layer_count"], 4)
        self.assertEqual(plan["retained_layer_indices"], [0, 1, 2, 3])
        self.assertEqual(
            plan["retained_layer_types"],
            ["linear_attention"] * 3 + ["full_attention"],
        )
        self.assertTrue(plan["preserves_visual_weights"])
        self.assertFalse(plan["tensor_bytes_transformed"])

    def test_rejects_broken_cycle_layout(self) -> None:
        config_path = self.source / "config.json"
        config = json.loads(config_path.read_text())
        config["text_config"]["layer_types"][3] = "linear_attention"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(
                builder.CheckpointError, "not 3 GDN"):
            builder.describe_plan(self.source, 1)

    def test_rejects_existing_output(self) -> None:
        output = self.root / "diagnostic"
        output.mkdir()
        sentinel = output / "keep"
        sentinel.write_text("user data", encoding="utf-8")
        with self.assertRaisesRegex(
                builder.CheckpointError, "output already exists"):
            builder.build_checkpoint(self.source, output, 1)
        self.assertEqual(sentinel.read_text(), "user data")

    def test_full_hash_detects_payload_tampering(self) -> None:
        output = self.root / "diagnostic"
        builder.build_checkpoint(self.source, output, 1)
        shard = sorted(output.glob("*.safetensors"))[0]
        with shard.open("r+b") as stream:
            stream.seek(-1, 2)
            value = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 0xFF]))
        with self.assertRaisesRegex(
                builder.CheckpointError, "SHA-256 differs"):
            verifier.verify_checkpoint(
                self.source, output, full_hash=True)

    def test_rejects_non_target_tensor_contract(self) -> None:
        config_path = self.source / "config.json"
        config = json.loads(config_path.read_text())
        config["text_config"]["num_experts"] = 8
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(
                builder.CheckpointError, "tensor contract"):
            builder.describe_plan(self.source, 1)


if __name__ == "__main__":
    unittest.main()
