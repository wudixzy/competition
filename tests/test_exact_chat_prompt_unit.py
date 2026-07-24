from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/exact_chat_prompt.py"
SPEC = importlib.util.spec_from_file_location("exact_prompt", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CharacterTokenizer:
    chat_template = "unit-template"

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, **kwargs):
        text = "|".join(
            message.get("content", "")
            for message in messages
            if isinstance(message.get("content"), str)
        )
        tools = json.dumps(kwargs.get("tools") or [], sort_keys=True)
        thinking = kwargs.get("enable_thinking")
        if thinking is None:
            thinking = (kwargs.get("chat_template_kwargs") or {}).get(
                "enable_thinking")
        rendered = f"chat:{text}:tools:{tools}:thinking:{int(bool(thinking))}"
        return self.encode(rendered)


class ConstantTemplateTokenizer(CharacterTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]


class ExactChatPromptTest(unittest.TestCase):

    def test_token_id_filler_revalidates_to_exact_template_length(self):
        tokenizer = CharacterTokenizer()

        def recipe(filler):
            return [{"role": "user", "content": "prefix:" + filler}], None

        messages, tools, evidence = MODULE.fit_exact_chat_prompt(
            tokenizer,
            4096,
            recipe,
            seed=20260724,
            namespace="unit",
            template_kwargs_mode="direct",
        )
        self.assertIsNone(tools)
        self.assertEqual(evidence["local_prompt_tokens"], 4096)
        self.assertEqual(MODULE.chat_template_token_count(
            tokenizer, messages, template_kwargs_mode="direct"), 4096)
        self.assertNotIn("prefix:", json.dumps(evidence))

    def test_direct_and_nested_template_modes_are_explicit(self):
        tokenizer = CharacterTokenizer()
        messages = [{"role": "user", "content": "probe"}]
        direct = MODULE.chat_template_token_ids(
            tokenizer, messages, thinking=True, template_kwargs_mode="direct")
        nested = MODULE.chat_template_token_ids(
            tokenizer, messages, thinking=True, template_kwargs_mode="nested")
        self.assertEqual(direct, nested)
        with self.assertRaisesRegex(
                MODULE.PromptConstructionError, "unsupported"):
            MODULE.chat_template_token_ids(
                tokenizer, messages, template_kwargs_mode="automatic")

    def test_unreachable_target_fails_with_construction_reason(self):
        tokenizer = ConstantTemplateTokenizer()
        with self.assertRaisesRegex(
                MODULE.PromptConstructionError, "closest_delta"):
            MODULE.fit_exact_chat_prompt(
                tokenizer,
                10,
                lambda filler: (
                    [{"role": "user", "content": filler}], None),
                seed=1,
                namespace="unreachable",
            )

    def test_tokenizer_identity_hashes_only_frozen_artifacts(self):
        tokenizer = CharacterTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text(
                '{"model_type":"unit"}\n', encoding="utf-8")
            (model_path / "tokenizer_config.json").write_text(
                '{"unit":true}\n', encoding="utf-8")
            identity = MODULE.tokenizer_identity(model_path, tokenizer)
        self.assertEqual(identity["tokenizer_class"], "CharacterTokenizer")
        self.assertEqual(
            [row["name"] for row in identity["files"]],
            ["config.json", "tokenizer_config.json"],
        )
        self.assertEqual(len(identity["artifact_set_sha256"]), 64)
        self.assertEqual(len(identity["chat_template_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
