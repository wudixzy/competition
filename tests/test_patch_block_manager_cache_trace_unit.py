import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).parents[1]
SCRIPT = ROOT / "qwen3_6_scripts" / "patch_block_manager_cache_trace.py"


def fake_package(root, source):
    (root / "vllm" / "core").mkdir(parents=True)
    for path in (root / "vllm", root / "vllm" / "core"):
        (path / "__init__.py").write_text("")
    (root / "vllm" / "core" / "block_manager_v2.py").write_text(source)


class PatchTest(unittest.TestCase):
    def test_subprocess_patch_is_valid_and_idempotent(self):
        source = (
            "import hashlib\n"
            "import struct\n"
            "from collections.abc import Mapping\n"
            "from typing import Any, Dict, List, Optional, Sequence as GenericSequence, Tuple\n"
            "class BlockSpaceManagerV2(BlockSpaceManager):\n"
            "    def allocate(self, seq_group):\n"
            "        block_table = BlockTable()\n"
            "        self.block_tables[seq.seq_id] = block_table\n\n"
            "        # Track seq\n"
            "    def free(self, seq):\n"
            "        seq_id = seq.seq_id\n"
            "        if seq_id not in self.block_tables:\n"
            "            return\n"
            "        self._last_access_blocks_tracker.update_seq_blocks_last_access(\n"
            "            seq_id, self.block_tables[seq_id].physical_block_ids)\n\n"
            "        # Untrack seq\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            fake_package(root, source)
            env = dict(
                os.environ,
                PYTHONPATH=str(ROOT / "qwen3_6_scripts")
                + os.pathsep
                + str(root),
            )
            subprocess.run([sys.executable, str(SCRIPT)], env=env, check=True,
                           capture_output=True)
            target = root / "vllm/core/block_manager_v2.py"
            first = target.read_text()
            self.assertEqual(first.count("def _bi100_capture_cache_trace("), 1)
            self.assertEqual(first.count("self._bi100_capture_cache_trace("), 1)
            self.assertEqual(first.count("_bi100_capture_cache_trace"), 2)
            self.assertIn(
                "self.block_tables[seq.seq_id] = block_table\n"
                "        self._bi100_capture_cache_trace(\n"
                "            seq_group, seq, block_table)",
                first,
            )
            self.assertEqual(first.count("def _bi100_emit_cache_trace("), 1)
            self.assertEqual(first.count("self._bi100_emit_cache_trace("), 1)
            self.assertEqual(first.count("_bi100_emit_cache_trace"), 2)
            self.assertIn(
                "self._last_access_blocks_tracker.update_seq_blocks_last_access(\n"
                "            seq_id, self.block_tables[seq_id].physical_block_ids)\n"
                "        self._bi100_emit_cache_trace(\n"
                "            seq, self.block_tables[seq_id])",
                first,
            )
            subprocess.run([sys.executable, str(SCRIPT)], env=env, check=True,
                           capture_output=True)
            self.assertEqual(first, target.read_text())
            subprocess.run([sys.executable, "-m", "py_compile", str(target)],
                           check=True)

    def test_unknown_layout_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            fake_package(root, "from typing import Dict, List, Optional\nclass Wrong: pass\n")
            env = dict(
                os.environ,
                PYTHONPATH=str(ROOT / "qwen3_6_scripts")
                + os.pathsep
                + str(root),
            )
            self.assertNotEqual(subprocess.run(
                [sys.executable, str(SCRIPT)], env=env, capture_output=True
            ).returncode, 0)

    def test_helper_runtime_contract(self):
        spec = importlib.util.spec_from_file_location("trace_patch", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "qwen3_6_scripts"))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
            sys.path.pop(0)

        namespace = {"os": os, "json": json, "base64": base64,
                     "hashlib": hashlib, "struct": __import__("struct")}
        exec("class Fake:\n" + module.HELPER, namespace)

        hash_a = hashlib.sha256((4).to_bytes(4, "big")).digest()
        hash_b = hashlib.sha256((7).to_bytes(4, "big")).digest()
        truncated_hash = b"0" * 16

        class Seq:
            seq_id = 7
            token_ids = [10, 11, 12, 13, 14]

            def get_token_ids(self):
                return self.token_ids

        class Group:
            request_id = "secret-request"

            def __init__(self):
                self.metrics = SimpleNamespace(
                    arrival_time=10.0,
                    first_token_time=12.0,
                    finished_time=14.0,
                    time_in_queue=0.25,
                    num_cached_tokens=4,
                )

        class Table:
            def __init__(self, values):
                self.values = values

            def get_content_hashes(self):
                return self.values

        fake = namespace["Fake"]()
        fake.block_size = 4
        fake.num_total_gpu_blocks = 9
        seq = Seq()
        output = io.StringIO()

        old = os.environ.pop("BI100_CACHE_TRACE", None)
        try:
            with contextlib.redirect_stdout(output):
                fake._bi100_capture_cache_trace(Group(), seq, Table([hash_a]))
                fake._bi100_emit_cache_trace(seq, Table([hash_a]))
            self.assertEqual(output.getvalue(), "")

            os.environ["BI100_CACHE_TRACE"] = "1"
            with contextlib.redirect_stdout(output):
                group = Group()
                fake._bi100_capture_cache_trace(group, seq, Table([hash_a]))
                fake._bi100_update_cache_trace(
                    seq, 2, (1, hash_a),
                    [((1, hash_a), "final_prefill")],
                    [(2, hash_b)], "admission64")
                seq.token_ids = [10, 11, 12, 13, 14, 15, 16, 17]
                fake._bi100_emit_cache_trace(seq, Table([hash_a, hash_b]))
            record = json.loads(output.getvalue().split(" ", 1)[1])
            self.assertEqual(record["version"], 4)
            self.assertEqual(record["ordinal"], 1)
            self.assertEqual(len(record["trace_session_sha256"]), 16)
            self.assertEqual(record["request_id_sha256"],
                             hashlib.sha256(
                                 str(Group.request_id).encode("utf-8")).hexdigest()[:16])
            self.assertEqual(record["block_size"], 4)
            self.assertEqual(record["capacity_blocks"], 9)
            self.assertEqual(record["prompt_tokens"], 5)
            self.assertEqual(record["prompt_allocated_blocks"], 2)
            self.assertEqual(record["total_tokens"], 8)
            self.assertEqual(record["allocated_blocks"], 2)
            self.assertEqual(record["full_blocks"], 2)
            self.assertEqual(record["hash_encoding"], "sha256_base64")
            self.assertEqual(record["gdn_policy"], "admission64")
            self.assertEqual(record["raw_kv_contiguous_hit_blocks"], 2)
            self.assertEqual(record["effective_gdn_hit_blocks"], 1)
            self.assertEqual(record["gdn_admissions"][0]["reason"],
                             "final_prefill")
            self.assertEqual(record["gdn_evictions"][0]["reason"],
                             "capacity_lru")
            self.assertEqual(record["ttft_s"], 2.0)
            self.assertEqual(record["request_latency_s"], 4.0)
            self.assertEqual(record["time_in_queue_s"], 0.25)
            self.assertEqual(record["observed_effective_cached_tokens"], 4)
            self.assertEqual(record["generated_tokens"], 3)
            self.assertEqual(
                base64.b64decode(record["block_hashes"]),
                hash_a + hash_b
            )
            self.assertNotIn("secret-request", output.getvalue())

            second = io.StringIO()
            with contextlib.redirect_stdout(second):
                seq.token_ids = [10, 11, 12, 13, 14]
                fake._bi100_capture_cache_trace(Group(), seq, Table([hash_a]))
                fake._bi100_emit_cache_trace(seq, Table([hash_a]))
            second_record = json.loads(second.getvalue().split(" ", 1)[1])
            self.assertEqual(second_record["ordinal"], 2)
            self.assertEqual(second_record["trace_session_sha256"],
                             record["trace_session_sha256"])

            bad = io.StringIO()
            with self.assertRaisesRegex(RuntimeError, "32-byte"):
                with contextlib.redirect_stdout(bad):
                    fake._bi100_capture_cache_trace(
                        Group(), seq, Table([hash_a]))
                    fake._bi100_emit_cache_trace(
                        seq, Table([truncated_hash]))
            self.assertEqual(bad.getvalue(), "")
        finally:
            if old is None:
                os.environ.pop("BI100_CACHE_TRACE", None)
            else:
                os.environ["BI100_CACHE_TRACE"] = old


if __name__ == "__main__":
    unittest.main()
