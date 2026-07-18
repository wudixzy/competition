"""Install the optional BI100 prefix-cache diagnostic trace."""
from patch_utils import package_root, replace_once

VLLM_ROOT = package_root("vllm")
TARGET = VLLM_ROOT / "core" / "block_manager_v2.py"

HELPER = '''
    def _bi100_capture_cache_trace(self, seq_group, seq, block_table) -> None:
        if os.getenv("BI100_CACHE_TRACE", "0") != "1":
            return

        session = getattr(self, "_bi100_trace_session", None)
        if session is None:
            session = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
            self._bi100_trace_session = session

        self._bi100_trace_ordinal = getattr(self, "_bi100_trace_ordinal", 0) + 1
        request_id_sha256 = hashlib.sha256(
            str(seq_group.request_id).encode("utf-8")).hexdigest()[:16]

        prompt_tokens = len(seq.get_token_ids())
        requests = getattr(self, "_bi100_trace_requests", None)
        if requests is None:
            requests = {}
            self._bi100_trace_requests = requests

        requests[seq.seq_id] = {
            "version": 4,
            "trace_session_sha256": session,
            "ordinal": self._bi100_trace_ordinal,
            "request_id_sha256": request_id_sha256,
            "prompt_tokens": prompt_tokens,
            "prompt_allocated_blocks": (
                (prompt_tokens + self.block_size - 1) // self.block_size
            ),
            "block_size": self.block_size,
            "capacity_blocks": self.num_total_gpu_blocks,
        }

    def _bi100_emit_cache_trace(self, seq, block_table) -> None:
        if os.getenv("BI100_CACHE_TRACE", "0") != "1":
            return

        requests = getattr(self, "_bi100_trace_requests", None)
        if not requests:
            return

        record = requests.pop(seq.seq_id, None)
        if record is None:
            return

        total_tokens = len(seq.get_token_ids())
        block_hashes = block_table.get_content_hashes()
        for block_hash in block_hashes:
            if not isinstance(block_hash, bytes) or len(block_hash) != 32:
                raise RuntimeError(
                    "BI100 cache trace requires 32-byte content hashes")
        full_blocks = len(block_hashes)
        record.update({
            "total_tokens": total_tokens,
            "allocated_blocks": (
                (total_tokens + self.block_size - 1) // self.block_size
            ),
            "full_blocks": full_blocks,
            "hash_encoding": "sha256_base64",
            "block_hashes": base64.b64encode(b"".join(block_hashes)).decode("ascii"),
        })
        print("[BI100_CACHE_TRACE] " + json.dumps(record, separators=(",", ":"),
                                             sort_keys=True), flush=True)
'''


def main():
    replace_once(TARGET, "from collections.abc import Mapping\n",
                 "from collections.abc import Mapping\nimport base64\nimport json\nimport os\n",
                 required=True, already_contains="import base64\n")
    replace_once(TARGET, "class BlockSpaceManagerV2(BlockSpaceManager):\n",
                 "class BlockSpaceManagerV2(BlockSpaceManager):\n" + HELPER,
                 required=True, already_contains="def _bi100_capture_cache_trace(")
    replace_once(TARGET,
                 "        self.block_tables[seq.seq_id] = block_table\n\n        # Track seq",
                 "        self.block_tables[seq.seq_id] = block_table\n        self._bi100_capture_cache_trace(\n            seq_group, seq, block_table)\n\n        # Track seq",
                 required=True,
                 already_contains="self.block_tables[seq.seq_id] = block_table\n"
                                "        self._bi100_capture_cache_trace(")
    replace_once(TARGET,
                 "        self._last_access_blocks_tracker.update_seq_blocks_last_access(\n"
                 "            seq_id, self.block_tables[seq_id].physical_block_ids)\n\n        # Untrack seq",
                 "        self._last_access_blocks_tracker.update_seq_blocks_last_access(\n"
                 "            seq_id, self.block_tables[seq_id].physical_block_ids)\n        self._bi100_emit_cache_trace(\n            seq, self.block_tables[seq_id])\n\n        # Untrack seq",
                 required=True,
                 already_contains="self._last_access_blocks_tracker.update_seq_blocks_last_access(\n"
                                "            seq_id, self.block_tables[seq_id].physical_block_ids)\n"
                                "        self._bi100_emit_cache_trace(")


if __name__ == "__main__":
    main()
