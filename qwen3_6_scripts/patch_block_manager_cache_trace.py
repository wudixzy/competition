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
            "_seq_group": seq_group,
        }

    def _bi100_update_cache_trace(
            self, seq, raw_kv_hit_blocks, restore_key, capture_actions,
            evict_keys, policy) -> None:
        if os.getenv("BI100_CACHE_TRACE", "0") != "1":
            return
        requests = getattr(self, "_bi100_trace_requests", None)
        if not requests or seq.seq_id not in requests:
            return
        record = requests[seq.seq_id]
        record["gdn_policy"] = policy
        record["raw_kv_contiguous_hit_blocks"] = max(
            int(raw_kv_hit_blocks),
            int(record.get("raw_kv_contiguous_hit_blocks", 0)))
        effective_blocks = int(restore_key[0]) if restore_key is not None else 0
        record["effective_gdn_hit_blocks"] = max(
            effective_blocks, int(record.get("effective_gdn_hit_blocks", 0)))

        admissions = record.setdefault("gdn_admissions", [])
        for key, reason in capture_actions:
            admissions.append({
                "block_count": int(key[0]),
                "digest_base64": base64.b64encode(key[1]).decode("ascii"),
                "reason": str(reason),
            })
        evictions = record.setdefault("gdn_evictions", [])
        for key in evict_keys:
            evictions.append({
                "block_count": int(key[0]),
                "digest_base64": base64.b64encode(key[1]).decode("ascii"),
                "reason": "capacity_lru",
            })

    def _bi100_emit_cache_trace(self, seq, block_table) -> None:
        if os.getenv("BI100_CACHE_TRACE", "0") != "1":
            return

        requests = getattr(self, "_bi100_trace_requests", None)
        if not requests:
            return

        record = requests.pop(seq.seq_id, None)
        if record is None:
            return

        seq_group = record.pop("_seq_group", None)
        metrics = getattr(seq_group, "metrics", None)
        if metrics is not None:
            arrival = getattr(metrics, "arrival_time", None)
            first_token = getattr(metrics, "first_token_time", None)
            finished = getattr(metrics, "finished_time", None)
            queue = getattr(metrics, "time_in_queue", None)
            cached = getattr(metrics, "num_cached_tokens", None)
            if arrival is not None and first_token is not None:
                record["ttft_s"] = max(0.0, float(first_token - arrival))
            if arrival is not None and finished is not None:
                record["request_latency_s"] = max(
                    0.0, float(finished - arrival))
            if queue is not None:
                record["time_in_queue_s"] = max(0.0, float(queue))
            if cached is not None:
                record["observed_effective_cached_tokens"] = max(
                    0, int(cached))

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
        generated_tokens = max(0, total_tokens - record["prompt_tokens"])
        record["generated_tokens"] = generated_tokens
        ttft_s = record.get("ttft_s")
        if isinstance(ttft_s, (int, float)) and ttft_s > 0:
            record["observed_input_tps"] = record["prompt_tokens"] / ttft_s
        if (metrics is not None and generated_tokens > 1
                and getattr(metrics, "first_token_time", None) is not None
                and getattr(metrics, "finished_time", None) is not None):
            decode_s = metrics.finished_time - metrics.first_token_time
            if decode_s > 0:
                record["observed_output_tps"] = (
                    (generated_tokens - 1) / decode_s)
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
