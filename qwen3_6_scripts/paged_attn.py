from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import sys
import torch
import traceback
from vllm import _custom_ops as ops
from vllm.bi100_env import env_bool, env_int
from vllm.bi100_profile import bi100_timer

try:
    from vllm import corex_paged_kv_gather as _corex_paged_kv_gather
except ImportError:
    _corex_paged_kv_gather = None

try:
    from vllm import corex_fused_paged_prefill as _corex_fused_paged_prefill
except ImportError:
    _corex_fused_paged_prefill = None

# from vllm.attention.ops.prefix_prefill import context_attention_fwd
# NOTE: context_attention_fwd (Triton kernel from prefix_prefill.py) is NOT
# imported here.  On Iluvatar BI-V100 that kernel hangs the GPU card
# permanently.  Chunked-prefill / prefix-caching attention is handled by
# _forward_prefix_pytorch below (pure PyTorch, no Triton dependency).

# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512
_PYTORCH_DECODE_THRESHOLD = env_int(
    "BI100_PYTORCH_DECODE_THRESHOLD", 32768, 1, 262144)
_PREFIX_BLOCKS_PER_TILE = env_int(
    "BI100_PREFIX_BLOCKS_PER_TILE", 32, 1, 1024)
_FORCE_PAGED_ATTN_V2 = env_bool("BI100_FORCE_PAGED_ATTN_V2", False)
_PAGED_ATTN_DIAGNOSTICS = env_bool(
    "BI100_PAGED_ATTN_DIAGNOSTICS", False)
_USE_COREX_PAGED_KV_GATHER = (
    _corex_paged_kv_gather is not None
    and env_bool("BI100_ATTN_COREX_PAGED_GATHER", True))
_ENABLE_COREX_FUSED_PAGED_PREFILL = env_bool(
    "BI100_ATTN_COREX_FUSED_PREFILL", False)
_FUSED_PREFILL_DIAGNOSTICS = env_bool(
    "BI100_ATTN_COREX_FUSED_PREFILL_DIAGNOSTICS", False)
_USE_COREX_FUSED_PAGED_PREFILL = (
    _corex_fused_paged_prefill is not None
    and _ENABLE_COREX_FUSED_PAGED_PREFILL)
_DECODE_LOG_INTERVAL = 8192 if _PAGED_ATTN_DIAGNOSTICS else 0
_DECODE_DISPATCH_LOGGED = set()
_PREFIX_DISPATCH_LOGGED = set()
_FUSED_PREFILL_DIAGNOSTICS_LOGGED = set()
_CACHE_WRITE_LOGGED = False


def _log_corex_fused_prefill_diagnostic(stage: str, **fields) -> None:
    """Emit one privacy-safe guard snapshot per stage and worker."""
    if not _FUSED_PREFILL_DIAGNOSTICS:
        return
    key = (os.getpid(), stage)
    if key in _FUSED_PREFILL_DIAGNOSTICS_LOGGED:
        return
    details = " ".join(f"{name}={value}" for name, value in fields.items())
    print(
        "[BI100 PAGED_ATTN] fused_prefill_guard "
        f"pid={os.getpid()} rank={os.environ.get('RANK', '?')} "
        f"local_rank={os.environ.get('LOCAL_RANK', '?')} stage={stage} "
        f"{details}",
        file=sys.stderr,
        flush=True,
    )
    _FUSED_PREFILL_DIAGNOSTICS_LOGGED.add(key)


def _validate_decode_layout(
    num_seqs: int,
    seq_lens_count: int,
    block_table_rows: int,
    block_table_width: int,
    actual_max: int,
    block_size: int,
    physical_key_blocks: int,
    physical_value_blocks: int,
    num_heads: int,
    num_kv_heads: int,
) -> int:
    """Validate host-visible decode metadata before a native kernel launch."""
    if num_seqs <= 0:
        raise RuntimeError(f"decode requires num_seqs > 0, got {num_seqs}")
    if seq_lens_count != num_seqs:
        raise RuntimeError(
            f"seq_lens has {seq_lens_count} entries for {num_seqs} sequences")
    if block_table_rows < num_seqs:
        raise RuntimeError(
            f"block table has {block_table_rows} rows for {num_seqs} sequences")
    if actual_max <= 0:
        raise RuntimeError(f"decode sequence length must be > 0, got {actual_max}")
    if block_size <= 0:
        raise RuntimeError(f"KV block_size must be > 0, got {block_size}")
    if physical_key_blocks != physical_value_blocks:
        raise RuntimeError(
            "key/value cache block counts differ: "
            f"{physical_key_blocks} != {physical_value_blocks}")
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise RuntimeError(
            f"invalid GQA layout: num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}")

    required_blocks = (actual_max + block_size - 1) // block_size
    if required_blocks > block_table_width:
        raise RuntimeError(
            f"decode needs {required_blocks} blocks for seq_len={actual_max}, "
            f"but block table width is {block_table_width}")
    return required_blocks


def _strict_prefix_query_segments(
    context_len: int,
    query_len: int,
    block_size: int,
) -> List[Tuple[int, int, int]]:
    """Split a query at the strict prefix-cache checkpoint, if it crosses it."""
    if query_len <= 0:
        return []
    total_len = context_len + query_len
    strict_prefix_len = ((total_len - 1) // block_size) * block_size
    split = strict_prefix_len - context_len
    if 0 < split < query_len:
        return [(0, split, context_len),
                (split, query_len, strict_prefix_len)]
    return [(0, query_len, context_len)]


def _can_enable_corex_fused_paged_prefill_request(
    kv_cache_dtype: str,
    max_query_len: int,
    total_query_len: int,
    alibi_slopes: Optional[torch.Tensor],
    sliding_window: Optional[int],
    k_scale: float,
    v_scale: float,
    is_causal_decoder: bool,
) -> bool:
    """Check request-wide properties that are outside the native ABI."""
    return bool(
        _USE_COREX_FUSED_PAGED_PREFILL
        and is_causal_decoder
        and _PREFIX_BLOCKS_PER_TILE == 32
        and kv_cache_dtype == "auto"
        and max_query_len == total_query_len
        and alibi_slopes is None
        and sliding_window is None
        and k_scale == 1.0
        and v_scale == 1.0
    )


def _is_single_sequence_fused_prefill_metadata(
    batch_size: int,
    block_table_rows: int,
    query_start_count: int,
    query_start_first: int,
    query_start_last: int,
    seq_lens_count: int,
    seq_len: int,
    context_lens_count: int,
    context_len: int,
    total_query_len: int,
) -> bool:
    """Validate the exact single-sequence metadata used by qualification."""
    return bool(
        batch_size == 1
        and block_table_rows == 1
        and query_start_count == 2
        and query_start_first == 0
        and query_start_last == total_query_len
        and seq_lens_count == 1
        and context_lens_count == 1
        and context_len >= 0
        and seq_len == context_len + total_query_len
        and seq_len <= 262144
    )


def _can_use_corex_fused_paged_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_index: int,
    block_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gqa_ratio: int,
    block_size: int,
) -> bool:
    """Accept only the fixed M1-47 shape covered by the CoreX gate."""
    if not _USE_COREX_FUSED_PAGED_PREFILL:
        return False
    query_len = query.shape[0]
    if (query_len <= 16 or query_len > 8192
            or block_context_len < 0
            or block_context_len % 16 != 0
            or block_context_len + query_len > 262144):
        return False
    if (num_q_heads, num_kv_heads, head_dim, gqa_ratio, block_size) != (
            4, 1, 256, 4, 16):
        return False
    if prefix_key.shape[0] != 0 or prefix_value.shape[0] != 0:
        return False
    if (tuple(query.shape) != (query_len, 4, 256)
            or tuple(key.shape) != (query_len, 1, 256)
            or tuple(value.shape) != (query_len, 1, 256)):
        return False
    if (len(key_cache.shape) != 5
            or tuple(key_cache.shape[1:]) != (1, 32, 16, 8)
            or len(value_cache.shape) != 4
            or tuple(value_cache.shape[1:]) != (1, 256, 16)
            or key_cache.shape[0] != value_cache.shape[0]):
        return False
    if (len(block_tables.shape) != 2
            or block_tables.shape[0] != 1
            or seq_index < 0
            or seq_index >= block_tables.shape[0]
            or block_tables.shape[1] < block_context_len // block_size):
        return False

    half_tensors = (query, key, value, key_cache, value_cache)
    if any(tensor.dtype != torch.float16 for tensor in half_tensors):
        return False
    if block_tables.dtype != torch.int32:
        return False
    tensors = half_tensors + (block_tables,)
    if any(not tensor.is_cuda for tensor in tensors):
        return False
    if any(tensor.device != query.device for tensor in tensors):
        return False
    if any(not tensor.is_contiguous() for tensor in tensors):
        return False
    return True


def _prefix_context_tile_spans(
    block_context_len: int,
    prefix_query_len: int,
    tile_size: int,
) -> List[Tuple[int, int, int, int]]:
    """Map context tiles to block-cache and preceding-query token ranges.

    Each tuple is ``(block_start, block_end, prefix_start, prefix_end)``.
    Concatenating both ranges reconstructs one tile in the logical context.
    Keeping tiles aligned to absolute token positions makes cold segmented
    prefill use the same online-softmax partitions as a warm cached request.
    """
    if block_context_len < 0 or prefix_query_len < 0 or tile_size <= 0:
        raise ValueError("context lengths must be non-negative and tile_size > 0")
    spans = []
    total_context_len = block_context_len + prefix_query_len
    for tile_start in range(0, total_context_len, tile_size):
        tile_end = min(tile_start + tile_size, total_context_len)
        block_start = min(tile_start, block_context_len)
        block_end = min(tile_end, block_context_len)
        prefix_start = max(0, tile_start - block_context_len)
        prefix_end = max(0, tile_end - block_context_len)
        spans.append((block_start, block_end, prefix_start, prefix_end))
    return spans


@dataclass
class PagedAttentionMetadata:
    """Metadata for PagedAttention."""
    # (batch_size,). The length of sequences (entire tokens seen so far) per
    # sequence.
    seq_lens_tensor: Optional[torch.Tensor]
    # Maximum sequence length in the batch. 0 if it is prefill-only batch.
    max_decode_seq_len: int
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]


class PagedAttention:

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 80, 96, 112, 120, 128, 192, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
    ) -> None:
        global _CACHE_WRITE_LOGGED
        flat_slots = slot_mapping.flatten()
        if key.shape[0] != value.shape[0]:
            raise RuntimeError(
                f"key/value token counts differ: {key.shape[0]} != "
                f"{value.shape[0]}")
        if flat_slots.numel() != key.shape[0]:
            raise RuntimeError(
                f"slot_mapping has {flat_slots.numel()} entries for "
                f"{key.shape[0]} KV tokens")
        if key_cache.shape[0] != value_cache.shape[0]:
            raise RuntimeError(
                "key/value cache block counts differ before cache write: "
                f"{key_cache.shape[0]} != {value_cache.shape[0]}")

        if _PAGED_ATTN_DIAGNOSTICS and flat_slots.numel() > 0:
            min_slot = int(flat_slots.min().item())
            max_slot = int(flat_slots.max().item())
            max_valid_slot = key_cache.shape[0] * value_cache.shape[3] - 1
            if min_slot < -1 or max_slot > max_valid_slot:
                raise RuntimeError(
                    f"slot_mapping range [{min_slot}, {max_slot}] outside "
                    f"[-1, {max_valid_slot}]")

        if _PAGED_ATTN_DIAGNOSTICS and not _CACHE_WRITE_LOGGED:
            print(
                "[BI100 PAGED_ATTN] cache_write "
                f"pid={os.getpid()} rank={os.environ.get('RANK', '?')} "
                f"local_rank={os.environ.get('LOCAL_RANK', '?')} "
                f"key={tuple(key.shape)} value={tuple(value.shape)} "
                f"slots={tuple(flat_slots.shape)} "
                f"key_cache={tuple(key_cache.shape)} "
                f"value_cache={tuple(value_cache.shape)}",
                file=sys.stderr,
                flush=True,
            )
            _CACHE_WRITE_LOGGED = True

        ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            flat_slots,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        if _PAGED_ATTN_DIAGNOSTICS:
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                print(
                    "[BI100 PAGED_ATTN] cache_write_sync_failed "
                    f"pid={os.getpid()} error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

    @staticmethod
    def _forward_decode_pytorch(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Pure-PyTorch decode attention for long contexts (no hardware kernel).

        paged_attention_v1 hangs on BI-V100 when max_seq_len > ~32K due to
        shared memory limits. For decode, q_len=1 per sequence so no Q-tiling
        is needed — the attention weight tensor is [H, 1, seq_len] which is
        trivially small (~5 MB at 50K).

        Shapes
        ------
        query       : [num_seqs, num_heads, head_dim]
        key_cache   : [num_blocks, num_kv_heads, head_dim//x, block_size, x]
        value_cache : [num_blocks, num_kv_heads, head_dim,    block_size]
        block_tables: [num_seqs, max_blocks_per_seq]
        seq_lens    : [num_seqs]
        """
        num_seqs, num_heads, head_dim = query.shape
        num_kv_heads = key_cache.shape[1]
        block_size = value_cache.shape[3]
        gqa_ratio = num_heads // num_kv_heads
        orig_dtype = query.dtype

        output = torch.empty_like(query)

        try:
            for i in range(num_seqs):
                seq_len = int(seq_lens[i].item())
                num_blocks = (seq_len + block_size - 1) // block_size
                blk_ids = block_tables[i, :num_blocks]

                use_corex_gather = (
                    _USE_COREX_PAGED_KV_GATHER
                    and query.dtype == torch.float16
                    and key_cache.dtype == torch.float16
                    and value_cache.dtype == torch.float16
                    and block_tables.dtype == torch.int32
                    and key_cache.is_contiguous()
                    and value_cache.is_contiguous()
                    and blk_ids.is_contiguous())
                if use_corex_gather:
                    k_t, v_t = _corex_paged_kv_gather.gather(
                        key_cache, value_cache, blk_ids, seq_len)
                else:
                    # Gather K: [kv_h, head_dim, seq_len] fp32 without GQA
                    # expansion. The CoreX path above fuses these layout copies
                    # and FP16-to-FP32 conversions into one kernel.
                    k_t = (key_cache[blk_ids]
                           .permute(0, 3, 1, 2, 4)
                           .contiguous()
                           .view(-1, num_kv_heads, head_dim))[:seq_len] \
                          .permute(1, 2, 0).contiguous().float()
                    v_t = (value_cache[blk_ids]
                           .permute(0, 3, 1, 2)
                           .contiguous()
                           .view(-1, num_kv_heads, head_dim))[:seq_len] \
                          .permute(1, 0, 2).contiguous().float()

                # Reshape Q for lazy GQA: [kv_h, gqa_ratio, 1, d]
                q_grouped = (query[i].float()
                             .view(num_kv_heads, gqa_ratio, head_dim)
                             .unsqueeze(2))

                # [kv_h, gqa_ratio, 1, seq_len]
                attn_w = torch.matmul(
                    q_grouped * scale,       # [kv_h, gqa, 1, d]
                    k_t.unsqueeze(1))        # [kv_h, 1, d, seq_len]
                attn_w = torch.softmax(attn_w, dim=-1)

                # [kv_h, gqa_ratio, 1, d] → [num_heads, head_dim]
                out_i = torch.matmul(attn_w, v_t.unsqueeze(1))
                output[i] = out_i.view(num_heads, head_dim).to(orig_dtype)

        except Exception as e:
            print(f"[decode_pytorch ERROR] {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            raise

        return output

    # paged_attention_v1 on BI-V100 fails for long contexts.
    # Route on actual sequence length (seq_lens.max()), not the max_seq_len
    # parameter which is inflated to max_model_len in CUDA graph mode.
    _PYTORCH_DECODE_THRESHOLD = _PYTORCH_DECODE_THRESHOLD
    _FORCE_PAGED_ATTN_V2 = _FORCE_PAGED_ATTN_V2

    @staticmethod
    def _should_use_paged_attention_v1(
        max_seq_len: int,
        max_num_partitions: int,
        num_seqs: int,
        num_heads: int,
    ) -> bool:
        if PagedAttention._FORCE_PAGED_ATTN_V2:
            return False
        # Keep the stable BI100 default: V1 is used unless long-context decode
        # has already routed to the PyTorch fallback above.
        return True

    @staticmethod
    def _validate_prefix_block_table(
        seq_index: int,
        num_ctx_blocks: int,
        block_table_width: int,
        ctx_len: int,
    ) -> int:
        if num_ctx_blocks <= block_table_width:
            return num_ctx_blocks
        msg = (
            f"seq {seq_index}: num_ctx_blocks={num_ctx_blocks} "
            f"> block_tables.shape[1]={block_table_width}, "
            f"ctx_len={ctx_len}. Block table is undersized; "
            "refusing to truncate context because attention would be incorrect.")
        if env_bool("BI100_ALLOW_PREFIX_GUARD_CAP", False):
            print(
                "[paged_attn RISK] BI100_ALLOW_PREFIX_GUARD_CAP=1; "
                f"{msg} Debug cap is enabled and may corrupt output.",
                file=sys.stderr,
                flush=True)
            return block_table_width
        raise RuntimeError(msg)

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        kv_cache_dtype: str,
        head_mapping: torch.Tensor,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> torch.Tensor:
        actual_max = int(seq_lens.max().item()) if seq_lens.numel() > 0 else max_seq_len
        block_size = value_cache.shape[3]
        num_seqs, num_heads, head_size = query.shape
        if key_cache.shape[1] != value_cache.shape[1]:
            raise RuntimeError(
                "key/value cache KV-head counts differ: "
                f"{key_cache.shape[1]} != {value_cache.shape[1]}")
        if head_mapping.numel() != num_heads:
            raise RuntimeError(
                f"head_mapping has {head_mapping.numel()} entries for "
                f"{num_heads} query heads")
        required_blocks = _validate_decode_layout(
            num_seqs=num_seqs,
            seq_lens_count=seq_lens.numel(),
            block_table_rows=block_tables.shape[0],
            block_table_width=block_tables.shape[1],
            actual_max=actual_max,
            block_size=block_size,
            physical_key_blocks=key_cache.shape[0],
            physical_value_blocks=value_cache.shape[0],
            num_heads=num_heads,
            num_kv_heads=key_cache.shape[1],
        )
        if actual_max > max_seq_len:
            raise RuntimeError(
                f"actual decode length {actual_max} exceeds max_seq_len "
                f"{max_seq_len}")

        if actual_max > PagedAttention._PYTORCH_DECODE_THRESHOLD:
            path = ("pytorch_corex_gather" if _USE_COREX_PAGED_KV_GATHER
                    else "pytorch")
        else:
            path = "native_v1"
        log_key = (path, None)
        if (_DECODE_LOG_INTERVAL > 0 and
                actual_max % _DECODE_LOG_INTERVAL == 0):
            log_key = (path, actual_max)
        if log_key not in _DECODE_DISPATCH_LOGGED:
            print(
                "[BI100 PAGED_ATTN] decode_dispatch "
                f"pid={os.getpid()} rank={os.environ.get('RANK', '?')} "
                f"local_rank={os.environ.get('LOCAL_RANK', '?')} "
                f"path={path} actual_max={actual_max} "
                f"max_seq_len={max_seq_len} query={tuple(query.shape)} "
                f"key_cache={tuple(key_cache.shape)} "
                f"value_cache={tuple(value_cache.shape)} "
                f"block_tables={tuple(block_tables.shape)} "
                f"required_blocks={required_blocks} "
                f"threshold={PagedAttention._PYTORCH_DECODE_THRESHOLD}",
                file=sys.stderr,
                flush=True,
            )
            _DECODE_DISPATCH_LOGGED.add(log_key)

        if _PAGED_ATTN_DIAGNOSTICS:
            for seq_index in range(num_seqs):
                seq_len = int(seq_lens[seq_index].item())
                if seq_len <= 0:
                    raise RuntimeError(
                        f"seq {seq_index}: decode length must be > 0, got {seq_len}")
                seq_blocks = (seq_len + block_size - 1) // block_size
                block_ids = block_tables[seq_index, :seq_blocks]
                min_block = int(block_ids.min().item())
                max_block = int(block_ids.max().item())
                if min_block < 0 or max_block >= key_cache.shape[0]:
                    raise RuntimeError(
                        f"seq {seq_index}: physical block range "
                        f"[{min_block}, {max_block}] outside "
                        f"[0, {key_cache.shape[0] - 1}]")

        if actual_max > PagedAttention._PYTORCH_DECODE_THRESHOLD:
            with bi100_timer("paged_attn.decode_pytorch"):
                return PagedAttention._forward_decode_pytorch(
                    query, key_cache, value_cache, block_tables, seq_lens,
                    scale)

        if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
            # use blocksparse paged attention
            block_size = value_cache.size(-1)
            assert (blocksparse_block_size > 0 and
                    blocksparse_block_size % block_size == 0), \
                (f"{blocksparse_block_size=} needs to be a multiple of"
                 f"{block_size=} used in block_tables.")

        output = torch.empty_like(query)
        max_num_partitions = ((max_seq_len + _PARTITION_SIZE - 1) //
                              _PARTITION_SIZE)
        # NOTE(woosuk): We use a simple heuristic to decide whether to use
        # PagedAttention V1 or V2. If the number of partitions is 1, we use
        # V1 to avoid the overhead of reduction. Also, if the number of
        # sequences or heads is large, we use V1 since there is enough work
        # to parallelize.
        # TODO(woosuk): Tune this heuristic.
        # For context len > 8192, use V2 kernel to avoid shared memory shortage.
        use_v1 = PagedAttention._should_use_paged_attention_v1(
            max_seq_len, max_num_partitions, num_seqs, num_heads)
        if use_v1:
            # Run PagedAttention V1.
            ops.paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                head_mapping,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
            )
        else:
            # Run PagedAttention V2.
            assert _PARTITION_SIZE % block_size == 0
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            ops.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                head_mapping,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
                tp_rank,
                blocksparse_local_blocks,
                blocksparse_vert_stride,
                blocksparse_block_size,
                blocksparse_head_sliding_step,
            )
        return output

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache_dtype: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        context_lens: torch.Tensor,
        max_query_len: int,
        alibi_slopes: Optional[torch.Tensor],
        sliding_window: Optional[int],
        k_scale: float,
        v_scale: float,
        is_causal_decoder: bool = False,
    ) -> torch.Tensor:
        # NOTE: The Triton context_attention_fwd kernel hangs on Iluvatar
        # BI-V100 hardware (same class of issue as cudnnFlashAttnForward).
        # Use a pure-PyTorch fallback that reads the paged KV cache directly.
        fused_request_eligible = (
            _can_enable_corex_fused_paged_prefill_request(
                kv_cache_dtype, max_query_len, query.shape[0], alibi_slopes,
                sliding_window, k_scale, v_scale, is_causal_decoder))
        _log_corex_fused_prefill_diagnostic(
            "request",
            eligible=fused_request_eligible,
            use_native=_USE_COREX_FUSED_PAGED_PREFILL,
            causal=is_causal_decoder,
            kv_cache_dtype=kv_cache_dtype,
            max_query_len=max_query_len,
            total_query_len=query.shape[0],
            tile_blocks=_PREFIX_BLOCKS_PER_TILE,
            alibi_none=alibi_slopes is None,
            sliding_window=sliding_window,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        return PagedAttention._forward_prefix_pytorch(
            query, key, value,
            key_cache, value_cache,
            block_tables, query_start_loc,
            seq_lens_tensor, context_lens,
            fused_request_eligible=fused_request_eligible,
        )

    @staticmethod
    def _forward_prefix_pytorch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        context_lens: torch.Tensor,
        fused_request_eligible: bool = False,
    ) -> torch.Tensor:
        """Pure-PyTorch prefix-attention with K-tiling (Flash-Attention online softmax).

        Memory complexity: O(q_len), independent of kv_len.
        Query segments end at the same strict block boundary used by prefix
        caching. This keeps online-softmax reduction partitions identical when
        an otherwise equivalent request reuses that prefix.

        Algorithm: Flash Attention online softmax.
        Q is reshaped once to [kv_h, gqa, q_len, d] (16 MB) and held for all
        K-tiles.  For each tile a running (m, l, o) accumulator is updated —
        the [q_len × kv_len] attention matrix is NEVER materialised in full.

        Tile budget (kv_h=1, gqa=4, q_len=4096, tile=256 tokens):
            q_seq   [1, 4, 4096, 256] fp32  16 MB  (held all tiles)
            o_acc   same shape               16 MB  (held all tiles)
            s       same shape               16 MB  (per tile, freed before exp_s)
            exp_s   same shape               16 MB  (per tile, brief overlap with s)
            Peak ≈ 64 MB  (s and exp_s briefly coexist during update).

        Shapes
        ------
        query          : [total_q_tokens, num_q_heads,  head_dim]
        key            : [total_q_tokens, num_kv_heads, head_dim]
        value          : [total_q_tokens, num_kv_heads, head_dim]
        key_cache      : [num_blocks, num_kv_heads, head_dim//x, block_size, x]
        value_cache    : [num_blocks, num_kv_heads, head_dim,    block_size]
        block_tables   : [batch_size, max_blocks_per_seq]
        query_start_loc: [batch_size + 1]
        seq_lens_tensor: [batch_size]  total length (context + query)
        context_lens   : [batch_size]  tokens already in KV cache
        """
        try:
            profile_name = "paged_attn.prefix_pytorch"
            # Paged-block tiles for context phase.
            # tile_sz = _BLOCKS_PER_TILE × block_size  (e.g. 16×16 = 256 tokens).
            # Score tensor [kv_h, gqa, q_len, tile_sz] fp32 = 16 MB per tile.
            # Same tile size reused for the current-chunk phase.
            _BLOCKS_PER_TILE = _PREFIX_BLOCKS_PER_TILE

            batch_size   = seq_lens_tensor.shape[0]
            num_q_heads  = query.shape[1]
            num_kv_heads = key_cache.shape[1]
            head_dim     = query.shape[2]
            gqa_ratio    = num_q_heads // num_kv_heads
            block_size   = value_cache.shape[3]
            tile_sz      = _BLOCKS_PER_TILE * block_size
            scale        = head_dim ** -0.5
            orig_dtype   = query.dtype
            output       = torch.empty_like(query)

            if fused_request_eligible:
                query_start_count = query_start_loc.numel()
                seq_lens_count = seq_lens_tensor.numel()
                context_lens_count = context_lens.numel()
                query_start_first = (
                    int(query_start_loc[0].item())
                    if query_start_count == 2 else -1)
                query_start_last = (
                    int(query_start_loc[1].item())
                    if query_start_count == 2 else -1)
                seq_len = (
                    int(seq_lens_tensor[0].item())
                    if seq_lens_count == 1 else -1)
                context_len = (
                    int(context_lens[0].item())
                    if context_lens_count == 1 else -1)
                metadata_eligible = (
                    _is_single_sequence_fused_prefill_metadata(
                        batch_size=batch_size,
                        block_table_rows=block_tables.shape[0],
                        query_start_count=query_start_count,
                        query_start_first=query_start_first,
                        query_start_last=query_start_last,
                        seq_lens_count=seq_lens_count,
                        seq_len=seq_len,
                        context_lens_count=context_lens_count,
                        context_len=context_len,
                        total_query_len=query.shape[0],
                    ))
                _log_corex_fused_prefill_diagnostic(
                    "metadata",
                    eligible=metadata_eligible,
                    batch_size=batch_size,
                    block_table_rows=block_tables.shape[0],
                    query_start_count=query_start_count,
                    query_start_first=query_start_first,
                    query_start_last=query_start_last,
                    seq_lens_count=seq_lens_count,
                    seq_len=seq_len,
                    context_lens_count=context_lens_count,
                    context_len=context_len,
                    total_query_len=query.shape[0],
                )
                fused_request_eligible = metadata_eligible

            for i in range(batch_size):
                ctx_len = int(context_lens[i].item())
                q_start = int(query_start_loc[i].item())
                q_end   = int(query_start_loc[i + 1].item())
                q_len   = q_end - q_start

                for seg_start, seg_end, _seg_ctx_len in (
                        _strict_prefix_query_segments(
                            ctx_len, q_len, block_size)):
                    absolute_start = q_start + seg_start
                    absolute_end = q_start + seg_end
                    with bi100_timer(profile_name):
                        output[absolute_start:absolute_end] = (
                            PagedAttention._forward_prefix_segment_pytorch(
                                query[absolute_start:absolute_end],
                                key[absolute_start:absolute_end],
                                value[absolute_start:absolute_end],
                                key[q_start:absolute_start],
                                value[q_start:absolute_start],
                                key_cache,
                                value_cache,
                                block_tables,
                                i,
                                ctx_len,
                                num_q_heads,
                                num_kv_heads,
                                head_dim,
                                gqa_ratio,
                                block_size,
                                tile_sz,
                                scale,
                                orig_dtype,
                                fused_request_eligible,
                            ))

        except Exception as e:
            print(f"[paged_attn ERROR] {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            raise
        return output

    @staticmethod
    def _forward_prefix_segment_pytorch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_index: int,
        block_context_len: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        gqa_ratio: int,
        block_size: int,
        tile_sz: int,
        scale: float,
        orig_dtype,
        fused_request_eligible: bool = False,
    ) -> torch.Tensor:
        """Run online-softmax attention for one strict-prefix query segment."""
        q_len = query.shape[0]
        segment_eligible = (
            fused_request_eligible
            and _can_use_corex_fused_paged_prefill(
                query, key, value, prefix_key, prefix_value,
                key_cache, value_cache, block_tables, seq_index,
                block_context_len, num_q_heads, num_kv_heads, head_dim,
                gqa_ratio, block_size))
        _log_corex_fused_prefill_diagnostic(
            "segment",
            eligible=segment_eligible,
            request_eligible=fused_request_eligible,
            query_shape=tuple(query.shape),
            key_shape=tuple(key.shape),
            value_shape=tuple(value.shape),
            prefix_key_shape=tuple(prefix_key.shape),
            prefix_value_shape=tuple(prefix_value.shape),
            key_cache_shape=tuple(key_cache.shape),
            value_cache_shape=tuple(value_cache.shape),
            block_table_shape=tuple(block_tables.shape),
            context_len=block_context_len,
            seq_index=seq_index,
            q_dtype=query.dtype,
            block_table_dtype=block_tables.dtype,
            query_cuda=query.is_cuda,
            query_contiguous=query.is_contiguous(),
            key_contiguous=key.is_contiguous(),
            value_contiguous=value.is_contiguous(),
            block_table_contiguous=block_tables.is_contiguous(),
            heads=f"{num_q_heads}/{num_kv_heads}/{head_dim}",
            gqa_ratio=gqa_ratio,
            block_size=block_size,
        )
        if segment_eligible:
            required_blocks = block_context_len // block_size
            active_block_table = block_tables[
                seq_index, :required_blocks].contiguous()
            fused_result = _corex_fused_paged_prefill.forward(
                query, key, value, key_cache, value_cache,
                active_block_table, block_context_len, scale)
            if not isinstance(fused_result, (list, tuple)) or len(fused_result) != 2:
                raise RuntimeError(
                    "corex fused paged-prefill returned an invalid result")
            fused_output = fused_result[0]
            if (tuple(fused_output.shape) != tuple(query.shape)
                    or fused_output.dtype != query.dtype
                    or fused_output.device != query.device):
                raise RuntimeError(
                    "corex fused paged-prefill returned an invalid output")
            log_key = "corex_split4"
            if log_key not in _PREFIX_DISPATCH_LOGGED:
                print(
                    "[BI100 PAGED_ATTN] prefix_dispatch "
                    f"pid={os.getpid()} rank={os.environ.get('RANK', '?')} "
                    f"local_rank={os.environ.get('LOCAL_RANK', '?')} "
                    f"path={log_key} context_len={block_context_len} "
                    f"query_len={q_len} required_blocks={required_blocks}",
                    file=sys.stderr,
                    flush=True,
                )
                _PREFIX_DISPATCH_LOGGED.add(log_key)
            return fused_output

        dev = query.device
        q_seq = (query.permute(1, 0, 2)
                      .float()
                      .view(num_kv_heads, gqa_ratio, q_len, head_dim)
                      .mul(scale))
        m = torch.full((num_kv_heads, gqa_ratio, q_len),
                       float('-inf'), dtype=torch.float32, device=dev)
        l = torch.zeros_like(m)
        o = torch.zeros((num_kv_heads, gqa_ratio, q_len, head_dim),
                        dtype=torch.float32, device=dev)

        if block_context_len > 0:
            num_ctx_blocks = (block_context_len + block_size - 1) // block_size
            num_ctx_blocks = PagedAttention._validate_prefix_block_table(
                seq_index, num_ctx_blocks, block_tables.shape[1],
                block_context_len)

        for block_start, block_end, prefix_start, prefix_end in (
                _prefix_context_tile_spans(
                    block_context_len, prefix_key.shape[0], tile_sz)):
            k_parts = []
            v_parts = []
            if block_end > block_start:
                first_block = block_start // block_size
                last_block = (block_end + block_size - 1) // block_size
                blk_ids = block_tables[seq_index, first_block:last_block]
                k_blocks = (key_cache[blk_ids]
                            .permute(0, 3, 1, 2, 4)
                            .contiguous()
                            .view(-1, num_kv_heads, head_dim))
                v_blocks = (value_cache[blk_ids]
                            .permute(0, 3, 1, 2)
                            .contiguous()
                            .view(-1, num_kv_heads, head_dim))
                offset = block_start - first_block * block_size
                length = block_end - block_start
                k_parts.append(k_blocks[offset:offset + length])
                v_parts.append(v_blocks[offset:offset + length])
            if prefix_end > prefix_start:
                k_parts.append(prefix_key[prefix_start:prefix_end])
                v_parts.append(prefix_value[prefix_start:prefix_end])
            k_context = (k_parts[0] if len(k_parts) == 1
                         else torch.cat(k_parts, dim=0))
            v_context = (v_parts[0] if len(v_parts) == 1
                         else torch.cat(v_parts, dim=0))
            k_t = (k_context.permute(1, 0, 2)
                   .unsqueeze(1).transpose(-1, -2).float())
            v_t = v_context.permute(1, 0, 2).unsqueeze(1).float()
            PagedAttention._update_online_softmax(q_seq, k_t, v_t, m, l, o)

        for key_start in range(0, q_len, tile_sz):
            key_end = min(key_start + tile_sz, q_len)
            k_t = (key[key_start:key_end].permute(1, 0, 2)
                   .unsqueeze(1).transpose(-1, -2).float())
            v_t = (value[key_start:key_end].permute(1, 0, 2)
                   .unsqueeze(1).float())
            scores = torch.matmul(q_seq, k_t)
            del k_t
            key_positions = torch.arange(key_start, key_end, device=dev)
            query_positions = torch.arange(q_len, device=dev)
            mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            del mask, key_positions, query_positions
            PagedAttention._update_online_softmax_from_scores(
                scores, v_t, m, l, o)

        o.div_(l.unsqueeze(-1))
        return (o.view(num_q_heads, q_len, head_dim)
                .permute(1, 0, 2).to(orig_dtype))

    @staticmethod
    def _update_online_softmax(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        running_max: torch.Tensor,
        running_sum: torch.Tensor,
        running_output: torch.Tensor,
    ) -> None:
        scores = torch.matmul(query, key)
        PagedAttention._update_online_softmax_from_scores(
            scores, value, running_max, running_sum, running_output)

    @staticmethod
    def _update_online_softmax_from_scores(
        scores: torch.Tensor,
        value: torch.Tensor,
        running_max: torch.Tensor,
        running_sum: torch.Tensor,
        running_output: torch.Tensor,
    ) -> None:
        block_max = scores.amax(dim=-1)
        new_max = torch.maximum(running_max, block_max)
        exp_scores = scores - new_max.unsqueeze(-1)
        del scores
        exp_scores.exp_()
        correction = torch.exp(running_max - new_max)
        running_max.copy_(new_max)
        running_sum.mul_(correction).add_(exp_scores.sum(dim=-1))
        running_output.mul_(correction.unsqueeze(-1)).add_(
            torch.matmul(exp_scores, value))

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)
