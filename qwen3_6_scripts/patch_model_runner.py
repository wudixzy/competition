"""Patch vLLM 0.6.3 prefix-cache and MRoPE chunk alignment bugs."""

from __future__ import annotations

import pathlib

from patch_utils import package_root, replace_once


HELPER_ANCHOR = """\
logger = init_logger(__name__)

LORA_WARMUP_RANK = 8"""

HELPER_REPLACEMENT = """\
logger = init_logger(__name__)


def _slice_mrope_positions(positions, start, stop, expected_len):
    if positions is None or len(positions) != 3:
        raise RuntimeError("MRoPE positions must contain three axes")
    sliced = [axis[start:stop] for axis in positions]
    lengths = [len(axis) for axis in sliced]
    if lengths != [expected_len] * 3:
        raise RuntimeError(
            "MRoPE/input token length mismatch after chunk alignment: "
            f"positions={lengths}, input_tokens={expected_len}, "
            f"slice=({start}, {stop})")
    return sliced


LORA_WARMUP_RANK = 8"""

PREFIX_PAST_ANCHOR = """\
        if prefix_cache_len <= context_len:
            # We already passed the cache hit region,
            # so do normal computation.
            pass"""

PREFIX_PAST_REPLACEMENT = """\
        if prefix_cache_len <= context_len:
            # We already passed the cache hit region,
            # so do normal computation.
            # Must clear prefix_cache_hit so _add_seq_group uses the full
            # block_tables (prefix + previous-chunk blocks) instead of only
            # computed_block_nums (prefix only).  Without this, block_tables
            # passed to _forward_prefix_pytorch is too narrow for context_len,
            # causing an empty blk_ids slice and a zero-dim amax() crash.
            inter_data.prefix_cache_hit = False"""

PARTIAL_HIT_ANCHOR = """\
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][uncomputed_start:]
            context_len = prefix_cache_len

            inter_data.context_lens[seq_idx] = context_len
            inter_data.query_lens[
                seq_idx] = inter_data.seq_lens[seq_idx] - context_len"""

PARTIAL_HIT_REPLACEMENT = """\
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][uncomputed_start:]
            context_len = prefix_cache_len

            inter_data.context_lens[seq_idx] = context_len
            inter_data.query_lens[
                seq_idx] = inter_data.seq_lens[seq_idx] - context_len
            if inter_data.mrope_input_positions is not None:
                positions = inter_data.mrope_input_positions[seq_idx]
                if positions is not None:
                    inter_data.mrope_input_positions[seq_idx] = \\
                        _slice_mrope_positions(
                            positions, uncomputed_start, None,
                            inter_data.query_lens[seq_idx])"""

FULL_HIT_ANCHOR = """\
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][-1:]
            inter_data.query_lens[seq_idx] = 1
            inter_data.context_lens[seq_idx] = inter_data.seq_lens[seq_idx] - 1"""

FULL_HIT_REPLACEMENT = """\
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][-1:]
            inter_data.query_lens[seq_idx] = 1
            inter_data.context_lens[seq_idx] = inter_data.seq_lens[seq_idx] - 1
            if inter_data.mrope_input_positions is not None:
                positions = inter_data.mrope_input_positions[seq_idx]
                if positions is not None:
                    inter_data.mrope_input_positions[seq_idx] = \\
                        _slice_mrope_positions(positions, -1, None, 1)"""

MULTIMODAL_MROPE_ANCHOR = """\
                mrope_input_positions, mrope_position_delta = \\
                    MRotaryEmbedding.get_input_positions(
                        token_ids,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        image_token_id=hf_config.image_token_id,
                        video_token_id=hf_config.video_token_id,
                        vision_start_token_id=hf_config.vision_start_token_id,
                        vision_end_token_id=hf_config.vision_end_token_id,
                        spatial_merge_size=hf_config.vision_config.
                        spatial_merge_size,
                        context_len=inter_data.context_lens[seq_idx],
                    )

                seq_data.mrope_position_delta = mrope_position_delta
                inter_data.mrope_input_positions[
                    seq_idx] = mrope_input_positions"""

MULTIMODAL_MROPE_REPLACEMENT = """\
                # vLLM 0.6.3 returns positions through the end of token_ids,
                # while chunked prefill sends only [context_len:seq_len].
                # Compute the full MRoPE map once so the delta remains tied to
                # the complete request, then select exactly the physical query.
                mrope_input_positions, mrope_position_delta = \\
                    MRotaryEmbedding.get_input_positions(
                        token_ids,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        image_token_id=hf_config.image_token_id,
                        video_token_id=hf_config.video_token_id,
                        vision_start_token_id=hf_config.vision_start_token_id,
                        vision_end_token_id=hf_config.vision_end_token_id,
                        spatial_merge_size=hf_config.vision_config.
                        spatial_merge_size,
                        context_len=0,
                    )
                mrope_input_positions = _slice_mrope_positions(
                    mrope_input_positions,
                    inter_data.context_lens[seq_idx],
                    inter_data.seq_lens[seq_idx],
                    len(inter_data.input_tokens[seq_idx]))

                seq_data.mrope_position_delta = mrope_position_delta
                inter_data.mrope_input_positions[
                    seq_idx] = mrope_input_positions"""


def patch_model_runner(model_runner: pathlib.Path) -> None:
    replace_once(
        model_runner,
        HELPER_ANCHOR,
        HELPER_REPLACEMENT,
        required=True,
        already_contains="def _slice_mrope_positions(",
    )
    replace_once(
        model_runner,
        PREFIX_PAST_ANCHOR,
        PREFIX_PAST_REPLACEMENT,
        required=True,
        already_contains="Must clear prefix_cache_hit so _add_seq_group",
    )
    replace_once(
        model_runner,
        PARTIAL_HIT_ANCHOR,
        PARTIAL_HIT_REPLACEMENT,
        required=True,
        already_contains="positions, uncomputed_start, None,",
    )
    replace_once(
        model_runner,
        FULL_HIT_ANCHOR,
        FULL_HIT_REPLACEMENT,
        required=True,
        already_contains="_slice_mrope_positions(positions, -1, None, 1)",
    )
    replace_once(
        model_runner,
        MULTIMODAL_MROPE_ANCHOR,
        MULTIMODAL_MROPE_REPLACEMENT,
        required=True,
        already_contains="Compute the full MRoPE map once",
    )


if __name__ == "__main__":
    patch_model_runner(package_root("vllm") / "worker" / "model_runner.py")
