"""Install the BI100 TP=4 sharded greedy argmax fast path."""

import ast

from patch_utils import package_root, replace_once


VLLM_ROOT = package_root("vllm")
LOGITS_PROCESSOR_PATH = (
    VLLM_ROOT / "model_executor" / "layers" / "logits_processor.py")
SAMPLER_PATH = VLLM_ROOT / "model_executor" / "layers" / "sampler.py"


_LOGITS_IMPORTS_OLD = """\
import inspect
from typing import Optional

import torch
import torch.nn as nn

from vllm.distributed import (tensor_model_parallel_all_gather,
                              tensor_model_parallel_gather)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.platforms import current_platform
"""

_LOGITS_IMPORTS_NEW = """\
import inspect
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from vllm.bi100_env import env_bool
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_gather)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingType


@dataclass
class ShardedGreedyResult:
    token_ids: torch.Tensor
"""

_LOGITS_FORWARD_OLD = """\
    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.logits_as_input:
            logits = hidden_states
        else:
            hidden_states = _prune_hidden_states(hidden_states,
                                                 sampling_metadata)

            # Get the logits for the next tokens.
            if hidden_states.shape[0] > 0:
                logits = self._get_logits(hidden_states, lm_head, embedding_bias)
            else:
                logits = torch.empty([0, lm_head.weight.shape[0]], device=hidden_states.device, dtype=hidden_states.dtype)
"""

_LOGITS_FORWARD_NEW = """\
    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[Union[torch.Tensor, ShardedGreedyResult]]:
        if self.logits_as_input:
            logits = hidden_states
        else:
            hidden_states = _prune_hidden_states(hidden_states,
                                                 sampling_metadata)

            if self._can_use_bi100_sharded_greedy(
                    hidden_states, sampling_metadata, embedding_bias):
                return self._bi100_sharded_greedy(
                    hidden_states, lm_head, sampling_metadata)

            # Get the logits for the next tokens.
            if hidden_states.shape[0] > 0:
                logits = self._get_logits(hidden_states, lm_head, embedding_bias)
            else:
                logits = torch.empty([0, lm_head.weight.shape[0]], device=hidden_states.device, dtype=hidden_states.dtype)
"""

_LOGITS_METHOD_ANCHOR = """\
    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
"""

_LOGITS_METHODS_NEW = """\
    def _can_use_bi100_sharded_greedy(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        embedding_bias: Optional[torch.Tensor],
    ) -> bool:
        if not env_bool("BI100_SHARDED_GREEDY_ARGMAX", False):
            return False
        if not env_bool("ENABLE_CUSTOM_IPC", False):
            return False
        if (self.logits_as_input or not self.use_gather
                or self.soft_cap is not None or self.scale != 1.0
                or embedding_bias is not None
                or self.org_vocab_size != self.vocab_size):
            return False

        world_size = get_tensor_model_parallel_world_size()
        if (world_size != 4 or self.org_vocab_size >= 2**24
                or self.org_vocab_size % world_size != 0):
            return False
        if sampling_metadata.seq_groups is None:
            return False
        if (sampling_metadata.num_prompts != 0
                or sampling_metadata.skip_sampler_cpu_output
                or sampling_metadata.reuse_sampling_tensors):
            return False

        seq_groups = sampling_metadata.seq_groups
        if not seq_groups or hidden_states.shape[0] != len(seq_groups):
            return False
        for row, seq_group in enumerate(seq_groups):
            params = seq_group.sampling_params
            if (seq_group.is_prompt or not seq_group.do_sample
                    or len(seq_group.seq_ids) != 1
                    or seq_group.sample_indices != [row]
                    or seq_group.prompt_logprob_indices):
                return False
            if (params.sampling_type != SamplingType.GREEDY
                    or params.n != 1
                    or params.best_of not in (None, 1)
                    or params.presence_penalty != 0.0
                    or params.frequency_penalty != 0.0
                    or params.repetition_penalty != 1.0
                    or params.min_tokens != 0
                    or params.logprobs is not None
                    or params.prompt_logprobs is not None
                    or params.logits_processors
                    or params.guided_decoding is not None
                    or params.logit_bias is not None
                    or params.allowed_token_ids is not None):
                return False
        return True

    def _bi100_sharded_greedy(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        sampling_metadata: SamplingMetadata,
    ) -> ShardedGreedyResult:
        local_logits = lm_head.linear_method.apply(
            lm_head, hidden_states, bias=None)
        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        local_vocab_size = self.org_vocab_size // world_size
        if local_logits.shape[-1] != local_vocab_size:
            raise RuntimeError(
                "BI100 sharded greedy requires an evenly sharded vocabulary")

        local_values, local_ids = torch.max(local_logits, dim=-1)
        global_ids = local_ids + rank * local_vocab_size

        # Each rank owns one zero-initialized row. SUM all-reduce therefore
        # acts as a tiny all-gather while using the proven IxFormer IPC path.
        candidates = torch.zeros(
            (local_logits.shape[0], world_size, 2),
            dtype=torch.float32,
            device=local_logits.device)
        candidates[:, rank, 0] = local_values.float()
        candidates[:, rank, 1] = global_ids.float()
        tensor_model_parallel_all_reduce(candidates)

        values = candidates[..., 0]
        token_ids = candidates[..., 1].long()
        invalid_id = self.org_vocab_size
        nan_mask = torch.isnan(values)
        first_nan = token_ids.masked_fill(~nan_mask, invalid_id).amin(dim=-1)
        finite_values = values.masked_fill(nan_mask, -float("inf"))
        max_values = finite_values.amax(dim=-1, keepdim=True)
        first_max = token_ids.masked_fill(
            finite_values != max_values, invalid_id).amin(dim=-1)
        selected = torch.where(nan_mask.any(dim=-1), first_nan, first_max)
        return ShardedGreedyResult(selected)

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
"""

_SAMPLER_IMPORT_OLD = """\
from vllm.model_executor.sampling_metadata import (SamplingMetadata,
                                                   SamplingTensors,
                                                   SequenceGroupToSample)
"""

_SAMPLER_IMPORT_NEW = """\
from vllm.model_executor.layers.logits_processor import ShardedGreedyResult
from vllm.model_executor.sampling_metadata import (SamplingMetadata,
                                                   SamplingTensors,
                                                   SequenceGroupToSample)
"""

_SAMPLER_FORWARD_OLD = """\
    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
"""

_SAMPLER_FORWARD_NEW = """\
    def forward(
        self,
        logits: Union[torch.Tensor, ShardedGreedyResult],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
"""

_SAMPLER_BODY_OLD = """\
        assert logits is not None
        _, vocab_size = logits.shape

        # Prepare sampling tensors with pinned memory to avoid blocking.
"""

_SAMPLER_BODY_NEW = """\
        assert logits is not None
        if isinstance(logits, ShardedGreedyResult):
            if (self.include_gpu_probs_tensor
                    or self._should_modify_greedy_probs_inplace
                    or sampling_metadata.skip_sampler_cpu_output
                    or sampling_metadata.reuse_sampling_tensors):
                raise RuntimeError(
                    "unsupported sampler mode for BI100 sharded greedy")
            sample_results = _greedy_sample(
                sampling_metadata.seq_groups, logits.token_ids)
            token_ids = logits.token_ids.tolist()
            prompt_logprobs = [None] * len(sample_results)
            sample_logprobs = [
                [{token_id: Logprob(inf)}] for token_id in token_ids
            ]
            return _build_sampler_output(
                sample_results,
                sampling_metadata,
                prompt_logprobs,
                sample_logprobs,
                on_device_tensors=None,
                skip_sampler_cpu_output=False)

        _, vocab_size = logits.shape

        # Prepare sampling tensors with pinned memory to avoid blocking.
"""


def patch_logits_processor(path=LOGITS_PROCESSOR_PATH):
    replace_once(
        path,
        _LOGITS_IMPORTS_OLD,
        _LOGITS_IMPORTS_NEW,
        already_contains="class ShardedGreedyResult:")
    replace_once(
        path,
        _LOGITS_FORWARD_OLD,
        _LOGITS_FORWARD_NEW,
        already_contains="self._can_use_bi100_sharded_greedy(")
    replace_once(
        path,
        _LOGITS_METHOD_ANCHOR,
        _LOGITS_METHODS_NEW,
        already_contains="def _bi100_sharded_greedy(")
    ast.parse(path.read_text())


def patch_sampler(path=SAMPLER_PATH):
    replace_once(
        path,
        _SAMPLER_IMPORT_OLD,
        _SAMPLER_IMPORT_NEW,
        already_contains="import ShardedGreedyResult")
    replace_once(
        path,
        _SAMPLER_FORWARD_OLD,
        _SAMPLER_FORWARD_NEW,
        already_contains="Union[torch.Tensor, ShardedGreedyResult]")
    replace_once(
        path,
        _SAMPLER_BODY_OLD,
        _SAMPLER_BODY_NEW,
        already_contains="unsupported sampler mode for BI100 sharded greedy")
    ast.parse(path.read_text())


def main():
    print("=== patch_sharded_greedy_argmax ===")
    print(f"Logits processor: {LOGITS_PROCESSOR_PATH}")
    patch_logits_processor()
    print(f"Sampler: {SAMPLER_PATH}")
    patch_sampler()
    print("Done.")


if __name__ == "__main__":
    main()
